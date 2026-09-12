"""Startup must survive a transient dependency failure.

Two paths are covered: the initial connect (Hypha, Ray client), which used to
exit the worker on the first refusal, and startup applications, where one
application's failure used to abort the whole worker.
"""

from types import SimpleNamespace

import pytest

from bioengine.apps import manager as manager_module
from bioengine.apps.manager import AppsManager
from bioengine.utils import connect_with_retry, is_transient_connect_error
from bioengine.utils import network as network_module
from bioengine.worker import worker as worker_module
from bioengine.worker.worker import BioEngineWorker


class _RecordingLogger:
    def __init__(self):
        self.messages = []

    def _record(self, message):
        self.messages.append(message)

    info = warning = error = debug = _record


def test_connection_failures_are_transient_but_auth_failures_are_not():
    assert is_transient_connect_error(
        ConnectionRefusedError(111, "Connect call failed ('10.43.210.131', 9520)")
    )
    assert is_transient_connect_error(
        Exception("server rejected WebSocket connection: HTTP 503")
    )
    # The checks the worker must still fail fast on.
    assert not is_transient_connect_error(
        ValueError("Provided token does not have admin permissions.")
    )
    assert not is_transient_connect_error(
        ValueError("Workspace mismatch: a (local) vs b (server)")
    )


async def test_retries_a_refused_connection_until_it_succeeds():
    attempts = []

    async def connect():
        attempts.append(None)
        if len(attempts) < 3:
            raise ConnectionRefusedError(111, "Connect call failed")
        return "connected"

    result = await connect_with_retry(
        connect,
        description="Connection to Hypha server",
        logger=_RecordingLogger(),
        total_seconds=5.0,
        initial_delay=0.01,
        max_delay=0.01,
    )

    assert result == "connected"
    assert len(attempts) == 3


async def test_does_not_retry_an_authentication_failure():
    attempts = []

    async def connect():
        attempts.append(None)
        raise ValueError("Provided token does not have admin permissions.")

    with pytest.raises(ValueError):
        await connect_with_retry(
            connect,
            description="Connection to Hypha server",
            logger=_RecordingLogger(),
            total_seconds=5.0,
            initial_delay=0.01,
        )

    assert len(attempts) == 1


async def test_gives_up_once_the_budget_is_spent():
    async def connect():
        raise ConnectionRefusedError(111, "Connect call failed")

    with pytest.raises(ConnectionRefusedError):
        await connect_with_retry(
            connect,
            description="Connection to Hypha server",
            logger=_RecordingLogger(),
            total_seconds=0.05,
            initial_delay=0.01,
            max_delay=0.01,
        )


async def test_worker_retries_the_initial_hypha_connection(monkeypatch):
    """The retry has to be wired into _connect_to_server, not just available."""

    class _Connected(Exception):
        """Raised past the connect call to end the test early."""

    attempts = []

    async def fake_connect_to_server(config):
        attempts.append(config)
        if len(attempts) < 3:
            raise ConnectionRefusedError(111, "Connect call failed")

        async def generate_token():
            raise _Connected

        return SimpleNamespace(generate_token=generate_token)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(worker_module, "connect_to_server", fake_connect_to_server)
    # network.py uses asyncio only for the backoff sleep.
    monkeypatch.setattr(network_module, "asyncio", SimpleNamespace(sleep=no_sleep))
    worker = SimpleNamespace(
        server=None,
        logger=_RecordingLogger(),
        server_url="http://hypha:9520",
        _token="token",
        workspace="ws",
        client_id="worker",
    )

    with pytest.raises(_Connected):
        await BioEngineWorker._connect_to_server(worker)

    assert len(attempts) == 3


def _startup_manager(startup_applications, deploy_app):
    """An AppsManager stand-in exposing only what the startup path touches."""
    deploy_app.__schema__ = {
        "parameters": {
            "properties": {
                "artifact_id": {},
                "version": {},
                "application_id": {},
                "hypha_token": {},
                "context": {},
            }
        }
    }

    async def generate_token(_config):
        return "startup-token"

    manager = SimpleNamespace(
        startup_applications=startup_applications,
        logger=_RecordingLogger(),
        server=SimpleNamespace(
            config=SimpleNamespace(workspace="ws"),
            generate_token=generate_token,
        ),
        admin_users=["admin@example.com"],
        deploy_app=deploy_app,
        _startup_retry_task=None,
        # Reporting-only helpers the startup path calls for their side effects.
        # Absorbed here so this test stays about retry, not about them.
        _warn_on_startup_pin_divergence=lambda _app_config: None,
    )
    manager._retry_startup_applications = (
        lambda configs: AppsManager._retry_startup_applications(manager, configs)
    )
    return manager


async def test_one_failing_startup_application_does_not_abort_the_others(monkeypatch):
    monkeypatch.setattr(manager_module, "_REDEPLOY_BACKOFF_INITIAL_SECONDS", 60.0)
    deployed = []

    async def deploy_app(**kwargs):
        artifact_id = kwargs["artifact_id"]
        if artifact_id == "ws/model-runner":
            raise OSError(37, "No locks available")
        deployed.append(artifact_id)
        return artifact_id.split("/")[-1]

    manager = _startup_manager(
        [
            {"artifact_id": "ws/model-runner"},
            {"artifact_id": "ws/annotation-broker"},
        ],
        deploy_app,
    )

    await AppsManager.deploy_startup_applications(manager)

    assert deployed == ["ws/annotation-broker"]
    assert manager._startup_retry_task is not None
    manager._startup_retry_task.cancel()


async def test_a_failed_startup_application_is_retried(monkeypatch):
    monkeypatch.setattr(manager_module, "_REDEPLOY_BACKOFF_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr(manager_module, "_REDEPLOY_BACKOFF_MAX_SECONDS", 0.01)
    attempts = []

    async def deploy_app(**kwargs):
        attempts.append(kwargs["artifact_id"])
        if len(attempts) < 3:
            raise OSError(37, "No locks available")
        return "model-runner"

    manager = _startup_manager([], deploy_app)

    await AppsManager._retry_startup_applications(
        manager, [{"artifact_id": "ws/model-runner"}]
    )

    assert attempts == ["ws/model-runner"] * 3
