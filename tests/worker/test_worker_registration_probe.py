"""The worker rebuilds its own Hypha client when the server evicts it.

Hypha can drop a client's registration while the socket stays open from the
container's side: hypha-rpc never sees a disconnect, never reconnects, and
``<workspace>/<client-id>:bioengine-worker`` stops resolving forever while the
process stays alive and every probe passes. ``echo("ping")`` cannot see this —
it kept succeeding for ten minutes against a server that had dropped the
registration — so the worker asks whether Hypha still serves its own service id
instead.

The probe rebuilds and never condemns. The sibling failure mode, where Hypha
briefly serves nothing at all, recovers by itself after roughly 200s; a
redundant rebuild there costs one reconnect, while wiring the probe into the
readiness path would cycle a pod that was about to come back.
"""

import asyncio
import inspect
import logging

import pytest

from bioengine.worker import worker as worker_module
from bioengine.worker.worker import BioEngineWorker

SERVICE_ID = "my-ws/pod-1:bioengine-worker"


class _Server:
    """Stand-in for the worker's Hypha client connection."""

    def __init__(self, *, serves: bool = True, disconnect_hangs: bool = False):
        self._serves = serves
        self._disconnect_hangs = disconnect_hangs
        self.echo_calls = 0
        self.probe_calls = 0
        self.disconnected = False

    async def echo(self, message):
        # Always answers: an evicted client keeps a working socket.
        self.echo_calls += 1
        return message

    async def get_service_info(self, service_id):
        self.probe_calls += 1
        if not self._serves:
            raise RuntimeError(f"Service not found: {service_id}")
        return {"id": service_id}

    async def disconnect(self):
        if self._disconnect_hangs:
            await asyncio.sleep(3600)
        self.disconnected = True


def _bare_worker(server, **attrs):
    worker = object.__new__(BioEngineWorker)
    worker.logger = logging.getLogger("test-bioengine-worker")
    worker.start_time = None  # keeps __del__ quiet on a hand-built instance
    worker.server = server
    worker.full_service_id = SERVICE_ID
    worker._registration_probe_due_at = 0.0
    worker._monitor_consecutive_errors = 0
    worker.reconnects = 0
    worker.registrations = 0

    async def _connect():
        worker.reconnects += 1

    async def _register():
        worker.registrations += 1

    worker._connect_to_server = _connect
    worker._register_bioengine_worker_service = _register
    for key, value in attrs.items():
        setattr(worker, key, value)
    return worker


@pytest.mark.asyncio
async def test_a_served_registration_does_not_rebuild():
    worker = _bare_worker(_Server(serves=True))

    await worker._check_service_registration()

    assert worker.server.probe_calls == 1
    assert worker.reconnects == 0
    assert worker.registrations == 0


@pytest.mark.asyncio
async def test_a_live_socket_with_a_dropped_registration_still_rebuilds():
    # The whole failure mode: the socket answers, the registration is gone.
    server = _Server(serves=False)
    worker = _bare_worker(server)

    assert await server.echo("ping") == "ping"
    await worker._check_service_registration()

    assert worker.reconnects == 1
    assert worker.registrations == 1


@pytest.mark.asyncio
async def test_a_failed_rebuild_is_not_reported_upwards():
    worker = _bare_worker(_Server(serves=False))
    attempts = []

    async def _connect():
        attempts.append(1)
        raise RuntimeError("Hypha is returning 500")

    worker._connect_to_server = _connect

    await worker._check_service_registration()

    assert attempts, "the rebuild was never attempted"
    assert worker.registrations == 0
    assert worker._monitor_consecutive_errors == 0


@pytest.mark.asyncio
async def test_a_failed_rebuild_is_retried_on_the_next_interval():
    worker = _bare_worker(_Server(serves=False))
    attempts = []

    async def _connect():
        attempts.append(len(attempts))
        raise RuntimeError("Hypha is returning 500")

    worker._connect_to_server = _connect

    await worker._check_service_registration()
    worker._registration_probe_due_at = 0.0
    await worker._check_service_registration()

    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_the_probe_costs_one_round_trip_per_interval():
    worker = _bare_worker(_Server(serves=True))

    await worker._check_service_registration()
    await worker._check_service_registration()
    await worker._check_service_registration()

    assert worker.server.probe_calls == 1


@pytest.mark.asyncio
async def test_the_probe_waits_until_the_service_is_registered():
    worker = _bare_worker(_Server(serves=True), full_service_id=None)

    await worker._check_service_registration()

    assert worker.server.probe_calls == 0
    assert worker.reconnects == 0


@pytest.mark.asyncio
async def test_a_disconnected_worker_is_left_to_the_connection_check():
    worker = _bare_worker(None)

    await worker._check_service_registration()

    assert worker.reconnects == 0


@pytest.mark.asyncio
async def test_a_hung_probe_does_not_stall_the_monitoring_loop(monkeypatch):
    monkeypatch.setattr(worker_module, "_REGISTRATION_PROBE_TIMEOUT_S", 0.05)

    class _HangingServer(_Server):
        async def get_service_info(self, service_id):
            self.probe_calls += 1
            await asyncio.sleep(3600)

    worker = _bare_worker(_HangingServer())

    await asyncio.wait_for(worker._check_service_registration(), timeout=10)

    assert worker.reconnects == 1


@pytest.mark.asyncio
async def test_a_hung_disconnect_does_not_stall_the_rebuild(monkeypatch):
    # The transport being closed on the rebuild path is the one already
    # suspected of being wedged, so the close has to be bounded.
    monkeypatch.setattr(worker_module, "_DISCONNECT_TIMEOUT_S", 0.05)

    connected = []

    async def _fake_connect_to_server(config):
        connected.append(config)
        return _Server()

    monkeypatch.setattr(worker_module, "connect_to_server", _fake_connect_to_server)

    worker = object.__new__(BioEngineWorker)
    worker.logger = logging.getLogger("test-bioengine-worker")
    worker.start_time = None
    worker.server = _Server(disconnect_hangs=True)
    worker.server_url = "https://hypha.example"
    worker._token = "token"
    worker.workspace = None
    worker.client_id = None

    with pytest.raises(Exception):
        # Stops at the first real step after the disconnect; what matters is
        # that the hung close did not swallow the call.
        await asyncio.wait_for(worker._connect_to_server(), timeout=10)

    assert connected, "the rebuild never reached connect_to_server"


def test_the_monitoring_loop_runs_the_registration_probe():
    source = inspect.getsource(BioEngineWorker._create_monitoring_task)

    assert "_check_service_registration()" in source


def test_the_probe_is_not_wired_into_the_readiness_backstop():
    # _monitor_consecutive_errors drives the not-ready flip that lets the k8s
    # liveness probe cycle the pod. The registration probe must never touch it.
    body = inspect.getsource(BioEngineWorker._check_service_registration).split('"""')[
        2
    ]

    assert "_monitor_consecutive_errors" not in body
    assert "raise" not in body
