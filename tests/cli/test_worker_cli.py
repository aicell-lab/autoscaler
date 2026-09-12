"""
Contract for ``bioengine worker`` — the container launcher.

The command's whole job is turning options into an argv list, so these tests
pin that argv: the shape documented in docs/deployment-guide.md for each
runtime, worker arguments forwarded verbatim, and the auth token never
appearing on a command line where `ps` would show it.
"""
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from bioengine import __version__
from bioengine.cli.worker import (
    CONTAINER_WORKSPACE_DIR,
    DEFAULT_IMAGE_REPO,
    _subprocess_env,
    build_command,
    worker_group,
)

WORKSPACE = Path("/home/someone/.bioengine")
IMAGE = f"{DEFAULT_IMAGE_REPO}:{__version__}"
WORKER_ARGS = ("--mode", "single-machine", "--head-num-cpus", "4")


def _build(runtime, **overrides):
    kwargs = dict(
        runtime=runtime,
        image=IMAGE,
        worker_args=WORKER_ARGS,
        workspace_dir=WORKSPACE,
        container_name="bioengine-worker",
        shm_size="8g",
        gpus=False,
        detach=False,
    )
    kwargs.update(overrides)
    return build_command(**kwargs)


def _run(args, env=None):
    base = {"HYPHA_TOKEN": "", "BIOENGINE_SERVER_URL": "", "BIOENGINE_TOKEN": ""}
    base.update(env or {})
    return CliRunner().invoke(worker_group, args, env=base)


# ── The documented invocation, per runtime ────────────────────────────────────


def test_the_docker_command_matches_the_deployment_guide():
    command = _build("docker", gpus=True)
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--user" in command and f"{os.getuid()}:{os.getgid()}" in command
    assert command[command.index("--shm-size") + 1] == "8g"
    assert "--gpus=all" in command
    assert f"{WORKSPACE}:{CONTAINER_WORKSPACE_DIR}" in command
    assert command[-len(WORKER_ARGS) - 4 :] == [
        IMAGE,
        "python",
        "-m",
        "bioengine.worker",
        *WORKER_ARGS,
    ]


def test_podman_uses_its_own_gpu_flag():
    command = _build("podman", gpus=True)
    assert command[:2] == ["podman", "run"]
    assert "--gpus=all" not in command
    assert ["--device", "nvidia.com/gpu=all"] == command[
        command.index("--device") : command.index("--device") + 2
    ]


def test_apptainer_binds_instead_of_mounting():
    command = _build("apptainer", gpus=True)
    assert command[:2] == ["apptainer", "exec"]
    assert "--nv" in command
    assert command[command.index("--bind") + 1] == f"{WORKSPACE}:{CONTAINER_WORKSPACE_DIR}"
    assert f"docker://{IMAGE}" in command
    # No container to name, detach or size — those flags belong to docker/podman.
    for flag in ("--name", "--detach", "--shm-size", "--user"):
        assert flag not in command


def test_native_runs_the_worker_without_a_container():
    command = _build("native", gpus=True)
    assert command == ["python", "-m", "bioengine.worker", *WORKER_ARGS]


def test_the_gpu_flag_is_omitted_when_gpus_are_off():
    for runtime in ("docker", "podman", "apptainer"):
        command = _build(runtime, gpus=False)
        assert "--gpus=all" not in command
        assert "--device" not in command
        assert "--nv" not in command


def test_detaching_replaces_the_interactive_flags():
    assert "-it" in _build("docker")
    detached = _build("docker", detach=True)
    assert "--detach" in detached and "-it" not in detached


# ── The token must never reach argv ───────────────────────────────────────────


def test_the_token_is_named_not_valued_in_the_container_command(monkeypatch):
    monkeypatch.setenv("HYPHA_TOKEN", "secret-token-value")
    for runtime in ("docker", "podman"):
        command = _build(runtime)
        assert "secret-token-value" not in command
        assert command[command.index("-e") + 1] == "HYPHA_TOKEN"


def test_the_token_never_appears_in_any_runtimes_command(monkeypatch):
    monkeypatch.setenv("HYPHA_TOKEN", "secret-token-value")
    for runtime in ("docker", "podman", "apptainer", "native"):
        assert "secret-token-value" not in " ".join(_build(runtime))


def test_the_token_is_passed_through_the_environment():
    env = _subprocess_env("docker", token="secret-token-value", server_url=None)
    assert env["HYPHA_TOKEN"] == "secret-token-value"


def test_apptainer_needs_the_prefixed_variable_to_forward_anything():
    env = _subprocess_env("apptainer", token="secret-token-value", server_url=None)
    assert env["APPTAINERENV_HYPHA_TOKEN"] == "secret-token-value"


def test_an_unset_variable_is_not_forwarded(monkeypatch):
    monkeypatch.delenv("HYPHA_TOKEN", raising=False)
    assert "HYPHA_TOKEN" not in _build("docker")
    assert "APPTAINERENV_HYPHA_TOKEN" not in _subprocess_env("apptainer", None, None)


# ── Worker arguments are forwarded, not interpreted ───────────────────────────


def test_worker_arguments_are_forwarded_verbatim():
    args = ("--mode", "slurm", "--admin-users", "a@x.org,b@y.org", "--debug")
    assert _build("native", worker_args=args)[3:] == list(args)


def test_an_option_the_cli_also_defines_still_reaches_the_worker():
    """``--workspace-dir`` after ``--`` configures the worker, not the container."""
    result = _run(
        ["start", "--runtime", "native", "--dry-run", "--", "--workspace-dir", "/data/ws"]
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip().endswith("--workspace-dir /data/ws")


def test_no_worker_arguments_still_starts_the_worker_module():
    assert _build("native", worker_args=())[-1] == "bioengine.worker"


# ── The CLI surface ───────────────────────────────────────────────────────────


def test_the_image_is_pinned_to_the_installed_version():
    result = _run(["start", "--runtime", "docker", "--dry-run", "--", "--mode", "single-machine"])
    assert result.exit_code == 0, result.output
    assert f"{DEFAULT_IMAGE_REPO}:{__version__}" in result.output


def test_a_dry_run_neither_creates_the_workspace_nor_needs_the_runtime(tmp_path):
    workspace = tmp_path / "never-created"
    result = _run(
        [
            "start",
            "--runtime",
            "podman",
            "--workspace-dir",
            str(workspace),
            "--dry-run",
            "--",
            "--mode",
            "single-machine",
        ]
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith("podman run")
    assert not workspace.exists()


def test_a_missing_runtime_is_refused_when_actually_starting(monkeypatch):
    monkeypatch.setattr("bioengine.cli.worker.shutil.which", lambda _: None)
    result = _run(["start", "--runtime", "podman", "--", "--mode", "single-machine"])
    assert result.exit_code == 1
    assert "not on PATH" in result.output


def test_stop_and_logs_refuse_runtimes_without_named_containers(monkeypatch):
    """On an apptainer-only host there is no container name to act on."""
    monkeypatch.setattr(
        "bioengine.cli.worker.shutil.which", lambda name: name if name == "apptainer" else None
    )
    for command in ("stop", "logs"):
        result = _run([command])
        assert result.exit_code == 1
        assert "no named containers" in result.output


@pytest.mark.parametrize("command", ["start", "stop", "logs"])
def test_every_subcommand_is_reachable(command):
    result = _run([command, "--help"])
    assert result.exit_code == 0
    assert "bioengine-worker" in result.output or "worker" in result.output
