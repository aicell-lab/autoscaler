"""Pin that an inherited version stops being silent.

``deploy_app`` with an ``application_id`` and no ``version`` inherits the
version already running (``manager.py``, the ``is_update`` branch). That rule is
deliberate and documented, but it used to leave no trace anywhere: a caller who
had just uploaded newer code got a deploy that reported success while
redeploying the code it had replaced, and every status field agreed. Reported
twice independently — over the API (svamp #0023) and from the CLI
(aicell-lab/bioengine#157).

Two pins here:

- The worker names the inherited version, and *warns* when it is not the
  artifact's newest — the case that is almost always a mistake. Pinning to an
  older version on purpose stays legal, it just announces itself.
- ``bioengine apps deploy`` passes the version it just uploaded, so pointing it
  at a running ``--app-id`` actually rolls that app forward instead of
  redeploying what was already there.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bioengine.apps.manager import AppsManager

ARTIFACT_ID = "bioimage-io/nuclei-seg"
APP_ID = "nuclei-seg"


def _make_manager(*, versions) -> AppsManager:
    """An AppsManager wired with only what the version report touches."""
    manager = object.__new__(AppsManager)
    manager.logger = logging.getLogger("test.deploy")

    artifact_manager = MagicMock()
    if versions is None:
        artifact_manager.read = AsyncMock(side_effect=RuntimeError("artifact gone"))
    else:
        artifact_manager.read = AsyncMock(return_value={"versions": versions})
    manager.artifact_manager = artifact_manager
    return manager


def _versions(*pairs):
    return [{"version": v, "created_at": t} for v, t in pairs]


@pytest.mark.asyncio
async def test_a_stale_inherited_version_warns_and_names_both(caplog) -> None:
    # rich-mole's case verbatim: 1.0.1 was uploaded, 1.0.0 is what runs.
    manager = _make_manager(versions=_versions(("1.0.0", 1), ("1.0.1", 2)))

    with caplog.at_level(logging.WARNING, logger="test.deploy"):
        await manager._report_inherited_version(APP_ID, ARTIFACT_ID, "1.0.0")

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    # Both numbers must be present — one of them alone is not actionable.
    assert "'1.0.0'" in warnings[0]
    assert "'1.0.1'" in warnings[0]
    assert "version='1.0.1'" in warnings[0], (
        "The warning has to name the way out, or it just restates the symptom."
    )


@pytest.mark.asyncio
async def test_inheriting_the_newest_version_does_not_warn(caplog) -> None:
    # Redeploying the running version *is* the request when nothing newer
    # exists; warning here would train people to ignore the warning.
    manager = _make_manager(versions=_versions(("1.0.0", 1), ("1.0.1", 2)))

    with caplog.at_level(logging.INFO, logger="test.deploy"):
        await manager._report_inherited_version(APP_ID, ARTIFACT_ID, "1.0.1")

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert any("keeping the running version" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_an_unpinned_running_app_is_not_reported_as_stale(caplog) -> None:
    # If the running app carries no version it was deployed as "latest", so the
    # update resolves latest again and genuinely does roll forward. Warning
    # would be false.
    manager = _make_manager(versions=_versions(("1.0.0", 1), ("1.0.1", 2)))

    with caplog.at_level(logging.INFO, logger="test.deploy"):
        await manager._report_inherited_version(APP_ID, ARTIFACT_ID, None)

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    manager.artifact_manager.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_artifact_never_blocks_the_deploy(caplog) -> None:
    # The report is observability. If the version lookup fails it must stay
    # quiet and let the deploy proceed, not raise into the caller.
    manager = _make_manager(versions=None)

    with caplog.at_level(logging.INFO, logger="test.deploy"):
        await manager._report_inherited_version(APP_ID, ARTIFACT_ID, "1.0.0")

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.asyncio
async def test_no_committed_versions_is_not_a_mismatch(caplog) -> None:
    manager = _make_manager(versions=[])

    with caplog.at_level(logging.INFO, logger="test.deploy"):
        await manager._report_inherited_version(APP_ID, ARTIFACT_ID, "1.0.0")

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_deploy_app_reports_the_version_it_inherited() -> None:
    # The report is only worth anything if the update path actually calls it,
    # and only when the version was inherited rather than requested.
    src = inspect.getsource(AppsManager.deploy_app)
    assert "version_inherited = version is None" in src
    assert "await self._report_inherited_version(" in src


# ── CLI: `bioengine apps deploy` must deploy what it just uploaded ────────────

MANIFEST = """\
format_version: 0.6.0
name: Nuclei Seg
id: nuclei-seg
id_emoji: "🔬"
description: Segment nuclei.
type: ray-serve
version: 1.0.1
entry: nuclei_seg.deployment:NucleiSeg
"""


@pytest.fixture
def deploy_cli(monkeypatch, tmp_path: Path):
    """Run ``apps deploy`` against a stub worker; yield the recorded kwargs."""
    from click.testing import CliRunner

    from bioengine.cli import apps as apps_cli

    app_dir = tmp_path / "nuclei-seg"
    app_dir.mkdir()
    (app_dir / "manifest.yaml").write_text(MANIFEST)
    (app_dir / "deployment.py").write_text("class NucleiSeg:\n    pass\n")

    recorded: dict = {}

    worker = MagicMock()
    worker.upload_app = AsyncMock(return_value=ARTIFACT_ID)

    async def _deploy_app(**kwargs):
        recorded.update(kwargs)
        return APP_ID

    worker.deploy_app = _deploy_app

    monkeypatch.setattr(
        apps_cli, "require_worker", lambda *a: ("https://hypha.test", "ws/w", "tok")
    )
    monkeypatch.setattr(apps_cli, "connect_worker", AsyncMock(return_value=worker))

    result = CliRunner().invoke(
        apps_cli.apps_group, ["deploy", str(app_dir), "--app-id", APP_ID]
    )
    assert result.exit_code == 0, result.output
    return recorded, result.output


def test_apps_deploy_pins_the_version_it_uploaded(deploy_cli) -> None:
    # Without this the command uploads 1.0.1, targets a running app, inherits
    # that app's 1.0.0 and reports success having deployed nothing new.
    recorded, _ = deploy_cli
    assert recorded["version"] == "1.0.1"
    assert recorded["application_id"] == APP_ID


def test_apps_deploy_tells_the_user_which_version(deploy_cli) -> None:
    _, output = deploy_cli
    assert "1.0.1" in output
