"""A logger bound at module scope must still reach the replica log.

``bioengine.logger`` resolves to ``ray.serve`` inside a replica and to
``bioengine.app`` everywhere else. The branch is decided by
``BIOENGINE_REPLICA``, which ``_setup_replica`` sets in-process — after the
user module has already been imported. A module-scope
``logger = bioengine.logger`` therefore took the fallback branch, and that
logger had neither a level nor a handler, so every ``INFO`` record it was
given was dropped before it could reach the replica log.
"""

from __future__ import annotations

import logging

import pytest

import bioengine
from bioengine._app import accessors


@pytest.fixture(autouse=True)
def _restore_app_logger():
    """Undo the process-global configuration the fallback branch installs."""
    logger = logging.getLogger("bioengine.app")
    handlers = list(logger.handlers)
    level, propagate = logger.level, logger.propagate
    yield
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_replica_env_var_selects_the_ray_serve_logger(monkeypatch) -> None:
    monkeypatch.setenv("BIOENGINE_REPLICA", "1")
    assert accessors._get_logger() is logging.getLogger("ray.serve")


def test_falls_back_outside_a_replica(monkeypatch) -> None:
    monkeypatch.delenv("BIOENGINE_REPLICA", raising=False)
    assert accessors._get_logger() is logging.getLogger("bioengine.app")


def test_the_branch_is_re_evaluated_not_cached(monkeypatch) -> None:
    """``_setup_replica`` sets the env var late; a cached fallback outlives it."""
    monkeypatch.delenv("BIOENGINE_REPLICA", raising=False)
    assert accessors._get_logger().name == "bioengine.app"

    monkeypatch.setenv("BIOENGINE_REPLICA", "1")
    assert accessors._get_logger().name == "ray.serve"


def test_module_scope_access_in_a_replica_reaches_ray_serve(monkeypatch) -> None:
    """The failing case: the env var comes from the replica's runtime_env, so
    it is already true when the user module is imported."""
    monkeypatch.setenv("BIOENGINE_REPLICA", "1")
    assert bioengine.logger is logging.getLogger("ray.serve")


def test_the_fallback_logger_actually_emits_info(monkeypatch) -> None:
    """Unconfigured, ``bioengine.app`` inherits the root level of WARNING and
    has no handler — an ``INFO`` call is discarded outright."""
    monkeypatch.delenv("BIOENGINE_REPLICA", raising=False)
    logger = logging.getLogger("bioengine.app")
    logger.handlers = []
    logger.setLevel(logging.NOTSET)

    logger = accessors._get_logger()

    assert logger.isEnabledFor(logging.INFO)
    assert logger.handlers


def test_the_fallback_logger_is_configured_once(monkeypatch) -> None:
    monkeypatch.delenv("BIOENGINE_REPLICA", raising=False)
    logging.getLogger("bioengine.app").handlers = []

    first = accessors._get_logger()
    accessors._get_logger()

    assert len(first.handlers) == 1
