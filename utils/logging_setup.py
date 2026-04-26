"""Logging — loguru-based, stderr sink + rotating file in config.logging.dir.

Spec: Project Spec 30min.md; fresh v2.0 glue per Decision v2.36.
"""
from __future__ import annotations

import sys
from functools import cache

from loguru import logger

from .config import load_config, resolve_path


@cache
def _setup() -> None:
    cfg = load_config().get("logging", {})
    level = cfg.get("level", "INFO")
    logs_dir = resolve_path(cfg.get("dir", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(
        logs_dir / "ml-bot-30m.log",
        level=level,
        rotation=cfg.get("rotation", "100 MB"),
        retention=cfg.get("retention", "30 days"),
        enqueue=True,
    )


def get_logger(name: str):
    """Return a loguru logger bound to `name`."""
    _setup()
    return logger.bind(name=name)
