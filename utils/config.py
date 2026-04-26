"""Config loader — reads config.yaml, resolves project paths, loads .env.

Spec: Project Spec 30min.md §6.1.1; fresh v2.0 glue per Decision v2.36.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@cache
def load_config() -> dict:
    """Load config.yaml once; load .env into the process environment."""
    load_dotenv(PROJECT_ROOT / ".env")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_path(rel: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p
