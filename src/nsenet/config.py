"""Configuration loading.

All tunable choices live in ``config.yaml`` at the repository root so that an
experiment is fully described by (commit hash, config file, cached data).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config and resolve the directories it mentions.

    Relative directories are interpreted relative to the repository root, so
    scripts behave identically no matter where they are launched from.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    for section, key in (("data", "raw_dir"), ("output", "figures_dir"), ("output", "tables_dir")):
        p = Path(cfg[section][key])
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        cfg[section][key] = p
    return cfg
