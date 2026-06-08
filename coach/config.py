"""Configuration loader: merge config.example.yaml with config.local.yaml.

config.local.yaml (gitignored) overrides the template. Secrets can also come
from the COACH_LLM_API_KEY environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None) -> dict:
    """Load merged config as a plain dict.

    Order: config.example.yaml (defaults) <- config.local.yaml (override)
    <- COACH_LLM_API_KEY env (secret override).
    """
    cfg: dict = {}
    example = REPO_ROOT / "config.example.yaml"
    if example.exists():
        cfg = yaml.safe_load(example.read_text(encoding="utf-8")) or {}
    local_path = Path(path) if path else (REPO_ROOT / "config.local.yaml")
    if local_path.exists():
        local = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
        cfg = _deep_merge(cfg, local)
    env_key = os.environ.get("COACH_LLM_API_KEY")
    if env_key:
        cfg.setdefault("llm", {})["api_key"] = env_key
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Read a nested value by dotted path, e.g. get(cfg, 'llm.model')."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def data_dir(cfg: dict) -> Path:
    d = REPO_ROOT / get(cfg, "paths.data_dir", "data")
    d.mkdir(parents=True, exist_ok=True)
    return d
