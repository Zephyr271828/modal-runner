"""Hydra-style override parsing.

Shared by src/config.py (Python entrypoint) and scripts/launch/lib.sh (shell
helpers) so the dispatch flag and the in-app cfg agree.

No modal dependency — safe to import from a thin shell helper.
"""
from __future__ import annotations

import shlex

import yaml


def _walk(cfg: dict, dotted: str, create: bool):
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            if not create:
                raise SystemExit(f"Override path not found: {dotted!r}")
            cur[p] = {}
        cur = cur[p]
    return cur, parts[-1]


def apply(cfg: dict, raw: str) -> list:
    """Mutate cfg with Hydra-style overrides parsed from `raw`. Returns diff lines."""
    diffs = []
    for tok in shlex.split(raw):
        if tok.startswith("~"):
            parent, leaf = _walk(cfg, tok[1:], create=False)
            if leaf in parent:
                parent.pop(leaf)
                diffs.append(f"~ {tok[1:]}")
            continue
        plus = tok.startswith("+")
        body = tok[1:] if plus else tok
        if "=" not in body:
            raise SystemExit(f"Bad override (need key=value, +key=value, or ~key): {tok!r}")
        key, raw_val = body.split("=", 1)
        try:
            val = yaml.safe_load(raw_val)
        except yaml.YAMLError as e:
            raise SystemExit(f"Bad override value for {key!r}: {e}") from None
        parent, leaf = _walk(cfg, key, create=True)
        if plus and leaf in parent:
            raise SystemExit(f"+{key}: already set (use plain {key}=... to overwrite)")
        parent[leaf] = val
        diffs.append(f"{'+' if plus else ''}{key} = {val!r}")
    return diffs
