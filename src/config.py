"""Config loading + validation.

Single source of truth for everything derived from the YAML file pointed to by
the `CONFIG` env var (default: configs/default.yml). All other modules read
from the constants exposed here.

Hydra-style overrides may be supplied via the `CONFIG_OVERRIDES` env var
(populated by launch.sh from positional CLI args):

    ./launch.sh gpu.count=4 duration_hours=12 interactive=false
    ./launch.sh +commands='[echo hi]' ~runlogs_dir

Tokens: `key=value` (set), `+key=value` (add, error if exists),
        `~key` (delete). Values parse as YAML for free typing.
"""
import os
from pathlib import Path

import modal
import yaml

from . import overrides as _overrides

CONFIG_PATH = os.environ.get("CONFIG", "configs/default.yml")
CONFIG_OVERRIDES = os.environ.get("CONFIG_OVERRIDES", "")
# Local path to a shell script to execute in non-interactive mode (alternative
# to inline `commands:`). Resolved at launch — NOT baked into the image, so
# editing the script doesn't bust the modal image cache.
SCRIPT_PATH = None  # populated below after cfg load

cfg: dict = yaml.safe_load(Path(CONFIG_PATH).read_text())
_diffs = _overrides.apply(cfg, CONFIG_OVERRIDES)
if _diffs and modal.is_local():
    print(f"[config] {CONFIG_PATH} + overrides:")
    for d in _diffs:
        print(f"  {d}")

SCRIPT_PATH = cfg.get("script")
if SCRIPT_PATH and cfg.get("commands"):
    raise SystemExit(
        "Config error: `script:` and `commands:` are mutually exclusive. "
        "Use `script:` for multi-step shell logic; use `commands:` for a list of one-liners."
    )
if SCRIPT_PATH and modal.is_local():
    if not Path(SCRIPT_PATH).is_file():
        raise SystemExit(f"Config error: script={SCRIPT_PATH!r} not found.")

GPU_SPEC = (
    f"{cfg['gpu']['type']}:{cfg['gpu']['count']}"
    if cfg["gpu"]["count"] > 1
    else cfg["gpu"]["type"]
)
TIMEOUT_SECONDS = int(float(cfg["duration_hours"]) * 3600)
INTERACTIVE = bool(cfg.get("interactive", True))

# Optional paths the user designates as "where outputs/logs live". Both MUST be
# placed under a path declared in `volumes:` so the bytes survive the container.
OUTPUT_DIR = cfg.get("output_dir")
RUNLOGS_DIR = cfg.get("runlogs_dir")

# Auto-resume scaffolding (flags only — wired but not yet enforced).
AUTO_RESUME = bool(cfg.get("auto_resume", False))
MAX_RESUMES = int(cfg.get("max_resumes", 5))
SOFT_TIMEOUT_MINUTES = cfg.get("soft_timeout_minutes")  # None → derive from duration_hours


def _under_mounted_volume(path: str, mounts: list) -> bool:
    if not path:
        return True
    p = os.path.normpath(path)
    for m in mounts:
        m_norm = os.path.normpath(m)
        if p == m_norm or p.startswith(m_norm.rstrip("/") + "/"):
            return True
    return False


def _validate():
    mounts = list((cfg.get("volumes") or {}).keys())
    for label, path in (("output_dir", OUTPUT_DIR), ("runlogs_dir", RUNLOGS_DIR)):
        if path and not _under_mounted_volume(path, mounts):
            raise SystemExit(
                f"Config error: {label}={path!r} is not under any mount in `volumes:`. "
                f"Currently mounted: {mounts or '(none)'}. "
                f"Add an entry to `volumes:` so outputs persist."
            )


if modal.is_local():
    _validate()

# Expanded SSH public-key paths, used by both the image builder and the local
# entrypoint (which prints the private-key path in the connect command).
PUBKEY_PATHS = [Path(p).expanduser() for p in cfg["ssh_public_keys"]]
if modal.is_local():
    for p in PUBKEY_PATHS:
        assert p.exists(), f"SSH public key not found: {p}"

# Private key counterpart of the first listed public key.
DEFAULT_PRIVATE_KEY = str(PUBKEY_PATHS[0]).removesuffix(".pub")


def volume_for(path: str) -> str:
    """Return the modal Volume name whose mount contains `path`, or '<vol>' if none."""
    if not path:
        return "<vol>"
    p = os.path.normpath(path)
    for m, n in (cfg.get("volumes") or {}).items():
        m_norm = os.path.normpath(m)
        if p == m_norm or p.startswith(m_norm.rstrip("/") + "/"):
            return n
    return "<vol>"
