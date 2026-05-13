"""Modal Volume construction."""
import modal

from .config import cfg


def build() -> dict:
    return {
        mount: modal.Volume.from_name(name, create_if_missing=True)
        for mount, name in (cfg.get("volumes") or {}).items()
    }
