"""Module-level Modal app wiring.

Generates the dedicated Modal SSH key locally if requested, builds the image
and volumes, and constructs the App. Importing this module also imports
`functions` so the @app.function decorators register both entrypoints.
"""
import modal

from . import image as _image
from . import volumes as _volumes
from .config import cfg

if modal.is_local() and cfg.get("auto_generate_modal_key", True):
    from .ssh_keys import ensure_modal_key
    ensure_modal_key()

image, REPO_DEST = _image.build()
volumes = _volumes.build()

app = modal.App(cfg["app_name"], image=image, volumes=volumes)

# Register @app.function definitions by importing the module.
from . import functions  # noqa: E402, F401
