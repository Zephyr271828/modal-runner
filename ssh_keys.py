import subprocess
from pathlib import Path

MODAL_KEY_PATH = Path.home() / ".ssh" / "id_ed25519_modal"


def ensure_modal_key(path: Path = MODAL_KEY_PATH) -> Path:
    """Generate a dedicated ed25519 keypair for Modal VMs if missing. Returns the public-key path."""
    pub = path.with_suffix(path.suffix + ".pub") if path.suffix else Path(str(path) + ".pub")
    if path.exists() and pub.exists():
        return pub

    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(path), "-N", "", "-C", "modal-vm"],
        check=True,
    )
    print(f"Generated new Modal SSH key: {path}")
    return pub
