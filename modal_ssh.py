import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import modal
import yaml

# ── Load config ───────────────────────────────────────────
CONFIG_PATH = os.environ.get("CONFIG", "configs/default.yml")
cfg = yaml.safe_load(Path(CONFIG_PATH).read_text())

GPU_SPEC = f"{cfg['gpu']['type']}:{cfg['gpu']['count']}" if cfg["gpu"]["count"] > 1 else cfg["gpu"]["type"]
TIMEOUT_SECONDS = int(float(cfg["duration_hours"]) * 3600)

# Auto-generate the dedicated Modal key locally if requested. ssh_keys.py is not shipped to the container.
if modal.is_local() and cfg.get("auto_generate_modal_key", True):
    from ssh_keys import ensure_modal_key
    ensure_modal_key()

pubkey_paths = [Path(p).expanduser() for p in cfg["ssh_public_keys"]]
if modal.is_local():
    for p in pubkey_paths:
        assert p.exists(), f"SSH public key not found: {p}"

# Use the first key's private counterpart in the printed ssh command.
default_private_key = str(pubkey_paths[0]).removesuffix(".pub")

# ── Image ─────────────────────────────────────────────────
_from_registry_kwargs = {}
if cfg.get("add_python"):
    _from_registry_kwargs["add_python"] = cfg["add_python"]

image = (
    modal.Image.from_registry(cfg["base_image"], **_from_registry_kwargs)
    .entrypoint([])
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC", "CONFIG": CONFIG_PATH})
    .apt_install(*cfg["apt_packages"])
    .pip_install("pyyaml")
    .add_local_dir("configs", "/root/configs", copy=True)
)

# Inject all configured public keys into authorized_keys.
for i, p in enumerate(pubkey_paths):
    image = image.add_local_file(str(p), f"/root/.ssh/keys/key_{i}.pub", copy=True)

image = image.run_commands(
    "mkdir -p /root/.ssh /run/sshd",
    "cat /root/.ssh/keys/*.pub > /root/.ssh/authorized_keys",
    "chmod 700 /root/.ssh",
    "chmod 600 /root/.ssh/authorized_keys",
    "ssh-keygen -A",
    "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
    "echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config",
)

# Upload a github private key and configure ssh so git over SSH works inside the VM.
github_key = cfg.get("github_ssh_key")
if github_key:
    github_key_local = Path(github_key).expanduser()
    if modal.is_local():
        assert github_key_local.exists(), f"github_ssh_key not found: {github_key_local}"
    image = image.add_local_file(str(github_key_local), "/root/.ssh/id_ed25519_github", copy=True)
    image = image.run_commands(
        "chmod 600 /root/.ssh/id_ed25519_github",
        "printf 'Host github.com\\n  HostName github.com\\n  User git\\n  IdentityFile /root/.ssh/id_ed25519_github\\n  StrictHostKeyChecking no\\n' >> /root/.ssh/config",
        "chmod 600 /root/.ssh/config",
        "ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null || true",
    )

# Upload arbitrary local files into the image (optional).
for local_path, remote_path in (cfg.get("local_files") or {}).items():
    local_expanded = Path(local_path).expanduser()
    if modal.is_local():
        assert local_expanded.exists(), f"local_files entry not found: {local_expanded}"
    image = image.add_local_file(str(local_expanded), remote_path, copy=True)

# Clone a single git repo into the image (optional).
repo_cfg = cfg.get("git_repo")
REPO_DEST = None
if repo_cfg:
    repo_url = repo_cfg["url"] if isinstance(repo_cfg, dict) else repo_cfg
    repo_dest = (
        repo_cfg.get("dest") if isinstance(repo_cfg, dict) else None
    ) or f"/root/{repo_url.rstrip('/').split('/')[-1].removesuffix('.git')}"
    REPO_DEST = repo_dest
    # `git_repo_ref` (optional) is embedded in the command string so updating it
    # busts Modal's image-step cache and forces a fresh clone. Use a branch/tag/sha
    # or just bump an integer when the upstream repo changes.
    repo_ref = cfg.get("git_repo_ref", "HEAD")
    image = image.run_commands(
        f"mkdir -p {os.path.dirname(repo_dest)}",
        f"git clone {repo_url} {repo_dest} && cd {repo_dest} && git checkout {repo_ref} # cachebust",
    )

if cfg.get("run_commands"):
    image = image.run_commands(*cfg["run_commands"])

# ── App ───────────────────────────────────────────────────
volumes = {
    mount: modal.Volume.from_name(name, create_if_missing=True)
    for mount, name in (cfg.get("volumes") or {}).items()
}

app = modal.App(cfg["app_name"], image=image, volumes=volumes)


def wait_for_port(host, port, q):
    start = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 22), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start >= 60.0:
                raise TimeoutError("Waited too long for SSH") from exc
    q.put((host, port))


@app.function(
    gpu=GPU_SPEC,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name(s) for s in (cfg.get("secrets") or [])],
)
def launch_ssh(q, shell_env: dict):
    with open("/root/.profile", "a") as f:
        for k, v in shell_env.items():
            # $VAR references resolve from the container's env (e.g. secret-injected vars).
            resolved = os.path.expandvars(v) if isinstance(v, str) else v
            f.write(f'export {k}="{resolved}"\n')

    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        threading.Thread(target=wait_for_port, args=(host, port, q)).start()
        subprocess.run(["/usr/sbin/sshd", "-D"])


@app.local_entrypoint()
def main():
    print(f"Config: {CONFIG_PATH}")
    print(f"GPU: {GPU_SPEC}  |  Duration: {cfg['duration_hours']}h")
    with modal.Queue.ephemeral() as q:
        launch_ssh.spawn(q, cfg.get("shell_env") or {})
        host, port = q.get(timeout=300)
        print(f"\nApp name: {app.name}")
        print("\nSSH server running. Connect with:")
        print(f"  ssh -i {default_private_key} -p {port} -o StrictHostKeyChecking=no root@{host}")
        # With --detach, the function keeps running after we return.
