"""Build the modal.Image from the loaded config.

Each step is its own helper so adding/reordering build steps is mechanical.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

from .config import CONFIG_OVERRIDES, CONFIG_PATH, PUBKEY_PATHS, cfg


def _from_registry() -> modal.Image:
    kwargs = {}
    if cfg.get("add_python"):
        kwargs["add_python"] = cfg["add_python"]
    return (
        modal.Image.from_registry(cfg["base_image"], **kwargs)
        .entrypoint([])
        .env({
            "DEBIAN_FRONTEND": "noninteractive",
            "TZ": "UTC",
            "CONFIG": CONFIG_PATH,
            # Propagate Hydra-style overrides so the container's config.py
            # produces the same cfg as the local entrypoint. Changes here
            # legitimately bust the image cache.
            "CONFIG_OVERRIDES": CONFIG_OVERRIDES,
            # Line-flush Python stdout so streamed training logs aren't held in 8KB block buffers.
            "PYTHONUNBUFFERED": "1",
        })
        .apt_install(*cfg["apt_packages"])
        .pip_install("pyyaml")
        .add_local_dir("configs", "/root/configs", copy=True)
    )


def _with_ssh_keys(image: modal.Image) -> modal.Image:
    for i, p in enumerate(PUBKEY_PATHS):
        image = image.add_local_file(str(p), f"/root/.ssh/keys/key_{i}.pub", copy=True)
    return image.run_commands(
        "mkdir -p /root/.ssh /run/sshd",
        "cat /root/.ssh/keys/*.pub > /root/.ssh/authorized_keys",
        "chmod 700 /root/.ssh",
        "chmod 600 /root/.ssh/authorized_keys",
        "ssh-keygen -A",
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
        "echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config",
    )


def _with_github_key(image: modal.Image) -> modal.Image:
    github_key = cfg.get("github_ssh_key")
    if not github_key:
        return image
    github_key_local = Path(github_key).expanduser()
    if modal.is_local():
        assert github_key_local.exists(), f"github_ssh_key not found: {github_key_local}"
    image = image.add_local_file(str(github_key_local), "/root/.ssh/id_ed25519_github", copy=True)
    return image.run_commands(
        "chmod 600 /root/.ssh/id_ed25519_github",
        "printf 'Host github.com\\n  HostName github.com\\n  User git\\n  IdentityFile /root/.ssh/id_ed25519_github\\n  StrictHostKeyChecking no\\n' >> /root/.ssh/config",
        "chmod 600 /root/.ssh/config",
        "ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null || true",
    )


def _with_local_files(image: modal.Image) -> modal.Image:
    for local_path, remote_path in (cfg.get("local_files") or {}).items():
        local_expanded = Path(local_path).expanduser()
        if modal.is_local():
            assert local_expanded.exists(), f"local_files entry not found: {local_expanded}"
        image = image.add_local_file(str(local_expanded), remote_path, copy=True)
    return image


def _with_git_repo(image: modal.Image) -> tuple[modal.Image, str | None]:
    repo_cfg = cfg.get("git_repo")
    if not repo_cfg:
        return image, None
    repo_url = repo_cfg["url"] if isinstance(repo_cfg, dict) else repo_cfg
    repo_dest = (
        repo_cfg.get("dest") if isinstance(repo_cfg, dict) else None
    ) or f"/root/{repo_url.rstrip('/').split('/')[-1].removesuffix('.git')}"
    # `git_repo_ref` (optional) is embedded in the command string so updating it
    # busts Modal's image-step cache and forces a fresh clone. Use a branch/tag/sha
    # or just bump an integer when the upstream repo changes.
    repo_ref = cfg.get("git_repo_ref", "HEAD")
    image = image.run_commands(
        f"mkdir -p {os.path.dirname(repo_dest)}",
        f"git clone {repo_url} {repo_dest} && cd {repo_dest} && git checkout {repo_ref} # cachebust",
    )
    return image, repo_dest


def _with_run_commands(image: modal.Image) -> modal.Image:
    if cfg.get("run_commands"):
        image = image.run_commands(*cfg["run_commands"])
    return image


def build() -> tuple[modal.Image, str | None]:
    """Build the image. Returns (image, repo_dest)."""
    image = _from_registry()
    image = _with_ssh_keys(image)
    image = _with_github_key(image)
    image = _with_local_files(image)
    image, repo_dest = _with_git_repo(image)
    image = _with_run_commands(image)
    return image, repo_dest
