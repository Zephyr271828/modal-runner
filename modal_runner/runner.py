"""Launch a shell script on Modal with retry, logging, and volume sync."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Optional

from . import queue

RETRY_PATTERNS = {
    "modal_timeout": re.compile(r"FunctionTimeoutError"),
    "nccl_watchdog": re.compile(
        r"Watchdog caught collective operation timeout"
        r"|ProcessGroupNCCL.*timeout"
        r"|ran for \d+ milliseconds before timing out"
    ),
    "control_plane": re.compile(
        r"modal\.exception\.ConnectionError: Deadline exceeded"
        r"|No address associated with hostname"
        r"|Connection reset by peer"
        r"|TimeoutError: Deadline exceeded"
    ),
}

# Env vars the user script declares; these are uploaded to / pulled from the
# Modal volume so the script sees identical paths on both ends.
SYNC_VARS = ("DATA_PATH", "MODEL_PATH", "OUTPUT_PATH")

# Host-system env vars that would break the container if forwarded. Everything
# else in the caller's environment IS forwarded, so users can pass arbitrary
# script config with `FOO=BAR BAZ=QUX modal-runner run ./train.sh`.
HOST_ONLY_VARS = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "PWD", "OLDPWD", "TERM",
    "DISPLAY", "HOSTNAME", "LANG", "LANGUAGE", "TMPDIR", "XDG_RUNTIME_DIR",
    "XDG_SESSION_ID", "XDG_SESSION_TYPE", "XDG_DATA_DIRS", "XDG_CONFIG_DIRS",
    "SSH_CLIENT", "SSH_CONNECTION", "SSH_TTY", "SSH_AUTH_SOCK", "MAIL",
    "SHLVL", "_", "PS1", "PS2", "PROMPT_COMMAND",
    "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL", "CONDA_PYTHON_EXE",
    "CONDA_EXE", "CONDA_PROMPT_MODIFIER",
    "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "LIBRARY_PATH", "CPATH",
    "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
})


def _collect_user_env() -> dict[str, str]:
    """Forward every env var the caller set, minus host-system noise and
    modal-runner's own MR_* flags."""
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in HOST_ONLY_VARS:
            continue
        if k.startswith((
            "MR_", "LC_", "SLURM_", "BASH_",
            "NVM_", "NODE_", "COREPACK_", "VSCODE_",
            "TERM_PROGRAM", "COLORTERM", "_CE_",
        )):
            continue
        out[k] = v
    return out

MODAL_APP_PATH = str(pathlib.Path(__file__).resolve().parent / "modal_app.py")


def classify_failure(log_path: pathlib.Path) -> Optional[str]:
    try:
        text = log_path.read_text(errors="replace")
    except FileNotFoundError:
        return None
    for name, pat in RETRY_PATTERNS.items():
        if pat.search(text):
            return name
    return None


def _vol_rel(host_path: str) -> str:
    """Stable volume-relative path derived from the host path.

    Hashing the host path keeps distinct local dirs from colliding while
    letting re-runs of the same job reuse uploads.
    """
    h = hashlib.sha1(host_path.encode()).hexdigest()[:10]
    base = pathlib.Path(host_path).name or "root"
    return f"mounts/{base}_{h}"


def _volume_put(volume: str, src: str, dst_rel: str) -> None:
    """Upload src directory into volume at dst_rel (idempotent)."""
    if not pathlib.Path(src).exists():
        return
    print(f"[modal-runner] uploading {src} -> {volume}:/{dst_rel}", flush=True)
    subprocess.run(
        ["modal", "volume", "put", "--force", volume, src, f"/{dst_rel}"],
        check=True,
    )


def _volume_get(volume: str, src_rel: str, dst: str) -> None:
    pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
    print(f"[modal-runner] downloading {volume}:/{src_rel} -> {dst}", flush=True)
    rc = subprocess.run(
        ["modal", "volume", "get", "--force", volume, f"/{src_rel}", dst],
    ).returncode
    if rc != 0:
        print(f"[modal-runner] warn: download rc={rc} (path may not yet exist)")


def _snapshot_repo(repo_dir: pathlib.Path) -> pathlib.Path:
    """Rsync the repo to a tempdir so mid-run edits don't break Modal uploads."""
    snap = pathlib.Path(tempfile.mkdtemp(prefix="modal-runner-repo-"))
    subprocess.run(
        [
            "rsync", "-a",
            "--exclude=__pycache__", "--exclude=*.pyc",
            "--exclude=.git", "--exclude=build/", "--exclude=dist/",
            f"{repo_dir}/", f"{snap}/",
        ],
        check=True,
    )
    return snap


def run(
    script: str,
    name: str,
    num_gpus: int,
    gpu_type: str,
    image: str,
    repo_dir: str,
    max_retries: int,
    max_modal_gpus: int,
    pip_install: str,
    volume: str,
    timeout: int,
    log_root: str,
) -> int:
    script_path = pathlib.Path(script).resolve()
    repo_root = pathlib.Path(repo_dir).resolve()
    if not script_path.is_file():
        raise SystemExit(f"script not found: {script_path}")
    try:
        script_rel = str(script_path.relative_to(repo_root))
    except ValueError:
        raise SystemExit(f"script {script_path} must live under --repo-dir {repo_root}")

    # Everything the caller set goes to the script (minus host noise).
    user_env = _collect_user_env()

    # Build the mounts dict: container path -> volume subpath.
    mounts: dict[str, str] = {}
    for var in SYNC_VARS:
        val = os.environ.get(var)
        if not val:
            continue
        mounts[val] = _vol_rel(val)

    app_name = queue.app_name_with_gpu(name, num_gpus, gpu_type)
    log_dir = pathlib.Path(log_root) / name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Gate on GPU availability before doing any uploads (uploads are cheap but
    # the queue wait should come first so we don't hold volume locks).
    queue.wait_for_slot(num_gpus, max_modal_gpus)

    # Stage the repo snapshot and push it to the volume under /repo.
    snap = _snapshot_repo(repo_root)
    try:
        _volume_put(volume, str(snap), "repo")

        # Upload DATA_PATH / MODEL_PATH once; OUTPUT_PATH also on first try if
        # it already has prior-run checkpoints locally.
        for var in SYNC_VARS:
            val = os.environ.get(var)
            if val and pathlib.Path(val).exists():
                _volume_put(volume, val, _vol_rel(val))

        env_for_script = user_env

        for attempt in range(1, max_retries + 2):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"{ts}.log"
            print(
                f"[modal-runner] attempt {attempt}/{max_retries + 1} — app={app_name} — log={log_path}",
                flush=True,
            )

            # Launch env for modal_app.py decorators.
            launch_env = {
                **os.environ,
                "MR_APP_NAME": app_name,
                "MR_GPU_TYPE": gpu_type,
                "MR_NUM_GPUS": str(num_gpus),
                "MR_TIMEOUT": str(timeout),
                "MR_IMAGE": image,
                "MR_PIP_INSTALL": pip_install,
                "MR_VOLUME": volume,
            }

            cmd = [
                "modal", "run", f"{MODAL_APP_PATH}::main",
                "--script-rel", script_rel,
                "--env-json", json.dumps(env_for_script),
                "--mounts-json", json.dumps(mounts),
            ]

            with open(log_path, "wb") as logf:
                proc = subprocess.Popen(
                    cmd,
                    env=launch_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                assert proc.stdout is not None
                for chunk in iter(lambda: proc.stdout.read(4096), b""):
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    logf.write(chunk)
                rc = proc.wait()

            # Always pull OUTPUT_PATH back so the local checkpoint dir is
            # current for the next retry and for the final result.
            out_path = os.environ.get("OUTPUT_PATH")
            if out_path:
                _volume_get(volume, _vol_rel(out_path), out_path)

            if rc == 0:
                print(f"[modal-runner] success on attempt {attempt}", flush=True)
                return 0

            cause = classify_failure(log_path)
            if cause and attempt <= max_retries:
                print(
                    f"[modal-runner] classified failure: {cause} — resuming (attempt {attempt + 1}/{max_retries + 1})",
                    flush=True,
                )
                # Push refreshed OUTPUT_PATH (it may have new checkpoints we
                # just pulled) back to volume so resumed container sees it.
                if out_path and pathlib.Path(out_path).exists():
                    _volume_put(volume, out_path, _vol_rel(out_path))
                # Small backoff to let Modal-side state settle.
                time.sleep(15)
                continue

            if not cause:
                print(
                    f"[modal-runner] unclassified failure (rc={rc}) — not retrying",
                    flush=True,
                )
            else:
                print(
                    f"[modal-runner] retries exhausted after {attempt} attempts (last cause: {cause})",
                    flush=True,
                )
            return rc
        return 1
    finally:
        shutil.rmtree(snap, ignore_errors=True)
