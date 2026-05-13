"""@app.function definitions for the SSH server and the batch runner."""
from __future__ import annotations

import datetime as _dt
import os
import shlex
import socket
import subprocess
import threading
import time

import modal

from .app import app, volumes
from .config import GPU_SPEC, TIMEOUT_SECONDS, cfg


def _wait_for_port(host, port, q):
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


def _write_shell_env(shell_env: dict):
    with open("/root/.profile", "a") as f:
        for k, v in shell_env.items():
            # $VAR references resolve from the container's env (e.g. secret-injected vars).
            resolved = os.path.expandvars(v) if isinstance(v, str) else v
            f.write(f'export {k}="{resolved}"\n')


def _commit_volumes():
    """Flush writes so a later `modal volume get` (or a resume attempt) sees them."""
    for v in volumes.values():
        try:
            v.commit()
        except Exception as e:
            print(f"(volume commit failed: {e})", flush=True)


_secrets = [modal.Secret.from_name(s) for s in (cfg.get("secrets") or [])]


@app.function(gpu=GPU_SPEC, timeout=TIMEOUT_SECONDS, secrets=_secrets)
def launch_ssh(q, shell_env: dict):
    _write_shell_env(shell_env)
    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        threading.Thread(target=_wait_for_port, args=(host, port, q)).start()
        subprocess.run(["/usr/sbin/sshd", "-D"])


@app.function(gpu=GPU_SPEC, timeout=TIMEOUT_SECONDS, secrets=_secrets)
def run_batch(shell_env: dict, commands: list, runlogs_dir: str | None):
    _write_shell_env(shell_env)
    if not commands:
        print("(no `commands` defined for non-interactive run — exiting)")
        return

    log_subdir = None
    if runlogs_dir:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        log_subdir = os.path.join(runlogs_dir, ts)
        os.makedirs(log_subdir, exist_ok=True)
        print(f"→ Per-command logs: {log_subdir}", flush=True)

    try:
        for idx, cmd in enumerate(commands):
            print(f"\n[{idx}] $ {cmd}", flush=True)
            if log_subdir:
                log_path = os.path.join(log_subdir, f"{idx:03d}.log")
                # pipefail → tee's success doesn't mask the real command's exit code.
                wrapped = f"set -o pipefail; ({cmd}) 2>&1 | tee {shlex.quote(log_path)}"
            else:
                wrapped = cmd
            # Login shell so /root/.profile (env vars, conda activate) is sourced.
            rc = subprocess.run(["bash", "-lc", wrapped]).returncode
            if rc != 0:
                raise SystemExit(f"command [{idx}] failed (rc={rc}): {cmd}")
    finally:
        _commit_volumes()
