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


def _open_log(runlogs_dir: str | None, label: str) -> str | None:
    if not runlogs_dir:
        return None
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    log_subdir = os.path.join(runlogs_dir, ts)
    os.makedirs(log_subdir, exist_ok=True)
    log_path = os.path.join(log_subdir, label)
    print(f"→ Log: {log_path}", flush=True)
    return log_path


def _run(wrapped: str, label: str):
    """Run a bash login-shell command; raise SystemExit on non-zero."""
    rc = subprocess.run(["bash", "-lc", wrapped]).returncode
    if rc != 0:
        raise SystemExit(f"{label} failed (rc={rc})")


@app.function(gpu=GPU_SPEC, timeout=TIMEOUT_SECONDS, secrets=_secrets)
def run_batch(
    shell_env: dict,
    commands: list,
    runlogs_dir: str | None,
    script_content: str | None = None,
    script_name: str | None = None,
):
    _write_shell_env(shell_env)

    try:
        if script_content is not None:
            # Materialize the uploaded script and execute it once. Bypasses the
            # image cache so editing the local script is instant.
            script_dir = "/root/.run"
            os.makedirs(script_dir, exist_ok=True)
            script_path = os.path.join(script_dir, script_name or "run.sh")
            with open(script_path, "w") as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            print(f"\n$ bash {script_path}", flush=True)
            log_path = _open_log(runlogs_dir, f"{(script_name or 'run.sh').rsplit('.', 1)[0]}.log")
            if log_path:
                wrapped = f"set -o pipefail; bash {shlex.quote(script_path)} 2>&1 | tee {shlex.quote(log_path)}"
            else:
                wrapped = f"bash {shlex.quote(script_path)}"
            _run(wrapped, f"script {script_name or 'run.sh'}")
            return

        if not commands:
            print("(no `commands` or `script` defined for non-interactive run — exiting)")
            return

        log_subdir = None
        if runlogs_dir:
            ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
            log_subdir = os.path.join(runlogs_dir, ts)
            os.makedirs(log_subdir, exist_ok=True)
            print(f"→ Per-command logs: {log_subdir}", flush=True)
        for idx, cmd in enumerate(commands):
            print(f"\n[{idx}] $ {cmd}", flush=True)
            if log_subdir:
                log_path = os.path.join(log_subdir, f"{idx:03d}.log")
                wrapped = f"set -o pipefail; ({cmd}) 2>&1 | tee {shlex.quote(log_path)}"
            else:
                wrapped = cmd
            _run(wrapped, f"command [{idx}]")
    finally:
        _commit_volumes()
