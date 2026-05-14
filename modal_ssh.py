"""Thin Modal entrypoint. All wiring lives in src/.

Run with:
    CONFIG=configs/<name>.yml modal run --detach modal_ssh.py
or via the launch.sh wrapper.
"""
from pathlib import Path

import modal

from src.app import app
from src.config import (
    CONFIG_PATH,
    DEFAULT_PRIVATE_KEY,
    GPU_SPEC,
    INTERACTIVE,
    OUTPUT_DIR,
    RUNLOGS_DIR,
    SCRIPT_PATH,
    cfg,
    volume_for,
)
from src.functions import launch_ssh, run_batch


@app.local_entrypoint()
def main():
    print(f"Config: {CONFIG_PATH}")
    print(f"GPU: {GPU_SPEC}  |  Duration: {cfg['duration_hours']}h  |  Interactive: {INTERACTIVE}")
    if INTERACTIVE:
        with modal.Queue.ephemeral() as q:
            launch_ssh.spawn(q, cfg.get("shell_env") or {})
            host, port = q.get(timeout=300)
            print(f"\nApp name: {app.name}")
            print("\nSSH server running. Connect with:")
            print(f"  ssh -i {DEFAULT_PRIVATE_KEY} -p {port} -o StrictHostKeyChecking=no root@{host}")
            # With --detach, the function keeps running after we return.
    else:
        print(f"\nApp name: {app.name}")
        script_content = None
        script_name = None
        if SCRIPT_PATH:
            script_content = Path(SCRIPT_PATH).read_text()
            script_name = Path(SCRIPT_PATH).name
            print(f"Running script (non-interactive): {SCRIPT_PATH}")
        else:
            print("Running batch commands (non-interactive)...")
        run_batch.remote(
            cfg.get("shell_env") or {},
            cfg.get("commands") or [],
            RUNLOGS_DIR,
            script_content,
            script_name,
        )
        print("Batch finished. Modal app will stop.")
        if RUNLOGS_DIR:
            print(f"  Per-command logs: modal volume get {volume_for(RUNLOGS_DIR)} / ./logs/{cfg['job_name']}/")
        if OUTPUT_DIR:
            print(f"  Outputs:          modal volume get {volume_for(OUTPUT_DIR)} / ./outputs/{cfg['job_name']}/")
