"""Launch a shell script on Modal with retry, logging, and volume sync."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import select
import shutil
import subprocess
import sys
import tempfile
import threading
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
    "silence_timeout": re.compile(r"\[modal-runner\] no log output for \d+s"),
}

# Env vars the user script declares; these are made visible on the Modal
# volume so the script sees identical paths on both ends.
#
#  - UPLOAD_VARS  : pushed local -> volume on launch.
#  - OUTPUT_PATH  : same — uploaded if a local copy exists (so Modal can resume
#                   from a prior local checkpoint). The caller is expected to
#                   make OUTPUT_PATH point to a per-run subdirectory (e.g.
#                   `checkpoints/<run_name>`) rather than a shared root, so
#                   this upload stays bounded to one run's tree.
#  - The pull-back from OUTPUT_PATH runs (a) periodically during the modal
#    run via `_start_output_poller` so intermediate checkpoints stream back,
#    and (b) once after the modal run exits.
SYNC_VARS = ("DATA_PATH", "MODEL_PATH", "OUTPUT_PATH")
UPLOAD_VARS = ("DATA_PATH", "MODEL_PATH", "OUTPUT_PATH")
OUTPUT_PULL_INTERVAL_S = int(os.environ.get("MR_OUTPUT_PULL_INTERVAL_S", "300"))
# Kill `modal run` if no log output for this many seconds. The Modal stream
# can hang silently after a container dies (Tasks=0) — without this watchdog
# the wrapper sits until Modal's 24h function timeout fires. Set to 0 to
# disable.
SILENCE_TIMEOUT_S = int(os.environ.get("MR_SILENCE_TIMEOUT_S", "600"))

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


def _modal_app_tasks(app_name: str) -> Optional[int]:
    """Return the running-task count for the most recent Modal app whose
    description matches ``app_name``. Returns None if the lookup fails or no
    matching app exists yet (e.g. still in image build / queue)."""
    try:
        out = subprocess.run(
            ["modal", "app", "list", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return None
        apps = json.loads(out.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None
    # Apps with the same description are listed newest-first by Modal.
    for a in apps:
        if a.get("Description", "").startswith(app_name) and a.get("Stopped at") in (None, ""):
            try:
                return int(a.get("Tasks", "0"))
            except (TypeError, ValueError):
                return None
    return None


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


def _ensure_volume(volume: str) -> None:
    """Create the Modal volume if it doesn't already exist (idempotent)."""
    r = subprocess.run(
        ["modal", "volume", "create", volume],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 and "already exists" not in (r.stdout + r.stderr):
        # Not a benign already-exists error; surface it.
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"failed to create modal volume {volume}")


def _volume_put(volume: str, src: str, dst_rel: str) -> None:
    """Upload src directory into volume at dst_rel (idempotent)."""
    if not pathlib.Path(src).exists():
        return
    print(f"[modal-runner] uploading {src} -> {volume}:/{dst_rel}", flush=True)
    subprocess.run(
        ["modal", "volume", "put", "--force", volume, src, f"/{dst_rel}"],
        check=True,
    )


def _volume_path_exists(volume: str, dst_rel: str) -> bool:
    """Return True if dst_rel exists (and is non-empty) on the Modal volume."""
    r = subprocess.run(
        ["modal", "volume", "ls", volume, f"/{dst_rel}"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return False
    # `modal volume ls` prints a table with at least a header even when empty;
    # treat any non-error response as "path exists".
    return bool(r.stdout.strip())


def _volume_get(volume: str, src_rel: str, dst: str, quiet: bool = False) -> None:
    pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"[modal-runner] downloading {volume}:/{src_rel} -> {dst}", flush=True)
    rc = subprocess.run(
        ["modal", "volume", "get", "--force", volume, f"/{src_rel}", dst],
        capture_output=quiet,
    ).returncode
    if rc != 0 and not quiet:
        print(f"[modal-runner] warn: download rc={rc} (path may not yet exist)")


def _incremental_volume_sync(volume_name: str, vol_rel: str, dst: str) -> None:
    """Mirror `volume_name:/vol_rel` into local `dst`, downloading only files
    that are missing locally or whose size differs. Returns silently if the
    volume slot doesn't yet exist (first poll ticks of a fresh run).

    Local-side files that no longer exist on the volume are NOT removed —
    keeping deletions one-way avoids accidental data loss on transient
    listing errors. The trainer's `save_total_limit` rotation thus keeps
    the volume slim while local accumulates; clean up manually if needed.
    """
    try:
        import modal as _m
        vol = _m.Volume.from_name(volume_name)
        remote_files: dict[str, int] = {}
        for e in vol.iterdir(f"/{vol_rel}", recursive=True):
            if "FILE" in str(getattr(e, "type", "")).upper():
                rel = str(pathlib.Path(e.path).relative_to(f"/{vol_rel}"))
                remote_files[rel] = int(getattr(e, "size", 0) or 0)
    except Exception:
        return  # slot missing or transient API error — try again next tick

    local_root = pathlib.Path(dst)
    local_root.mkdir(parents=True, exist_ok=True)
    local_files: dict[str, int] = {}
    for f in local_root.rglob("*"):
        if f.is_file():
            local_files[str(f.relative_to(local_root))] = f.stat().st_size

    missing = [p for p, sz in remote_files.items() if local_files.get(p) != sz]
    if missing:
        total_bytes = sum(remote_files[p] for p in missing)
        print(
            f"[modal-runner] poller: fetching {len(missing)} file(s) "
            f"({total_bytes / 1e9:.2f} GB) <- {volume_name}:/{vol_rel}",
            flush=True,
        )
        for rel in missing:
            full_local = local_root / rel
            full_local.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(full_local, "wb") as wf:
                    for chunk in vol.read_file(f"/{vol_rel}/{rel}"):
                        wf.write(chunk)
            except Exception as ex:
                print(f"[modal-runner] poller: skip {rel}: {ex}", flush=True)

    # Bounded prune: mirror the trainer's save_total_limit rotation by
    # removing local `checkpoint-N/` dirs that no longer exist on the
    # volume. Scoped to dirs matching `checkpoint-<int>` so we never
    # delete state files (trainer_state.json, wandb logs, etc.) that the
    # trainer doesn't rotate. Skipped if the remote listing was empty,
    # which we treat as "couldn't see remote" rather than "remote is
    # genuinely empty" — otherwise a single API hiccup could nuke local.
    if remote_files:
        ckpt_re = re.compile(r"(.*?checkpoint-\d+)(?:/|$)")
        def _ckpt_roots(paths):
            roots = set()
            for p in paths:
                m = ckpt_re.match(p)
                if m:
                    roots.add(m.group(1))
            return roots
        stale = _ckpt_roots(local_files) - _ckpt_roots(remote_files)
        for rel in stale:
            full = local_root / rel
            if full.is_dir():
                shutil.rmtree(full, ignore_errors=True)
                print(
                    f"[modal-runner] poller: pruned stale {rel} (rotated off volume)",
                    flush=True,
                )


def _start_output_poller(
    volume: str, vol_rel: str, dst: str, interval_s: int
) -> tuple[threading.Event, threading.Thread]:
    """Periodically diff-sync `volume:/vol_rel` to local `dst` while a modal
    run is in flight. Caller signals the returned event to stop the loop,
    then joins the thread.
    """
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval_s):
            try:
                _incremental_volume_sync(volume, vol_rel, dst)
            except Exception as e:
                print(f"[modal-runner] poller error (continuing): {e}", flush=True)

    t = threading.Thread(target=_loop, daemon=True, name="mr-output-poller")
    t.start()
    return stop, t


DEFAULT_SNAPSHOT_EXCLUDES = [
    "__pycache__", "*.pyc", "*.pyo", "*.so", "*.o",
    ".git", ".venv", "venv", "node_modules",
    "build/", "dist/", "*.egg-info",
    # Training/data dirs that must never end up in the code snapshot (they
    # are handled separately via DATA_PATH / MODEL_PATH / OUTPUT_PATH).
    "checkpoints/", "checkpoint-*/", "wandb/",
    "outputs/", "output/", "runs/", "tb_logs/", "tensorboard/",
    "tokenized_cache/", "precomputed_states/", "cache/", ".cache/",
    # Heavy weight/archive files anywhere in the tree.
    "*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt",
    "*.tar", "*.tar.gz", "*.zip", "*.gz", "*.parquet", "*.arrow",
]


def _snapshot_root() -> pathlib.Path:
    """Where to stage repo snapshots. Default: ~/.cache/modal-runner/snapshots.

    `/tmp` is often on the root FS, which is tiny on shared-storage clusters.
    The user can override via MR_SNAPSHOT_ROOT if they have a faster local FS.
    """
    root = os.environ.get("MR_SNAPSHOT_ROOT") or os.path.expanduser(
        "~/.cache/modal-runner/snapshots"
    )
    p = pathlib.Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _snapshot_repo(repo_dir: pathlib.Path, extra_excludes: list[str]) -> pathlib.Path:
    """Rsync the repo to a tempdir so mid-run edits don't break Modal uploads.

    The default exclude list keeps weights, checkpoints, datasets, and caches
    out of the snapshot — those are uploaded separately through the Modal
    volume via DATA_PATH / MODEL_PATH / OUTPUT_PATH.
    """
    snap = pathlib.Path(tempfile.mkdtemp(prefix="modal-runner-repo-", dir=_snapshot_root()))
    excludes: list[str] = []
    for pat in DEFAULT_SNAPSHOT_EXCLUDES + extra_excludes:
        excludes += ["--exclude", pat]
    subprocess.run(
        ["rsync", "-a", *excludes, f"{repo_dir}/", f"{snap}/"],
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

    # Claim a GPU slot under an exclusive flock so concurrent launches
    # serialize through the cap check. The slot is held for the entire
    # lifetime of this run (across retries) and released in `finally`.
    slot = queue.acquire_slot(num_gpus, max_modal_gpus)

    # Stage the repo snapshot and push it to the volume under /repo.
    # Additionally exclude any SYNC_VARS path that lives inside the repo
    # (e.g. OUTPUT_PATH=<repo>/checkpoints) — those are uploaded separately
    # through the Modal volume and would otherwise fill /tmp.
    extra_excludes: list[str] = []
    for var in SYNC_VARS:
        val = os.environ.get(var)
        if not val:
            continue
        try:
            rel = pathlib.Path(val).resolve().relative_to(repo_root)
            extra_excludes.append(f"/{rel}/")
        except ValueError:
            pass
    snap = _snapshot_repo(repo_root, extra_excludes)
    try:
        _ensure_volume(volume)
        _volume_put(volume, str(snap), "repo")

        # DATA_PATH / MODEL_PATH are uploaded only the first time (when the
        # volume path doesn't yet exist). Re-uploading on every launch is
        # slow. Set MR_FORCE_UPLOAD=1 to override.
        #
        # OUTPUT_PATH is intentionally NOT uploaded — see SYNC_VARS comment.
        force_upload = os.environ.get("MR_FORCE_UPLOAD") == "1"
        for var in UPLOAD_VARS:
            val = os.environ.get(var)
            if not (val and pathlib.Path(val).exists()):
                continue
            dst_rel = _vol_rel(val)
            if not force_upload and _volume_path_exists(volume, dst_rel):
                print(
                    f"[modal-runner] skipping {var} upload — {volume}:/{dst_rel} already present "
                    f"(set MR_FORCE_UPLOAD=1 to override)",
                    flush=True,
                )
            else:
                _volume_put(volume, val, dst_rel)

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
                "MR_REQUIREMENTS": os.environ.get("MR_REQUIREMENTS", ""),
                "MR_EDITABLE": os.environ.get("MR_EDITABLE", ""),
                "MR_VOLUME": volume,
            }

            cmd = [
                "modal", "run", f"{MODAL_APP_PATH}::main",
                "--script-rel", script_rel,
                "--env-json", json.dumps(env_for_script),
                "--mounts-json", json.dumps(mounts),
            ]

            # Stream intermediate checkpoints back during training so
            # caller's local OUTPUT_PATH stays current. Only enabled when
            # OUTPUT_PATH is set; safe even if the volume slot doesn't yet
            # exist (the poller just retries on the next tick).
            poller_stop = poller_thread = None
            out_path = os.environ.get("OUTPUT_PATH")
            if out_path and OUTPUT_PULL_INTERVAL_S > 0:
                poller_stop, poller_thread = _start_output_poller(
                    volume, _vol_rel(out_path), out_path, OUTPUT_PULL_INTERVAL_S
                )

            try:
                with open(log_path, "wb") as logf:
                    proc = subprocess.Popen(
                        cmd,
                        env=launch_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    assert proc.stdout is not None
                    fd = proc.stdout.fileno()
                    last_data = time.monotonic()
                    while True:
                        ready, _, _ = select.select([fd], [], [], 30.0)
                        if ready:
                            chunk = os.read(fd, 65536)
                            if not chunk:
                                break  # EOF
                            last_data = time.monotonic()
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                            logf.write(chunk)
                            logf.flush()
                        else:
                            if proc.poll() is not None:
                                break
                            silent_for = time.monotonic() - last_data
                            if SILENCE_TIMEOUT_S > 0 and silent_for > SILENCE_TIMEOUT_S:
                                # Only kill if the app is actually running
                                # (Tasks >= 1). During image build / GPU queue
                                # the log can be silent for many minutes — that
                                # is not a hang, so we wait it out.
                                tasks = _modal_app_tasks(app_name)
                                if tasks is None or tasks < 1:
                                    note = (
                                        f"[modal-runner] silent for {int(silent_for)}s but "
                                        f"app not in running state (tasks={tasks}); "
                                        f"watchdog deferred\n"
                                    ).encode()
                                    sys.stdout.buffer.write(note)
                                    sys.stdout.buffer.flush()
                                    logf.write(note)
                                    logf.flush()
                                    last_data = time.monotonic()
                                    continue
                                msg = (
                                    f"[modal-runner] no log output for {int(silent_for)}s "
                                    f"(threshold {SILENCE_TIMEOUT_S}s, tasks={tasks}) — "
                                    f"killing modal run and stopping app {app_name}\n"
                                ).encode()
                                sys.stdout.buffer.write(msg)
                                sys.stdout.buffer.flush()
                                logf.write(msg)
                                logf.flush()
                                subprocess.run(
                                    ["modal", "app", "stop", app_name],
                                    check=False,
                                    capture_output=True,
                                )
                                proc.terminate()
                                try:
                                    proc.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                break
                    rc = proc.wait()
            finally:
                if poller_stop is not None:
                    poller_stop.set()
                    if poller_thread is not None:
                        poller_thread.join(timeout=10)

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
                # OUTPUT_PATH is NOT pushed back — the Modal volume retains it
                # across retries, so the resumed container already sees the
                # latest checkpoints. Pushing the local copy back would re-
                # upload the whole checkpoints tree on every retry.
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
        queue.release_slot(slot)
