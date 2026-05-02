"""GPU gating via `modal app list` + a local file-lock semaphore.

We don't have a first-class GPU-usage API from Modal, so we approximate:
ask for the set of running apps and sum their declared GPU counts using the
conventional `<app_name>__gpu<N>x<TYPE>` suffix that modal_runner stamps onto
every app it launches. Apps without that suffix contribute 0 — good enough
for a single user's budget.

Concurrent local launches can't be serialized by polling alone (they all
read the same in-flight count before any has registered an app). To make
the cap effective, `acquire_slot()` claims a local reservation file under
an exclusive flock; `release_slot()` drops it on success or failure. Stale
reservations from crashed processes are reaped automatically (PID check).
"""

from __future__ import annotations

import fcntl
import json
import os
import pathlib
import re
import subprocess
import time

_SLOTS_BASE = pathlib.Path.home() / ".cache" / "modal-runner" / "slots"


def _slots_dir() -> pathlib.Path:
    """Per-Modal-profile slots dir.

    Each Modal account has independent GPU quota, so there's no reason a
    heavyball job should block a yucheng job (or vice versa). The active
    profile is set by ``cli._apply_user`` via ``MODAL_PROFILE``; if unset
    (default profile), we use a stable ``__default__`` namespace.
    """
    profile = os.environ.get("MODAL_PROFILE") or "__default__"
    return _SLOTS_BASE / profile

GPU_SUFFIX_RE = re.compile(r"__gpu(\d+)x([A-Za-z0-9_-]+)$")


def app_name_with_gpu(base: str, num_gpus: int, gpu_type: str) -> str:
    return f"{base}__gpu{num_gpus}x{gpu_type}"


def _list_apps_json() -> list[dict]:
    # `modal app list --json` prints a JSON array.
    try:
        out = subprocess.run(
            ["modal", "app", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return json.loads(out)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"[modal-runner] warn: could not read modal app list: {e}")
        return []


def current_modal_gpus() -> tuple[int, list[tuple[str, int, str]]]:
    """Returns (total_gpus_in_flight, [(app_name, n_gpus, gpu_type), ...])."""
    apps = _list_apps_json()
    total = 0
    breakdown: list[tuple[str, int, str]] = []
    for a in apps:
        # Field names vary across Modal versions; be lenient.
        state = (a.get("State") or a.get("state") or "").lower()
        # `modal app list --json` returns the app name under "Description"
        # (not "Name"). Keep the older keys as fallbacks across Modal versions.
        name = a.get("Description") or a.get("Name") or a.get("name") or ""
        if "stop" in state or "terminated" in state or state in {"done", "ephemeral"}:
            continue
        m = GPU_SUFFIX_RE.search(name)
        if not m:
            continue
        n = int(m.group(1))
        total += n
        breakdown.append((name, n, m.group(2)))
    return total, breakdown


def wait_for_slot(need: int, cap: int, poll_s: int = 30) -> None:
    """Block until `need + current <= cap`. Polling-only — racy under
    concurrent launches. Prefer `acquire_slot` / `release_slot`.
    """
    if need > cap:
        raise SystemExit(
            f"requested {need} GPUs exceeds cap {cap}; raise --max-modal-gpus"
        )
    while True:
        cur, _ = current_modal_gpus()
        if cur + need <= cap:
            return
        print(
            f"[modal-runner] waiting: {cur} GPUs in flight, need {need}, cap {cap} — sleeping {poll_s}s",
            flush=True,
        )
        time.sleep(poll_s)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_local_reservations() -> tuple[int, list[pathlib.Path]]:
    """Sum reserved GPUs across alive launches, reaping stale entries."""
    slots_dir = _slots_dir()
    slots_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    live: list[pathlib.Path] = []
    for f in sorted(slots_dir.glob("slot-*.json")):
        try:
            d = json.loads(f.read_text())
            pid = int(d.get("pid", -1))
            n = int(d.get("n", 0))
        except Exception:
            f.unlink(missing_ok=True)
            continue
        if not _pid_alive(pid):
            f.unlink(missing_ok=True)
            continue
        total += n
        live.append(f)
    return total, live


def acquire_slot(need: int, cap: int, poll_s: int = 30) -> pathlib.Path:
    """Claim a GPU reservation. Returns a path the caller must pass to
    `release_slot()` after `modal run` finishes (success or failure).

    Concurrent launches serialize on an exclusive flock so only one is
    inside the check-and-claim critical section at a time. The lock is
    held only briefly (long enough to read reservations, query Modal,
    and write our reservation file); we sleep *outside* the lock so
    other waiters can make progress.

    The current in-flight GPU count is taken as max(local-reservations,
    modal-app-list) — local catches our own concurrent launches before
    Modal sees them; modal-app-list catches anything launched outside
    this runner.
    """
    if need > cap:
        raise SystemExit(
            f"requested {need} GPUs exceeds cap {cap}; raise --max-modal-gpus"
        )
    slots_dir = _slots_dir()
    slots_dir.mkdir(parents=True, exist_ok=True)
    lock_path = slots_dir / ".lock"
    while True:
        with open(lock_path, "a+") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                cur_local, _ = _read_local_reservations()
                cur_modal, _ = current_modal_gpus()
                cur = max(cur_local, cur_modal)
                if cur + need <= cap:
                    slot = slots_dir / f"slot-{os.getpid()}-{int(time.time()*1000)}.json"
                    slot.write_text(json.dumps({"pid": os.getpid(), "n": need}))
                    return slot
                print(
                    f"[modal-runner] waiting: {cur} GPUs in flight "
                    f"(local={cur_local}, modal={cur_modal}), need {need}, "
                    f"cap {cap} — sleeping {poll_s}s",
                    flush=True,
                )
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        time.sleep(poll_s)


def release_slot(slot: pathlib.Path) -> None:
    """Drop a reservation. Safe to call on a missing path."""
    try:
        slot.unlink()
    except FileNotFoundError:
        pass
