"""GPU gating via `modal app list`.

We don't have a first-class GPU-usage API from Modal, so we approximate:
ask for the set of running apps and sum their declared GPU counts using the
conventional `<app_name>__gpu<N>x<TYPE>` suffix that modal_runner stamps onto
every app it launches. Apps without that suffix contribute 0 — good enough
for a single user's budget.
"""

from __future__ import annotations

import json
import re
import subprocess
import time

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
        name = a.get("Name") or a.get("name") or ""
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
    """Block until `need + current <= cap`."""
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
