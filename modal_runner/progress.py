"""Snapshot every active modal-runner job and print a summary table.

Scans ``<log_dir>/<name>/`` (one dir per app) and classifies each job by
combining (a) whether a live ``modal_runner.cli ... --name <name>`` process
exists, and (b) the contents of the latest ``launch_*.log`` and
``YYYYMMDD_HHMMSS.log`` files in that dir.

States:
  - DONE        run log contains ``train_runtime`` (HF Trainer summary line)
  - RUNNING     run log has a recent tqdm progress line
  - STARTING    launcher alive, no run output yet (or run log is empty)
  - QUEUED      launcher alive, last launch line is a "waiting: ... GPUs" notice
  - RESTARTING  launcher alive, run log has a Modal 24h-timeout marker;
                draining means the local ``modal`` child hasn't yet exited,
                resuming means modal-runner already emitted the retry line
  - FAILED      run log shows a hard failure (signal, NCCL hang, etc.) and
                no recoverable cause
  - DEAD        no live launcher process and no terminal marker in the log

The classifier is project-agnostic: it works on any directory tree produced
by ``modal-runner run --log-dir <dir>``. Optional ``--filter`` is a substring
match against the job name.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TQDM_RE = re.compile(r"(\d+)/(\d+) \[([^\]]+)\]")
RATE_RE = re.compile(r"([0-9.]+)\s*(it/s|s/it)")
# tqdm bracket body: "elapsed<remaining, rate" e.g. "00:30<22:37:19, 1.24s/it"
TQDM_TIMES_RE = re.compile(r"([^<,\s]+)\s*<\s*([^,\s]+)")
TIME_PART_RE = re.compile(r"(?:(\d+)\s*days?,\s*)?(?:(\d+):)?(\d+):(\d+)")
WAITING_RE = re.compile(r"waiting:.*GPUs in flight")
TRAIN_RUNTIME_RE = re.compile(r"['\"]?train_runtime['\"]?\s*[:=]\s*[0-9.]+")
RUN_LOG_RE = re.compile(r"^\d{6,}_\d{4,}\.log$")  # e.g. 20260425_084251.log
FILENAME_TS_RE = re.compile(r"(\d{6,})_(\d{4,})")


def _filename_ts(name: str) -> str | None:
    """Extract the 'YYYYMMDD_HHMMSS' timestamp embedded in a log filename.

    Used to compare 'when was this log created' between launch_*.log and the
    run log, without relying on mtime (which can be updated by the launcher
    long after the run log's last write).
    """
    m = FILENAME_TS_RE.search(name)
    return f"{m.group(1)}_{m.group(2)}" if m else None

# Modal 24h timeout markers — checked BEFORE FAIL_PATTERNS so noisy SIGINT
# shutdown traces (pin_memory FileNotFoundError, NCCL aborts) don't mask the
# real cause. These jobs auto-resume on the launcher's next attempt.
TIMEOUT_PATTERNS = [
    re.compile(r"FunctionTimeoutError.*hit its timeout of \d+s"),
    re.compile(r"Runner has been running for too long"),
    re.compile(r"max runtime:\s*\d+\s*seconds"),
]
FAIL_PATTERNS = [
    re.compile(r"Signal (?:6|9|15) "),
    re.compile(r"exitcode\s*[:=]\s*-?(?:6|9|15)"),
    re.compile(r"SystemExit:\s*1"),
    re.compile(r"RemoteError"),
    re.compile(r"NCCL .*timeout", re.I),
]
LAUNCHER_RETRY_RE = re.compile(r"classified failure:\s*\w+.*resuming")


@dataclass
class JobStatus:
    name: str
    state: str
    step: int = 0
    total: int = 0
    rate: str = ""
    eta_h: float = 0.0
    note: str = ""

    @property
    def pct(self) -> str:
        return f"{100 * self.step / self.total:.0f}%" if self.total else "-"

    @property
    def progress(self) -> str:
        if self.total:
            return f"{self.step:,}/{self.total:,}"
        return "-"

    @property
    def eta(self) -> str:
        if self.eta_h <= 0:
            return "-"
        return f"~{self.eta_h:.1f}h" if self.eta_h < 10 else f"~{self.eta_h:.0f}h"


def live_launcher_names() -> dict[str, int]:
    """Map active modal-runner app name -> launcher PID."""
    out = subprocess.run(
        ["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False
    ).stdout
    names: dict[str, int] = {}
    for line in out.splitlines():
        if "modal_runner.cli" not in line or " run " not in line:
            continue
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_str, args = parts
        m = re.search(r"--name\s+(\S+)", args)
        if m and pid_str.isdigit():
            names[m.group(1)] = int(pid_str)
    return names


def latest(d: Path, pattern: str, predicate=None) -> Path | None:
    cands = [p for p in d.glob(pattern) if predicate is None or predicate(p)]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _parse_time_to_seconds(s: str) -> float:
    m = TIME_PART_RE.fullmatch(s.strip())
    if not m:
        return 0.0
    days, hours, minutes, seconds = m.groups()
    total = int(minutes) * 60 + int(seconds)
    if hours:
        total += int(hours) * 3600
    if days:
        total += int(days) * 86400
    return float(total)


def parse_tqdm_from_logs(candidates: list[Path]) -> tuple[int, int, str, float, float]:
    """Walk run logs newest-first; return tqdm progress from the first log
    that has any. Latest log has highest priority; older logs are a fallback
    so terminal-state jobs (DONE/FAILED/RESTARTING/DEAD) still show how far
    they got even if the most recent log got truncated or holds only
    shutdown noise.
    """
    for p in candidates:
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if not text:
            continue
        # Cap by lines for speed; \r-normalize so in-place tqdm updates
        # split into separate lines.
        normalized = "\n".join(text.splitlines()[-4000:]).replace("\r", "\n")
        step, total, rate, elapsed_s, remaining_s = parse_tqdm(normalized)
        if total > 0:
            return step, total, rate, elapsed_s, remaining_s
    return 0, 0, "", 0.0, 0.0


def parse_tqdm(text: str) -> tuple[int, int, str, float, float]:
    """Return (step, total, rate, elapsed_s, remaining_s) from the last tqdm line."""
    last_step = last_total = 0
    last_rate = ""
    last_elapsed = last_remaining = 0.0
    for m in TQDM_RE.finditer(text):
        last_step, last_total = int(m.group(1)), int(m.group(2))
        body = m.group(3)
        rate_m = RATE_RE.search(body)
        if rate_m:
            last_rate = f"{rate_m.group(1)}{rate_m.group(2)}"
        times_m = TQDM_TIMES_RE.search(body)
        if times_m:
            last_elapsed = _parse_time_to_seconds(times_m.group(1))
            last_remaining = _parse_time_to_seconds(times_m.group(2))
    return last_step, last_total, last_rate, last_elapsed, last_remaining


def eta_hours(step: int, total: int, rate: str, elapsed_s: float, remaining_s: float) -> float:
    """Prefer tqdm's smoothed remaining; fall back to step/elapsed average; last
    resort: instantaneous rate (which can be skewed by transient slow steps)."""
    if total <= step:
        return 0.0
    if remaining_s > 0:
        return remaining_s / 3600
    if step > 0 and elapsed_s > 0:
        return (total - step) * (elapsed_s / step) / 3600
    if rate:
        m = re.match(r"([0-9.]+)(it/s|s/it)", rate)
        if m:
            val = float(m.group(1))
            sec_per_step = (1 / val) if m.group(2) == "it/s" else val
            return (total - step) * sec_per_step / 3600
    return 0.0


def classify(d: Path, live_names: dict[str, int]) -> JobStatus | None:
    name = d.name
    js = JobStatus(name=name, state="?")

    launch = latest(d, "launch_*.log")
    if not launch:
        return None
    launcher_pid = live_names.get(name)
    launcher_alive = launcher_pid is not None

    launch_text = launch.read_text(errors="replace")
    launch_tail_lines = launch_text.splitlines()[-50:]
    last_launch_line = launch_tail_lines[-1] if launch_tail_lines else ""

    # QUEUED: still in GPU-pool waiter (only meaningful if launcher is alive)
    if launcher_alive and WAITING_RE.search(last_launch_line):
        js.state = "QUEUED"
        js.note = "GPU pool"
        return js

    # All run logs in the same dir, newest first. Used both to pick the
    # latest non-stale log for state classification AND as a fallback chain
    # for surfacing tqdm progress on non-RUNNING rows.
    run_candidates = sorted(
        (p for p in d.glob("*.log") if RUN_LOG_RE.match(p.name)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    runlog = run_candidates[0] if run_candidates else None

    # Compare the latest run log's filename timestamp against the launcher's;
    # if the run log was created BEFORE the active launcher started, it's
    # stale (e.g. a still-queueing relaunch should not surface a previous
    # attempt's failure). Filename timestamps are used rather than mtimes
    # because launchers update their own log on exit, which would otherwise
    # mark completed jobs as stale. Skip the staleness check when the
    # launcher is gone — for dead launchers, the most recent run log IS the
    # terminal record.
    if runlog is not None and launcher_alive:
        rl_ts = _filename_ts(runlog.name)
        ll_ts = _filename_ts(launch.name)
        if rl_ts and ll_ts and rl_ts < ll_ts:
            runlog = None

    # Best-effort progress for non-RUNNING rows: pull tqdm from the newest
    # run log that has any (older logs as fallback) so DONE / FAILED /
    # RESTARTING / DEAD still show how far the job got. ETA is only
    # meaningful while RUNNING, so we leave it blank elsewhere.
    def _attach_progress() -> None:
        step, total, rate, _e, _r = parse_tqdm_from_logs(run_candidates)
        if total > 0:
            js.step, js.total, js.rate = step, total, rate

    # If we have a fresh run log, check terminal states first.
    if runlog is not None and runlog.stat().st_size > 0:
        run_tail = "\n".join(runlog.read_text(errors="replace").splitlines()[-4000:])
        if TRAIN_RUNTIME_RE.search(run_tail):
            js.state = "DONE"
            if not launcher_alive:
                js.note = "training complete"
            _attach_progress()
            # Pin DONE rows to 100% — HF Trainer prints train_runtime AFTER
            # the final tqdm tick, so the bar may read total-1/total.
            if js.total > 0:
                js.step = js.total
            return js
        for pat in TIMEOUT_PATTERNS:
            if pat.search(run_tail):
                if launcher_alive:
                    js.state = "RESTARTING"
                    retried = bool(LAUNCHER_RETRY_RE.search(launch_text))
                    js.note = "modal timeout (resuming)" if retried else "modal timeout (draining)"
                else:
                    js.state = "FAILED"
                    js.note = "modal timeout, launcher gone"
                _attach_progress()
                return js
        for pat in FAIL_PATTERNS:
            if pat.search(run_tail):
                js.state = "FAILED"
                js.note = pat.pattern
                _attach_progress()
                return js

    if not launcher_alive:
        js.state = "DEAD"
        js.note = "no live launcher proc"
        _attach_progress()
        return js

    if runlog is None or runlog.stat().st_size == 0:
        js.state = "STARTING"
        if runlog is not None:
            age_min = (datetime.now().timestamp() - runlog.stat().st_mtime) / 60
            js.note = f"silent {age_min:.0f}m"
        else:
            js.note = "no run output yet"
        # Surface progress from prior attempts (autoresume will pick up
        # from there once the new container starts).
        _attach_progress()
        return js

    run_tail_norm = "\n".join(
        runlog.read_text(errors="replace").splitlines()[-4000:]
    ).replace("\r", "\n")
    step, total, rate, elapsed_s, remaining_s = parse_tqdm(run_tail_norm)
    if step == 0 and total == 0:
        js.state = "STARTING"
        age_min = (datetime.now().timestamp() - runlog.stat().st_mtime) / 60
        js.note = f"silent {age_min:.0f}m"
        _attach_progress()
        return js

    js.state = "RUNNING"
    js.step, js.total, js.rate = step, total, rate
    js.eta_h = eta_hours(step, total, rate, elapsed_s, remaining_s)
    return js


STATE_ORDER = {
    "RUNNING": 0, "RESTARTING": 1, "STARTING": 2,
    "QUEUED": 3, "FAILED": 4, "DONE": 5, "DEAD": 6,
}


def status(log_dir: str, name_filter: str | None = None) -> list[JobStatus]:
    root = Path(log_dir)
    if not root.is_dir():
        return []
    live = live_launcher_names()
    rows: list[JobStatus] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if name_filter and name_filter not in d.name:
            continue
        if not any(d.glob("launch_*.log")):
            continue  # not a modal-runner job dir
        js = classify(d, live)
        if js is not None:
            rows.append(js)
    rows.sort(key=lambda r: (STATE_ORDER.get(r.state, 99), r.name))
    return rows


def render(rows: list[JobStatus]) -> str:
    headers = ["job", "state", "step/total", "%", "rate", "ETA", "note"]
    cols = [
        [r.name, r.state, r.progress, r.pct, r.rate or "-", r.eta, r.note]
        for r in rows
    ]
    widths = [max(len(h), max((len(c[i]) for c in cols), default=0)) for i, h in enumerate(headers)]

    def fmt(parts):
        return "  ".join(p.ljust(w) for p, w in zip(parts, widths))

    lines = [fmt(headers), fmt(["-" * w for w in widths])]
    lines += [fmt(c) for c in cols]
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    rows = status(args.log_dir, args.filter)
    if not rows:
        print(f"no modal-runner jobs found under {args.log_dir}")
        return 0
    print(render(rows))
    return 0
