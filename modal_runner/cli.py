"""CLI entrypoint: `modal-runner run|jobs|clean|status|kill`."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
from datetime import datetime

from . import progress, queue, runner


def _default_name(script: str) -> str:
    return pathlib.Path(script).stem


def _detach(args: argparse.Namespace, name: str) -> int:
    """Re-exec ourselves in the background, detached from the terminal.

    Parent returns immediately after printing the child PID + launch-log path.
    The child re-enters `cmd_run` via `--foreground` and runs the full pipeline.
    """
    log_dir = pathlib.Path(args.log_dir) / name
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    launch_log = log_dir / f"launch_{ts}.log"

    argv = [sys.executable, "-m", "modal_runner.cli", "run", "--foreground"]
    # Forward every flag the user passed (and the positional script).
    passthrough = {
        "--name": name,
        "--num-gpus": str(args.num_gpus),
        "--gpu-type": args.gpu_type,
        "--image": args.image,
        "--pip-install": args.pip_install,
        "--repo-dir": args.repo_dir,
        "--max-retries": str(args.max_retries),
        "--max-modal-gpus": str(args.max_modal_gpus),
        "--volume": args.volume,
        "--timeout": str(args.timeout),
        "--log-dir": args.log_dir,
    }
    for k, v in passthrough.items():
        argv += [k, v]
    argv.append(args.script)

    fd = open(launch_log, "ab", buffering=0)
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=fd,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
        cwd=os.getcwd(),
        env=os.environ.copy(),
    )
    print(f"[modal-runner] launched {name} in background  pid={proc.pid}")
    print(f"[modal-runner] launch log: {launch_log}")
    print(f"[modal-runner] attempt logs: {log_dir}/<timestamp>.log")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    name = args.name or _default_name(args.script)
    if not args.foreground:
        return _detach(args, name)
    return runner.run(
        script=args.script,
        name=name,
        num_gpus=args.num_gpus,
        gpu_type=args.gpu_type,
        image=args.image,
        repo_dir=args.repo_dir,
        max_retries=args.max_retries,
        max_modal_gpus=args.max_modal_gpus,
        pip_install=args.pip_install,
        volume=args.volume,
        timeout=args.timeout,
        log_root=args.log_dir,
    )


def cmd_jobs(args: argparse.Namespace) -> int:
    total, breakdown = queue.current_modal_gpus()
    print(f"In-flight modal-runner GPUs: {total}")
    for name, n, gpu in breakdown:
        print(f"  {name:60s} {n}x{gpu}")
    # Also show raw `modal app list` for context.
    print("\n$ modal app list")
    subprocess.run(["modal", "app", "list"])
    return 0


def _launcher_args(pid: int) -> dict[str, str]:
    """Parse the launcher's argv from /proc/<pid>/cmdline into --flag dict."""
    try:
        raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError):
        return {}
    parts = raw.split(b"\0")
    out: dict[str, str] = {}
    i = 0
    while i < len(parts) - 1:
        tok = parts[i].decode(errors="replace")
        if tok.startswith("--") and parts[i + 1] and not parts[i + 1].startswith(b"--"):
            out[tok] = parts[i + 1].decode(errors="replace")
            i += 2
        else:
            i += 1
    return out


def cmd_kill(args: argparse.Namespace) -> int:
    """Stop one or more in-flight modal-runner jobs and free their GPUs.

    For each matched job we (1) ``modal app stop <app>`` to release Modal-
    side GPUs, then (2) SIGTERM the launcher so it exits its retry loop and
    its ``finally`` block drops the local slot reservation. SIGKILL after a
    short grace period if it doesn't exit.
    """
    live = progress.live_launcher_names()
    if args.name:
        targets = {n: pid for n, pid in live.items() if n in args.name}
    elif args.filter:
        targets = {n: pid for n, pid in live.items() if args.filter in n}
    else:
        print("[kill] specify at least one NAME or --filter", file=sys.stderr)
        return 2
    if not targets:
        print("[kill] no live launcher matched")
        return 1

    print("[kill] will stop:")
    for n in sorted(targets):
        print(f"  - {n}  (pid={targets[n]})")
    if not args.yes:
        if input("proceed? [y/N] ").strip().lower() != "y":
            print("[kill] aborted")
            return 0

    rc = 0
    for name, pid in sorted(targets.items()):
        # Recover the Modal app suffix from the launcher's argv so we stop
        # the right per-GPU-shape app (e.g. foo__gpu8xB200).
        argv = _launcher_args(pid)
        try:
            n_gpu = int(argv.get("--num-gpus", "0"))
        except ValueError:
            n_gpu = 0
        gpu_type = argv.get("--gpu-type", "")
        app_name = (
            queue.app_name_with_gpu(name, n_gpu, gpu_type) if n_gpu and gpu_type else name
        )

        print(f"[kill] {name}: modal app stop {app_name}")
        subprocess.run(["modal", "app", "stop", app_name], check=False)

        print(f"[kill] {name}: SIGTERM launcher pid={pid}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"[kill] {name}: launcher already gone")
            continue
        # Wait briefly for graceful exit (the launcher's finally releases the
        # local slot file). Force-kill if it ignores SIGTERM.
        deadline = time.monotonic() + args.grace
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"[kill] {name}: SIGKILL (grace expired)")
            except ProcessLookupError:
                pass
            rc = max(rc, 1)
    return rc


def cmd_clean(args: argparse.Namespace) -> int:
    total, breakdown = queue.current_modal_gpus()
    # Best-effort: let the user stop any stale modal-runner app by name.
    for name, _, _ in breakdown:
        if args.yes or input(f"stop app {name}? [y/N] ").lower() == "y":
            subprocess.run(["modal", "app", "stop", name])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="modal-runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a shell script on Modal with retries")
    pr.add_argument("script", help="path to the training shell script")
    pr.add_argument("--name", help="app name (default: script basename)")
    pr.add_argument("--num-gpus", type=int, default=int_env("NUM_GPUS", 1))
    pr.add_argument("--gpu-type", default="H100")
    pr.add_argument(
        "--image",
        default="nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04",
        help="base registry image",
    )
    pr.add_argument(
        "--pip-install",
        default="",
        help="space-separated pip packages to install into the image",
    )
    pr.add_argument(
        "--repo-dir",
        default=".",
        help="local repo rsynced into the container (script must live inside)",
    )
    pr.add_argument("--max-retries", type=int, default=5)
    pr.add_argument("--max-modal-gpus", type=int, default=64)
    pr.add_argument("--volume", default="modal-runner-vol")
    pr.add_argument("--timeout", type=int, default=86400)
    pr.add_argument("--log-dir", default="logs")
    pr.add_argument(
        "-f", "--foreground",
        action="store_true",
        help="stay attached to the terminal (default: detach & run in background)",
    )
    pr.set_defaults(func=cmd_run)

    pj = sub.add_parser("jobs", help="show modal-runner GPU usage and app list")
    pj.set_defaults(func=cmd_jobs)

    pc = sub.add_parser("clean", help="stop stale modal-runner apps")
    pc.add_argument("--yes", "-y", action="store_true")
    pc.set_defaults(func=cmd_clean)

    ps = sub.add_parser("status", help="snapshot per-job state, progress, ETA")
    ps.add_argument("--log-dir", default="logs", help="root containing per-job dirs")
    ps.add_argument("--filter", default=None, help="substring filter on job name")
    ps.set_defaults(func=progress.cmd_status)

    pk = sub.add_parser("kill", help="stop in-flight jobs and release their GPUs")
    pk.add_argument("name", nargs="*", help="exact job name(s); omit to use --filter")
    pk.add_argument("--filter", default=None, help="substring filter on job name")
    pk.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    pk.add_argument("--grace", type=int, default=15, help="SIGTERM grace period (s) before SIGKILL")
    pk.set_defaults(func=cmd_kill)

    return p


def int_env(var: str, default: int) -> int:
    import os
    v = os.environ.get(var)
    return int(v) if v else default


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
