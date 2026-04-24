# modal-runner

A tiny orchestrator for running **any plain shell training script** on
[Modal](https://modal.com/), with automatic resume on transient failures,
GPU-budget queueing, and per-app structured logs.

## Why

- You already have a shell script like `torchrun … train.py` that runs locally.
- You want it to run on Modal H100s without authoring a purpose-built Modal app.
- You want it to auto-resume on Modal timeouts, NCCL watchdog blips, and
  Modal control-plane connection errors.
- You want to cap total Modal GPUs in flight so you don't exceed your quota.

## Install

```bash
pip install -e .
# also requires: modal CLI authenticated (`modal setup`)
```

## Contract with the user script

Before invoking `modal-runner`, the **user** exports these env vars. They are
preserved verbatim inside the container and mirrored between local disk and
a persistent Modal volume, so the script sees identical paths locally and
remotely.

| Env var       | Purpose                                                        |
|---------------|----------------------------------------------------------------|
| `DATA_PATH`   | Dataset dir; uploaded to volume before the run.                |
| `MODEL_PATH`  | Pretrained-model dir; uploaded before the run.                 |
| `OUTPUT_PATH` | Checkpoint/output dir. Uploaded on retry, pulled back after every attempt. |

Resume is the script's responsibility: it should check `OUTPUT_PATH` for a
latest checkpoint (LLaMA-Factory, HF Trainer, etc. do this by default).

**All other env vars are forwarded verbatim** into the container, so you can
pass arbitrary script config inline:

```bash
LR=1e-3 LOSS_TYPE=kd_diff mtp_init_std=0.2 modal-runner run ./train.sh \
    --name kd_diff_run --num-gpus 8 --gpu-type H100
```

A small denylist of host-system vars (`PATH`, `HOME`, `LD_LIBRARY_PATH`,
`CUDA_VISIBLE_DEVICES`, conda/virtualenv state, …) is stripped so they
don't leak into the container. `MR_*`, `LC_*`, `SLURM_*`, `BASH_*` are
also stripped. See `HOST_ONLY_VARS` in `runner.py` for the exact list.

## Usage

```bash
export DATA_PATH=/data/ultrachat
export MODEL_PATH=/models/sdar-1_7b
export OUTPUT_PATH=/ckpts/sdar-1_7b-run42

modal-runner run ./scripts/train.sh \
    --name sdar-1_7b-run42 \
    --num-gpus 8 --gpu-type H100 \
    --repo-dir . \
    --image nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04 \
    --pip-install "torch==2.4.0 transformers accelerate" \
    --max-retries 5 \
    --max-modal-gpus 64
```

Logs land at `logs/sdar-1_7b-run42/<YYYYMMDD_HHMMSS>.log` — one file per
attempt, grouped by app name.

**Backgrounding.** `modal-runner run` **detaches by default** (like
`nohup-queue run`): the parent prints the child PID + launch-log path and
returns immediately. Pass `-f` / `--foreground` to stay attached for
interactive debugging. The detached process is a session leader (survives
terminal disconnect), reads from `/dev/null`, and tees its own output to
`logs/<name>/launch_<timestamp>.log` in addition to the per-attempt logs.

### Other commands

```bash
modal-runner jobs        # show in-flight GPUs + `modal app list`
modal-runner clean -y    # stop stale modal-runner apps
```

## What it does on each run

1. **Queue** — parses `modal app list --json`, sums GPUs across apps whose
   names carry the modal-runner `__gpu<N>x<TYPE>` suffix, and waits until
   `in_flight + requested <= --max-modal-gpus`.
2. **Snapshot** — rsyncs `--repo-dir` to a tempdir (so mid-run edits
   don't corrupt the Modal image), uploads the snapshot to the volume at
   `/repo`.
3. **Upload** — `modal volume put` for `DATA_PATH`, `MODEL_PATH`, and any
   existing `OUTPUT_PATH` (idempotent; `--force` upsert).
4. **Launch** — `modal run modal_runner/modal_app.py::main` with the GPU,
   app name, and env injected through `MR_*` env vars. The in-container
   function creates symlinks so the script sees its original paths.
5. **Stream + log** — stdout/stderr tee'd to both terminal and
   `logs/<app>/<ts>.log`.
6. **Download** — pulls `OUTPUT_PATH` back after every attempt.
7. **Classify & retry** — on non-zero exit, grep the log for one of the
   patterns below. Retry if matched, surface otherwise.

## Failure classification (retryable)

| Cause              | Pattern (regex)                                          |
|--------------------|----------------------------------------------------------|
| `modal_timeout`    | `FunctionTimeoutError`                                   |
| `nccl_watchdog`    | `Watchdog caught collective operation timeout` / `ProcessGroupNCCL.*timeout` / `ran for \d+ milliseconds before timing out` |
| `control_plane`    | `modal.exception.ConnectionError: Deadline exceeded` / `No address associated with hostname` / `Connection reset by peer` / `TimeoutError: Deadline exceeded` |

Unclassified non-zero exits do **not** retry (prevents burning the budget
on legitimate training bugs).

## Known limitations

- GPU-count accounting relies on the `__gpu<N>x<TYPE>` suffix in app names.
  Apps launched outside `modal-runner` contribute 0 to the counter.
- `modal app list --json` is the programmatic path. If a future Modal CLI
  renames fields, `queue.py` falls back to 0 and prints a warning.
- `modal volume put --force` upserts the whole directory each retry. For
  very large `OUTPUT_PATH` dirs, a smarter rsync-to-volume would be nicer.
