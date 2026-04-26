# modal-runner

A tiny orchestrator for running **any plain shell training script** on
[Modal](https://modal.com/), with auto-resume on transient failures,
GPU-budget queueing, and per-app structured logs.

## Install

```bash
pip install -e .
modal setup   # authenticate the Modal CLI
```

## What you must provide

### 1. A training shell script

A script (e.g. `train.sh`) that runs your training when executed with `bash`.
It must:

- **Live inside `--repo-dir`** (the local repo rsynced into the container).
- **Be resume-safe**: on restart, detect the latest checkpoint under
  `OUTPUT_PATH` and continue from it. Frameworks like LLaMA-Factory and
  HF Trainer do this by default.
- **Read its paths from `DATA_PATH` / `MODEL_PATH` / `OUTPUT_PATH`**
  (see below). The script sees the same path strings locally and inside the
  container.

### 2. Three required env vars

Exported before invoking `modal-runner`:

| Env var       | Purpose                                                                                  |
|---------------|------------------------------------------------------------------------------------------|
| `DATA_PATH`   | Dataset dir. Uploaded to the Modal volume on first run (skipped on later runs unless `MR_FORCE_UPLOAD=1`). |
| `MODEL_PATH`  | Pretrained-model dir. Same upload semantics as `DATA_PATH`.                              |
| `OUTPUT_PATH` | Checkpoint/output dir. **Not** uploaded; pulled back to local after every attempt.       |

These three are mirrored into the container via symlinks so the script sees
its original paths.

### 3. (Optional) any other env vars your script needs

Every other env var in the caller's environment is forwarded verbatim:

```bash
LR=1e-3 LOSS_TYPE=kd_diff modal-runner run ./train.sh --name kd_run --num-gpus 8
```

A denylist of host-system vars (`PATH`, `HOME`, `LD_LIBRARY_PATH`,
`CUDA_VISIBLE_DEVICES`, conda/virtualenv state, `MR_*`, `LC_*`, `SLURM_*`,
`BASH_*`, …) is stripped. See `HOST_ONLY_VARS` in `runner.py`.

### 4. (Optional) image / dependencies

Default base image: `nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04`. Override
with `--image`. Install Python deps with one of:

- `--pip-install "torch==2.4.0 transformers accelerate"` — inline list.
- `MR_REQUIREMENTS=requirements.txt` — a requirements file. `-r` includes
  are inlined; local wheels (e.g. `./flash_attn-…whl`) are auto-baked into
  the image.
- `MR_EDITABLE=path/to/pkg1:path/to/pkg2` — colon-separated local dirs to
  install editable. Each is copied into the image at `/editable/<basename>`
  and `pip install -e`'d in the same pass as the requirements file. Use
  this for in-tree forks of libraries you want resolved at image-build
  time (no script-time `pip install -e .` needed).

## Usage

```bash
export DATA_PATH=/data/ultrachat
export MODEL_PATH=/models/sdar-1_7b
export OUTPUT_PATH=/ckpts/sdar-1_7b-run42

modal-runner run ./scripts/train.sh \
    --name sdar-1_7b-run42 \
    --num-gpus 8 --gpu-type H100 \
    --repo-dir . \
    --pip-install "torch==2.4.0 transformers accelerate" \
    --max-retries 5 \
    --max-modal-gpus 64
```

`modal-runner run` **detaches by default**: it prints the child PID and
launch-log path, then returns. Pass `-f` / `--foreground` to stay attached.

Logs land at `logs/<name>/<YYYYMMDD_HHMMSS>.log` — one per attempt.

### Other commands

```bash
modal-runner jobs                       # show in-flight GPUs + `modal app list`
modal-runner status                     # per-job state, latest-log progress, ETA
modal-runner status --filter sdar       # restrict to names containing "sdar"
modal-runner kill sdar-1_7b-run42 -y    # stop one job (modal app stop + SIGTERM launcher)
modal-runner kill --filter sdar         # kill all live launchers matching substring
modal-runner clean -y                   # stop stale modal-runner apps
```

`kill` first runs `modal app stop` to release Modal-side GPUs, then SIGTERMs
the local launcher (SIGKILL after `--grace` seconds, default 15) so its
`finally` block drops the local GPU-slot reservation.

## What it does on each run

1. **Queue** — waits until `in_flight_gpus + requested <= --max-modal-gpus`
   (counts apps with the `__gpu<N>x<TYPE>` suffix).
2. **Snapshot** — rsyncs `--repo-dir` to a tempdir (mid-run edits don't
   corrupt the upload), pushes it to the volume at `/repo`.
3. **Upload** — `modal volume put` for `DATA_PATH` and `MODEL_PATH` (first
   run only).
4. **Launch** — runs the script in a Modal container with the three paths
   symlinked to their volume locations.
5. **Stream + log** — stdout/stderr tee'd to terminal and per-attempt log.
6. **Download** — pulls `OUTPUT_PATH` back after every attempt.
7. **Classify & retry** — on non-zero exit, grep the log; retry only on a
   matched pattern.

## Retryable failures

| Cause              | Pattern (regex)                                          |
|--------------------|----------------------------------------------------------|
| `modal_timeout`    | `FunctionTimeoutError`                                   |
| `nccl_watchdog`    | `Watchdog caught collective operation timeout` / `ProcessGroupNCCL.*timeout` / `ran for \d+ milliseconds before timing out` |
| `control_plane`    | `modal.exception.ConnectionError: Deadline exceeded` / `No address associated with hostname` / `Connection reset by peer` / `TimeoutError: Deadline exceeded` |

Unclassified failures do **not** retry (so legitimate training bugs don't
burn the budget).

## Known limitations

- GPU accounting relies on the `__gpu<N>x<TYPE>` suffix; apps launched
  outside `modal-runner` count as 0.
- `modal volume put --force` upserts whole dirs — large `OUTPUT_PATH`
  pulls/pushes can be slow.
