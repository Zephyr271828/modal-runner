"""Generic Modal app driven entirely by env vars set by modal_runner.runner.

The runner sets these before `modal run` so decorators pick them up at import:

    MR_APP_NAME      unique app name (becomes `modal.App(name=...)`)
    MR_GPU_TYPE      H100 / A100 / L40S / ...
    MR_NUM_GPUS      e.g. 8
    MR_TIMEOUT       per-invocation timeout (seconds). Modal caps at 86400.
    MR_IMAGE         base registry image tag
    MR_PIP_INSTALL   space-separated pip packages (optional)
    MR_REQUIREMENTS  path to a requirements.txt to pip install (optional).
                     Resolved on the local launching machine. Local-file
                     entries (e.g. `./flash_attn-…whl`) are auto-baked into
                     the image via `add_local_file(copy=True)` and rewritten
                     to in-image paths so pip resolves them at build time.
                     Editable (`-e`) entries that point into the uploaded
                     repo still won't work — the repo isn't there yet at
                     image-build time; use PYTHONPATH or a script-time
                     `pip install -e ...` for those.
    MR_VOLUME        persistent modal.Volume name (default: modal-runner-vol)
"""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys
import tempfile

import modal

APP_NAME = os.environ.get("MR_APP_NAME", "modal-runner")
GPU_TYPE = os.environ.get("MR_GPU_TYPE", "H100")
NUM_GPUS = int(os.environ.get("MR_NUM_GPUS", "1"))
TIMEOUT = int(os.environ.get("MR_TIMEOUT", "86400"))
IMAGE = os.environ.get("MR_IMAGE", "nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04")
PIP_INSTALL = os.environ.get("MR_PIP_INSTALL", "").split()
REQUIREMENTS = os.environ.get("MR_REQUIREMENTS", "").strip()
VOLUME_NAME = os.environ.get("MR_VOLUME", "modal-runner-vol")

_image = modal.Image.from_registry(IMAGE, add_python="3.11").apt_install(
    "rsync", "git", "curl"
)
if REQUIREMENTS:
    req_path = pathlib.Path(REQUIREMENTS)
    if not req_path.is_file():
        raise SystemExit(f"MR_REQUIREMENTS={REQUIREMENTS!r} is not a file")

    # Build a flat, self-contained requirements file:
    #   - inline `-r other.txt` references (so the rewritten file stands
    #     alone when Modal copies it into the build container);
    #   - rewrite local-file entries (e.g. ./flash_attn-...whl) to in-image
    #     paths after baking them via `add_local_file(copy=True)`.
    rewritten: list[str] = []

    def _inline(path: pathlib.Path, _seen: set[pathlib.Path]) -> None:
        path = path.resolve()
        if path in _seen:
            return  # avoid recursive cycles
        _seen.add(path)
        base = path.parent
        for raw in path.read_text().splitlines():
            line = raw.strip()
            # `-r foo.txt` / `--requirement foo.txt` -> inline the file.
            if line.startswith(("-r ", "--requirement ", "-c ", "--constraint ")):
                _, _, ref = line.partition(" ")
                ref = ref.strip()
                ref_path = pathlib.Path(ref) if os.path.isabs(ref) else (base / ref)
                if ref_path.is_file():
                    rewritten.append(f"# === inlined: {ref_path} ===")
                    _inline(ref_path, _seen)
                    continue
            # Detect local-file entries and bake into the image.
            if line and not line.startswith(("#", "-")):
                cand = pathlib.Path(line) if os.path.isabs(line) else (base / line)
                if cand.is_file():
                    in_image = f"/wheels/{cand.name}"
                    global _image
                    _image = _image.add_local_file(
                        str(cand.resolve()), in_image, copy=True
                    )
                    rewritten.append(in_image)
                    continue
            rewritten.append(raw)

    _inline(req_path, set())

    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", prefix="mr-req-", delete=False
    )
    tmp.write("\n".join(rewritten) + "\n")
    tmp.close()
    _image = _image.pip_install_from_requirements(tmp.name)
if PIP_INSTALL:
    _image = _image.pip_install(*PIP_INSTALL)

app = modal.App(name=APP_NAME, image=_image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

GPU_SPEC = f"{GPU_TYPE}:{NUM_GPUS}" if NUM_GPUS > 1 else GPU_TYPE
VOLUME_MNT = "/mr"


@app.function(
    gpu=GPU_SPEC,
    timeout=TIMEOUT,
    volumes={VOLUME_MNT: volume},
)
def run_script(
    script_rel: str,
    env: dict[str, str],
    mounts: dict[str, str],
) -> int:
    """Execute a shell script inside the Modal container.

    `script_rel` is the path (relative to the repo snapshot uploaded on the
    volume at `/mr/repo`) of the entry shell script.
    `mounts` maps container paths -> volume subpaths; a symlink per entry is
    created so the user script sees its original local paths (DATA_PATH etc.).
    """
    # Reload so we see any changes the runner pushed since import.
    volume.reload()

    repo = pathlib.Path(VOLUME_MNT) / "repo"
    for container_path, vol_rel in mounts.items():
        cp = pathlib.Path(container_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        if cp.is_symlink() or cp.exists():
            if cp.is_symlink():
                cp.unlink()
            else:
                subprocess.run(["rm", "-rf", str(cp)], check=True)
        target = pathlib.Path(VOLUME_MNT) / vol_rel
        target.mkdir(parents=True, exist_ok=True)
        cp.symlink_to(target)

    full_env = {**os.environ, **env}
    script = repo / script_rel
    print(f"[modal-runner] exec: bash {script}", flush=True)
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=str(repo),
        env=full_env,
    )
    # Commit any writes the script made to mounted volume paths.
    volume.commit()
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    return 0


@app.local_entrypoint()
def main(
    script_rel: str,
    env_json: str = "{}",
    mounts_json: str = "{}",
):
    import json

    env = json.loads(env_json)
    mounts = json.loads(mounts_json)
    run_script.remote(script_rel, env, mounts)
