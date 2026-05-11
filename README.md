# modal-ssh

Launch a GPU VM on [Modal](https://modal.com) you can SSH into. Config-driven via YAML.

## Setup

```bash
pip install modal pyyaml
modal setup
```

Create a Modal secret named `huggingface-secret` with an `HF_TOKEN` key (or remove the `secrets:` block from your config).

### VSCode Remote-SSH (one-time)

If you'll use `launch.sh` to auto-open VSCode, add this to your VSCode `settings.json` so it skips the "Select platform" prompt for every Modal host:

```json
"remote.SSH.remotePlatform": {
    "*": "linux"
}
```

Open settings.json via Cmd+Shift+P → "Preferences: Open User Settings (JSON)".

## Run

**Foreground** (prints ssh command, you connect manually):

```bash
modal run --detach modal_ssh.py                          # configs/default.yml
CONFIG=configs/sglang.yml modal run --detach modal_ssh.py
```

**Background + auto-open VSCode** (recommended):

```bash
./launch.sh
CONFIG=configs/sglang.yml ./launch.sh
```

`launch.sh` backgrounds `modal run`, logs to `logs/<job_name>/<timestamp>.log`, writes a managed entry to `~/.ssh/config`, waits for the tunnel to be reachable, then opens VSCode Remote-SSH (if `open_vscode: true` in the config).

Stop the VM with `modal app stop <app-name>`, or let `duration_hours` elapse.

## Configure

Edit `configs/default.yml` or copy it. Fields:

| key | what |
| --- | --- |
| `app_name` | Modal app name |
| `job_name` | log subdirectory (`logs/<job_name>/...`) |
| `open_vscode` | auto-open VSCode Remote-SSH after launch |
| `base_image` | Docker image to start from |
| `add_python` | Python version Modal should bundle (omit if base has one) |
| `gpu.type` / `gpu.count` | e.g. `H100` / `4` |
| `duration_hours` | container timeout |
| `ssh_public_keys` | local pubkey paths injected into `authorized_keys` |
| `github_ssh_key` | optional: local **private** key uploaded to the VM and wired up for `github.com` |
| `git_repo` | optional: `url` (string) or `{url, dest}`. Cloned at build time; VSCode opens here. |
| `apt_packages` | installed at build time |
| `run_commands` | extra shell commands at build time |
| `volumes` | `mount_path: volume_name` (persists across runs) |
| `secrets` | Modal secret names |
| `shell_env` | vars exported in `/root/.profile`; `$VAR` resolves container-side |

See `configs/sglang.yml` for an example with conda + a repo clone baked in.
