#!/bin/bash
# Shared helpers for launch.sh. Sourced — never executed directly.
#
# Exports (after load_config):
#   CONFIG_PATH  JOB_NAME  INTERACTIVE  REPO_DEST  REMOTE_DIR
#   LOG_DIR      LOG_FILE  SSH_CONFIG   KEY_PATH

# Extract a single top-level field from the config YAML via python, applying
# any CONFIG_OVERRIDES first so the shell dispatch agrees with src/config.py.
read_cfg() {
    CONFIG_PATH="$CONFIG_PATH" KEY="$1" DEFAULT="$2" python3 -c "
import os, yaml
from src.overrides import apply
cfg = yaml.safe_load(open(os.environ['CONFIG_PATH']))
apply(cfg, os.environ.get('CONFIG_OVERRIDES', ''))
print(cfg.get(os.environ['KEY'], os.environ['DEFAULT']))
"
}

# Compute the repo destination from `git_repo` (string or {url, dest}).
# Empty when no git_repo is configured.
_compute_repo_dest() {
    CONFIG_PATH="$CONFIG_PATH" python3 -c "
import os, yaml
from src.overrides import apply
cfg = yaml.safe_load(open(os.environ['CONFIG_PATH']))
apply(cfg, os.environ.get('CONFIG_OVERRIDES', ''))
r = cfg.get('git_repo')
if not r:
    print('')
else:
    if isinstance(r, dict):
        url = r['url']
        dest = r.get('dest')
    else:
        url, dest = r, None
    if not dest:
        dest = '/root/' + os.path.basename(url.rstrip('/')).removesuffix('.git')
    print(dest)
"
}

load_config() {
    CONFIG_PATH="${1:-configs/default.yml}"
    JOB_NAME=$(read_cfg job_name default)
    INTERACTIVE=$(read_cfg interactive true)
    REPO_DEST=$(_compute_repo_dest)

    # Modal profile to use (from ~/.modal.toml). Overridable: `modal_profile=research`.
    export MODAL_PROFILE=$(read_cfg modal_profile default)
    echo "→ Modal profile: $MODAL_PROFILE"

    LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/$JOB_NAME}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_FILE:-$LOG_DIR/$(date +%Y%m%d_%H%M%S).log}"
    SSH_CONFIG="$HOME/.ssh/config"
    KEY_PATH="${KEY_PATH:-$HOME/.ssh/id_ed25519_modal}"
    REMOTE_DIR="${REMOTE_DIR:-${REPO_DEST:-/root}}"

    : > "$LOG_FILE"
    echo "→ Launching Modal VM (logs: $LOG_FILE)"
    echo "→ Config: $CONFIG_PATH  |  Job: $JOB_NAME  |  interactive: $INTERACTIVE"
}

is_interactive() {
    [ "$INTERACTIVE" = "True" ] || [ "$INTERACTIVE" = "true" ]
}
