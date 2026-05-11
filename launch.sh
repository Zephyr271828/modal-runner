#!/bin/bash
# Launches the Modal VM in the background, waits for SSH details,
# updates ~/.ssh/config with a `modal-vm` host entry, and opens VSCode Remote-SSH.
#
# Usage:
#   ./launch.sh                          # uses configs/default.yml
#   CONFIG=configs/sglang.yml ./launch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG:-configs/default.yml}"

# Extract fields from the YAML config via python.
read_cfg() {
    python3 -c "
import sys, yaml
cfg = yaml.safe_load(open('$CONFIG_PATH'))
print(cfg.get('$1', '$2'))
"
}
JOB_NAME=$(read_cfg job_name default)
OPEN_VSCODE=$(read_cfg open_vscode true)

# Compute repo dest from git_repo (string or {url, dest}). Empty if not set.
REPO_DEST=$(python3 -c "
import yaml, os
cfg = yaml.safe_load(open('$CONFIG_PATH'))
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
")

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/$JOB_NAME}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$(date +%Y%m%d_%H%M%S).log}"
SSH_CONFIG="$HOME/.ssh/config"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/id_ed25519_modal}"
# If a git_repo is configured, open VSCode in that directory; otherwise /root.
REMOTE_DIR="${REMOTE_DIR:-${REPO_DEST:-/root}}"

: > "$LOG_FILE"
echo "→ Launching Modal VM (logs: $LOG_FILE)"
echo "→ Config: $CONFIG_PATH  |  Job: $JOB_NAME  |  open_vscode: $OPEN_VSCODE"

# Run modal in the background. --detach keeps the container alive after this process exits.
nohup modal run --detach modal_ssh.py >> "$LOG_FILE" 2>&1 &
MODAL_PID=$!
disown "$MODAL_PID" 2>/dev/null || true

cleanup_on_fail() {
    echo "✗ Modal process exited before SSH was ready. Tail of log:"
    tail -n 30 "$LOG_FILE"
    exit 1
}

echo "→ Waiting for SSH endpoint..."
while ! grep -qE '^\s*ssh -i ' "$LOG_FILE" 2>/dev/null; do
    kill -0 "$MODAL_PID" 2>/dev/null || cleanup_on_fail
    sleep 1
done

SSH_LINE=$(grep -E '^\s*ssh -i ' "$LOG_FILE" | head -1 | sed 's/^[[:space:]]*//')
HOST=$(echo "$SSH_LINE" | awk -F'root@' '{print $2}' | awk '{print $1}')
PORT=$(echo "$SSH_LINE" | grep -oE '\-p [0-9]+' | awk '{print $2}')

echo "→ Host: $HOST  Port: $PORT"

# Update ~/.ssh/config: replace the single managed block (keyed by the marker,
# not the host, so each run cleans up the previous entry).
mkdir -p "$HOME/.ssh"
touch "$SSH_CONFIG"
BEGIN_MARK="# >>> modal-vm >>>"
END_MARK="# <<< modal-vm <<<"
awk -v b="$BEGIN_MARK" -v e="$END_MARK" '
    $0 == b {skip=1; next}
    $0 == e {skip=0; next}
    !skip {print}
' "$SSH_CONFIG" > "$SSH_CONFIG.tmp" && mv "$SSH_CONFIG.tmp" "$SSH_CONFIG"

cat >> "$SSH_CONFIG" <<EOF
$BEGIN_MARK
Host $HOST
    Port $PORT
    User root
    IdentityFile $KEY_PATH
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
$END_MARK
EOF

echo "→ Updated $SSH_CONFIG (Host $HOST)"

# Probe the SSH endpoint end-to-end. The container thinks sshd is up the moment it binds
# localhost:22, but the Modal tunnel takes a couple more seconds to route public traffic.
echo "→ Probing SSH endpoint..."
for i in $(seq 1 30); do
    if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no \
           -o UserKnownHostsFile=/dev/null \
           "$HOST" true 2>/dev/null; then
        echo "→ SSH reachable."
        SSH_OK=1
        break
    fi
    sleep 2
done

echo
echo "Connect with:"
echo "  ssh $HOST"
echo "  $SSH_LINE"
echo

if [ -z "${SSH_OK:-}" ]; then
    echo "⚠ Could not reach SSH after 60s. Not opening VSCode. Tail of log:"
    tail -n 20 "$LOG_FILE"
    exit 1
fi

if [ "$OPEN_VSCODE" = "True" ] || [ "$OPEN_VSCODE" = "true" ]; then
    if command -v code >/dev/null 2>&1; then
        echo "→ Opening VSCode Remote-SSH..."
        code --disable-workspace-trust --folder-uri "vscode-remote://ssh-remote+${HOST}${REMOTE_DIR}"
    else
        echo "(VSCode 'code' CLI not on PATH — skipping window open)"
    fi
else
    echo "(open_vscode=false — skipping VSCode launch)"
fi

echo
echo "Modal app keeps running in the background (--detach). Stop with:"
echo "  modal app list && modal app stop <app-name>"
