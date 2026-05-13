#!/bin/bash
# Interactive launch: background `modal run --detach`, wait for SSH endpoint,
# rewrite ~/.ssh/config, probe the tunnel, open VSCode. Sourced from launch.sh.

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

if command -v code >/dev/null 2>&1; then
    echo "→ Opening VSCode Remote-SSH..."
    code --disable-workspace-trust --folder-uri "vscode-remote://ssh-remote+${HOST}${REMOTE_DIR}"
else
    echo "(VSCode 'code' CLI not on PATH — skipping window open)"
fi

echo
echo "Modal app keeps running in the background (--detach). Stop with:"
echo "  modal app list && modal app stop <app-name>"
