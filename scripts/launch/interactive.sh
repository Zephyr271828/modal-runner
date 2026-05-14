#!/bin/bash
# Interactive launch: backgrounds modal AND all the SSH/VSCode follow-up work
# so launch.sh returns immediately. Watch `$LOG_FILE` to follow progress.

# Run modal in the background. --detach keeps the container alive after this
# process exits.
nohup modal run --detach modal_ssh.py >> "$LOG_FILE" 2>&1 &
MODAL_PID=$!
disown "$MODAL_PID" 2>/dev/null || true

# Wait briefly to surface immediate auth/config errors instead of returning
# while modal is about to die.
for i in 1 2 3 4 5; do
    sleep 1
    if ! kill -0 "$MODAL_PID" 2>/dev/null; then
        echo "✗ Modal process exited within 5s. Tail of log:"
        tail -n 30 "$LOG_FILE"
        exit 1
    fi
done

# Fork the SSH-poll → ssh-config update → tunnel probe → VSCode open work into
# a detached subshell. Its output streams into the same $LOG_FILE so the user
# can follow with `tail -f`.
(
    set +e
    echo "→ Waiting for SSH endpoint..." >> "$LOG_FILE"
    while ! grep -qE '^\s*ssh -i ' "$LOG_FILE" 2>/dev/null; do
        kill -0 "$MODAL_PID" 2>/dev/null || {
            echo "✗ Modal process exited before SSH was ready." >> "$LOG_FILE"
            exit 1
        }
        sleep 1
    done

    SSH_LINE=$(grep -E '^\s*ssh -i ' "$LOG_FILE" | head -1 | sed 's/^[[:space:]]*//')
    HOST=$(echo "$SSH_LINE" | awk -F'root@' '{print $2}' | awk '{print $1}')
    PORT=$(echo "$SSH_LINE" | grep -oE '\-p [0-9]+' | awk '{print $2}')
    echo "→ Host: $HOST  Port: $PORT" >> "$LOG_FILE"

    # Update ~/.ssh/config — single managed block, keyed by the marker so each
    # run cleans up the previous entry.
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
    echo "→ Updated $SSH_CONFIG (Host $HOST)" >> "$LOG_FILE"

    # Probe the tunnel end-to-end. sshd binding to :22 inside the container
    # happens a few seconds before Modal's public tunnel actually routes.
    echo "→ Probing SSH endpoint..." >> "$LOG_FILE"
    SSH_OK=""
    for i in $(seq 1 30); do
        if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no \
               -o UserKnownHostsFile=/dev/null \
               "$HOST" true 2>/dev/null; then
            SSH_OK=1
            break
        fi
        sleep 2
    done

    if [ -z "$SSH_OK" ]; then
        echo "⚠ Could not reach SSH after 60s. Skipping VSCode launch." >> "$LOG_FILE"
        exit 1
    fi
    echo "→ SSH reachable." >> "$LOG_FILE"

    if command -v code >/dev/null 2>&1; then
        echo "→ Opening VSCode Remote-SSH..." >> "$LOG_FILE"
        code --disable-workspace-trust \
             --folder-uri "vscode-remote://ssh-remote+${HOST}${REMOTE_DIR}" \
             >> "$LOG_FILE" 2>&1
    else
        echo "(VSCode 'code' CLI not on PATH — skipping window open)" >> "$LOG_FILE"
    fi
) >/dev/null 2>&1 &
disown $! 2>/dev/null || true

echo "→ Job submitted. PID=$MODAL_PID"
echo "  Follow logs:  tail -f $LOG_FILE"
echo "  VSCode will open automatically when the SSH tunnel is up (~30-60s)."
echo "  Stop early:   modal app list  &&  modal app stop <app-name>"
