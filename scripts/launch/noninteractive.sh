#!/bin/bash
# Non-interactive launch: foreground `modal run`, tee logs locally, and exit
# when the modal app stops. Sourced from launch.sh.
#
# This is the file to customize when adding things like:
#   - auto-pulling outputs via `modal volume get` after a successful run
#   - posting a Slack/email notification on completion
#   - running multiple configs in sequence

echo "→ Running in non-interactive mode (background). Logs: $LOG_FILE"

# --detach: keep the app running on Modal's infra after this local process exits.
# When run_batch finishes (success or failure), the function returns and the
# app stops automatically — no need for the user to clean up.
nohup modal run --detach modal_ssh.py >> "$LOG_FILE" 2>&1 &
MODAL_PID=$!
disown "$MODAL_PID" 2>/dev/null || true

# Wait briefly to surface any immediate auth/config errors instead of
# silently returning while the modal CLI is about to die.
for i in 1 2 3 4 5; do
    sleep 1
    if ! kill -0 "$MODAL_PID" 2>/dev/null; then
        echo "✗ Modal process exited within 5s. Tail of log:"
        tail -n 30 "$LOG_FILE"
        exit 1
    fi
done

echo "→ Job submitted. PID=$MODAL_PID"
echo "  Follow logs: tail -f $LOG_FILE"
echo "  Stop early:  modal app list  &&  modal app stop <app-name>"
