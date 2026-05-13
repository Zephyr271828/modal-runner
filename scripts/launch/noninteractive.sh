#!/bin/bash
# Non-interactive launch: foreground `modal run`, tee logs locally, and exit
# when the modal app stops. Sourced from launch.sh.
#
# This is the file to customize when adding things like:
#   - auto-pulling outputs via `modal volume get` after a successful run
#   - posting a Slack/email notification on completion
#   - running multiple configs in sequence

echo "→ Running in non-interactive mode. Streaming logs to $LOG_FILE..."
modal run modal_ssh.py 2>&1 | tee -a "$LOG_FILE"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "✗ Modal run failed (rc=$rc). Tail of log:"
    tail -n 30 "$LOG_FILE"
    exit "$rc"
fi
echo "→ Non-interactive run complete. Modal app stopped."
