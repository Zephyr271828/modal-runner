#!/bin/bash
# Launches a Modal VM described by a YAML config. Dispatches to interactive
# (VSCode Remote-SSH) or non-interactive (batch run) flow based on the
# `interactive:` flag in the config.
#
# Usage:
#   ./launch.sh                                              # uses configs/default.yml
#   CONFIG=configs/sglang.yml ./launch.sh
#   ./launch.sh gpu.count=4 duration_hours=12 interactive=false
#   ./launch.sh +commands='[echo hi]' ~runlogs_dir           # Hydra-style edits
#
# Override syntax (Hydra subset):
#   key=value, nested.key=value   set/overwrite (value parsed as YAML)
#   +key=value                    add (error if already set)
#   ~key                          delete
#
# Customization points:
#   scripts/launch/lib.sh             — config reading, derived paths
#   scripts/launch/interactive.sh     — background modal + ssh + VSCode
#   scripts/launch/noninteractive.sh  — foreground modal + log tee + post-run hooks

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Collect positional overrides into CONFIG_OVERRIDES (consumed by src/config.py
# locally and baked into the image env so the container produces the same cfg).
# Quote each arg to preserve list/dict literals when shlex re-tokenizes.
overrides=""
for arg in "$@"; do
    overrides+=" $(printf '%q' "$arg")"
done
export CONFIG_OVERRIDES="${overrides# }"

source "$SCRIPT_DIR/scripts/launch/lib.sh"
load_config "${CONFIG:-configs/default.yml}"

if is_interactive; then
    source "$SCRIPT_DIR/scripts/launch/interactive.sh"
else
    source "$SCRIPT_DIR/scripts/launch/noninteractive.sh"
fi
