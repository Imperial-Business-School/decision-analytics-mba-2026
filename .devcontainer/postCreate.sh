#!/usr/bin/env bash
# Runs once when the Codespace is created. Sets up the one shared Python
# environment + Jupyter kernel everyone (teacher and students alike) uses.
set -e

python3 -m pip install --upgrade pip

# Editable install: decision-suite/ is committed straight into this repo, so
# dragging a newer version of it in later (delete the old folder, paste the
# new one in whole) takes effect immediately, no reinstall needed — just
# restart the Jupyter kernel afterward, since an already-running kernel
# keeps already-imported modules cached in memory.
pip install -e "./decision-suite[examples]"

# gurobipy's own bundled free "size-limited" license needs no setup beyond
# this install (see 2026/ROADMAP.md Section 11 for the verification).
pip install gurobipy ipykernel

python3 -m ipykernel install --user --name decision-suite \
  --display-name "Python (decision-suite)"

npm install -g @anthropic-ai/claude-code

echo "Setup complete. Kernel 'Python (decision-suite)' is ready."
