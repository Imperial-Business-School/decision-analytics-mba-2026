# Business Analytics — Codespaces environment

One blank template: launch this exact same Codespace whether you're the
teacher or a student. It carries the environment only — Python, Jupyter,
`decision-suite` (editable), `gurobipy`, VS Code + Copilot extensions,
Claude Code CLI — no course content.

## After launching

Drag and drop your own files in:

- **Teacher:** the contents of `teacher-view/` from the main
  `decision-analytics` repo (slides, case PDFs, `CLASS-GUIDE.md`,
  solution notebooks).
- **Student:** whatever's provided for the current class
  (`student-view/<class>`).

Nothing needs installing or configuring — the environment is already set
up. Open a notebook, pick the kernel named **Python (decision-suite)**,
and go (each notebook's own saved kernel selection should pick this up
automatically once it exists).

## Updating the package later

`decision-suite/` is committed directly into this repo and installed
*editable* — so a newer version can be dropped straight in:

1. Delete the existing `decision-suite/` folder first (don't paste on top
   of it — a plain overwrite won't remove files the old version had that
   the new one doesn't).
2. Paste the new `decision-suite/` folder in.
3. **Restart the Jupyter kernel** in any notebook that already imported
   it — a running kernel keeps already-imported modules cached in memory,
   so it won't see the change until it restarts. No reinstall needed.

If the update adds a new dependency (not just changed code), also rerun:

```
pip install -e "./decision-suite[examples]"
```

## Updating the container itself

Edit `.devcontainer/devcontainer.json` / `postCreate.sh`, then rebuild the
container (Codespaces: Command Palette → "Codespaces: Rebuild Container").
