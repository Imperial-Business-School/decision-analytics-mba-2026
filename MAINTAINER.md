# Maintainer notes

This repo is a blank template. The teacher and every student launch the
exact same Codespace, then drag in their own files (`teacher-view/` from
the main `decision-analytics` repo, or `student-view/<class>` for
students). No course content is committed here, only the environment:
Python, Jupyter, `decision-suite` (editable), `gurobipy`, VS Code +
Copilot extensions, Claude Code CLI.

## Updating the package

`decision-suite/` is committed directly into this repo and installed
*editable*, so a newer version can be dropped straight in:

1. Delete the existing `decision-suite/` folder first. Don't paste on top
   of it, a plain overwrite won't remove files the old version had that
   the new one doesn't.
2. Paste the new `decision-suite/` folder in.
3. Restart the Jupyter kernel in any notebook that already imported it. A
   running kernel keeps already-imported modules cached in memory, so it
   won't see the change until it restarts. No reinstall needed.

If the update adds a new dependency, not just changed code, also rerun:

```
pip install -e "./decision-suite[examples]"
```

## Updating the container itself

Edit `.devcontainer/devcontainer.json` / `postCreate.sh`, then rebuild the
container (Codespaces: Command Palette, "Codespaces: Rebuild Container").
