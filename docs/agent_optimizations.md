# Agent Optimizations and Conventions

Recommendations from reviewing past agent conversations, project rules, and docs. Implemented items and optional follow-ups.

---

## Implemented (this pass)

### 1. `tasks/lessons.md` created

- **Purpose:** Self-improvement loop (AGENTS.md §3) requires updating `tasks/lessons.md` after user corrections and reviewing it at session start.
- **Change:** Created `tasks/lessons.md` with a short template and format for new entries. Path matches `.cursor/rules/project.mdc` and AGENTS.md.

### 2. AGENTS.md — Subagent (mcp_task) clarification

- **Purpose:** Past conversations showed uncertainty about "spinning off subagents"; the agent has the `mcp_task` tool and can use it.
- **Change:** Replaced the generic "use subagents liberally" bullet list with a short **Subagent Strategy (mcp_task)** section that:
  - States the agent *can* use subagents via `mcp_task`.
  - Maps subagent types to use cases: **explore** (fast codebase search), **generalPurpose** (multi-step research), **shell** (git, scripts, commands), **test-runner** (run tests, fix failures).
  - Keeps "one task per subagent" and "use in parallel when independent."

### 3. Project rule — Scripts & environment (§7)

- **Purpose:** Recurring issue: scripts fail with `ModuleNotFoundError: No module named 'torch'` when run with a different Python. No single place said "use project venv/pyenv" or "run from repo root."
- **Change:** In `.cursor/rules/project.mdc`:
  - **§6 (Testing):** Added "run pytest from repo root" and "use the same Python that has project dependencies installed."
  - **§7 (Scripts & environment):** New section: scripts need `requirements.txt` installed; `ModuleNotFoundError` usually means wrong interpreter; prefer `python -m bazi_engine` over running `calculator.py` directly (see code review doc §3.3).

---

## Implemented (follow-up pass)

### 4. Cursor skill: run-bazi-scripts-and-tests

- **Location:** `.cursor/skills/run-bazi-scripts-and-tests/SKILL.md`
- **Content:** Run CLI scripts and tests from repo root with the project Python; pytest commands; script examples (inference, training); pointer to `docs/scripts_guide.md`. Use when running/debugging scripts or when the user sees `ModuleNotFoundError` for torch/deps.

### 5. Pre-commit config

- **Location:** `.pre-commit-config.yaml`
- **Hooks:** Standard pre-commit-hooks (trailing-whitespace, end-of-file-fixer, check-yaml, check-json, check-added-large-files, check-merge-conflict, detect-private-key) plus a **local pytest** hook that runs a fast subset: `test_clashes`, `test_encoding_consistency`, `test_harms`, `test_destructions`, `test_punishments`, `test_three_harmony` with `-x -q`.
- **Setup:** `pip install pre-commit && pre-commit install`. Skip pytest on commit: `SKIP=pytest git commit ...`.

### 6. Code-review checklist rule

- **Location:** `.cursor/rules/project.mdc` §9 (Tech debt / before-merge).
- **Content:** When touching `bazi_engine/models.py` (BaziChart, to_feature_arrays, to_json, serialization) or cross-context/dynamic JSON, check `docs/code_review_issues_and_simplifications.md` for open Fix/Simplify items.

### 7. Session start: review lessons

- **AGENTS.md:** Under Self-Improvement Loop, added explicit line: "**At session start:** Read `tasks/lessons.md` and apply any listed patterns for this project."
- **Rule:** `.cursor/rules/session-start.mdc` (alwaysApply) — at session start, read `tasks/lessons.md` and apply listed patterns.

---

## References

- **Agent workflow:** `AGENTS.md`
- **Project rules:** `.cursor/rules/project.mdc`
- **Lessons log:** `tasks/lessons.md`
- **Scripts and pipeline:** `docs/scripts_guide.md`
- **Code review and simplifications:** `docs/code_review_issues_and_simplifications.md`
- **Design decisions:** `docs/design_decisions.md`
