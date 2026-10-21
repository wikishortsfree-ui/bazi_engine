## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task [3+ steps or architectural decisions]
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy (mcp_task)
- **You can use subagents** via the `mcp_task` tool; use it liberally to keep main context clean.
- **One task per subagent** for focused execution. Prefer parallel subagents when tasks are independent.
- **When to use which type:**
  - **explore**: Fast codebase search (files by pattern, keywords, "how does X work?"). Use for quick exploration; specify thoroughness: quick / medium / very thorough.
  - **generalPurpose**: Multi-step research, code search when not confident of a quick match, complex multi-step tasks.
  - **shell**: Git, terminal commands, running scripts. Use for execution that doesn’t need deep code reasoning.
  - **test-runner**: Running tests, analyzing failures, fixing tests while preserving intent. Use proactively when code changes are made.
- Offload research, exploration, and parallel analysis to subagents; for complex problems, use multiple subagents rather than one overloaded thread.

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write global rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- **At session start:** Read `tasks/lessons.md` and apply any listed patterns for this project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: 'Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### 7. Retrospective (extract lessons from conversation)
- **Universal:** The workflow lives in **~/.cursor/rules/retrospective.mdc** and **~/.cursor/skills/retro/** — use **/retro** or say "retro" / "extract lessons"; the rule and skill are self-contained and work in any project.
- **In this project:** Lessons go in `tasks/lessons.md`. The checklist format below is duplicated in the universal rule for reference.
- **Do not auto-apply:** Only output the checklist. Do not edit `tasks/lessons.md`, create skills, or add rules unless the user explicitly says "apply" or "add these".
- **Optional:** When you have just completed a non-trivial task, you may add: "Want a recap? Say **/retro** to extract lessons, skills, and rules from this chat."

**Output format (same as ~/.cursor/rules/retrospective.mdc):**

```markdown
## Retrospective

### 1. Lessons (for tasks/lessons.md)
- **What happened:** (one line)
- **Rule/fix:** (what to do instead / which rule to follow)
(Add more bullets if needed; else "—")

### 2. Skills (candidate .cursor/skills/)
- **Trigger/name:** (when would the agent use this?)
- **One-line description:** (scope and when to use)
(Repeat or "—")

### 3. Rules (candidate .cursor/rules/ or AGENTS.md)
- **Rule:** (short directive)
- **Where:** (project.mdc | AGENTS.md | new file)
(Repeat or "—")

### 4. Subagent usage
- **Pattern:** (when to use explore / generalPurpose / shell / test-runner based on this conversation)
(Or "—")

### 5. Commands / scripts
- **What:** (command or script to document or automate)
- **When:** (when it would have helped)
(Or "—")
```

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.