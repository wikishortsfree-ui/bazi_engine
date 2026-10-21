# Agent Lessons (Self-Improvement)

Patterns and corrections to avoid repeating. **After any user correction**, append a short entry here and add or update rules to prevent the same mistake.

---

## Format for new entries

- **What happened:** One-line description.
- **Rule/fix:** What to do instead (or which rule to follow).
- **Date:** YYYY-MM-DD (optional).

---

## Lessons

- **What happened:** Jinja2 template used `grp.items` to access a dict key named `"items"`, but Jinja2 resolved it to the `dict.items()` method — caused a 500 error on page load.
- **Rule/fix:** In Jinja2, when a dict key collides with a Python dict method name (`items`, `keys`, `values`, `get`, `update`, `pop`), always use bracket notation `obj['key']` instead of dot access.
- **Date:** 2026-03-19

- **What happened:** CJK characters (壬, 丁, 丑, 寅) rendered incorrectly in the browser because `font-weight: 700` triggered synthetic bold, corrupting strokes.
- **Rule/fix:** When displaying CJK text in HTML, use `font-weight: 500` max and specify an explicit CJK font stack (`PingFang SC`, `Noto Sans SC`, etc.). Always visually verify CJK rendering in a browser before marking done.
- **Date:** 2026-03-19

- **What happened:** Flask dev server failed to start on macOS because port 5000 is used by AirPlay Receiver.
- **Rule/fix:** On macOS, default to port 5050 or 8080 for dev servers. Mention this in usage docs.
- **Date:** 2026-03-19

- **What happened:** The 64-token projection trained on Phase 2 prose CE plateaued at ~30% factual probe accuracy with a 0% / 75% split between answer-in-one-token categories (pillar identity, ten gods) and answer-in-binary-vector categories (interactions); long-form CE smeared per-fact gradient signal across hundreds of prose tokens.
- **Rule/fix:** Train prefix/projection adapters on the **downstream decoding objective** (short factual QA with the loss masked to the answer span only) before falling back to scaling data or unfreezing more parameters. When a probe metric splits cleanly along answer-length lines, suspect the loss-mask geometry, not model capacity.
- **Date:** 2026-04-27
