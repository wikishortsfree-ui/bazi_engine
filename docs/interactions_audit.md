# Interactions module audit

Review of `bazi_engine/interactions/`: potential issues, redundancy, and optimizations.

---

## 1. Potential issues

### 1.1 Mutation of detector return values (fix at cause)

In `interaction_result.detect_all_interactions` we **mutate in place** the dicts returned by detectors:

- `branch_six_combinations`: we set `d["is_blocked"] = True/False` on each dict.
- `destructions`: we set `d["is_weakened"] = ...` on each dict.

**Risk**: If callers ever cache or reuse the raw detector output (e.g. `detect_branch_six_combinations(branches)` and later use the same list), or if we refactor to reuse references, those objects get mutated. The contract is unclear: “detectors return base result; consolidation adds flags by mutating.”

**Recommendation**: Build **new** dicts when adding `is_blocked` / `is_weakened` (e.g. `{**d, "is_blocked": True}`) so detector return values stay immutable and consolidation is explicit.

### 1.2 Inconsistent result shape across detectors

- **Clashes** (stem/branch) include `pos_a_label`, `pos_b_label` (e.g. `"year"`, `"month"`).
- Other detectors (harms, destructions, branch_six_combinations, etc.) only have `pos_a`, `pos_b`.

Not a bug, but the API is inconsistent. If we ever expose these to consumers, consider either adding labels everywhere (using a shared `POSITION_LABELS`) or documenting that only clash results have labels.

### 1.3 `to_binary_vector` relies on default for missing keys

We use `d.get("is_blocked", False)` and `d.get("is_weakened", False)` so missing keys are OK. After fixing 1.1, every consolidated result will always have these keys, so the `.get(..., False)` is just defensive. No change required, but the default is what makes the current in-place mutation “work” when reading.

---

## 2. Redundancy

### 2.1 Length validation (addressed)

Shared `_constants.py` now provides `PILLAR_COUNT`, `require_four_pillars(items, name)`, and `iter_position_pairs()`. All detectors use these; validation remains in each detector for defensive, clear errors on direct calls.

### 2.2 Pair iteration (addressed)

`_constants.iter_position_pairs()` uses `itertools.combinations(range(PILLAR_COUNT), 2)` and is used by all branch/stem pair detectors.

### 2.3 Constant type in `harms.py`

`BRANCH_SIX_HARMS` is typed and implemented as `set[frozenset[str]]`. For a fixed constant, `frozenset` of `frozenset`s is more appropriate (immutable, hashable). Minor cleanup.

---

## 3. Optimizations

### 3.1 `to_binary_vector`: avoid repeated O(n) scans

Currently we:

- For each canonical pair, call `_in_list(...)` (full scan), then often `_get_from_list(...)` (second full scan).
- So for each category we do **two** O(n) passes per pair.

**Optimization**:
- Build **lookup dicts once** at the start of `to_binary_vector`: e.g. map `frozenset({a, b}) -> dict` for stem_combinations, branch_six_combinations, stem_clashes, branch_clashes, harms, destructions.
- Then for each slot we do O(1) lookup. No need for both `_in_list` and `_get_from_list`; use a single “get” and check for presence.

This makes `to_binary_vector` O(total lists size) instead of O(64 × list lengths).

### 3.2 Single lookup instead of _in_list + _get_from_list

Where we do:

```python
if _in_list((sa, sb), self.stem_combinations, "stem_a", "stem_b"):
    vec[i] = 1.0
    d = _get_from_list(...)
```

we can do:

```python
d = _get_from_list(...)
if d is not None:
    vec[i] = 1.0
    ...
```

So one scan per pair instead of two (or, with 3.1, one O(1) lookup).

---

## 4. Summary

| Item | Severity | Action |
|------|----------|--------|
| Mutate detector outputs (1.1) | Medium | Build new dicts when adding is_blocked / is_weakened |
| to_binary_vector double scan + no index (3.1, 3.2) | Low (perf) | Build lookup dicts once; use single get per slot |
| BRANCH_SIX_HARMS as set (2.3) | Low | Use frozenset for the constant |
| Repeated len==4 (2.1) | Optional | Keep as-is or introduce PILLAR_COUNT |
| Inconsistent labels (1.2) | Optional | Document or add labels everywhere |
| Pair loop helper (2.2) | Optional | Extract if we add more detectors |

Implementing the cause-level fix (1.1) and the to_binary_vector + constant cleanup is recommended; the rest can be done incrementally.
