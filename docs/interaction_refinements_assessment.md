# Interaction Model Refinements Assessment

## Executive Summary

ChatGPT's critique is **mostly valid**. Your current interaction model captures the *basic presence* of interactions but misses important classical refinements about *precedence, transformation conditions, and absorption patterns*. However, these are **not fatal gaps** for v1 — they are refinements that improve classical accuracy.

**Bottom line:** You should implement refinements #1, #2, and #6 (estimated +3 days to Phase 0). The others can wait for v2 or be learned implicitly by the LLM.

---

## Detailed Assessment

### ✅ 1. 三合 Overrides 六合 — **Valid and Important**

**What's missing:** When 申子辰 forms a full 三合, any 六合 using those branches (e.g., 子丑合) should be marked as *absorbed into* the larger pattern, not counted as an independent interaction.

**Current model:** Detects both independently as separate binary features.

**Impact:** Without this, your model will over-count harmonious interactions and miss that 三合局 has hierarchical dominance.

**Fix complexity:** LOW (0.5 days)
- After detecting all interactions, run a post-processing pass
- For each active 三合, mark any 六合 using the same branches as `absorbed=True`

**Recommendation:** ✅ **IMPLEMENT IN V1**

---

### ⚠️ 2. 合化 vs 合 Distinction — **Valid, Should Add**

**What's missing:** Classical bāzì distinguishes:
- **合 (meeting):** The stems/branches come together
- **化 (transformation):** They actually transform into a new element

**Transformation requires:**
1. Month branch seasonal support (月令司令)
2. Sufficient elemental strength
3. No clash interference (冲破)

**Current model:** Only detects 合 as binary presence.

**Impact:** Your model will over-report transformations that don't actually occur. Many 合 meetings don't reach the 化 threshold.

**Fix complexity:** MEDIUM (1 day)
```python
def check_transformation(combination, month_branch, element_strengths, clashes):
    # 1. Check if combination's result element is supported by month
    result_element = combination.result_element
    if not month_supports_element(month_branch, result_element):
        return False
    
    # 2. Check if result element has sufficient strength
    if element_strengths[result_element] < TRANSFORMATION_THRESHOLD:  # e.g., 0.30
        return False
    
    # 3. Check if either stem is clashed
    if any(stem in clashes for stem in combination.stems):
        return False
    
    return True
```

**Recommendation:** ✅ **IMPLEMENT IN V1** — This is a core classical distinction.

---

### ✅ 3. 冲 Can Break 三合 — **Valid, Easy to Add**

**What's missing:** A clash targeting one leg of a 三合 disrupts the formation.

**Current model:** Detects both clashes and 三合 independently.

**Example:** If you have 申子辰 (full Water 三合) but 午 is present (午子冲), the 三合 is broken.

**Fix complexity:** LOW (0.5 days)
```python
def check_sanhe_blocked(sanhe_branches, all_clashes):
    for branch in sanhe_branches:
        if branch in [clash.branch1 or clash.branch2 for clash in all_clashes]:
            return True  # blocked
    return False
```

**Recommendation:** ✅ **IMPLEMENT IN V1** — Simple and important.

---

### ⚠️ 4. Half-Combinations Are Directional — **Valid but Advanced**

**What's missing:** Classical theory distinguishes:
- **生地半合** (growth stage): 申子 (birth → growth)
- **墓地半合** (storage): 子辰 (growth → storage)
- **拱合** (arching): Missing branch must not be clashed

**Current model:** "Half = any 2 of 3 present"

**Impact:** Your model treats all half-combinations equally, missing that they have different qualities and strengths.

**Fix complexity:** MEDIUM (1 day)
- Add categorical labels for which pair is present
- Encode directional type in the feature

**Recommendation:** ⚠️ **OPTIONAL FOR V1**
- If time permits: add categorical type labels
- Otherwise: the LLM can learn differential meanings from training data if you have good examples

**Priority:** Lower than #1, #2, #3, #6

---

### ⚠️ 5. Punishments Are Conditional — **Valid but Complex**

**What's missing:** Some 刑 only activate when:
- The same element is excessive in the chart
- No combination neutralizes them

**Current model:** Binary detection of punishment patterns.

**Impact:** You'll report punishments that aren't actually "active" in the chart's dynamic.

**Fix complexity:** MEDIUM-HIGH (1-2 days)
- Requires elemental strength to be computed first
- Creates circular dependency (interactions affect strength, strength affects interaction activation)
- May need iterative resolution

**Recommendation:** ⚠️ **DEFER TO V2**
- For v1: detect 刑 as binary + add a flag `potentially_conditional=True`
- Let the LLM learn from context when they're actually relevant
- Implement full conditional logic in v2 after you have training data to verify behavior

**Priority:** Lowest of the six

---

### ✅ 6. Interaction Hierarchy Is Missing — **Most Important Gap**

**What's missing:** An explicit precedence resolver that determines which interactions are "active" vs "absorbed/suppressed."

**Classical hierarchy:**
```
1. 月令 (Seasonal authority) — dominates everything
2. 成局 (三合/三会) — formations
3. 化气 (Transformations) — stem/branch 化
4. 冲 (Clashes) — dynamic breakers
5. 六合 (Combinations) — secondary binding
6. 刑害破 (Punishments/harms/destructions) — residual effects
```

**Current model:** All interactions are independent binary features.

**Impact:** This is the **CORE CONCEPTUAL GAP**. Without hierarchy, you're giving the LLM a flat list of interactions when classical bāzì has a very specific order of precedence.

**Fix complexity:** MEDIUM (2-3 days)

**Two implementation approaches:**

**Option A — Let LLM Learn Implicitly (Recommended for v1):**
- Provide all raw interaction flags as input
- LLM learns hierarchy from training data
- Pros: No additional Phase 0 work; flexible
- Cons: Requires high-quality training data; black box

**Option B — Explicit Hierarchy Resolver:**
```python
def resolve_interaction_hierarchy(raw_interactions, month_branch):
    active = []
    absorbed = []
    
    # 1. Month seasonal authority always applies
    seasonal_element = get_seasonal_element(month_branch)
    
    # 2. Mark 三合/三会 as primary
    for sanhe in raw_interactions.sanhe:
        if not sanhe.blocked_by_clash:
            active.append(sanhe)
            # Absorb any 六合 using same branches
            absorbed.extend(find_liuhe_with_branches(sanhe.branches))
    
    # 3. Process 化气 transformations
    for combo in raw_interactions.combinations:
        if combo.transformation_active:
            active.append(combo)
        else:
            active.append(combo.as_meeting_only())
    
    # 4. Clashes are always active (they break other things)
    active.extend(raw_interactions.clashes)
    
    # 5-6. Remaining 合/刑害破 only if not absorbed
    # ...
    
    return {
        'active_interactions': active,
        'absorbed_interactions': absorbed,
        'raw_interactions': raw_interactions  # Keep for reference
    }
```

**Recommendation:** ✅ **IMPLEMENT OPTION B IN V1**
- This is the most important refinement
- It's what separates a "feature detector" from a "classical bāzì calculator"
- The explicit resolver makes your system interpretable and debuggable

**Priority:** **HIGHEST**

---

## Revised Phase 0 Timeline

### Original Estimate: 2-3 weeks

### With Refinements:

| Refinement | Priority | Time | Cumulative |
|------------|----------|------|------------|
| #1 — 三合 absorbs 六合 | High | +0.5 days | 2.5-3.5 weeks |
| #2 — 合化 conditions | High | +1 day | 2.6-3.6 weeks |
| #3 — 冲 breaks 三合 | High | +0.5 days | 2.7-3.7 weeks |
| #6 — Hierarchy resolver | **Critical** | +2 days | **3-4 weeks** |
| #4 — Directional 半合 | Medium | +1 day | 3.1-4.1 weeks (optional) |
| #5 — Conditional 刑 | Low | +1 day | 3.2-4.2 weeks (optional) |

### Recommended v1 Scope:
**Implement: #1, #2, #3, #6**
**Defer: #4, #5**

**New total estimate:** 3-4 weeks (up from 2-3)

---

## Should You Implement These Now?

### Yes, for #1, #2, #3, #6:

**Reasons:**
1. **Classical accuracy** — These are not esoteric edge cases; they're mainstream classical theory
2. **Data quality** — If your synthetic training data uses these rules, your model needs to encode them
3. **Interpretability** — The hierarchy resolver (#6) makes your system's logic transparent
4. **Downstream dependency** — Elemental strength computation depends on knowing which interactions are *actually active*

**Cost:** +3-4 days to Phase 0

**Benefit:** Proper classical foundation; better alignment with training data; debuggable logic

---

### No, for #4 and #5 (defer to v2):

**Reasons:**
1. **Diminishing returns** — The LLM can learn these subtleties from good training examples
2. **Complexity** — Conditional 刑 creates circular dependencies
3. **Ambiguity** — Even classical texts disagree on some directional 半合 nuances

**Alternative:** Flag these cases in the output and let human curators refine during Phase 3

---

## Final Answer to Your Question

> "Is ChatGPT right? Are these rules missing?"

**Yes, ChatGPT is mostly correct.** Your current interaction model is a **first-order approximation** that detects presence but misses:
- Transformation conditions (合 vs 化)
- Absorption patterns (三合 absorbs 六合)
- Breaking dynamics (冲 breaks 三合)
- Hierarchical precedence (what wins when multiple interactions compete)

**However**, not all of these are equally critical:

**Critical for v1:** #1, #2, #3, #6 (add 3-4 days)

**Optional/v2:** #4, #5 (can be learned implicitly or added later)

**The hierarchy resolver (#6) is the most important addition** — it's the difference between a feature extraction tool and a classical bāzì calculator.

---

## Recommended Next Step

Update your Phase 0 plan to add a new task:

**Task 0.3.5 — Interaction Hierarchy Resolver (NEW)**

Subtasks:
1. 三合 absorption check for 六合
2. 合化 transformation conditions
3. 冲 breaking 三合 check
4. Master hierarchy resolver function
5. Unit tests with known complex charts

**Est. time:** 3 days
**Status:** ⬜ Todo (HIGH PRIORITY)

This brings Phase 0 to **3-4 weeks total**, which is still within your original "2-3 weeks" estimate if you work efficiently.

The investment is worth it — you're building a **classical foundation**, not just a feature extractor.
