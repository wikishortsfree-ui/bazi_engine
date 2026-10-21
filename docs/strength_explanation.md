# Simple Explanation: Elemental Strength in 合化

## The Core Question

When two stems meet (合), do they actually TRANSFORM (化) into a new element?

**Example: 甲己合**
- 甲 (Wood) + 己 (Earth) meet → they want to transform into Earth (土)
- But do they actually complete the transformation?

## What "Strength" Means Here

**Elemental strength = How much of that element is already present in the chart**

Think of it like chemical reactions:
- You need enough of the "product" element already in the environment for the reaction to complete
- If the target element is weak/absent, the transformation fizzles out

## Concrete Example

### Chart 1: Transformation SUCCEEDS (化)

```
Pillars: 甲申 戊辰 己卯 甲戌
         Wood/Metal  Earth/Earth  Earth/Wood  Wood/Earth

Elemental breakdown:
- Wood: 30% (甲甲, 卯, hidden stems)
- Earth: 35% ← HIGH (戊己, 辰戌, hidden stems)
- Metal: 20%
- Fire: 10%
- Water: 5%

Month: 辰 (Earth season)

甲己合 analysis:
✅ Result element is Earth
✅ Earth already has 35% strength (above 30% threshold)
✅ Month is Earth season (seasonal support)
✅ No clash on 甲 or 己

Result: transformation_active = TRUE
→ This is 合化 (meeting AND transformation)
→ 甲 and 己 actually become Earth energy in function
```

### Chart 2: Transformation FAILS (合而不化)

```
Pillars: 甲子 丁卯 己巳 甲子
         Wood/Water  Fire/Wood  Earth/Fire  Wood/Water

Elemental breakdown:
- Wood: 40% (甲甲, 卯, hidden stems)
- Water: 25% (子子, hidden stems)
- Fire: 20% (丁, 巳)
- Earth: 10% ← LOW (己, some hidden stems)
- Metal: 5%

Month: 卯 (Wood season, NOT Earth season)

甲己合 analysis:
✅ Result element is Earth
❌ Earth only has 10% strength (below 30% threshold)
❌ Month is Wood season (no seasonal support for Earth)
✅ No clash on 甲 or 己

Result: transformation_active = FALSE
→ This is 合而不化 (meeting but NO transformation)
→ 甲 stays Wood, 己 stays Earth - they just "like each other" but don't merge
```

## The Simple Rule

For 化 (transformation) to happen, you need **THREE conditions**:

### 1. Seasonal Support (月令)
The month must support the target element: the month branch's element either **equals** the result element or **生 (generates)** it (五行相生: 木→火, 火→土, 土→金, 金→水, 水→木).
- Earth transformations: supported by Earth months (辰戌丑未) or Fire months (巳午未), because 火生土.
- Fire transformations: supported by Fire months (巳午未) or Wood months (寅卯).
- Etc.

### 2. Elemental Strength Threshold
**The target element must already be strong in the chart (≥30%)**

Why? Classical theory says: "化气需要有根" (transformation needs roots)
- You can't create something from nothing
- The new element needs an existing foundation to "land" on
- Think of it like: the chart needs to already be "receptive" to that element

### 3. No Clash Interference
Neither stem in the combination can be clashed (冲破合)

## Why This Matters

### If transformation_active = TRUE (化):
The stems actually CHANGE FUNCTION:
- 甲 stops acting like Wood
- 己 stops acting like Earth  
- They both act as Earth energy now
- This affects:
  - Elemental balance calculation
  - Useful god determination
  - Interpretation of their ten-god roles

### If transformation_active = FALSE (合而不化):
The stems keep their original nature:
- 甲 still acts as Wood
- 己 still acts as Earth
- They just have a "harmonious relationship" (合)
- Like two people who are friendly but remain separate individuals

## Practical Impact

**Chart with 甲己合化 (transformed):**
> "Your 甲木 day master has transformed through the 甲己合化土. This means you've adopted Earth qualities - you're grounded, stable, focused on building security. The Wood's growth drive has been tempered into Earth's consolidation."

**Chart with 甲己合而不化 (meeting only):**
> "Your 甲木 day master has a 合 relationship with 己土. This creates harmony between your Wood nature (growth, ambition) and Earth energy (stability, carefulness). They work together but maintain their distinct qualities - you balance initiative with practicality."

## The Threshold: 30%

Why 30%?

This is a practical threshold used in 子平 methodology:
- Below 20%: Element is very weak, unlikely to support transformation
- 20-30%: Borderline, some schools disagree
- Above 30%: Element is strong enough to "receive" the transformation
- Above 40%: Very strong, transformation is certain

**Your implementation uses 30% as the cutoff** - this is mainstream and defensible.

## In Code

```python
def _check_transformation_conditions(
    combo: StemCombination,
    month_branch: str,
    element_strengths: dict[str, float],
    clashes: List[Clash]
) -> bool:
    
    # Condition 1: Seasonal support
    seasonal_element = _get_seasonal_element(month_branch)
    result_element = combo.result_element  # e.g., "Earth" for 甲己合
    
    if not _element_supports(seasonal_element, result_element):
        return False  # Wrong season, no transformation
    
    # Condition 2: STRENGTH THRESHOLD
    TRANSFORMATION_THRESHOLD = 0.30  # 30%
    if element_strengths.get(result_element, 0) < TRANSFORMATION_THRESHOLD:
        return False  # Not enough of the target element present
    
    # Condition 3: No clash
    for clash in clashes:
        if clash.type == "stem":
            if combo.stem1 in (clash.element1, clash.element2):
                return False
            if combo.stem2 in (clash.element1, clash.element2):
                return False
    
    return True  # All three conditions met → transformation happens
```

## Analogy

Think of transformation like trying to start a fire:

**合 (meeting)**: You have wood and a spark - they've met
**化 (transformation)**: The fire actually catches and burns

For the fire to catch, you need:
1. Good conditions (dry weather = right season)
2. Enough fuel already present (kindling = existing elemental strength)
3. No water thrown on it (no clash)

If any condition is missing, you get 合而不化 - the spark touches the wood but doesn't ignite.

## Summary in One Sentence

**"Strength" = how much of the target element already exists in the chart; you need at least 30% for the transformation to complete, because classical theory says transformations need existing "roots" to succeed.**
