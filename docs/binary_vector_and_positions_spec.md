# Binary Vector and Position Encoding Spec

**Stability:** This document is the single source of truth for (1) pillar position encoding and (2) the 64‑dimension binary feature vector layout. Changing the vector layout breaks any trained models that use it; changes must be versioned and documented here.

---

## 1. Position encoding

All pillar indices are **0‑indexed**:

| Index | Pillar | Label (English) |
|-------|--------|------------------|
| 0     | 年柱   | year             |
| 1     | 月柱   | month            |
| 2     | 日柱   | day              |
| 3     | 時柱   | hour             |

Constants: `PILLAR_COUNT = 4`, `POSITION_LABELS = ("year", "month", "day", "hour")`. Input lists `stems` and `branches` are ordered `[Year, Month, Day, Hour]`, so `stems[0]` is year stem, `branches[3]` is hour branch.

### 1.1 Pair-based interactions (pos_a, pos_b)

For **binary** interactions (one pair of positions), each result dict uses:

- **`pos_a`**, **`pos_b`**: integer indices in `0..3`.
- **`pos_a_label`**, **`pos_b_label`** (where provided): e.g. `"year"`, `"day"`.

Used by: 天干合, 六合, 天干冲, 六冲, 六害, 无礼之刑 (子卯), 破. For 六害/六合/破, `pos_a < pos_b` (canonical order). For 冲/天干合, `pos_a` is the position of the canonical first item (`item_a`/`stem_a`), `pos_b` of the second.

### 1.2 Triplet punishments (无恩之刑, 持势之刑)

These use **all positions in a list**, not `pos_a`/`pos_b`:

- **`positions`**: `list[int]` — indices of the 2 or 3 branches that form the triplet, **in cycle order** (e.g. 寅→巳→申 for 无恩之刑, 丑→戌→未 for 持势之刑). So length is 2 or 3.
- **`members_present`**: list of branch names in the same order.
- **`is_full`**: `True` when all 3 members are present.

So triplet punishments store **all two or three positions** in one list per punishment type.

### 1.3 自刑 (self-punishment)

One dict **per self-punishment branch** (辰, 午, 酉, 亥) that appears in **two or more** positions:

- **`positions`**: `list[int]` — **all** indices where that branch appears (length ≥ 2, e.g. `[0, 2]` or `[0, 1, 2]`).
- **`branch`**: the branch character (e.g. `"辰"`).

So 自刑 always uses a **list** of positions; there can be up to 4 instances (one per self-punishment branch), each with its own `positions` list.

### 1.4 三合 (three harmony)

- **`positions`**: `list[int]` — indices of the 2 or 3 branches that form the full or half 三合 (same order as `present_branches`).

---

## 2. 64‑dimension binary vector layout

**Total length:** 64. **Dtype:** float64. **Values:** 0.0 or 1.0.

The vector encodes **presence (and some flags) per interaction type/slot**. It does **not** encode which positions are involved or how many instances; e.g. 0–3 stem clashes are collapsed to one bit per clash **pair type** (present or not). So variable-length lists (e.g. multiple clashes) map to a **fixed set of bits**: one bit per pair/type, set to 1 if that pair/type appears anywhere.

### 2.1 Index ranges and ordering

| Indices   | Content | Count | Order / notes |
|-----------|---------|-------|----------------|
| 0–4      | Stem combinations **present** (天干合) | 5 | `STEM_COMBINATION_PAIRS.keys()`: 甲己, 乙庚, 丙辛, 丁壬, 戊癸 |
| 5–9      | Stem combinations **is_pure** | 5 | Same order as above |
| 10–15    | Branch six combinations **present** (六合) | 6 | `BRANCH_SIX_PAIRS.keys()`: 子丑, 寅亥, 卯戌, 辰酉, 巳申, 午未 |
| 16–21    | Branch six combinations **is_blocked** | 6 | Same order |
| 22–25    | Three harmony **full** (三合局) | 4 | `THREE_HARMONY_GROUPS.keys()`: Water, Fire, Metal, Wood |
| 26–29    | Three harmony **half** (半合) | 4 | Same order |
| 30–35    | Stem clashes **present** (天干冲) | 6 | `sorted(STEM_CLASH_PAIRS)`: 丁癸, 丙壬, 乙辛, 己乙, 戊甲, 甲庚 (lex) |
| 36–41    | Branch clashes **present** (六冲) | 6 | `sorted(BRANCH_CLASH_PAIRS)`: 丑未, 寅申, 卯酉, 辰戌, 巳亥, 子午 (lex) |
| 42–47    | Harms **present** (六害) | 6 | 子未, 丑午, 寅巳, 卯辰, 申亥, 酉戌 (hardcoded in `to_binary_vector`) |
| 48–51    | Punishments **present** (刑, by type) | 4 | 无恩之刑, 持势之刑, 无礼之刑, 自刑 |
| 52–57    | Destructions **present** (破) | 6 | `sorted(BRANCH_DESTRUCTION_PAIRS)` |
| 58–63    | Destructions **is_weakened** | 6 | Same order as 52–57 |

**Summary order:** stem combinations → branch six → three harmony → stem clashes → branch clashes → harms → punishments → destructions (each with their secondary bits as above).

### 2.2 How variable-length lists map to the vector

- **Stem/branch clashes:** At most one detection per **pair type** (e.g. 甲庚). The vector has one bit per pair; that bit is 1 if that pair appears in **any** position pair. Positions and count are **not** encoded.
- **Harms, destructions, 六合, 天干合:** Same idea: one bit per pair (or per type for 刑); 1 = present somewhere, 0 = absent. No position or multiplicity.
- **三合:** One bit per element (Water/Fire/Metal/Wood) for “full” and one for “half”; no position list in the vector.
- **刑 (punishments):** One bit per **type** (无恩之刑, 持势之刑, 无礼之刑, 自刑). Multiple 自刑 branches or multiple triplets still only set the same type bit; no position or count.

So the 64‑dim vector is **fixed-length and presence-only**; for position or count you must use the raw `InteractionResult` lists (which do store `pos_a`/`pos_b` or `positions` as above).

### 2.3 Implementation reference

- **Code:** `bazi_engine.interactions.interaction_result.InteractionResult.to_binary_vector()`.
- **Pair/type order:** Determined by the constants and sort order in that method (see table above). Any change to ordering or to the list of pair/type slots must be reflected here and in code to keep models stable.
