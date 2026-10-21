# Fix: Data Leakage in Training Pipeline

**Issue**: Current training has catastrophic data leakage. The model receives ground truth answers as input features, allowing it to simply copy inputs to outputs instead of learning bāzì logic. This explains the impossibly low initial training loss (0.054 instead of expected ~2.3).

**Root cause**: `BaziChart.to_feature_arrays()` returns ALL computed features (ten-gods, interactions, elemental strengths, useful god) as input, when these should ONLY be ground truth targets for validation.

---

## Required Changes

### 1. Create Separate Input and Target Methods in `BaziChart`

**File**: `bazi_engine/chart.py` (or wherever `BaziChart` is defined)

**Add two new methods** to replace or supplement `to_feature_arrays()`:

```python
def to_input_features(self, task_name: str) -> dict:
    """
    Returns ONLY the minimal input features needed for a specific task.
    These are the raw chart structure that the encoder must learn from.
    
    NEVER include ground truth targets (ten-gods, interactions, strengths, useful_god) as input!
    
    Args:
        task_name: One of 'ten_god', 'element_strength', 'interaction', 'useful_god', 'cross_temporal'
    
    Returns:
        Dictionary of numpy arrays with minimal input features only
    """
    # Base features: raw chart structure (always included)
    base_features = {
        'stem_ids': np.array([STEM_TO_ID[p.stem] for p in self.natal.pillars], dtype=np.int64),
        'branch_ids': np.array([BRANCH_TO_ID[p.branch] for p in self.natal.pillars], dtype=np.int64),
        'stem_elements': np.array([ELEMENT_TO_ID[STEM_ELEMENTS[p.stem]] for p in self.natal.pillars], dtype=np.int64),
        'stem_polarities': np.array([POLARITY_TO_ID[STEM_POLARITIES[p.stem]] for p in self.natal.pillars], dtype=np.int64),
        'pillar_positions': np.array([0, 1, 2, 3], dtype=np.int64),  # year, month, day, hour
    }
    
    # Task-specific additions
    if task_name == 'cross_temporal':
        # For cross-temporal task, add temporal pillar structure (but NOT computed interactions!)
        if self.dynamic is None:
            raise ValueError("Cross-temporal task requires dynamic context, but chart has none")
        
        base_features.update({
            'dayun_stem_id': np.array([STEM_TO_ID[self.dynamic.dayun.stem]], dtype=np.int64),
            'dayun_branch_id': np.array([BRANCH_TO_ID[self.dynamic.dayun.branch]], dtype=np.int64),
            'liunian_stem_id': np.array([STEM_TO_ID[self.dynamic.liunian.stem]], dtype=np.int64),
            'liunian_branch_id': np.array([BRANCH_TO_ID[self.dynamic.liunian.branch]], dtype=np.int64),
            'liuyue_stem_id': np.array([STEM_TO_ID[self.dynamic.liuyue.stem] if self.dynamic.liuyue else -1], dtype=np.int64),
            'liuyue_branch_id': np.array([BRANCH_TO_ID[self.dynamic.liuyue.branch] if self.dynamic.liuyue else -1], dtype=np.int64),
            'pillar_presence_mask': np.array([
                1 if self.dynamic.dayun else 0,
                1 if self.dynamic.liunian else 0,
                1 if self.dynamic.liuyue else 0
            ], dtype=np.float32),
        })
    
    return base_features


def to_ground_truth(self, task_name: str) -> dict:
    """
    Returns ground truth targets for validation/training loss for a specific task.
    These are the answers the model must learn to predict.
    
    Args:
        task_name: One of 'ten_god', 'element_strength', 'interaction', 'useful_god', 'cross_temporal'
    
    Returns:
        Dictionary of numpy arrays with ground truth targets
    """
    if task_name == 'ten_god':
        # Target: ten-god class for each of 4 pillars
        ten_god_ids = np.array([
            TEN_GOD_TO_ID[self.natal.ten_gods[i]] for i in range(4)
        ], dtype=np.int64)
        return {'ten_god_ids': ten_god_ids}
    
    elif task_name == 'element_strength':
        # Target: 5-dimensional elemental strength vector (sums to 1.0)
        elemental_strengths = np.array([
            self.natal.elemental_strengths['木'],
            self.natal.elemental_strengths['火'],
            self.natal.elemental_strengths['土'],
            self.natal.elemental_strengths['金'],
            self.natal.elemental_strengths['水'],
        ], dtype=np.float32)
        return {'elemental_strengths': elemental_strengths}
    
    elif task_name == 'interaction':
        # Target: 64-dimensional binary interaction vector
        interaction_vector = self.natal.interaction_result.to_binary_vector()
        return {'interaction_vector': interaction_vector}
    
    elif task_name == 'useful_god':
        # Target: useful god class (0-4 for 5 types, or 5 for "ambiguous" if id=-1)
        useful_god_id = self.natal.useful_god_id if self.natal.useful_god_id != -1 else 5
        return {'useful_god_id': np.array([useful_god_id], dtype=np.int64)}
    
    elif task_name == 'cross_temporal':
        # Target: 66-dimensional binary cross-temporal interaction vector
        if self.dynamic is None:
            raise ValueError("Cross-temporal task requires dynamic context, but chart has none")
        
        dynamic_interaction_vector = self.dynamic.cross_interaction_vector  # 66-dim binary
        return {'dynamic_interaction_vector': dynamic_interaction_vector}
    
    else:
        raise ValueError(f"Unknown task_name: {task_name}")
```

---

### 2. Update All Dataset Classes

**Files**: `bazi_model/data/dataset.py` (or wherever `ChartDataset` is defined)

**Modify `__getitem__` method** to use the new separated methods:

```python
class ChartDataset(Dataset):
    def __init__(self, jsonl_path: str, task_name: str, temporal_only: bool = False):
        """
        Args:
            jsonl_path: Path to charts.jsonl
            task_name: One of 'ten_god', 'element_strength', 'interaction', 'useful_god', 'cross_temporal'
            temporal_only: If True, filter to only charts with dynamic context (for cross_temporal task)
        """
        self.task_name = task_name
        self.temporal_only = temporal_only
        
        # Load charts from JSONL
        self.charts = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                chart_dict = json.loads(line)
                chart = BaziChart.from_dict(chart_dict)
                
                # Filter for temporal_only if needed
                if temporal_only and chart.dynamic is None:
                    continue
                
                self.charts.append(chart)
    
    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        chart = self.charts[idx]
        
        # Get minimal input features (NO ground truth!)
        inputs = chart.to_input_features(self.task_name)
        
        # Get ground truth targets (for loss calculation)
        targets = chart.to_ground_truth(self.task_name)
        
        # Convert to tensors
        inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
        targets = {k: torch.from_numpy(v) for k, v in targets.items()}
        
        return inputs, targets
    
    def __len__(self) -> int:
        return len(self.charts)
```

---

### 3. Update All Training Scripts

**Files**: `scripts/train_ten_god.py`, `scripts/train_element_strength.py`, `scripts/train_interaction.py`, `scripts/train_useful_god.py`, `scripts/train_cross_temporal.py`, `scripts/train_multi_task.py`

**Modify dataset initialization** to specify task_name:

#### Example: `scripts/train_ten_god.py`

```python
# OLD (wrong):
# train_dataset = ChartDataset(args.dataset)

# NEW (correct):
train_dataset = ChartDataset(
    jsonl_path=args.dataset,
    task_name='ten_god',  # Specifies which input/target pair to use
    temporal_only=False
)

val_dataset = ChartDataset(
    jsonl_path=args.dataset,
    task_name='ten_god',
    temporal_only=False
)
```

#### Example: `scripts/train_element_strength.py`

```python
train_dataset = ChartDataset(
    jsonl_path=args.data,
    task_name='element_strength',
    temporal_only=False
)
```

#### Example: `scripts/train_interaction.py`

```python
train_dataset = ChartDataset(
    jsonl_path=args.data,
    task_name='interaction',
    temporal_only=False
)
```

#### Example: `scripts/train_useful_god.py`

```python
train_dataset = ChartDataset(
    jsonl_path=args.data,
    task_name='useful_god',
    temporal_only=False
)
```

#### Example: `scripts/train_cross_temporal.py`

```python
train_dataset = ChartDataset(
    jsonl_path=args.data,
    task_name='cross_temporal',
    temporal_only=True  # Only charts with dynamic context
)
```

#### Example: `scripts/train_multi_task.py`

For multi-task, you need **separate datasets per task** (or modify the dataset to handle multiple tasks):

```python
# Option A: One dataset per task (simpler)
ten_god_dataset = ChartDataset(args.data, task_name='ten_god', temporal_only=False)
element_dataset = ChartDataset(args.data, task_name='element_strength', temporal_only=False)
interaction_dataset = ChartDataset(args.data, task_name='interaction', temporal_only=False)
useful_god_dataset = ChartDataset(args.data, task_name='useful_god', temporal_only=False)

# Then in training loop, sample from each dataset and combine losses

# Option B: Multi-task dataset (more complex, returns all targets)
# Create MultiTaskDataset that returns inputs once + all 4 ground truths
```

---

### 4. Verify Encoder Input Shapes

**Files**: `bazi_model/encoders/natal_encoder.py`, `bazi_model/encoders/dynamic_encoder.py`

Ensure encoders expect ONLY the minimal input features:

```python
class NatalEncoder(nn.Module):
    def forward(self, inputs: dict) -> torch.Tensor:
        """
        Args:
            inputs: Dictionary with keys:
                - stem_ids: (batch, 4) int64
                - branch_ids: (batch, 4) int64
                - stem_elements: (batch, 4) int64
                - stem_polarities: (batch, 4) int64
                - pillar_positions: (batch, 4) int64
                
                MUST NOT contain: ten_god_ids, interaction_vector, elemental_strengths, useful_god_id
        
        Returns:
            natal_emb: (batch, d_model) float32
        """
        # Embed each input feature
        stem_emb = self.stem_embedding(inputs['stem_ids'])  # (batch, 4, d_emb)
        branch_emb = self.branch_embedding(inputs['branch_ids'])  # (batch, 4, d_emb)
        # ... etc
        
        # Concatenate and pass through transformer
        # ...
        
        return natal_emb
```

---

### 5. Update Task Heads

**Files**: `bazi_model/tasks/ten_god_head.py`, `bazi_model/tasks/element_strength_head.py`, etc.

Ensure heads receive ONLY encoder embeddings, not raw features:

```python
class TenGodHead(nn.Module):
    def forward(self, natal_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            natal_emb: (batch, d_model) - encoder output ONLY
        
        Returns:
            logits: (batch, 4, 11) - ten-god logits for 4 pillars × 11 classes
        """
        logits = self.classifier(natal_emb)  # (batch, 4*11)
        logits = logits.view(-1, 4, 11)
        return logits
```

---

## Testing the Fix

### 1. Sanity Check: Print Input Features

Add this to your training script before the first epoch:

```python
# After creating dataset
train_dataset = ChartDataset(args.data, task_name='ten_god', temporal_only=False)

# Get one sample
inputs, targets = train_dataset[0]

print("=== INPUT FEATURES (should NOT contain answers) ===")
print(f"Input keys: {inputs.keys()}")
for k, v in inputs.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

print("\n=== GROUND TRUTH TARGETS (the answers) ===")
print(f"Target keys: {targets.keys()}")
for k, v in targets.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

# Verify no overlap
assert set(inputs.keys()).isdisjoint(set(targets.keys())), "ERROR: Input and targets overlap!"
print("\n✅ No overlap between inputs and targets")
```

**Expected output for ten_god task:**
```
=== INPUT FEATURES (should NOT contain answers) ===
Input keys: dict_keys(['stem_ids', 'branch_ids', 'stem_elements', 'stem_polarities', 'pillar_positions'])
  stem_ids: shape=torch.Size([4]), dtype=torch.int64
  branch_ids: shape=torch.Size([4]), dtype=torch.int64
  stem_elements: shape=torch.Size([4]), dtype=torch.int64
  stem_polarities: shape=torch.Size([4]), dtype=torch.int64
  pillar_positions: shape=torch.Size([4]), dtype=torch.int64

=== GROUND TRUTH TARGETS (the answers) ===
Target keys: dict_keys(['ten_god_ids'])
  ten_god_ids: shape=torch.Size([4]), dtype=torch.int64

✅ No overlap between inputs and targets
```

### 2. Verify Initial Loss is Correct

After fixing, the first epoch should show:

```
Epoch 1/30  train_loss=2.303  val_loss=2.305
```

Not:
```
Epoch 1/30  train_loss=0.054  val_loss=0.000143  ❌ WRONG (data leakage!)
```

**Initial loss calculation:**
- 10-class classification: `-log(1/10) = 2.303`
- 11-class (if including day master): `-log(1/11) = 2.398`

If your initial loss is still <0.1, there's still leakage somewhere.

---

## Expected Training Behavior After Fix

### Ten-God Task (should look like this):
```
Epoch 1/30  train_loss=2.303  val_loss=2.305
Epoch 2/30  train_loss=1.856  val_loss=1.921
Epoch 3/30  train_loss=1.234  val_loss=1.356
Epoch 4/30  train_loss=0.876  val_loss=1.012
Epoch 5/30  train_loss=0.654  val_loss=0.823
...
Epoch 25/30  train_loss=0.045  val_loss=0.089
Epoch 26/30  train_loss=0.042  val_loss=0.087
```

**Key properties:**
- ✅ Initial loss ~2.3 (random 10-class)
- ✅ train_loss decreases smoothly and monotonically
- ✅ val_loss ≥ train_loss (normal generalization gap of ~2x)
- ✅ Final train_loss << val_loss (model memorizes training data)
- ✅ Training takes real effort (25+ epochs to converge)

---

## Checklist

- [ ] Add `to_input_features()` method to `BaziChart`
- [ ] Add `to_ground_truth()` method to `BaziChart`
- [ ] Update `ChartDataset.__getitem__()` to use both methods
- [ ] Update all 6 training scripts to pass `task_name` parameter
- [ ] Add sanity check print statements to verify no input/target overlap
- [ ] Re-run `train_ten_god.py` and verify initial loss is ~2.3
- [ ] If initial loss is still wrong, debug by printing actual inputs/targets
- [ ] Once ten_god is fixed, apply same pattern to other 4 tasks

---

## Why This Matters

From the Phase 1 plan (Section 1, "Why Pre-train the Encoders?"):

> "The encoders are new, randomly-initialized neural networks that must learn to represent complex symbolic bāzì logic. [...] By pre-training on supervised tasks with deterministic ground truth, we ensure the encoders learn robust internal representations that are verifiable and interpretable."

**The encoder must LEARN the logic from raw stems/branches, not be given pre-computed features!** Otherwise:
- ❌ Encoder learns nothing (just copies inputs)
- ❌ Phase 2 (LLM integration) will fail (encoder embeddings are meaningless)
- ❌ Final model cannot generalize (never learned bāzì rules)

**This fix is critical before any further training.**
