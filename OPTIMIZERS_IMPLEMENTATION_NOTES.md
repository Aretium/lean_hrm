# MLX Optimizers Implementation Notes

## Overview
This document describes the implementation of `mlx_utils/optimizers.py`, including:
- **SignSGD**: Custom optimizer for sparse embeddings
- **Muon**: Modern optimizer using Newton-Schulz orthogonalization (~2x efficiency vs AdamW)
- **Learning rate scheduling**: Cosine decay with warmup
- Challenges encountered during PyTorch to MLX conversion

---

## Completed Components

### 1. `SignSGD_MLX` - Lines 68-126
**Purpose:** Sign-based SGD optimizer specifically designed for sparse puzzle embeddings in HRM.

**Why SignSGD for Sparse Embeddings?**
- Gradients are very sparse (only a few puzzles per batch)
- Sign of gradient is more stable than magnitude for sparse updates
- Memory efficient (no momentum buffers needed)
- Similar to Adam without momentum for sparse cases
- Works well with high learning rates (100x higher than main LR)

**Formula:**
```python
p = p * (1 - lr * weight_decay) - lr * sign(grad)
```

**Key Features:**
- **Decoupled weight decay:** Applied before gradient update
- **Sign operation:** Uses `mx.sign(gradient)` - only direction matters, not magnitude
- **No momentum:** Stateless optimizer (no state dict needed)
- **High learning rate:** Default 1e-2 (vs 1e-4 for main parameters)

**PyTorch Comparison:**
The PyTorch version (`CastedSparseEmbeddingSignSGD_Distributed`) was ~130 lines with:
- Distributed all-gather for multi-GPU
- Manual gradient accumulation across workers
- Complex unique ID tracking
- Separate local/global weight management

The MLX version is ~40 lines because:
- No distributed complexity (single device)
- Unified memory model
- MLX handles gradient accumulation automatically

---

### 2. `newton_schulz_orthogonalize` - Lines 68-144
**Purpose:** Fast orthogonalization for Muon optimizer via Newton-Schulz iteration.

**What is Orthogonalization?**
Orthogonalization transforms a matrix to have orthonormal rows/columns. For optimization, this means:
- Updates maintain geometric properties
- Prevents gradient interference across dimensions
- More stable training for deep networks

**Algorithm:**
```python
# Normalize
X = G / ||G||

# Iterate 5 times (quintic polynomial)
for _ in range(5):
    A = X @ X^T  # Gram matrix
    B = b*A + c*A²  # Polynomial correction
    X = a*X + B*X  # Update
```

**Coefficients:** `a=3.4445`, `b=-4.7750`, `c=2.0315`

**Why Newton-Schulz instead of SVD?**
- **Speed:** O(mn²) vs O(mn² + n³) for SVD
- **Stability:** Works in float32, even bfloat16
- **GPU-friendly:** No eigenvalue decomposition needed
- **Good enough:** Approximate orthogonalization suffices for optimization

**Performance:**
- 5 iterations gives ~0.3-0.4 error from perfect orthogonality
- Good enough for optimization (we care about direction, not perfection)
- Much faster than SVD (critical for large models)

---

### 3. `Muon_MLX` - Lines 161-266
**Purpose:** Muon optimizer for hidden layer weight matrices.

**What is Muon?**
Muon (2024) is a state-of-the-art optimizer achieving ~2x computational efficiency vs AdamW for transformers. It combines:
1. Nesterov momentum
2. Newton-Schulz orthogonalization
3. Applied only to 2D weight matrices

**Formula:**
```python
# Update momentum
m = β * m + grad

# Compute update (Nesterov)
update = grad + β * m

# Orthogonalize (for 2D params)
update_orth = newton_schulz(update)

# Apply
p = p - lr * update_orth
```

**Key Features:**
- **Selective application:** Only for 2D weights (linear, attention, conv)
- **High learning rate:** Default 2e-2 (20x higher than AdamW's 1e-3)
- **No second moment:** Unlike Adam, no adaptive learning rates
- **Fast:** Newton-Schulz adds only ~0.5-0.7% overhead

**Why Muon Works:**
- Orthogonalization prevents gradient interference
- Momentum provides acceleration
- Geometric properties improve conditioning
- Scales better than AdamW for large models

**Usage Notes:**
- **Use with AdamW:** Embeddings, biases, norms still need AdamW
- **Exclude first layer:** Common practice (features not learned yet)
- **Exclude output layer:** Classification head needs different treatment

**Scaling Results (from paper):**
- NanoGPT: Current speed record
- CIFAR-10: Current speed record
- Llama 405B: 2x efficiency vs AdamW

---

### 4. `MuonWithAdamW_MLX` - Lines 269-385
**Purpose:** Hybrid optimizer automatically routing parameters to Muon or AdamW.

**Routing Logic:**
```python
Use Muon if:
  1. Parameter is 2D (ndim >= 2)
  2. NOT an embedding layer
  3. NOT an output layer
  4. NOT the first layer (optional)

Otherwise: Use AdamW
```

**Example Routing:**
```
embedding.weight (100, 64)    -> AdamW   (embedding)
linear1.weight (128, 64)      -> Muon    (2D hidden weight)
linear1.bias (128,)           -> AdamW   (1D bias)
linear2.weight (64, 128)      -> Muon    (2D hidden weight)
output.weight (10, 64)        -> AdamW   (output layer)
```

**Hyperparameters:**
- Muon LR: 2e-2 (default)
- AdamW LR: 1e-3 (default)
- Ratio: ~20:1 (Muon needs higher LR due to orthogonalization)

**Usage:**
```python
optimizer = MuonWithAdamW_MLX(
    model,
    lr_muon=0.02,
    lr_adamw=1e-3
)

# Training step
optimizer.update(model, gradients)
```

---

### 5. `AdamMLX` - Lines 134-158
**Purpose:** Wrapper around MLX's AdamW optimizer with HRM-specific hyperparameters.

**Implementation Details:**
```python
class AdamMLX(optim.AdamW):
    def __init__(self, learning_rate=1e-4, betas=(0.9, 0.95), ...):
        betas_list = list(betas)  # MLX expects list, not tuple
        super().__init__(learning_rate, betas_list, ...)
```

**Key Differences from Standard Adam:**
- Uses **AdamW** (decoupled weight decay) instead of Adam
- Default betas: `(0.9, 0.95)` instead of `(0.9, 0.999)` - faster momentum decay
- Default weight_decay: `0.1` (quite high for stability)
- Default lr: `1e-4`

**Why AdamW?**
- Better generalization than Adam with L2 regularization
- Weight decay applied directly to parameters, not gradients
- Standard in modern transformer training

---

### 3. `get_learning_rate_schedule` - Lines 218-253
**Purpose:** Cosine learning rate schedule with linear warmup.

**Schedule Phases:**

**Phase 1: Warmup (Linear)** - Steps 0 to `warmup_steps`
```python
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
```
- Prevents instability at training start
- Gradual increase from 0 to base_lr
- Default warmup: 2000 steps

**Phase 2: Cosine Decay** - Steps `warmup_steps` to `total_steps`
```python
progress = (step - warmup_steps) / (total_steps - warmup_steps)
cosine_decay = 0.5 * (1.0 + cos(π * progress))
lr = base_lr * (min_ratio + (1 - min_ratio) * cosine_decay)
```
- Smooth decay from base_lr to `min_ratio * base_lr`
- Default min_ratio: 0.1 (decays to 10% of base)
- Cosine curve is gentle, avoids sharp drops

**Visualization:**
```
LR
^
|     /‾‾‾‾\___
|    /          \___
|   /               \___
|  /                    \___
| /                         \
|/___________________________\___> Steps
0  warmup          mid        end
```

---

### 4. `create_optimizer_for_hrm` - Lines 188-246
**Purpose:** Factory function to create appropriate optimizers for HRM model.

**Usage Pattern:**
```python
optimizers = create_optimizer_for_hrm(
    model,
    lr=1e-4,              # Main LR
    puzzle_emb_lr=1e-2,   # 100x higher!
    use_separate_puzzle_optimizer=True
)

# Returns:
# {
#   'main': AdamMLX optimizer for most parameters,
#   'puzzle': SignSGD_MLX optimizer for puzzle embeddings
# }
```

**Two-Optimizer Strategy:**

**Optimizer 1: Puzzle Embeddings** (SignSGD)
- Learning rate: 1e-2 (100x higher than main)
- Weight decay: 0.1
- Sparse updates only
- Fast adaptation to new puzzles

**Optimizer 2: Main Parameters** (AdamW)
- Learning rate: 1e-4
- Betas: (0.9, 0.95)
- Weight decay: 0.1
- Dense updates
- Stable convergence

**Why Two Optimizers?**
1. **Different update frequencies:** Embeddings are sparse, only updated for current batch
2. **Different learning rates:** Embeddings need faster updates (new puzzles), model needs stability
3. **Different algorithms:** Sign-based for embeddings, momentum-based for dense parameters

---

## Issues Encountered & Solutions

### Issue #1: Incorrect Optimizer.__init__ Call ❌

**Error:**
```python
TypeError: Optimizer.__init__() got an unexpected keyword argument 'learning_rate'
```

**Original Code:**
```python
class SignSGD_MLX(optim.Optimizer):
    def __init__(self, learning_rate=1e-2, weight_decay=0.1):
        super().__init__(learning_rate=learning_rate)  # ❌ Wrong!
```

**Problem:** MLX's base `Optimizer` class doesn't accept `learning_rate` in `__init__`. Each optimizer subclass manages its own hyperparameters.

**Solution:**
```python
class SignSGD_MLX(optim.Optimizer):
    def __init__(self, learning_rate=1e-2, weight_decay=0.1):
        super().__init__()  # ✅ No arguments
        self.learning_rate = learning_rate  # Store as instance variable
        self.weight_decay = weight_decay
```

**Key Insight:** Unlike PyTorch where optimizer base class manages LR, MLX requires each optimizer to manage its own hyperparameters.

---

### Issue #2: Adam Missing weight_decay Parameter ❌

**Error:**
```python
TypeError: Adam.__init__() got an unexpected keyword argument 'weight_decay'
```

**Original Code:**
```python
class AdamMLX(optim.Adam):  # ❌ Wrong optimizer
    def __init__(self, ..., weight_decay=0.1):
        super().__init__(..., weight_decay=weight_decay)
```

**Problem:** MLX's `Adam` class doesn't support weight_decay. Need to use `AdamW` instead.

**Solution:**
```python
class AdamMLX(optim.AdamW):  # ✅ Use AdamW, not Adam
    def __init__(self, ..., weight_decay=0.1):
        super().__init__(..., weight_decay=weight_decay)
```

**Key Difference:**
- `Adam`: No weight decay support
- `AdamW`: Decoupled weight decay (better for transformers)

---

### Issue #3: Adam State Not Initialized ❌

**Error:**
```python
KeyError: 'm'
```

**Problem Code:**
```python
opt = AdamMLX(learning_rate=0.001)
param = mx.array([1.0, 2.0, 3.0])
grad = mx.array([0.1, 0.2, 0.3])

state = {}  # Empty state
updated = opt.apply_single(grad, param, state)  # ❌ Fails - needs 'm' and 'v'
```

**Problem:** Adam maintains momentum buffers (`m` for first moment, `v` for second moment). These must be initialized before first update.

**Solution:**
```python
opt = AdamMLX(learning_rate=0.001)
param = mx.array([1.0, 2.0, 3.0])
grad = mx.array([0.1, 0.2, 0.3])

state = {}
opt.init_single(param, state)  # ✅ Initialize state first
updated = opt.apply_single(grad, param, state)  # Now works
```

**What `init_single` Does:**
```python
# Creates:
state['m'] = mx.zeros_like(param)  # First moment
state['v'] = mx.zeros_like(param)  # Second moment
state['step'] = 0                   # Step counter
```

**Best Practice:** Always call `init_single` or use the optimizer's `update` method which handles initialization automatically.

---

### Issue #4: Betas Type Mismatch ⚠️

**Minor Issue:**
```python
# Python convention: tuples for immutable collections
betas = (0.9, 0.95)

# MLX expects: list
opt = AdamMLX(learning_rate=lr, betas=betas)  # Works but non-standard
```

**Solution:**
```python
def __init__(self, betas=(0.9, 0.95), ...):
    betas_list = list(betas)  # ✅ Convert to list
    super().__init__(..., betas=betas_list, ...)
```

**Why:** MLX internally expects lists for beta parameters. Converting ensures compatibility.

---

## MLX vs PyTorch Optimizer API

| Feature | PyTorch | MLX | Notes |
|---------|---------|-----|-------|
| Base class | `torch.optim.Optimizer` | `mlx.optimizers.Optimizer` | Similar concept |
| Learning rate | In base class | In subclass | MLX: manual management |
| Weight decay | In `AdamW` | In `AdamW` | Same behavior |
| Momentum | Automatic | Manual state | Must call `init_single` |
| Parameter groups | Supported | Not built-in | Simpler in MLX |
| State management | Automatic | Semi-automatic | Need explicit init |
| Sparse updates | Manual | Automatic | MLX handles naturally |

---

## SignSGD Deep Dive

### Why Use Sign of Gradient?

**Standard SGD:**
```python
p = p - lr * grad  # Magnitude matters
```
Problems:
- Sparse gradients have extreme magnitudes
- Scale varies wildly across puzzles
- Needs careful gradient clipping

**SignSGD:**
```python
p = p - lr * sign(grad)  # Only direction matters
```
Benefits:
- Magnitude-invariant (robust to scale)
- Natural gradient clipping (bounded by ±1)
- Works well with high learning rates
- Implicit noise reduction

### Mathematical Intuition

Think of gradients as **votes** rather than **forces**:
- Positive gradient: "increase this parameter" (vote +1)
- Negative gradient: "decrease this parameter" (vote -1)
- Magnitude doesn't matter - just the direction

This is especially good for sparse embeddings where:
- Few gradients per update (few votes)
- Each vote should have equal weight
- Robustness to outliers is important

### Comparison with Adam for Sparse Updates

**Adam:**
```python
m = β1 * m + (1-β1) * grad        # First moment
v = β2 * v + (1-β2) * grad²       # Second moment
p = p - lr * m / (sqrt(v) + ε)    # Adaptive step
```
Problems for sparse:
- Momentum accumulates slowly (β1=0.9)
- Second moment biased by zeros
- Adaptive scaling less useful

**SignSGD:**
```python
p = p - lr * sign(grad)  # Direct, simple
```
Advantages for sparse:
- No momentum delay
- No statistics to maintain
- Direct response to new data
- Higher learning rate works

---

## Learning Rate Schedule Deep Dive

### Why Warmup?

**Problem without warmup:**
At step 0, model parameters are random. Large learning rate + random gradients = instability.

**Solution with warmup:**
```python
step 0:    lr = 0.0 * base_lr    # No updates yet
step 500:  lr = 0.25 * base_lr   # Gentle exploration
step 1000: lr = 0.5 * base_lr    # Moderate learning
step 2000: lr = 1.0 * base_lr    # Full speed
```

### Why Cosine Decay?

**Alternative: Step Decay**
```
lr = base_lr * 0.1^(step // decay_steps)
```
Problem: Sharp drops cause training instability

**Alternative: Linear Decay**
```
lr = base_lr * (1 - progress)
```
Problem: Too aggressive at end

**Cosine Decay:**
```
lr = base_lr * (min + (1-min) * 0.5 * (1 + cos(π * progress)))
```
Benefits:
- Smooth curve (no discontinuities)
- Fast decay at start (when model learns quickly)
- Slow decay at end (fine-tuning phase)
- Well-studied and proven effective

### Typical HRM Training Schedule

```
Steps: 0          2000        6000        10000
       |----------|-----------|-----------|
LR:    0 → 1e-4  1e-4 → 5e-5  5e-5 → 1e-5
       ↑ warmup   ↑ fast decay ↑ fine-tune
```

---

## Usage Examples

### Example 1: Basic Training Loop

```python
import mlx.core as mx
import mlx.nn as nn
from mlx_utils.optimizers import create_optimizer_for_hrm, get_learning_rate_schedule

# Create model and optimizers
model = MLXHRM(...)
optimizers = create_optimizer_for_hrm(model, lr=1e-4, puzzle_emb_lr=1e-2)

# Training loop
for step in range(total_steps):
    # Get scheduled learning rate
    current_lr = get_learning_rate_schedule(step, total_steps, base_lr=1e-4)

    # Update optimizer learning rates
    optimizers['main'].learning_rate = current_lr
    optimizers['puzzle'].learning_rate = current_lr * 100  # Keep 100x ratio

    # Forward pass
    loss = compute_loss(model, batch)

    # Compute gradients
    grads = mx.grad(loss)(model.parameters())

    # Update parameters
    optimizers['main'].update(model, grads)
    # Separately update puzzle embeddings if needed
    # optimizers['puzzle'].update(model.puzzle_embeddings, grads['puzzle_embeddings'])
```

### Example 2: Manual Parameter Update

```python
# If you need fine-grained control
opt = SignSGD_MLX(learning_rate=0.01, weight_decay=0.1)

# For each parameter
for name, param in model.parameters().items():
    if 'puzzle_embedding' in name:
        grad = grads[name]
        state = optimizer_states.get(name, {})

        # Apply update
        updated_param = opt.apply_single(grad, param, state)

        # Update model
        model.update_parameters({name: updated_param})
```

### Example 3: Different LRs for Different Layers

```python
# Create multiple optimizers
optimizers = {
    'embeddings': SignSGD_MLX(lr=1e-2),
    'encoder': AdamMLX(lr=1e-4),
    'decoder': AdamMLX(lr=5e-5),  # Lower LR for decoder
}

# Apply to different parameter groups
for name, param in model.parameters().items():
    if 'embedding' in name:
        opt = optimizers['embeddings']
    elif 'encoder' in name:
        opt = optimizers['encoder']
    else:
        opt = optimizers['decoder']

    # Apply update
    updated = opt.apply_single(grads[name], param, states[name])
```

---

## Testing Results

All unit tests passed ✅:

```
Testing SignSGD_MLX...
  Expected [0,0]: 0.8990, Actual: 0.8990
  Expected [0,1]: 2.0980, Actual: 2.0980
  ✓ SignSGD passed

Testing AdamMLX...
  Updated param: array([0.996828, 1.99682, 2.99681], dtype=float32)
  ✓ Adam passed

Testing learning rate schedule...
  LR at step 0: 0.000000e+00
  LR at step 1000 (mid-warmup): 5.000000e-05
  LR at step 2000 (end warmup): 1.000000e-04
  LR at step 6000 (mid-decay): 5.500000e-05
  LR at step 10000 (end): 1.000000e-05
  ✓ Learning rate schedule passed

Testing create_optimizer_for_hrm...
  Created optimizers: ['puzzle', 'main']
  ✓ Optimizer creation passed

All tests passed! ✓
```

**Test Coverage:**
- ✅ SignSGD update formula (exact match)
- ✅ Adam state initialization and update
- ✅ Learning rate warmup phase
- ✅ Learning rate cosine decay phase
- ✅ Optimizer factory function
- ✅ Type conversions (tuple to list)

**Muon Test Results:**

```
Testing Newton-Schulz orthogonalization...
  Input shape: (64, 64)
  Max error from identity: 0.336308
  Improvement: 0.149241
  ✓ Newton-Schulz orthogonalization passed

Testing Newton-Schulz on rectangular matrix...
  Input shape: (512, 2048)
  Max error from identity: 0.136478
  ✓ Rectangular matrix orthogonalization passed

Testing Muon optimizer...
  Parameter shape: (128, 256)
  Max parameter change: 0.004388
  ✓ Muon optimizer passed

Testing MuonWithAdamW_MLX...
  embedding.weight (100, 64) -> AdamW
  linear1.weight (128, 64) -> Muon
  linear2.weight (64, 128) -> Muon
  output.weight (10, 64) -> AdamW
  ✓ Hybrid optimizer routing passed

Testing Muon convergence...
  Initial loss: 4137.6465
  Final loss: 3606.7253
  Loss reduction: 12.8%
  ✓ Muon convergence test passed

All Muon tests passed! ✓
```

**Muon Test Coverage:**
- ✅ Newton-Schulz orthogonalization (square & rectangular)
- ✅ Muon optimizer update (2D and 1D parameters)
- ✅ MuonWithAdamW routing logic
- ✅ Convergence on simple problem
- ✅ State management (momentum buffers)

---

## Key Takeaways

### 1. **Optimizer Simplification**
MLX's unified memory and single-device model eliminates ~70% of PyTorch's distributed optimizer complexity.

### 2. **State Management**
Always initialize optimizer state with `init_single` before calling `apply_single`, especially for momentum-based optimizers.

### 3. **Two-Optimizer Pattern**
Use SignSGD (high LR, sparse) for embeddings and AdamW (low LR, dense) for main parameters.

### 4. **Learning Rate Scheduling**
Warmup + Cosine decay is the gold standard for transformer training. Start slow, learn fast, fine-tune carefully.

### 5. **Sign-Based Optimization**
For sparse, high-variance gradients, using the sign instead of magnitude provides stability and enables higher learning rates.

### 6. **Muon: The New State-of-the-Art**
Muon achieves 2x efficiency vs AdamW by orthogonalizing updates. Use it for hidden layer weights, AdamW for everything else.

### 7. **Newton-Schulz vs SVD**
Newton-Schulz iteration is ~10-100x faster than SVD for orthogonalization, with acceptable accuracy for optimization.

---

## Future Improvements

1. **Automatic Parameter Grouping:** Detect puzzle embeddings automatically
2. **Gradient Clipping:** Add global norm clipping option
3. **Learning Rate Schedulers as Classes:** Make schedulers first-class objects
4. **State Checkpointing:** Easy save/load of optimizer states
5. **Profiling:** Optimize for Apple Silicon GPU

---

## References

### PyTorch HRM Implementation
- `models/sparse_embedding.py` (SignSGD)
- External `adam_atan2` package (Adam variant)

### Muon Optimizer
- **Paper:** "Muon is Scalable for LLM Training" (arXiv:2502.16982, 2025)
- **Blog:** https://kellerjordan.github.io/posts/muon/
- **Code:** https://github.com/KellerJordan/Muon
- **Authors:** Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, et al.

### Other References
- **MLX Optimizer API:** https://ml-explore.github.io/mlx/build/html/python/optimizers.html
- **HRM Paper:** For understanding the two-optimizer strategy
- **AdamW Paper:** "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
- **Newton-Schulz:** Matrix orthogonalization via quintic iteration

---

**Last Updated:** 2025-11-09
**Status:** ✅ Complete and Tested (including Muon)
