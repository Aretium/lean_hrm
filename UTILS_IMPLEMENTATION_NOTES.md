## MLX Utilities Implementation Notes

## Overview
This document describes the implementation of `mlx_utils/utils.py`, a comprehensive collection of helper functions for MLX development including:
- **Initialization**: Random seeds, parameter counting, model summaries
- **Array Manipulation**: PyTree operations (map, flatten, detach)
- **Debugging**: NaN/Inf checking, statistics printing, gradient inspection
- **Profiling**: Timing, memory tracking, GPU synchronization
- **Visualization**: Attention heatmaps, learning curves
- **Configuration**: YAML loading/saving

---

## Completed Components

### 1. Initialization Utilities

#### `set_random_seed(seed: int)`
**Purpose:** Set random seed for reproducibility across MLX and NumPy.

**Implementation:**
```python
mx.random.seed(seed)
np.random.seed(seed)
```

**Usage:**
```python
set_random_seed(42)
# Now all random operations are reproducible
```

---

#### `count_parameters(model: nn.Module) -> int`
**Purpose:** Count total number of parameters in a model.

**Key Features:**
- Handles nested parameter dictionaries
- Recursively traverses model structure
- Works with any MLX nn.Module

**Algorithm:**
```python
def count_in_tree(tree):
    if isinstance(tree, dict):
        return sum(count_in_tree(v) for v in tree.values())
    elif isinstance(tree, mx.array):
        return tree.size
    return 0
```

**Example:**
```python
model = MyTransformer()
total = count_parameters(model)
print(f"Total parameters: {total:,}")
# Output: Total parameters: 124,439,808
```

---

#### `print_model_summary(model: nn.Module)`
**Purpose:** Print detailed model architecture summary.

**Output Format:**
```
================================================================================
Model Summary
================================================================================
embedding.weight                                   (50000, 512)      25,600,000
encoder.layer0.attention.weight                    (512, 512)           262,144
encoder.layer0.attention.bias                      (512,)                   512
...
================================================================================
Total Parameters                                                  124,439,808
Trainable Parameters                                              124,439,808
================================================================================
```

**Features:**
- Flattens nested parameter structure
- Shows parameter name, shape, and count
- Formatted with proper alignment

---

### 2. Array Manipulation Utilities

#### `tree_map(fn, tree)`
**Purpose:** Apply function to all arrays in a PyTree structure.

**What is a PyTree?**
A PyTree is a nested structure of dicts, lists, tuples, and arrays. Common in MLX for representing model parameters, gradients, and optimizer states.

**Implementation:**
```python
if isinstance(tree, dict):
    return {k: tree_map(fn, v) for k, v in tree.items()}
elif isinstance(tree, list):
    return [tree_map(fn, v) for v in tree]
elif isinstance(tree, mx.array):
    return fn(tree)
else:
    return tree  # Pass through non-arrays
```

**Examples:**
```python
# Multiply all parameters by 0.9
params = model.parameters()
scaled = tree_map(lambda x: x * 0.9, params)

# Convert all arrays to float32
params_f32 = tree_map(lambda x: x.astype(mx.float32), params)

# Compute norms
norms = tree_map(lambda x: mx.sqrt(mx.sum(x * x)), params)
```

---

#### `tree_flatten(tree) -> List[mx.array]`
**Purpose:** Flatten a PyTree into a list of arrays.

**Use Cases:**
- Computing global gradient norm
- Counting total elements
- Applying operations to all arrays sequentially

**Example:**
```python
params = model.parameters()
flat_params = tree_flatten(params)

# Compute global norm
global_norm = mx.sqrt(sum(mx.sum(p * p) for p in flat_params))

# Count total parameters
total_params = sum(p.size for p in flat_params)
```

---

#### `detach_tree(tree)`
**Purpose:** Detach all arrays in tree from gradient graph.

**When to Use:**
- After computing metrics (prevent memory leaks)
- When caching intermediate values
- For logging/visualization

**Implementation:**
```python
return tree_map(mx.stop_gradient, tree)
```

**Example:**
```python
# Compute loss and metrics
loss, metrics = compute_loss_and_metrics(model, batch)

# Detach metrics to prevent gradient accumulation
metrics = detach_tree(metrics)

# Safe to store/log without memory leaks
metrics_history.append(metrics)
```

---

### 3. Debugging Utilities

#### `check_nan_inf(x: mx.array, name: str)`
**Purpose:** Check for NaN or Inf values, raise error if found.

**Why Important:**
- NaN/Inf values indicate training instability
- Early detection prevents wasted training time
- Helps locate problematic operations

**Usage:**
```python
# After each major operation
logits = model(inputs)
check_nan_inf(logits, "model_outputs")

loss = compute_loss(logits, labels)
check_nan_inf(loss, "loss")

# In training loop
for param, grad in zip(params, grads):
    check_nan_inf(grad, f"grad_{param_name}")
```

**Example Error:**
```
ValueError: model_outputs contains NaN values!
```

---

#### `print_array_stats(x: mx.array, name: str)`
**Purpose:** Print comprehensive statistics about an array.

**Output:**
```
Array Statistics: activations
  Shape: (32, 128, 512)
  Dtype: mlx.core.float32
  Min: -3.142857
  Max: 4.285714
  Mean: 0.003421
  Std: 1.002341
```

**With Warnings:**
```
Array Statistics: suspicious_values
  Shape: (100,)
  Dtype: mlx.core.float32
  Min: -inf
  Max: inf
  Mean: nan
  Std: nan
  ⚠️  WARNING: Contains NaN=True, Inf=True
```

---

#### `debug_gradient(grad: mx.array, name: str)`
**Purpose:** Print gradient statistics with health checks.

**Output:**
```
Gradient Debug: encoder.weight
  Shape: (512, 512)
  L2 norm: 0.124567
  Min: -2.341234e-02
  Max: 3.456789e-02
  Mean: -1.234567e-04
  Std: 8.901234e-03
```

**Health Warnings:**
```
Gradient Debug: problematic_grad
  Shape: (256, 256)
  L2 norm: 12345.678900
  ...
  ⚠️  WARNING: Very large gradient (norm=1.23e+04)
```

**Thresholds:**
- `norm < 1e-8`: Very small gradient (vanishing)
- `norm > 1e3`: Very large gradient (exploding)
- NaN/Inf: Critical error

---

### 4. Profiling Utilities

#### `Timer` Context Manager
**Purpose:** Time code blocks with clean syntax.

**Usage:**
```python
with Timer("data_loading"):
    batch = load_batch()

with Timer("forward_pass"):
    output = model(batch)
    mx.eval(output)  # Force evaluation

# Output:
# [Timer] data_loading: 0.1234s
# [Timer] forward_pass: 0.0567s
```

**Silent Mode:**
```python
with Timer("operation", verbose=False) as timer:
    result = expensive_op()

print(f"Operation took {timer.elapsed:.4f}s")
```

**Nested Timing:**
```python
with Timer("training_step"):
    with Timer("forward"):
        loss = model(batch)
    with Timer("backward"):
        grads = mx.grad(loss)(model.parameters())
    with Timer("optimizer"):
        optimizer.update(model, grads)
```

---

#### `@profile` Decorator
**Purpose:** Profile function execution time automatically.

**Usage:**
```python
@profile
def train_epoch(model, dataloader):
    for batch in dataloader:
        loss = train_step(model, batch)
    return loss

# Automatically prints:
# [Profile] train_epoch: 45.1234s
```

**Composable:**
```python
@profile
def outer():
    inner()

@profile
def inner():
    expensive_computation()

# Output:
# [Profile] inner: 1.2345s
# [Profile] outer: 1.2456s
```

---

#### `memory_usage() -> Dict[str, float]`
**Purpose:** Get current GPU memory usage in MB.

**Returns:**
```python
{
    "active_mb": 1234.56,    # Currently allocated
    "peak_mb": 2048.00,      # Peak allocation
    "cache_mb": 512.00,      # Cached memory
    "total_mb": 1746.56      # Active + Cache
}
```

**Usage:**
```python
# Before training
mem_before = memory_usage()

# Train
train_model()

# After training
mem_after = memory_usage()

print(f"Memory increase: {mem_after['active_mb'] - mem_before['active_mb']:.2f} MB")
```

**Tracking Memory Leaks:**
```python
mem_history = []
for step in range(1000):
    loss = train_step()
    mem = memory_usage()
    mem_history.append(mem['active_mb'])

    # Check for leaks
    if step > 100 and mem['active_mb'] > mem_history[0] * 2:
        print("⚠️  Possible memory leak detected!")
```

---

### 5. Metal GPU Utilities

#### `get_device_info() -> Dict`
**Purpose:** Get information about Metal GPU device.

**Returns:**
```python
{
    "device_type": "DeviceType.gpu",
    "device_id": 0,
    "device_str": "Device(gpu, 0)",
    "active_mb": 1234.56,
    "peak_mb": 2048.00,
    "cache_mb": 512.00,
    "total_mb": 1746.56
}
```

**Usage:**
```python
info = get_device_info()
print(f"Running on: {info['device_str']}")
print(f"Memory available: {info['total_mb']:.2f} MB")
```

---

#### `synchronize()`
**Purpose:** Force evaluation of all pending GPU operations.

**Why Needed:**
MLX uses **lazy evaluation** - operations aren't computed until their results are needed. This is great for performance but can make timing inaccurate.

**Example Without Synchronization:**
```python
start = time.time()
result = expensive_computation()  # Returns immediately (lazy!)
elapsed = time.time() - start
print(f"Time: {elapsed:.4f}s")  # ❌ Wrong! Only measures ~0.0001s
```

**Example With Synchronization:**
```python
start = time.time()
result = expensive_computation()
synchronize()  # Force computation
elapsed = time.time() - start
print(f"Time: {elapsed:.4f}s")  # ✅ Correct! Measures actual compute time
```

**Best Practice:**
```python
with Timer("gpu_operation"):
    result = gpu_intensive_op()
    synchronize()  # Ensure timing is accurate
```

---

### 6. Visualization Utilities

#### `visualize_attention(attn_weights, tokens, save_path)`
**Purpose:** Create heatmap visualization of attention weights.

**Input:**
- `attn_weights`: `[num_heads, seq_len, seq_len]` or `[seq_len, seq_len]`
- `tokens`: List of token strings
- `save_path`: Optional path to save figure

**Features:**
- Automatically averages over multiple heads
- Colorbar shows attention strength
- Token labels on both axes
- Saves to file or displays interactively

**Usage:**
```python
# Get attention from model
attn = model.get_attention_weights(inputs)

# Visualize
tokens = ["The", "cat", "sat", "on", "mat"]
visualize_attention(attn, tokens, "attention_viz.png")
```

**Output:**
Beautiful heatmap showing which tokens attend to which!

---

#### `plot_learning_curves(metrics_history, save_path)`
**Purpose:** Plot training metrics over time.

**Input:**
```python
metrics_history = {
    "loss": [2.5, 2.3, 2.1, 1.9, 1.7],
    "accuracy": [0.5, 0.6, 0.65, 0.7, 0.75],
    "perplexity": [12.0, 10.0, 8.5, 7.0, 6.0]
}
```

**Features:**
- Creates subplots for each metric
- Grid lines for readability
- Saves to file or displays
- Automatic layout

**Usage:**
```python
# During training
history = {"loss": [], "accuracy": []}
for epoch in range(num_epochs):
    loss, acc = train_epoch()
    history["loss"].append(loss)
    history["accuracy"].append(acc)

# Plot results
plot_learning_curves(history, "training_curves.png")
```

---

### 7. Configuration Utilities

#### `load_config(config_path) -> Dict`
**Purpose:** Load configuration from YAML file.

**Example Config (`config.yaml`):**
```yaml
model:
  hidden_size: 512
  num_layers: 6
  num_heads: 8

training:
  learning_rate: 0.0001
  batch_size: 32
  num_epochs: 100

optimizer:
  type: adamw
  weight_decay: 0.1
```

**Usage:**
```python
config = load_config("config.yaml")

model = create_model(
    hidden_size=config['model']['hidden_size'],
    num_layers=config['model']['num_layers']
)

optimizer = create_optimizer(
    lr=config['training']['learning_rate'],
    weight_decay=config['optimizer']['weight_decay']
)
```

---

#### `save_config(config, save_path)`
**Purpose:** Save configuration to YAML file.

**Usage:**
```python
config = {
    "model": {"hidden_size": 512},
    "training": {"learning_rate": 0.0001}
}

save_config(config, "experiment_config.yaml")
```

**Why Use Config Files:**
1. **Reproducibility**: Easy to share and reproduce experiments
2. **Version Control**: Track changes to hyperparameters
3. **Experimentation**: Quick to modify without changing code
4. **Documentation**: Self-documenting experiments

---

## Testing Results

All tests passed! ✅

```
Testing random seed...
  ✓ Random seed test passed

Testing parameter counting...
  Total parameters: 325
  ✓ Parameter counting test passed

Testing model summary...
  ✓ Model summary test passed

Testing tree operations...
  ✓ Tree operations test passed

Testing NaN/Inf checking...
  NaN detected correctly
  Inf detected correctly
  ✓ NaN/Inf checking test passed

Testing array statistics...
  ✓ Array statistics test passed

Testing gradient debugging...
  ⚠️  WARNING: Very small gradient (norm=5.00e-10)
  ⚠️  WARNING: Very large gradient (norm=5.00e+04)
  ✓ Gradient debugging test passed

Testing Timer...
  Measured time: 0.1050s
  ✓ Timer test passed

Testing profile decorator...
  ✓ Profile decorator test passed

Testing memory usage...
  Active memory: 0.00 MB
  Peak memory: 0.00 MB
  ✓ Memory usage test passed

Testing device info...
  ✓ Device info test passed

Testing synchronization...
  ✓ Synchronization test passed

All utility tests passed! ✓
```

**Test Coverage:**
- ✅ Random seed reproducibility
- ✅ Parameter counting accuracy
- ✅ Model summary formatting
- ✅ Tree operations (map, flatten, detach)
- ✅ NaN/Inf detection
- ✅ Array statistics
- ✅ Gradient debugging with health checks
- ✅ Timer accuracy
- ✅ Profile decorator
- ✅ Memory tracking
- ✅ Device info retrieval
- ✅ GPU synchronization

---

## Usage Patterns

### Pattern 1: Debugging Training Instability

```python
@profile
def train_step(model, batch):
    # Forward pass
    logits = model(batch["inputs"])
    check_nan_inf(logits, "logits")

    # Compute loss
    loss = compute_loss(logits, batch["labels"])
    check_nan_inf(loss, "loss")

    # Backward pass
    grads = mx.grad(loss)(model.parameters())

    # Debug problematic gradients
    for name, grad in grads.items():
        if "problematic_layer" in name:
            debug_gradient(grad, name)

    # Check for exploding gradients
    flat_grads = tree_flatten(grads)
    global_norm = mx.sqrt(sum(mx.sum(g * g) for g in flat_grads))
    if global_norm > 10.0:
        print(f"⚠️  Large gradient norm: {global_norm:.2f}")

    return loss
```

### Pattern 2: Memory Profiling

```python
def profile_memory_usage(model, dataloader):
    print("Profiling memory usage...")

    for i, batch in enumerate(dataloader):
        with Timer(f"step_{i}"):
            loss = train_step(model, batch)
            synchronize()

        mem = memory_usage()
        print(f"Step {i}: {mem['active_mb']:.2f} MB active, "
              f"{mem['peak_mb']:.2f} MB peak")

        if i >= 10:
            break
```

### Pattern 3: Experiment Management

```python
def run_experiment(config_path):
    # Load config
    config = load_config(config_path)
    set_random_seed(config['seed'])

    # Create model
    model = create_model(**config['model'])
    print_model_summary(model)

    # Track metrics
    metrics_history = {"loss": [], "accuracy": []}

    # Training loop
    for epoch in range(config['training']['epochs']):
        with Timer(f"epoch_{epoch}"):
            for batch in dataloader:
                loss, metrics = train_step(model, batch)

                # Detach metrics to prevent memory leaks
                metrics = detach_tree(metrics)

                metrics_history["loss"].append(metrics["loss"])
                metrics_history["accuracy"].append(metrics["accuracy"])

    # Save results
    plot_learning_curves(metrics_history, "results.png")

    # Save config with results
    config['final_loss'] = metrics_history["loss"][-1]
    save_config(config, "experiment_results.yaml")
```

### Pattern 4: Model Inspection

```python
def inspect_model(model):
    print("=" * 80)
    print("Model Inspection")
    print("=" * 80)

    # Basic info
    print_model_summary(model)

    # Parameter statistics
    params = model.parameters()
    flat_params = tree_flatten(params)

    print(f"\nParameter Statistics:")
    for name, param in params.items():
        print_array_stats(param, name)

    # Check for issues
    print(f"\nHealth Check:")
    for name, param in params.items():
        try:
            check_nan_inf(param, name)
            print(f"  ✓ {name}: OK")
        except ValueError as e:
            print(f"  ❌ {name}: {e}")
```

---

## Key Takeaways

### 1. **Always Use Synchronize for Timing**
MLX's lazy evaluation means timing without synchronization is meaningless.

### 2. **Detach Metrics to Prevent Memory Leaks**
Use `detach_tree()` on metrics before storing them in history.

### 3. **Check for NaN/Inf Early and Often**
Early detection saves hours of wasted training time.

### 4. **Profile Memory Usage Regularly**
Memory leaks are common in deep learning - catch them early.

### 5. **Use Config Files for Experiments**
YAML configs make experiments reproducible and shareable.

### 6. **Debug Gradients, Not Just Loss**
Gradient statistics reveal issues before they affect loss.

### 7. **Tree Operations are Powerful**
`tree_map`, `tree_flatten`, and `detach_tree` simplify working with nested structures.

---

## MLX-Specific Tips

### Lazy Evaluation
```python
# ❌ Wrong - no computation happens
result = expensive_op()
print("Done")  # Prints immediately

# ✅ Right - force evaluation
result = expensive_op()
mx.eval(result)
print("Done")  # Prints after computation
```

### Memory Management
```python
# Clear unused memory
mx.eval(model.parameters())  # Force cleanup

# Check memory
mem = memory_usage()
if mem['active_mb'] > threshold:
    print("⚠️  High memory usage!")
```

### Device Info
```python
info = get_device_info()
if info['device_type'] == 'DeviceType.cpu':
    print("⚠️  Running on CPU, expect slow performance")
```

---

## Future Improvements

1. **Gradient Clipping Utilities**: Helper functions for global norm clipping
2. **Checkpoint Management**: Save/load model checkpoints with metadata
3. **Distributed Utilities**: Tools for multi-GPU training (when MLX supports it)
4. **Advanced Profiling**: Integration with Metal System Trace
5. **Automatic Mixed Precision**: Helpers for FP16/BF16 training
6. **Learning Rate Finder**: Automatic optimal LR discovery
7. **Model Compression**: Quantization and pruning utilities

---

## Dependencies

**Required:**
- `mlx >= 0.0.1`
- `numpy`

**Optional:**
- `matplotlib`: For visualization (`visualize_attention`, `plot_learning_curves`)
- `pyyaml`: For config management (`load_config`, `save_config`)

**Installation:**
```bash
pip install mlx numpy matplotlib pyyaml
```

---

## References

- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **MLX Examples**: https://github.com/ml-explore/mlx-examples
- **PyTree Concept**: JAX documentation on pytrees
- **Metal Performance Tools**: Xcode Instruments documentation

---

**Last Updated:** 2025-11-09
**Status:** ✅ Complete and Tested
