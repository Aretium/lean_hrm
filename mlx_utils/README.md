# MLX Utils - HRM Implementation for Apple Silicon

This directory contains the complete MLX implementation of the Hierarchical Reasoning Model (HRM).

## 📁 File Structure

```
mlx_utils/
├── __init__.py          # Package initialization
├── README.md            # This file
│
├── layers.py            # Core neural network layers
│   ├── MLXAttention     # Replaces FlashAttention
│   ├── MLXSwiGLU        # Activation function
│   ├── MLXRotaryEmbedding  # Position encoding
│   ├── rms_norm         # Normalization
│   └── MLXTransformerBlock  # Complete block
│
├── embeddings.py        # Embedding layers
│   ├── MLXEmbedding     # Token embeddings
│   ├── MLXSparseEmbedding   # Puzzle-specific embeddings
│   └── MLXCombinedEmbeddings  # Full input embeddings
│
├── hrm_model.py         # HRM architecture
│   ├── HRMInnerCarry    # State management
│   ├── MLXReasoningModule  # H/L level modules
│   ├── MLXHRMInner      # Core reasoning engine
│   └── MLXHRM           # Complete model with ACT
│
├── losses.py            # Loss functions
│   ├── stablemax_cross_entropy  # Numerically stable CE
│   ├── softmax_cross_entropy    # Standard CE
│   └── MLXACTLossHead   # Complete loss computation
│
├── optimizers.py        # Optimization algorithms
│   ├── SignSGD_MLX      # For sparse embeddings
│   ├── AdamMLX          # Standard Adam
│   └── create_optimizer_for_hrm  # Optimizer factory
│
├── dataset.py           # Data loading
│   ├── MLXPuzzleDataset # Main dataset class
│   ├── DatasetMetadata  # Metadata structure
│   └── collate_batch    # Batching utilities
│
├── training.py          # Training loop
│   ├── train_step       # Single training step
│   ├── evaluate         # Evaluation logic
│   ├── cosine_schedule_with_warmup  # LR scheduling
│   ├── save/load_checkpoint  # Checkpointing
│   └── train            # Main training loop
│
└── utils.py             # Helper utilities
    ├── set_random_seed  # Reproducibility
    ├── count_parameters # Model info
    ├── Timer            # Profiling
    ├── tree_map         # PyTree utilities
    └── check_nan_inf    # Debugging
```

## 🚀 Implementation Status

| Module | Status | Priority | Notes |
|--------|--------|----------|-------|
| `layers.py` | 📝 Scaffolded | HIGH | Core building blocks |
| `embeddings.py` | 📝 Scaffolded | HIGH | Input representations |
| `hrm_model.py` | 📝 Scaffolded | HIGH | Main architecture |
| `losses.py` | 📝 Scaffolded | MEDIUM | Loss computation |
| `optimizers.py` | 📝 Scaffolded | MEDIUM | Training algorithms |
| `dataset.py` | 📝 Scaffolded | MEDIUM | Data loading |
| `training.py` | 📝 Scaffolded | HIGH | Training loop |
| `utils.py` | 📝 Scaffolded | LOW | Helpers |

**Legend:**
- 📝 Scaffolded: Interface defined, needs implementation
- 🚧 In Progress: Partially implemented
- ✅ Complete: Fully implemented and tested
- ✓ Tested: Passing all tests

## 📊 Implementation Order

### Phase 1: Core Layers (Week 1)
1. `rms_norm` in `layers.py`
2. `MLXLinear` in `layers.py`
3. `MLXSwiGLU` in `layers.py`
4. `MLXRotaryEmbedding` in `layers.py`
5. `MLXAttention` in `layers.py`
6. `MLXTransformerBlock` in `layers.py`

**Test:** Single transformer block forward pass

### Phase 2: Embeddings (Week 1)
1. `trunc_normal_init` in `embeddings.py`
2. `MLXEmbedding` in `embeddings.py`
3. `MLXLearnedPositionEmbedding` in `embeddings.py`
4. `MLXSparseEmbedding` in `embeddings.py`
5. `MLXCombinedEmbeddings` in `embeddings.py`

**Test:** Input embedding generation

### Phase 3: HRM Model (Week 2)
1. `HRMInnerCarry`, `HRMCarry` in `hrm_model.py`
2. `MLXReasoningModule` in `hrm_model.py`
3. `MLXHRMInner` in `hrm_model.py`
4. `MLXHRM` in `hrm_model.py`

**Test:** Full model forward pass (single step)

### Phase 4: Losses & Optimizers (Week 2)
1. `stablemax`, `log_stablemax` in `losses.py`
2. `stablemax_cross_entropy` in `losses.py`
3. `MLXACTLossHead` in `losses.py`
4. `SignSGD_MLX` in `optimizers.py`
5. `create_optimizer_for_hrm` in `optimizers.py`

**Test:** Loss computation and gradient flow

### Phase 5: Data & Training (Week 3)
1. `DatasetMetadata`, data loading in `dataset.py`
2. `MLXPuzzleDataset` in `dataset.py`
3. `train_step` in `training.py`
4. `evaluate` in `training.py`
5. `train` in `training.py`
6. Utilities in `utils.py`

**Test:** Full training loop on small dataset

## 🎯 Key Design Decisions

### 1. No Distributed Training
**Rationale:** MLX targets single Apple Silicon chips with unified memory.  
**Impact:** ~50% less code complexity, simpler API, easier debugging.

### 2. Native Attention (No FlashAttention)
**Rationale:** FlashAttention is CUDA-specific. MLX attention is Metal-optimized.  
**Impact:** Similar or better performance on Apple GPUs.

### 3. Simplified Sparse Embeddings
**Rationale:** MLX's unified memory eliminates need for complex buffer management.  
**Impact:** Cleaner implementation, easier to understand and maintain.

### 4. Single Optimizer API
**Rationale:** MLX optimizers work differently than PyTorch's.  
**Impact:** Use `apply_single` method instead of separate step() calls.

### 5. Lazy Evaluation
**Rationale:** MLX evaluates lazily for efficiency.  
**Impact:** Explicit `mx.eval()` calls needed in training loop.

## 🔧 MLX-Specific Features

### Compilation
```python
@mx.compile
def compiled_attention(q, k, v):
    return scaled_dot_product_attention(q, k, v)
```

### Gradient Computation
```python
def loss_fn(model, batch):
    output, metrics = model(batch)
    return output["loss"], metrics

(loss, metrics), grads = mx.value_and_grad(loss_fn, has_aux=True)(model, batch)
```

### Unified Memory
```python
# No .cuda() or .to(device) needed!
# Data already accessible by GPU
batch = load_batch()
output = model(batch)  # Just works!
```

## 📝 Coding Conventions

1. **Type hints:** Use typing annotations for all functions
2. **Docstrings:** Google-style docstrings with Args/Returns
3. **Comments:** Explain "why", not "what"
4. **Naming:** 
   - Classes: `PascalCase` with `MLX` prefix
   - Functions: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
5. **Structure:** Group related functions, clear section headers

## 🧪 Testing Strategy

Each module should have corresponding tests:
```
tests/
├── test_layers.py
├── test_embeddings.py
├── test_hrm_model.py
├── test_losses.py
├── test_optimizers.py
├── test_dataset.py
└── test_training.py
```

Test types:
1. **Unit tests:** Individual functions/classes
2. **Integration tests:** Multiple components together
3. **Gradient tests:** Verify gradients flow correctly
4. **Numerical tests:** Compare with PyTorch version
5. **Performance tests:** Benchmark speed and memory

## 📚 Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
- [Original HRM Paper](https://arxiv.org/abs/2506.21734)
- [Original PyTorch Code](../models/)

## 🐛 Known Limitations

1. **No multi-GPU:** MLX is single-device only
2. **macOS only:** Requires Apple Silicon (M1/M2/M3/M4)
3. **Python 3.9+:** MLX requires recent Python
4. **Memory:** Unified memory shared with system (no dedicated VRAM)

## 💡 Optimization Tips

1. Use `@mx.compile` for hot paths
2. Batch operations where possible
3. Minimize Python loops over sequences
4. Profile with Metal System Trace
5. Use `mx.eval()` at checkpoints to free graph memory
6. Consider bf16 for speed (automatic in MLX)

## 🤝 Contributing

When implementing a module:
1. Follow the scaffolded interface
2. Add comprehensive docstrings
3. Include type hints
4. Write unit tests
5. Verify gradients flow correctly
6. Benchmark against PyTorch version

## 📧 Contact

Questions? Check the parent README or open an issue.

---

**Status:** 📝 Scaffolding Complete  
**Next:** Begin Phase 1 - Core Layers  
**Target:** Full implementation by Week 3

