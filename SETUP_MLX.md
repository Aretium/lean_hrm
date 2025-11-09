# MLX Environment Setup for Lean-HRM

This document describes the MLX-based virtual environment for the Lean Hierarchical Reasoning Model.

## ✅ Environment Status

**Environment Created:** ✓ `venv_mlx/`  
**MLX Version:** 0.29.3  
**Python Version:** 3.13  
**Target Hardware:** Apple Silicon (M-series chips)

## 📦 Installed Packages

### Core ML Framework
- `mlx` (0.29.3) - Apple's ML framework for Metal acceleration
- `mlx-data` (0.2.0) - Data loading utilities
- `numpy` (2.3.4) - Numerical computing

### Model Components
- `einops` - Tensor manipulation
- `pydantic` - Configuration validation
- `omegaconf` & `hydra-core` - Hierarchical configuration

### Training & Logging
- `wandb` - Experiment tracking
- `tqdm` - Progress bars
- `coolname` - Run name generation

### Development Tools
- `jupyter`, `notebook` - Interactive development
- `matplotlib` - Visualization
- `ipython` - Enhanced REPL

## 🚀 Quick Start

### Activate the environment:
```bash
cd /Users/umang/Users/umang/Projects/Lean-HRM-MLX
source venv_mlx/bin/activate
```

### Or use the direct python path:
```bash
./venv_mlx/bin/python your_script.py
```

### Test the setup:
```bash
./venv_mlx/bin/python test_mlx_setup.py
```

## 📝 Key Differences from PyTorch Version

| Component | PyTorch (Original) | MLX (New) |
|-----------|-------------------|-----------|
| **Backend** | CUDA (NVIDIA) | Metal (Apple) |
| **Attention** | FlashAttention | Native MLX attention |
| **Precision** | bfloat16 + casting | Automatic dtype handling |
| **Distributed** | torch.distributed | Single device optimized |
| **Compilation** | torch.compile | @mx.compile decorator |
| **Memory** | Explicit CUDA mgmt | Unified memory |

## 🔧 What We Replaced

### 1. FlashAttention → MLX Attention
```python
# OLD (PyTorch)
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, causal=False)

# NEW (MLX)
import mlx.core as mx
# Custom scaled_dot_product_attention (see test_mlx_setup.py)
output = scaled_dot_product_attention(q, k, v)
```

### 2. Dtype Casting → Automatic
```python
# OLD (PyTorch)
class CastedLinear(nn.Module):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), ...)

# NEW (MLX)
# No explicit casting needed - MLX handles it!
class MLXLinear(nn.Module):
    def __call__(self, x):
        return x @ self.weight.T + self.bias
```

### 3. Distributed Training → Single Device
```python
# OLD (PyTorch)
dist.init_process_group(backend="nccl")
dist.all_reduce(param.grad)

# NEW (MLX)
# Not needed! MLX optimized for single Apple Silicon chip
# Unified memory architecture removes need for multi-GPU
```

## 🧪 Verified Features

All tests passed successfully:
- ✅ Basic tensor operations
- ✅ Matrix multiplication on GPU
- ✅ Scaled dot-product attention
- ✅ Metal GPU memory allocation
- ✅ Automatic differentiation
- ✅ Compilation with @mx.compile

**Performance observed:**
- 100x100 matrix multiply: ~167ms
- Attention (batch=4, seq=16, heads=8): ~142ms → ~100ms (compiled)

## 📂 File Structure

```
Lean-HRM-MLX/
├── venv_mlx/                    # Virtual environment
├── requirements_mlx.txt         # Human-readable deps
├── requirements_mlx_frozen.txt  # Exact versions
├── test_mlx_setup.py           # Setup verification
├── SETUP_MLX.md                # This file
│
├── models/                      # Model implementations
│   ├── hrm/                    
│   │   └── hrm_act_v1.py      # Original PyTorch version
│   │   └── hrm_mlx.py         # MLX version (to be created)
│   ├── layers.py               # Original layers
│   └── mlx_layers.py          # MLX layers (to be created)
│
└── ...
```

## 🎯 Next Steps

1. **Create MLX Layers** (`models/mlx_layers.py`)
   - MLXAttention (replaces FlashAttention)
   - MLXSwiGLU
   - MLXRotaryEmbedding
   - RMS normalization

2. **Port HRM Model** (`models/hrm/hrm_mlx.py`)
   - Convert hierarchical reasoning loops
   - Adapt Q-learning halting mechanism
   - Implement carry state management

3. **Create MLX Dataset** (`dataset/mlx_dataset.py`)
   - Use mlx-data for efficient loading
   - Remove torch-specific code

4. **Training Loop** (`pretrain_mlx.py`)
   - Remove distributed training code
   - Add MLX-specific optimizations
   - Keep W&B logging

## 💡 Performance Tips

1. **Use @mx.compile** for hot paths
2. **Batch operations** where possible
3. **Use unified memory** efficiently (no CPU↔GPU copies)
4. **Profile with Metal System Trace**
5. **Leverage automatic mixed precision**

## 🐛 Troubleshooting

### Import errors:
```bash
# Make sure you're using the venv
which python
# Should show: .../venv_mlx/bin/python
```

### Metal errors:
```bash
# Check GPU is available
python -c "import mlx.core as mx; print(mx.default_device())"
# Should show: Device(gpu, 0)
```

### Memory issues:
```python
# MLX uses unified memory - monitor with Activity Monitor
# Reduce batch size if needed
```

## 📚 Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
- [Original HRM Paper](https://arxiv.org/abs/2506.21734)
- [Lean-HRM-MLX GitHub](https://github.com/your-repo)

---

**Created:** November 9, 2025  
**Last Updated:** November 9, 2025  
**Maintainer:** @umang

