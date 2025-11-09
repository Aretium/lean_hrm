"""
MLX Utilities for Lean-HRM

This package contains MLX-specific adaptations of the HRM model components.
It replaces PyTorch/CUDA dependencies with Apple Silicon-optimized MLX implementations.

Directory Structure:
-------------------
mlx_utils/
├── __init__.py           # This file - package initialization
├── layers.py             # MLX attention, transformers, activations
├── embeddings.py         # MLX embeddings (token, position, puzzle)
├── hrm_model.py          # MLX HRM architecture
├── losses.py             # MLX loss functions
├── optimizers.py         # MLX optimizers (Adam, SignSGD)
├── dataset.py            # MLX data loading
├── training.py           # MLX training loop
└── utils.py              # Helper functions

Key Replacements:
-----------------
1. flash_attn → MLX scaled dot-product attention
2. torch.nn → mlx.nn
3. torch.distributed → Single device optimization
4. CastedLinear/Embedding → Native MLX layers
5. Custom CUDA kernels → MLX Metal kernels
"""

__version__ = "0.1.0"
__author__ = "Lean-HRM-MLX Team"

# Will be populated as we implement components
__all__ = [
    # Layers (to be implemented)
    # "MLXAttention",
    # "MLXSwiGLU", 
    # "MLXRotaryEmbedding",
    # "rms_norm",
    
    # Embeddings (to be implemented)
    # "MLXEmbedding",
    # "MLXSparseEmbedding",
    
    # Model (to be implemented)
    # "HRM_MLX",
    
    # Losses (to be implemented)
    # "StableMaxCrossEntropy",
    # "ACTLossHead",
    
    # Optimizers (to be implemented)
    # "AdamMLX",
    # "SignSGD_MLX",
    
    # Reference & Validation (available now)
    "OriginalSpecs",
    "NumericalValidator",
    "compare_outputs",
]

# Import reference utilities
try:
    from .reference import OriginalSpecs, NumericalValidator, compare_outputs
except ImportError:
    pass  # Not critical during initial setup

