"""
MLX Layers - Core neural network building blocks

This module contains MLX implementations of the core layers used in HRM:
- Attention mechanisms (replacing FlashAttention)
- SwiGLU activation
- Rotary Position Embeddings (RoPE)
- RMS Normalization
- Linear layers

Key differences from PyTorch version:
- No manual dtype casting needed
- No CUDA kernel dependencies
- Optimized for Apple Metal GPU
- Simpler API with @mx.compile support

ORIGINAL PYTORCH REFERENCE:
---------------------------
File: models/layers.py (lines 1-158)

Original Components:
1. flash_attn_func (lines 8-11)
   → Replaced by MLXAttention
   → FlashAttention CUDA kernel → Native MLX attention
   
2. RotaryEmbedding (lines 80-96)
   → Replicated as MLXRotaryEmbedding
   → Same algorithm, MLX arrays instead of torch.Tensor
   
3. apply_rotary_pos_emb (lines 30-40)
   → Same function signature in MLX
   → rotate_half helper (lines 23-27)
   
4. Attention class (lines 98-136)
   → Replicated as MLXAttention
   → QKV projection → attention → output projection
   
5. SwiGLU (lines 138-149)
   → Replicated as MLXSwiGLU
   → gate_up_proj → silu(gate) * up → down_proj
   
6. rms_norm function (lines 151-158)
   → Critical: matches PyTorch exactly
   → Convert to float32 → compute → convert back
   
7. CastedLinear (lines 43-60)
   → Simplified to MLXLinear (no casting needed!)
   
8. CastedEmbedding (lines 62-78)
   → See embeddings.py

NUMERICAL VALIDATION:
Use mlx_utils.reference.NumericalValidator to verify outputs match!
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional
import math

# Type alias for RoPE
CosSin = Tuple[mx.array, mx.array]


# ============================================================================
# ATTENTION
# ============================================================================

class MLXAttention(nn.Module):
    """
    Multi-head attention with optional causal masking.
    
    Replaces: flash_attn_func from PyTorch version
    
    Features:
    - Scaled dot-product attention
    - RoPE support
    - Causal masking
    - Metal-optimized computation
    """
    
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        causal: bool = False
    ):
        super().__init__()
        # TODO: Implement initialization
        pass
    
    def __call__(
        self,
        hidden_states: mx.array,
        cos_sin: Optional[CosSin] = None
    ) -> mx.array:
        """
        Forward pass.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos_sin: Optional (cos, sin) tensors for RoPE
            
        Returns:
            [batch, seq_len, hidden_size]
        """
        # TODO: Implement attention computation
        pass


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

class MLXSwiGLU(nn.Module):
    """
    SwiGLU activation function.
    
    Replaces: SwiGLU from PyTorch layers.py
    
    Formula: SwiGLU(x) = Swish(xW_gate+b) ⊙ (xW_up+b)W_down
    """
    
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        # TODO: Implement initialization
        self.hidden_size = hidden_size
        self.expansion = expansion
        self.expanded_size = int(hidden_size * expansion )

        #projections
        self.gate_up_proj = nn.Linear(hidden_size, self.expanded_size * 2, bias=False)
        self.down_proj = nn.Linear(self.expanded_size, hidden_size, bias=False)
    def Swish(self, x: mx.array) -> mx.array:
        return x * mx.nn.functional.sigmoid(x)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, hidden_size]
            
        Returns:
            [batch, seq_len, hidden_size]
        """
        
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        swiglu_x=self.Swish(gate) * up
        swiglu_x=self.down_proj(swiglu_x)
        return swiglu_x


# ============================================================================
# POSITIONAL ENCODINGS
# ============================================================================

class MLXRotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    Replaces: RotaryEmbedding from PyTorch layers.py
    
    Provides rotational position information without adding to embeddings.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float = 10000.0
    ):
        super().__init__()
        # TODO: Implement initialization
        pass
    
    def __call__(self) -> CosSin:
        """
        Returns precomputed (cos, sin) tensors.
        
        Returns:
            Tuple of (cos, sin) arrays for RoPE application
        """
        # TODO: Implement RoPE computation
        pass


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array
) -> Tuple[mx.array, mx.array]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_heads, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]
        
    Returns:
        Rotated (q, k) tensors
    """
    # TODO: Implement rotation logic
    pass


# ============================================================================
# NORMALIZATION
# ============================================================================

def rms_norm(
    hidden_states: mx.array,
    variance_epsilon: float = 1e-5
) -> mx.array:
    """
    Root Mean Square Layer Normalization.
    
    Replaces: rms_norm from PyTorch layers.py
    
    More efficient than LayerNorm (no mean centering).
    
    Args:
        hidden_states: Input tensor
        variance_epsilon: Small constant for numerical stability
        
    Returns:
        Normalized tensor with same shape as input
    """
    
    return hidden_states / mx.sqrt(mx.mean(hidden_states**2, axis=-1, keepdims=True) + variance_epsilon)


# ============================================================================
# LINEAR LAYERS
# ============================================================================

class MLXLinear(nn.Module):
    """
    Linear layer with custom initialization.
    
    Replaces: CastedLinear from PyTorch layers.py
    
    Note: MLX doesn't need explicit dtype casting!
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()
        # TODO: Implement initialization
        pass
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        # TODO: Implement linear transformation
        pass


# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================

class MLXTransformerBlock(nn.Module):
    """
    Single transformer block: Attention + MLP with post-norm.
    
    Replaces: HierarchicalReasoningModel_ACTV1Block from hrm_act_v1.py
    
    Architecture:
    - Post-norm (different from typical pre-norm!)
    - RMS normalization
    - SwiGLU activation in MLP
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        rms_norm_eps: float = 1e-5,
        causal: bool = False
    ):
        super().__init__()
        # TODO: Implement initialization
        pass
    
    def __call__(
        self,
        hidden_states: mx.array,
        cos_sin: Optional[CosSin] = None
    ) -> mx.array:
        """
        Forward pass.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos_sin: Optional RoPE (cos, sin)
            
        Returns:
            [batch, seq_len, hidden_size]
        """
        # TODO: Implement transformer block
        pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_multiple(a: int, b: int) -> int:
    """Find smallest multiple of b that is >= a."""
    return (-(a // -b)) * b


# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================

"""
Implementation Priority:
1. rms_norm - simplest, no state
2. MLXLinear - basic building block
3. MLXSwiGLU - uses MLXLinear
4. MLXRotaryEmbedding - standalone utility
5. MLXAttention - core component (most complex)
6. MLXTransformerBlock - combines above

Key MLX Features to Use:
- @mx.compile for speed
- Automatic broadcasting
- Efficient matmul with @
- mx.fast namespace for optimized ops
- No need for .to(dtype) casting

Performance Tips:
- Batch operations where possible
- Avoid Python loops over batch/sequence
- Use mx.eval() explicitly when needed
- Profile with Metal System Trace
"""

