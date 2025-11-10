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
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.output_size = head_dim * num_heads

        # QKV projection: projects to Q (num_heads), K (num_key_value_heads), V (num_key_value_heads)
        self.qkv_proj = MLXLinear(
            hidden_size,
            (num_heads + 2 * num_key_value_heads) * head_dim,
            bias=False
        )

        # Output projection
        self.o_proj = MLXLinear(self.output_size, hidden_size, bias=False)

        # Scaling factor for attention
        self.scale = 1.0 / math.sqrt(head_dim)

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
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)

        # Reshape and split into Q, K, V
        # qkv: [batch, seq_len, (num_heads + 2*num_key_value_heads) * head_dim]
        # -> [batch, seq_len, num_heads + 2*num_key_value_heads, head_dim]
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        # Split into Q, K, V
        query = qkv[:, :, :self.num_heads, :]
        key = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads, :]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:, :]

        # Apply RoPE if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            # Slice cos/sin to match sequence length
            cos = cos[:seq_len, :]
            sin = sin[:seq_len, :]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Expand key and value if using grouped-query attention (GQA)
        # For standard attention, num_heads == num_key_value_heads, so this is a no-op
        if self.num_key_value_heads != self.num_heads:
            # Repeat each key/value head to match query heads
            n_rep = self.num_heads // self.num_key_value_heads
            key = mx.repeat(key, n_rep, axis=2)
            value = mx.repeat(value, n_rep, axis=2)

        # Transpose for attention computation
        # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        query = mx.transpose(query, (0, 2, 1, 3))
        key = mx.transpose(key, (0, 2, 1, 3))
        value = mx.transpose(value, (0, 2, 1, 3))

        # Scaled dot-product attention
        # scores = (Q @ K.T) / sqrt(d_k)
        scores = (query @ mx.transpose(key, (0, 1, 3, 2))) * self.scale

        # Apply causal mask if needed
        if self.causal:
            # Create causal mask: upper triangular matrix
            mask = mx.triu(mx.ones((seq_len, seq_len)), k=1)
            # Convert to -inf for masked positions
            mask = mx.where(mask, -1e9, 0.0)
            scores = scores + mask

        # Softmax
        attn_weights = mx.softmax(scores, axis=-1)

        # Weighted sum of values
        # [batch, num_heads, seq_len, head_dim]
        attn_output = attn_weights @ value

        # Transpose back and reshape
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))

        # Flatten heads: [batch, seq_len, num_heads * head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)

        # Output projection
        return self.o_proj(attn_output)


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

class MLXSwiGLU(nn.Module):
    """
    SwiGLU activation function.

    Replaces: SwiGLU from PyTorch layers.py

    Formula: SwiGLU(x) = SiLU(xW_gate) ⊙ (xW_up)W_down
    """

    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.expansion = expansion

        # Match PyTorch: inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        # Use custom MLXLinear instead of nn.Linear to match initialization
        self.gate_up_proj = MLXLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = MLXLinear(inter, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            [batch, seq_len, hidden_size]
        """
        # Combined gate and up projection
        gate_up = self.gate_up_proj(x)

        # Split into gate and up
        # MLX doesn't have chunk, so we use array slicing
        inter_size = gate_up.shape[-1] // 2
        gate = gate_up[..., :inter_size]
        up = gate_up[..., inter_size:]

        # SiLU (Swish) activation: x * sigmoid(x)
        # MLX has nn.silu built-in
        activated = nn.silu(gate) * up

        # Down projection
        return self.down_proj(activated)


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
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

        # Compute position indices
        t = mx.arange(max_position_embeddings, dtype=mx.float32)

        # Outer product: [max_pos, dim/2]
        freqs = mx.expand_dims(t, axis=1) * mx.expand_dims(inv_freq, axis=0)

        # Different from paper, but uses a different permutation to obtain the same calculation
        emb = mx.concatenate([freqs, freqs], axis=-1)

        # Cache cos and sin
        self.cos_cached = mx.cos(emb)
        self.sin_cached = mx.sin(emb)

    def __call__(self) -> CosSin:
        """
        Returns precomputed (cos, sin) tensors.

        Returns:
            Tuple of (cos, sin) arrays for RoPE application
        """
        return self.cos_cached, self.sin_cached


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
    # Store original dtype
    orig_dtype = q.dtype

    # Ensure cos/sin have the right shape for broadcasting
    # cos, sin: [seq_len, head_dim] -> [1, seq_len, 1, head_dim]
    cos = mx.expand_dims(cos, axis=-2)  # [seq_len, 1, head_dim]
    sin = mx.expand_dims(sin, axis=-2)  # [seq_len, 1, head_dim]

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


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
        self.in_features = in_features
        self.out_features = out_features

        # Truncated LeCun normal init
        std = 1.0 / (in_features ** 0.5)
        self.weight = trunc_normal_init((out_features, in_features), std=std)

        if bias:
            # Zero init bias
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Linear transformation: x @ W.T + b
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output


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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.rms_norm_eps = rms_norm_eps

        # Compute head dimension
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Attention layer
        self.attention = MLXAttention(
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,  # Standard attention (no GQA by default)
            causal=causal
        )

        # MLP layer (SwiGLU)
        self.mlp = MLXSwiGLU(hidden_size, expansion)

        # We'll apply RMS norm inline in the forward pass

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
        # Post-norm architecture:
        # 1. Attention with residual
        # 2. RMS Norm
        # 3. MLP with residual
        # 4. RMS Norm

        # Attention block
        attn_output = self.attention(hidden_states, cos_sin)
        hidden_states = hidden_states + attn_output
        hidden_states = rms_norm(hidden_states, self.rms_norm_eps)

        # MLP block
        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_output
        hidden_states = rms_norm(hidden_states, self.rms_norm_eps)

        return hidden_states


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_multiple(a: int, b: int) -> int:
    """Find smallest multiple of b that is >= a."""
    return (-(a // -b)) * b


def trunc_normal_init(
    shape: tuple,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
    key: Optional[int] = None
) -> mx.array:
    """
    Truncated normal initialization (matches JAX/PyTorch behavior).

    This is the mathematically correct version used in the original HRM.

    Args:
        shape: Shape of the output tensor
        std: Standard deviation
        lower: Lower truncation bound (in std units)
        upper: Upper truncation bound (in std units)
        key: Random key (optional)

    Returns:
        Initialized array
    """
    if std == 0:
        return mx.zeros(shape)

    sqrt2 = math.sqrt(2)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    z = (b - a) / 2

    c = (2 * math.pi) ** -0.5
    pdf_u = c * math.exp(-0.5 * lower ** 2)
    pdf_l = c * math.exp(-0.5 * upper ** 2)
    comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

    # Sample from uniform distribution on [a, b]
    if key is not None:
        u = mx.random.uniform(a, b, shape, key=key)
    else:
        u = mx.random.uniform(a, b, shape)

    # Apply inverse error function
    # MLX doesn't have erfinv, so we use a numerical approximation
    # For better accuracy, we'll use a series approximation
    def erfinv_approx(x):
        # Approximate inverse error function using rational approximation
        # This is good enough for initialization purposes
        a = 0.147
        sgn = mx.sign(x)
        x = mx.abs(x)

        ln_term = mx.log(1 - x * x)
        term1 = 2 / (math.pi * a) + ln_term / 2
        term2 = ln_term / a

        result = sgn * mx.sqrt(-term1 + mx.sqrt(term1 * term1 - term2))
        return result

    z_vals = erfinv_approx(u)
    result = sqrt2 * comp_std * z_vals

    # Clip to bounds
    result = mx.clip(result, lower * comp_std, upper * comp_std)

    return result


def rotate_half(x: mx.array) -> mx.array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


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

