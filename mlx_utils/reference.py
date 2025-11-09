"""
Reference Implementations - Original PyTorch Components

This module contains reference information and comparison utilities for the original
PyTorch HRM implementation. Use these to verify MLX implementations match the original.

NOT FOR PRODUCTION - Only for testing and validation!
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Any, Optional, Tuple
import numpy as np


# ============================================================================
# ORIGINAL PYTORCH SPECIFICATIONS
# ============================================================================

class OriginalSpecs:
    """
    Specifications from the original PyTorch implementation.
    
    Use these constants to ensure MLX implementation matches exactly.
    """
    
    # From models/hrm/hrm_act_v1.py
    ARCHITECTURE = {
        "name": "HierarchicalReasoningModel_ACTV1",
        "H_cycles": 2,          # High-level reasoning cycles
        "L_cycles": 2,          # Low-level reasoning cycles per H cycle
        "H_layers": 4,          # Transformer layers in H module
        "L_layers": 4,          # Transformer layers in L module
        "hidden_size": 512,     # Hidden dimension
        "num_heads": 8,         # Attention heads
        "expansion": 4.0,       # MLP expansion ratio
        "halt_max_steps": 16,   # Maximum ACT steps
        "halt_exploration_prob": 0.1,  # Exploration probability for Q-learning
        "pos_encodings": "rope",  # Position encoding type
        "rms_norm_eps": 1e-5,   # RMS norm epsilon
        "rope_theta": 10000.0,  # RoPE base frequency
        "forward_dtype": "bfloat16",  # Forward pass precision
    }
    
    # From models/layers.py
    ATTENTION = {
        "impl": "flash_attn_func",  # Original uses FlashAttention
        "causal": False,            # Non-causal attention
        "head_dim": 64,            # hidden_size // num_heads
        "num_key_value_heads": 8,  # Same as num_heads (no GQA)
    }
    
    # From models/sparse_embedding.py
    SPARSE_EMBEDDING = {
        "init_std": 0.0,           # Zero initialization for puzzle embeddings
        "optimizer": "SignSGD",    # Sign-based SGD
        "distributed": True,       # Original uses all-gather
        "local_weights_buffer": True,  # Separate local buffer for batch
    }
    
    # From models/losses.py
    LOSS = {
        "lm_loss_type": "stablemax_cross_entropy",  # Default loss
        "q_halt_loss": "binary_cross_entropy",      # BCE for halting
        "q_continue_loss": "binary_cross_entropy",  # BCE for continue
        "loss_weights": {
            "lm": 1.0,
            "q_halt": 0.5,
            "q_continue": 0.5,
        },
        "ignore_label_id": -100,
    }
    
    # From pretrain.py (config/cfg_pretrain.yaml)
    TRAINING = {
        "global_batch_size": 768,
        "lr": 1e-4,
        "lr_min_ratio": 1.0,
        "lr_warmup_steps": 2000,
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "puzzle_emb_lr": 1e-2,      # 100x higher for embeddings!
        "puzzle_emb_weight_decay": 0.1,
        "epochs": 100000,
        "eval_interval": 10000,
    }
    
    # From dataset/common.py
    DATASET = {
        "vocab_size": None,  # Task-dependent
        "seq_len": None,     # Task-dependent
        "pad_id": 0,
        "ignore_label_id": -100,
    }


# ============================================================================
# ORIGINAL COMPONENT SIGNATURES
# ============================================================================

class OriginalSignatures:
    """
    Function signatures from PyTorch implementation.
    Use these to ensure MLX APIs match.
    """
    
    # From models/layers.py
    ATTENTION_SIGNATURE = """
    def flash_attn_func(
        q: Tensor,  # [batch, seq_len, num_heads, head_dim]
        k: Tensor,  # [batch, seq_len, num_heads, head_dim]
        v: Tensor,  # [batch, seq_len, num_heads, head_dim]
        causal: bool = False,
        softmax_scale: Optional[float] = None  # 1/sqrt(head_dim)
    ) -> Tensor:  # [batch, seq_len, num_heads, head_dim]
    """
    
    # From models/layers.py
    RMS_NORM_SIGNATURE = """
    def rms_norm(
        hidden_states: Tensor,
        variance_epsilon: float = 1e-5
    ) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        return hidden_states.to(input_dtype)
    """
    
    # From models/layers.py
    ROPE_SIGNATURE = """
    def apply_rotary_pos_emb(
        q: Tensor,  # [batch, seq_len, num_heads, head_dim]
        k: Tensor,  # [batch, seq_len, num_heads, head_dim]
        cos: Tensor,  # [seq_len, head_dim]
        sin: Tensor   # [seq_len, head_dim]
    ) -> Tuple[Tensor, Tensor]:
        q = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        k = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        return q, k
    """
    
    # From models/losses.py
    STABLEMAX_SIGNATURE = """
    def s(x, epsilon=1e-30):
        return torch.where(
            x < 0,
            1/(1-x+epsilon),
            x + 1
        )
    
    def log_stablemax(x, dim=-1):
        s_x = s(x)
        return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))
    """
    
    # From models/sparse_embedding.py
    SIGNSGD_SIGNATURE = """
    def _sparse_emb_signsgd_dist(
        local_weights_grad: Tensor,  # [batch, emb_dim]
        local_ids: Tensor,           # [batch]
        weights: Tensor,             # [num_embeddings, emb_dim]
        lr: float,
        weight_decay: float,
        world_size: int
    ):
        # All-gather gradients
        # Unique IDs
        # p = p * (1 - lr * wd) - lr * sign(grad)
        # Update weights[unique_ids]
    """


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

def compare_outputs(
    mlx_output: mx.array,
    pytorch_output: np.ndarray,
    name: str = "output",
    rtol: float = 1e-5,
    atol: float = 1e-5
) -> Dict[str, Any]:
    """
    Compare MLX output with PyTorch reference.
    
    Args:
        mlx_output: MLX array
        pytorch_output: PyTorch tensor as numpy
        name: Name for logging
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Dict with comparison metrics
    """
    mlx_np = np.array(mlx_output)
    
    # Check shapes match
    shape_match = mlx_np.shape == pytorch_output.shape
    
    # Check values match
    close = np.allclose(mlx_np, pytorch_output, rtol=rtol, atol=atol)
    
    # Compute metrics
    abs_diff = np.abs(mlx_np - pytorch_output)
    rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-10)
    
    result = {
        "name": name,
        "shape_match": shape_match,
        "values_close": close,
        "mlx_shape": mlx_np.shape,
        "pytorch_shape": pytorch_output.shape,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
    }
    
    return result


def print_comparison(comparison: Dict[str, Any]):
    """Pretty print comparison results."""
    print(f"\n{'='*70}")
    print(f"Comparison: {comparison['name']}")
    print(f"{'='*70}")
    print(f"✓ Shape match:    {comparison['shape_match']}")
    print(f"  MLX shape:      {comparison['mlx_shape']}")
    print(f"  PyTorch shape:  {comparison['pytorch_shape']}")
    print(f"✓ Values close:   {comparison['values_close']}")
    print(f"  Max abs diff:   {comparison['max_abs_diff']:.2e}")
    print(f"  Mean abs diff:  {comparison['mean_abs_diff']:.2e}")
    print(f"  Max rel diff:   {comparison['max_rel_diff']:.2e}")
    print(f"  Mean rel diff:  {comparison['mean_rel_diff']:.2e}")
    print(f"{'='*70}")


def compare_gradients(
    mlx_grads: Dict[str, mx.array],
    pytorch_grads: Dict[str, np.ndarray],
    rtol: float = 1e-4,
    atol: float = 1e-4
) -> Dict[str, Dict[str, Any]]:
    """
    Compare gradients between MLX and PyTorch.
    
    Args:
        mlx_grads: Dict of parameter name -> MLX gradient
        pytorch_grads: Dict of parameter name -> PyTorch gradient (numpy)
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Dict of parameter name -> comparison results
    """
    results = {}
    
    for name in mlx_grads.keys():
        if name not in pytorch_grads:
            print(f"Warning: {name} in MLX but not PyTorch")
            continue
        
        results[name] = compare_outputs(
            mlx_grads[name],
            pytorch_grads[name],
            name=f"grad/{name}",
            rtol=rtol,
            atol=atol
        )
    
    return results


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

def generate_test_batch(
    batch_size: int = 4,
    seq_len: int = 16,
    vocab_size: int = 256,
    num_puzzle_identifiers: int = 100,
    seed: int = 42
) -> Dict[str, mx.array]:
    """
    Generate a test batch matching PyTorch format.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        num_puzzle_identifiers: Number of puzzles
        seed: Random seed
        
    Returns:
        Batch dict with MLX arrays
    """
    np.random.seed(seed)
    
    return {
        "inputs": mx.array(np.random.randint(0, vocab_size, (batch_size, seq_len)), dtype=mx.int32),
        "labels": mx.array(np.random.randint(-100, vocab_size, (batch_size, seq_len)), dtype=mx.int32),
        "puzzle_identifiers": mx.array(np.random.randint(0, num_puzzle_identifiers, batch_size), dtype=mx.int32),
    }


def generate_attention_test_data(
    batch_size: int = 2,
    seq_len: int = 8,
    num_heads: int = 4,
    head_dim: int = 32,
    seed: int = 42
) -> Dict[str, mx.array]:
    """
    Generate test data for attention comparison.
    
    Returns:
        Dict with Q, K, V arrays
    """
    np.random.seed(seed)
    
    return {
        "q": mx.array(np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)),
        "k": mx.array(np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)),
        "v": mx.array(np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)),
    }


# ============================================================================
# REFERENCE IMPLEMENTATIONS (Simplified PyTorch Logic)
# ============================================================================

def reference_rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """
    Reference RMS norm implementation (matches PyTorch).
    
    Use this to validate MLX implementation.
    """
    # Convert to float32 for computation (like PyTorch)
    variance = mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True)
    normalized = x.astype(mx.float32) * mx.rsqrt(variance + eps)
    return normalized.astype(x.dtype)


def reference_stablemax(x: mx.array, epsilon: float = 1e-30) -> mx.array:
    """
    Reference StableMax transformation (matches PyTorch).
    """
    return mx.where(
        x < 0,
        1.0 / (1.0 - x + epsilon),
        x + 1.0
    )


def reference_log_stablemax(x: mx.array, axis: int = -1) -> mx.array:
    """
    Reference log-stablemax (matches PyTorch).
    """
    s_x = reference_stablemax(x)
    return mx.log(s_x / mx.sum(s_x, axis=axis, keepdims=True))


# ============================================================================
# NUMERICAL VALIDATION SUITE
# ============================================================================

class NumericalValidator:
    """
    Validate MLX implementations match PyTorch numerically.
    
    Usage:
        validator = NumericalValidator()
        validator.test_rms_norm()
        validator.test_attention()
        validator.print_summary()
    """
    
    def __init__(self):
        self.results = {}
    
    def test_rms_norm(
        self,
        mlx_fn,
        test_shapes: list = [(4, 16, 512), (2, 32, 256)]
    ):
        """Test RMS norm implementation."""
        results = []
        
        for shape in test_shapes:
            x = mx.random.normal(shape)
            
            mlx_out = mlx_fn(x)
            ref_out = reference_rms_norm(x)
            
            comp = compare_outputs(
                mlx_out,
                np.array(ref_out),
                name=f"rms_norm_{shape}"
            )
            results.append(comp)
        
        self.results["rms_norm"] = results
        return results
    
    def test_stablemax(
        self,
        mlx_fn,
        test_shapes: list = [(4, 16, 256), (2, 32, 128)]
    ):
        """Test StableMax implementation."""
        results = []
        
        for shape in test_shapes:
            x = mx.random.normal(shape) * 2.0  # Test range
            
            mlx_out = mlx_fn(x)
            ref_out = reference_stablemax(x)
            
            comp = compare_outputs(
                mlx_out,
                np.array(ref_out),
                name=f"stablemax_{shape}"
            )
            results.append(comp)
        
        self.results["stablemax"] = results
        return results
    
    def print_summary(self):
        """Print summary of all tests."""
        print("\n" + "="*70)
        print("NUMERICAL VALIDATION SUMMARY")
        print("="*70)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, results in self.results.items():
            print(f"\n{test_name}:")
            for result in results:
                total_tests += 1
                if result["values_close"]:
                    passed_tests += 1
                    print(f"  ✓ {result['name']}: PASS")
                else:
                    print(f"  ✗ {result['name']}: FAIL (max diff: {result['max_abs_diff']:.2e})")
        
        print(f"\n{'='*70}")
        print(f"Total: {passed_tests}/{total_tests} tests passed")
        print(f"{'='*70}\n")


# ============================================================================
# PYTORCH CONVERSION HELPERS
# ============================================================================

def pytorch_to_mlx(pytorch_tensor) -> mx.array:
    """
    Convert PyTorch tensor to MLX array.
    
    Args:
        pytorch_tensor: PyTorch tensor or numpy array
        
    Returns:
        MLX array
    """
    if hasattr(pytorch_tensor, 'detach'):
        # PyTorch tensor
        numpy_array = pytorch_tensor.detach().cpu().numpy()
    else:
        # Already numpy
        numpy_array = pytorch_tensor
    
    return mx.array(numpy_array)


def mlx_to_pytorch(mlx_array: mx.array):
    """
    Convert MLX array to format for PyTorch comparison.
    
    Args:
        mlx_array: MLX array
        
    Returns:
        Numpy array suitable for PyTorch comparison
    """
    return np.array(mlx_array)


def load_pytorch_checkpoint(checkpoint_path: str) -> Dict[str, np.ndarray]:
    """
    Load PyTorch checkpoint for comparison.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        
    Returns:
        Dict of parameter name -> numpy array
    """
    # TODO: Implement when needed for actual comparison
    # Will need torch import, but only in this utility file
    raise NotImplementedError(
        "PyTorch checkpoint loading not implemented. "
        "Install PyTorch in a separate environment and use "
        "torch.load() to load checkpoint, then save as numpy."
    )


# ============================================================================
# NOTES FOR USAGE
# ============================================================================

"""
Usage Examples:

1. Validate RMS Norm:
    ```python
    from mlx_utils.reference import NumericalValidator, OriginalSpecs
    from mlx_utils.layers import rms_norm
    
    validator = NumericalValidator()
    validator.test_rms_norm(rms_norm)
    validator.print_summary()
    ```

2. Compare with PyTorch output:
    ```python
    from mlx_utils.reference import compare_outputs, print_comparison
    
    mlx_output = mlx_model(batch)
    pytorch_output = pytorch_model(batch)  # Run in separate env
    
    result = compare_outputs(mlx_output, pytorch_output, "model_output")
    print_comparison(result)
    ```

3. Check architecture specs:
    ```python
    from mlx_utils.reference import OriginalSpecs
    
    print(f"H cycles: {OriginalSpecs.ARCHITECTURE['H_cycles']}")
    print(f"Loss type: {OriginalSpecs.LOSS['lm_loss_type']}")
    ```

4. Generate test data:
    ```python
    from mlx_utils.reference import generate_test_batch
    
    batch = generate_test_batch(batch_size=4, seq_len=16)
    output = model(batch)
    ```

Best Practices:
- Always validate numerical correctness before training
- Use reference implementations for unit tests
- Compare gradients, not just outputs
- Test edge cases (zeros, large values, etc.)
- Keep PyTorch in separate environment to avoid conflicts
"""

