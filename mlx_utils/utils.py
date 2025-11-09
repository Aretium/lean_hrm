"""
MLX Utilities - Helper functions

General utility functions:
- Initialization helpers
- Array manipulation
- Debugging tools
- Profiling helpers
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import time
from functools import wraps


# ============================================================================
# INITIALIZATION
# ============================================================================

def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    mx.random.seed(seed)
    np.random.seed(seed)


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of parameters in model.

    Args:
        model: MLX model

    Returns:
        Total parameter count
    """
    def count_in_tree(tree):
        """Recursively count parameters in nested dict."""
        total = 0
        if isinstance(tree, dict):
            for value in tree.values():
                total += count_in_tree(value)
        elif isinstance(tree, mx.array):
            total += tree.size
        return total

    return count_in_tree(model.parameters())


def print_model_summary(model: nn.Module):
    """
    Print model architecture summary.

    Args:
        model: MLX model
    """
    def flatten_params(params, prefix=""):
        """Flatten nested parameter dict."""
        flat = {}
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(flatten_params(value, full_key))
            else:
                flat[full_key] = value
        return flat

    print("=" * 80)
    print("Model Summary")
    print("=" * 80)

    flat_params = flatten_params(model.parameters())

    total_params = 0
    for name, param in flat_params.items():
        param_count = param.size
        total_params += param_count
        print(f"{name:50s} {str(param.shape):20s} {param_count:>15,d}")

    print("=" * 80)
    print(f"{'Total Parameters':50s} {total_params:>35,d}")
    print(f"{'Trainable Parameters':50s} {total_params:>35,d}")
    print("=" * 80)


# ============================================================================
# ARRAY MANIPULATION
# ============================================================================

def tree_map(fn, tree):
    """
    Apply function to all arrays in a pytree.

    Args:
        fn: Function to apply
        tree: PyTree (dict, list, or nested structure)

    Returns:
        PyTree with function applied to all arrays
    """
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [tree_map(fn, v) for v in tree]
    elif isinstance(tree, tuple):
        return tuple(tree_map(fn, v) for v in tree)
    elif isinstance(tree, mx.array):
        return fn(tree)
    else:
        return tree


def tree_flatten(tree) -> List:
    """
    Flatten a pytree into a list of arrays.

    Args:
        tree: PyTree structure

    Returns:
        List of arrays
    """
    result = []

    def _flatten(subtree):
        if isinstance(subtree, dict):
            for value in subtree.values():
                _flatten(value)
        elif isinstance(subtree, (list, tuple)):
            for value in subtree:
                _flatten(value)
        elif isinstance(subtree, mx.array):
            result.append(subtree)

    _flatten(tree)
    return result


def detach_tree(tree):
    """
    Detach all arrays in tree from gradient graph.

    Args:
        tree: PyTree of arrays

    Returns:
        PyTree with detached arrays
    """
    return tree_map(mx.stop_gradient, tree)


# ============================================================================
# DEBUGGING
# ============================================================================

def check_nan_inf(x: mx.array, name: str = "array"):
    """
    Check for NaN or Inf values in array.

    Args:
        x: Array to check
        name: Name for error message

    Raises:
        ValueError if NaN or Inf found
    """
    # Force evaluation
    mx.eval(x)

    has_nan = mx.any(mx.isnan(x))
    has_inf = mx.any(mx.isinf(x))

    if has_nan or has_inf:
        issues = []
        if has_nan:
            issues.append("NaN")
        if has_inf:
            issues.append("Inf")
        raise ValueError(f"{name} contains {' and '.join(issues)} values!")


def print_array_stats(x: mx.array, name: str = "array"):
    """
    Print statistics about an array.

    Args:
        x: Array to analyze
        name: Name to display
    """
    # Force evaluation
    mx.eval(x)

    print(f"\nArray Statistics: {name}")
    print(f"  Shape: {x.shape}")
    print(f"  Dtype: {x.dtype}")
    print(f"  Min: {float(mx.min(x)):.6f}")
    print(f"  Max: {float(mx.max(x)):.6f}")
    print(f"  Mean: {float(mx.mean(x)):.6f}")
    print(f"  Std: {float(mx.std(x)):.6f}")

    # Check for NaN/Inf
    has_nan = bool(mx.any(mx.isnan(x)))
    has_inf = bool(mx.any(mx.isinf(x)))
    if has_nan or has_inf:
        print(f"  ⚠️  WARNING: Contains NaN={has_nan}, Inf={has_inf}")


def debug_gradient(grad: mx.array, name: str = "gradient"):
    """
    Print gradient statistics for debugging.

    Args:
        grad: Gradient array
        name: Name to display
    """
    # Force evaluation
    mx.eval(grad)

    print(f"\nGradient Debug: {name}")
    print(f"  Shape: {grad.shape}")
    print(f"  L2 norm: {float(mx.sqrt(mx.sum(grad * grad))):.6f}")
    print(f"  Min: {float(mx.min(grad)):.6e}")
    print(f"  Max: {float(mx.max(grad)):.6e}")
    print(f"  Mean: {float(mx.mean(grad)):.6e}")
    print(f"  Std: {float(mx.std(grad)):.6e}")

    # Gradient health checks
    norm = float(mx.sqrt(mx.sum(grad * grad)))
    if norm < 1e-8:
        print(f"  ⚠️  WARNING: Very small gradient (norm={norm:.2e})")
    elif norm > 1e3:
        print(f"  ⚠️  WARNING: Very large gradient (norm={norm:.2e})")

    has_nan = bool(mx.any(mx.isnan(grad)))
    has_inf = bool(mx.any(mx.isinf(grad)))
    if has_nan or has_inf:
        print(f"  ❌ ERROR: Contains NaN={has_nan}, Inf={has_inf}")


# ============================================================================
# PROFILING
# ============================================================================

class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("my_operation"):
            result = expensive_computation()
    """

    def __init__(self, name: str = "operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"[Timer] {self.name}: {self.elapsed:.4f}s")


def profile(fn):
    """
    Decorator to profile function execution time.

    Usage:
        @profile
        def my_function():
            ...
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[Profile] {fn.__name__}: {elapsed:.4f}s")
        return result
    return wrapper


def memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.

    Returns:
        Dict with memory stats (in MB)
    """
    try:
        # MLX provides memory info (newer API)
        # Try new API first, fall back to deprecated if needed
        try:
            active_memory = mx.get_active_memory() / 1024 / 1024  # Convert to MB
            peak_memory = mx.get_peak_memory() / 1024 / 1024
            cache_memory = mx.get_cache_memory() / 1024 / 1024
        except AttributeError:
            # Fall back to deprecated API
            active_memory = mx.metal.get_active_memory() / 1024 / 1024
            peak_memory = mx.metal.get_peak_memory() / 1024 / 1024
            cache_memory = mx.metal.get_cache_memory() / 1024 / 1024

        return {
            "active_mb": active_memory,
            "peak_mb": peak_memory,
            "cache_mb": cache_memory,
            "total_mb": active_memory + cache_memory
        }
    except Exception as e:
        # Fallback if memory API not available
        return {
            "active_mb": 0.0,
            "peak_mb": 0.0,
            "cache_mb": 0.0,
            "total_mb": 0.0,
            "error": str(e)
        }


# ============================================================================
# METAL GPU UTILITIES
# ============================================================================

def get_device_info() -> Dict[str, Any]:
    """
    Get information about Metal GPU device.

    Returns:
        Dict with device info
    """
    device = mx.default_device()

    info = {
        "device_type": str(device.type),
        "device_id": device.id if hasattr(device, 'id') else 0,
        "device_str": str(device)
    }

    # Try to get memory info
    try:
        memory_info = memory_usage()
        info.update(memory_info)
    except:
        pass

    return info


def synchronize():
    """
    Synchronize GPU operations (force evaluation).

    MLX is lazily evaluated, so this forces all pending operations
    to complete. Useful for accurate timing and debugging.
    """
    # Create a dummy operation and evaluate it to force synchronization
    dummy = mx.array([0.0])
    mx.eval(dummy)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_attention(
    attn_weights: mx.array,
    tokens: List[str],
    save_path: Optional[str] = None
):
    """
    Visualize attention weights as a heatmap.

    Args:
        attn_weights: Attention weights [num_heads, seq_len, seq_len]
        tokens: Token strings
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Convert to numpy
        attn_np = np.array(attn_weights)

        # Handle multiple heads by averaging
        if attn_np.ndim == 3:
            attn_np = attn_np.mean(axis=0)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot heatmap
        im = ax.imshow(attn_np, cmap='viridis', aspect='auto')

        # Set ticks
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

        # Add colorbar
        plt.colorbar(im, ax=ax)

        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        ax.set_title('Attention Weights')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("Warning: matplotlib not installed. Cannot visualize attention.")
        print("Install with: pip install matplotlib")


def plot_learning_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot learning curves from training history.

    Args:
        metrics_history: Dict mapping metric name to values over time
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt

        num_metrics = len(metrics_history)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3 * num_metrics))

        if num_metrics == 1:
            axes = [axes]

        for ax, (name, values) in zip(axes, metrics_history.items()):
            ax.plot(values)
            ax.set_xlabel('Step')
            ax.set_ylabel(name)
            ax.set_title(f'{name} over time')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved learning curves to {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("Warning: matplotlib not installed. Cannot plot learning curves.")
        print("Install with: pip install matplotlib")


# ============================================================================
# CONFIG UTILITIES
# ============================================================================

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Config dictionary
    """
    try:
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"Loaded config from {config_path}")
        return config

    except ImportError:
        print("Warning: PyYAML not installed. Cannot load YAML config.")
        print("Install with: pip install pyyaml")
        return {}
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def save_config(config: Dict, save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Config dictionary
        save_path: Path to save to
    """
    try:
        import yaml

        with open(save_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)

        print(f"Saved config to {save_path}")

    except ImportError:
        print("Warning: PyYAML not installed. Cannot save YAML config.")
        print("Install with: pip install pyyaml")
    except Exception as e:
        print(f"Error saving config: {e}")


# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================

"""
Implementation Priority:
1. set_random_seed - needed for reproducibility
2. count_parameters - useful for model info
3. Timer - profiling helper
4. tree_map, detach_tree - gradient utilities
5. check_nan_inf - debugging
6. Everything else as needed

MLX-Specific Utilities:

1. Evaluation forcing:
   ```python
   # MLX is lazy - force computation
   result = expensive_op(x)
   mx.eval(result)  # Now actually computed
   ```

2. Memory management:
   ```python
   # Clear computation graph
   mx.eval(model.parameters())
   
   # Check memory
   # Use Activity Monitor or Metal System Trace
   ```

3. Device info:
   ```python
   device = mx.default_device()
   print(device)  # Device(gpu, 0)
   ```

4. Profiling:
   ```python
   # Use Metal System Trace (Instruments)
   # Or simple Python timing
   with Timer("forward_pass"):
       output = model(input)
       mx.eval(output)
   ```

Useful MLX Functions to Wrap:
- mx.eval() - force evaluation
- mx.stop_gradient() - detach from graph
- mx.compile() - JIT compilation
- mx.metal.get_active_memory() - memory usage
- mx.metal.get_peak_memory() - peak memory

Debug Tips:
1. Check for NaN/Inf after each major operation
2. Print array shapes frequently
3. Use mx.eval() to force computation at checkpoints
4. Profile with Metal System Trace for GPU bottlenecks
5. Compare MLX vs NumPy results for correctness
"""

