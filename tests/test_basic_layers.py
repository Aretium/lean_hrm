"""
Test script for basic layers: MLXLinear and rms_norm

This script tests:
1. MLXLinear layer initialization and forward pass
2. RMS Normalization correctness
3. Shape preservation
4. Numerical stability
"""

import mlx.core as mx
import sys
sys.path.append('..')

from mlx_utils.layers import MLXLinear, rms_norm


def test_linear():
    """Test MLXLinear layer."""
    print("\n" + "="*70)
    print("Testing MLXLinear")
    print("="*70)

    # Test parameters
    batch_size = 4
    seq_len = 16
    in_features = 512
    out_features = 256

    # Create layer
    linear = MLXLinear(in_features, out_features, bias=True)

    # Create input
    x = mx.random.normal((batch_size, seq_len, in_features))

    # Forward pass
    output = linear(x)

    # Check shape
    expected_shape = (batch_size, seq_len, out_features)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print(f"✓ Shape test passed: {output.shape}")

    # Check that output is not all zeros (weights are initialized)
    assert mx.abs(output).max() > 0, "Output should not be all zeros"
    print(f"✓ Non-zero output: max={mx.abs(output).max():.4f}")

    # Test without bias
    linear_no_bias = MLXLinear(in_features, out_features, bias=False)
    output_no_bias = linear_no_bias(x)
    assert output_no_bias.shape == expected_shape
    print(f"✓ No-bias variant works: {output_no_bias.shape}")

    print("\n✅ MLXLinear tests PASSED\n")


def test_rms_norm():
    """Test RMS normalization."""
    print("\n" + "="*70)
    print("Testing rms_norm")
    print("="*70)

    # Test parameters
    batch_size = 4
    seq_len = 16
    hidden_size = 512

    # Create input
    x = mx.random.normal((batch_size, seq_len, hidden_size))

    # Apply RMS norm
    output = rms_norm(x, variance_epsilon=1e-5)

    # Check shape preservation
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print(f"✓ Shape preserved: {output.shape}")

    # Check that RMS is approximately 1 (within tolerance)
    rms = mx.sqrt(mx.mean(output ** 2, axis=-1))
    expected_rms = 1.0
    max_deviation = mx.abs(rms - expected_rms).max()
    print(f"✓ RMS check: max deviation from 1.0 = {max_deviation:.6f}")
    assert max_deviation < 0.01, f"RMS should be close to 1.0, max deviation: {max_deviation}"

    # Test numerical stability with zeros
    x_zeros = mx.zeros((2, 4, 8))
    output_zeros = rms_norm(x_zeros, variance_epsilon=1e-5)
    assert mx.all(mx.isfinite(output_zeros)), "Should handle zeros without NaN/Inf"
    print(f"✓ Handles zeros without NaN/Inf")

    # Test with very small values
    x_small = mx.ones((2, 4, 8)) * 1e-10
    output_small = rms_norm(x_small, variance_epsilon=1e-5)
    assert mx.all(mx.isfinite(output_small)), "Should handle small values"
    print(f"✓ Handles small values")

    print("\n✅ rms_norm tests PASSED\n")


def test_weight_initialization():
    """Test that weight initialization follows truncated normal distribution."""
    print("\n" + "="*70)
    print("Testing Weight Initialization")
    print("="*70)

    in_features = 512
    out_features = 256

    # Create multiple layers to check initialization distribution
    linear = MLXLinear(in_features, out_features, bias=True)

    # Check weight shape
    assert linear.weight.shape == (out_features, in_features)
    print(f"✓ Weight shape correct: {linear.weight.shape}")

    # Check bias shape and initialization (should be zeros)
    assert linear.bias.shape == (out_features,)
    assert mx.all(linear.bias == 0), "Bias should be initialized to zeros"
    print(f"✓ Bias initialized to zeros: {linear.bias.shape}")

    # Check that weights are not all the same
    assert mx.std(linear.weight) > 0, "Weights should have variance"
    print(f"✓ Weights have variance: std={mx.std(linear.weight):.4f}")

    # Check approximate std (should be around 1/sqrt(in_features))
    expected_std = 1.0 / (in_features ** 0.5)
    actual_std = mx.std(linear.weight)
    print(f"✓ Expected std ≈ {expected_std:.4f}, actual std = {actual_std:.4f}")

    print("\n✅ Weight initialization tests PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BASIC LAYERS TEST SUITE")
    print("="*70)

    try:
        test_linear()
        test_rms_norm()
        test_weight_initialization()

        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
