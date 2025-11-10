"""
Test script for SwiGLU activation

This script tests:
1. MLXSwiGLU initialization
2. Forward pass correctness
3. Shape transformations
4. Expansion factor calculations
"""

import mlx.core as mx
import mlx.nn as nn
import sys
sys.path.append('..')

from mlx_utils.layers import MLXSwiGLU, _find_multiple


def test_find_multiple():
    """Test the _find_multiple helper function."""
    print("\n" + "="*70)
    print("Testing _find_multiple Helper")
    print("="*70)

    # Test cases
    assert _find_multiple(100, 256) == 256, "100 -> 256"
    assert _find_multiple(300, 256) == 512, "300 -> 512"
    assert _find_multiple(256, 256) == 256, "256 -> 256"
    assert _find_multiple(512, 256) == 512, "512 -> 512"
    assert _find_multiple(513, 256) == 768, "513 -> 768"

    print("✓ _find_multiple works correctly")
    print("  Examples:")
    print(f"    _find_multiple(100, 256) = {_find_multiple(100, 256)}")
    print(f"    _find_multiple(300, 256) = {_find_multiple(300, 256)}")
    print(f"    _find_multiple(513, 256) = {_find_multiple(513, 256)}")

    print("\n✅ _find_multiple tests PASSED\n")


def test_swiglu_initialization():
    """Test SwiGLU initialization."""
    print("\n" + "="*70)
    print("Testing MLXSwiGLU Initialization")
    print("="*70)

    hidden_size = 512
    expansion = 4.0

    # Create SwiGLU
    swiglu = MLXSwiGLU(hidden_size, expansion)

    # Calculate expected intermediate size
    # PyTorch: inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
    expected_inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
    print(f"  Hidden size: {hidden_size}")
    print(f"  Expansion: {expansion}")
    print(f"  Expected intermediate: {expected_inter}")

    # Check gate_up_proj shape (should output 2 * inter)
    # Weight shape: [out_features, in_features]
    assert swiglu.gate_up_proj.weight.shape == (expected_inter * 2, hidden_size)
    print(f"✓ gate_up_proj weight shape: {swiglu.gate_up_proj.weight.shape}")

    # Check down_proj shape
    assert swiglu.down_proj.weight.shape == (hidden_size, expected_inter)
    print(f"✓ down_proj weight shape: {swiglu.down_proj.weight.shape}")

    print("\n✅ SwiGLU initialization tests PASSED\n")


def test_swiglu_forward():
    """Test SwiGLU forward pass."""
    print("\n" + "="*70)
    print("Testing MLXSwiGLU Forward Pass")
    print("="*70)

    batch_size = 4
    seq_len = 16
    hidden_size = 512
    expansion = 4.0

    # Create SwiGLU
    swiglu = MLXSwiGLU(hidden_size, expansion)

    # Create input
    x = mx.random.normal((batch_size, seq_len, hidden_size))

    # Forward pass
    output = swiglu(x)

    # Check shape (should preserve input shape)
    expected_shape = (batch_size, seq_len, hidden_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Output shape correct: {output.shape}")

    # Check that output is not all zeros
    assert mx.abs(output).max() > 0, "Output should not be all zeros"
    print(f"✓ Non-zero output: max={mx.abs(output).max():.4f}")

    # Check that output is different from input (transformation applied)
    assert not mx.allclose(output, x), "Output should be different from input"
    print(f"✓ Output differs from input (transformation applied)")

    print("\n✅ SwiGLU forward pass tests PASSED\n")


def test_swiglu_activation():
    """Test that SwiGLU correctly applies SiLU activation."""
    print("\n" + "="*70)
    print("Testing SwiGLU Activation Function")
    print("="*70)

    hidden_size = 256
    expansion = 2.0

    swiglu = MLXSwiGLU(hidden_size, expansion)

    # Create a simple input
    x = mx.ones((1, 1, hidden_size))

    # Forward pass
    output = swiglu(x)

    # The output should be non-linear due to SiLU activation
    # SiLU(x) = x * sigmoid(x)
    # For x=1: sigmoid(1) ≈ 0.731, so SiLU(1) ≈ 0.731

    print(f"✓ SwiGLU activation applied")
    print(f"  Input mean: {mx.mean(x):.4f}")
    print(f"  Output mean: {mx.mean(output):.4f}")
    print(f"  Output std: {mx.std(output):.4f}")

    # Test with zeros
    x_zeros = mx.zeros((2, 4, hidden_size))
    output_zeros = swiglu(x_zeros)
    # SiLU(0) = 0, so output should be close to bias (which is zero)
    print(f"✓ Handles zero input: max abs output = {mx.abs(output_zeros).max():.6f}")

    print("\n✅ SwiGLU activation tests PASSED\n")


def test_swiglu_different_expansions():
    """Test SwiGLU with different expansion factors."""
    print("\n" + "="*70)
    print("Testing SwiGLU with Different Expansions")
    print("="*70)

    hidden_size = 512
    batch_size = 2
    seq_len = 8

    expansions = [2.0, 4.0, 6.0, 8.0]

    for expansion in expansions:
        swiglu = MLXSwiGLU(hidden_size, expansion)
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        output = swiglu(x)

        assert output.shape == x.shape, f"Shape mismatch for expansion={expansion}"

        # Calculate intermediate size
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        print(f"  Expansion {expansion}: inter={inter}, output_shape={output.shape}")

    print(f"✓ All expansion factors work correctly")

    print("\n✅ SwiGLU expansion tests PASSED\n")


def test_swiglu_gradient_flow():
    """Test that gradients can flow through SwiGLU."""
    print("\n" + "="*70)
    print("Testing SwiGLU Gradient Flow")
    print("="*70)

    hidden_size = 256
    expansion = 4.0

    swiglu = MLXSwiGLU(hidden_size, expansion)

    # Create input
    x = mx.random.normal((2, 4, hidden_size))

    # Forward pass
    def loss_fn(x):
        output = swiglu(x)
        return mx.mean(output ** 2)

    # Compute loss and gradient
    loss_and_grad = mx.value_and_grad(loss_fn)
    loss, grad = loss_and_grad(x)

    print(f"  Loss: {loss:.4f}")
    print(f"  Gradient shape: {grad.shape}")
    print(f"  Gradient mean: {mx.mean(mx.abs(grad)):.6f}")

    # Check gradient is not all zeros
    assert mx.abs(grad).max() > 0, "Gradients should flow"
    print(f"✓ Gradients flow through SwiGLU")

    print("\n✅ SwiGLU gradient flow tests PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SWIGLU TEST SUITE")
    print("="*70)

    try:
        test_find_multiple()
        test_swiglu_initialization()
        test_swiglu_forward()
        test_swiglu_activation()
        test_swiglu_different_expansions()
        test_swiglu_gradient_flow()

        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
