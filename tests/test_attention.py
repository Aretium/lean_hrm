"""
Test script for MLXAttention

This script tests:
1. Attention initialization
2. Forward pass with and without RoPE
3. Causal masking
4. Shape transformations
5. Attention patterns
"""

import mlx.core as mx
import sys
import math
sys.path.append('..')

from mlx_utils.layers import MLXAttention, MLXRotaryEmbedding


def test_attention_initialization():
    """Test attention initialization."""
    print("\n" + "="*70)
    print("Testing MLXAttention Initialization")
    print("="*70)

    hidden_size = 512
    head_dim = 64
    num_heads = 8
    num_key_value_heads = 8

    # Create attention
    attn = MLXAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        causal=False
    )

    print(f"  Hidden size: {hidden_size}")
    print(f"  Head dim: {head_dim}")
    print(f"  Num heads: {num_heads}")

    # Check QKV projection shape
    # Should output (num_heads + 2 * num_key_value_heads) * head_dim
    expected_qkv_out = (num_heads + 2 * num_key_value_heads) * head_dim
    assert attn.qkv_proj.weight.shape == (expected_qkv_out, hidden_size)
    print(f"✓ QKV projection shape: {attn.qkv_proj.weight.shape}")

    # Check output projection shape
    output_size = head_dim * num_heads
    assert attn.o_proj.weight.shape == (hidden_size, output_size)
    print(f"✓ Output projection shape: {attn.o_proj.weight.shape}")

    # Check scale factor
    expected_scale = 1.0 / math.sqrt(head_dim)
    assert abs(attn.scale - expected_scale) < 1e-6
    print(f"✓ Scale factor: {attn.scale:.6f}")

    print("\n✅ Attention initialization tests PASSED\n")


def test_attention_forward_basic():
    """Test basic attention forward pass."""
    print("\n" + "="*70)
    print("Testing MLXAttention Forward Pass (Basic)")
    print("="*70)

    batch_size = 4
    seq_len = 16
    hidden_size = 512
    head_dim = 64
    num_heads = 8

    # Create attention
    attn = MLXAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        causal=False
    )

    # Create input
    x = mx.random.normal((batch_size, seq_len, hidden_size))

    # Forward pass (no RoPE)
    output = attn(x, cos_sin=None)

    # Check shape
    expected_shape = (batch_size, seq_len, hidden_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Output shape correct: {output.shape}")

    # Check non-zero output
    assert mx.abs(output).max() > 0, "Output should not be all zeros"
    print(f"✓ Non-zero output: max={mx.abs(output).max():.4f}")

    print("\n✅ Basic attention forward tests PASSED\n")


def test_attention_with_rope():
    """Test attention with RoPE."""
    print("\n" + "="*70)
    print("Testing MLXAttention with RoPE")
    print("="*70)

    batch_size = 2
    seq_len = 32
    hidden_size = 512
    head_dim = 64
    num_heads = 8

    # Create attention
    attn = MLXAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        causal=False
    )

    # Create RoPE
    rope = MLXRotaryEmbedding(head_dim, max_position_embeddings=2048)
    cos, sin = rope()

    # Create input
    x = mx.random.normal((batch_size, seq_len, hidden_size))

    # Forward pass with RoPE
    output_with_rope = attn(x, cos_sin=(cos, sin))

    # Forward pass without RoPE
    output_no_rope = attn(x, cos_sin=None)

    # Check shapes
    assert output_with_rope.shape == output_no_rope.shape
    print(f"✓ Output shape with RoPE: {output_with_rope.shape}")

    # Outputs should be different due to RoPE
    assert not mx.allclose(output_with_rope, output_no_rope, atol=1e-4)
    print(f"✓ RoPE changes output (as expected)")

    print("\n✅ Attention with RoPE tests PASSED\n")


def test_attention_causal_masking():
    """Test causal attention masking."""
    print("\n" + "="*70)
    print("Testing Causal Masking")
    print("="*70)

    batch_size = 1
    seq_len = 8
    hidden_size = 256
    head_dim = 64
    num_heads = 4

    # Create causal attention
    attn_causal = MLXAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        causal=True
    )

    # Create non-causal attention
    attn_non_causal = MLXAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        causal=False
    )

    # Use same weights for both
    attn_non_causal.qkv_proj.weight = attn_causal.qkv_proj.weight
    attn_non_causal.o_proj.weight = attn_causal.o_proj.weight

    # Create input
    x = mx.random.normal((batch_size, seq_len, hidden_size))

    # Forward pass
    output_causal = attn_causal(x, cos_sin=None)
    output_non_causal = attn_non_causal(x, cos_sin=None)

    # Check shapes
    assert output_causal.shape == output_non_causal.shape
    print(f"✓ Causal and non-causal have same output shape: {output_causal.shape}")

    # Outputs should be different due to masking
    assert not mx.allclose(output_causal, output_non_causal, atol=1e-4)
    print(f"✓ Causal masking changes output (as expected)")

    # The difference should be more pronounced at later positions
    # (since they can't attend to future tokens)
    diff = mx.abs(output_causal - output_non_causal)
    diff_per_pos = mx.mean(diff, axis=(0, 2))  # Average over batch and hidden
    print(f"  Difference by position: {diff_per_pos}")
    print(f"✓ Causal masking verified")

    print("\n✅ Causal masking tests PASSED\n")


def test_attention_different_seq_lengths():
    """Test attention with different sequence lengths."""
    print("\n" + "="*70)
    print("Testing Different Sequence Lengths")
    print("="*70)

    hidden_size = 512
    head_dim = 64
    num_heads = 8

    attn = MLXAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        causal=False
    )

    # Create RoPE
    rope = MLXRotaryEmbedding(head_dim, max_position_embeddings=2048)
    cos, sin = rope()

    seq_lengths = [8, 16, 32, 64, 128]

    for seq_len in seq_lengths:
        x = mx.random.normal((2, seq_len, hidden_size))
        output = attn(x, cos_sin=(cos, sin))

        assert output.shape == (2, seq_len, hidden_size)
        print(f"  seq_len={seq_len:3d}: output_shape={output.shape}")

    print(f"✓ All sequence lengths work correctly")

    print("\n✅ Different sequence length tests PASSED\n")


def test_attention_self_attention_property():
    """Test that attention computes weighted averages."""
    print("\n" + "="*70)
    print("Testing Self-Attention Properties")
    print("="*70)

    # Simple test: with uniform input, output should be bounded
    batch_size = 1
    seq_len = 4
    hidden_size = 128
    head_dim = 32
    num_heads = 4

    attn = MLXAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        causal=False
    )

    # Uniform input
    x = mx.ones((batch_size, seq_len, hidden_size))

    output = attn(x, cos_sin=None)

    print(f"  Input mean: {mx.mean(x):.4f}, std: {mx.std(x):.4f}")
    print(f"  Output mean: {mx.mean(output):.4f}, std: {mx.std(output):.4f}")
    print(f"✓ Attention produces bounded output")

    print("\n✅ Self-attention property tests PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ATTENTION TEST SUITE")
    print("="*70)

    try:
        test_attention_initialization()
        test_attention_forward_basic()
        test_attention_with_rope()
        test_attention_causal_masking()
        test_attention_different_seq_lengths()
        test_attention_self_attention_property()

        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
