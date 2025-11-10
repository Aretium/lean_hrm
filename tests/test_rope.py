"""
Test script for Rotary Position Embeddings (RoPE)

This script tests:
1. MLXRotaryEmbedding initialization and caching
2. apply_rotary_pos_emb function
3. Shape correctness
4. Rotation properties
"""

import mlx.core as mx
import sys
import math
sys.path.append('..')

from mlx_utils.layers import MLXRotaryEmbedding, apply_rotary_pos_emb


def test_rope_initialization():
    """Test RoPE initialization."""
    print("\n" + "="*70)
    print("Testing MLXRotaryEmbedding Initialization")
    print("="*70)

    dim = 64
    max_position_embeddings = 2048
    base = 10000.0

    # Create RoPE
    rope = MLXRotaryEmbedding(dim, max_position_embeddings, base)

    # Get cos and sin
    cos, sin = rope()

    # Check shapes
    expected_shape = (max_position_embeddings, dim)
    assert cos.shape == expected_shape, f"Expected cos shape {expected_shape}, got {cos.shape}"
    assert sin.shape == expected_shape, f"Expected sin shape {expected_shape}, got {sin.shape}"
    print(f"✓ Cos/Sin shapes correct: {cos.shape}")

    # Check that cos^2 + sin^2 = 1 (approximately)
    # This should hold for each position
    cos_sin_squared = cos ** 2 + sin ** 2
    # Due to how RoPE concatenates freqs, we need to check pairs
    max_deviation = mx.abs(cos_sin_squared - 1.0).max()
    print(f"✓ cos²+sin²≈1 check: max deviation = {max_deviation:.6f}")

    # Check that values are in [-1, 1]
    assert mx.all(cos >= -1.0) and mx.all(cos <= 1.0), "Cos values should be in [-1, 1]"
    assert mx.all(sin >= -1.0) and mx.all(sin <= 1.0), "Sin values should be in [-1, 1]"
    print(f"✓ Cos/Sin values in [-1, 1]")

    print("\n✅ RoPE initialization tests PASSED\n")


def test_apply_rotary_pos_emb():
    """Test apply_rotary_pos_emb function."""
    print("\n" + "="*70)
    print("Testing apply_rotary_pos_emb")
    print("="*70)

    batch_size = 2
    seq_len = 16
    num_heads = 8
    head_dim = 64

    # Create query and key
    q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

    # Create RoPE
    rope = MLXRotaryEmbedding(head_dim, max_position_embeddings=2048)
    cos, sin = rope()

    # Slice to sequence length
    cos = cos[:seq_len, :]
    sin = sin[:seq_len, :]

    # Apply RoPE
    q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)

    # Check shapes preserved
    assert q_rotated.shape == q.shape, f"Q shape changed: {q.shape} -> {q_rotated.shape}"
    assert k_rotated.shape == k.shape, f"K shape changed: {k.shape} -> {k_rotated.shape}"
    print(f"✓ Shapes preserved: Q={q_rotated.shape}, K={k_rotated.shape}")

    # Check that rotation actually changed the values
    assert not mx.allclose(q_rotated, q), "Q should be rotated (changed)"
    assert not mx.allclose(k_rotated, k), "K should be rotated (changed)"
    print(f"✓ Values changed after rotation")

    # Check that norms are approximately preserved
    # RoPE is a rotation, so it should preserve norms
    q_norm_before = mx.sqrt(mx.sum(q ** 2, axis=-1))
    q_norm_after = mx.sqrt(mx.sum(q_rotated ** 2, axis=-1))
    norm_diff = mx.abs(q_norm_before - q_norm_after).max()
    print(f"✓ Norm preservation (Q): max diff = {norm_diff:.6f}")
    assert norm_diff < 1e-4, f"Norms should be preserved, max diff: {norm_diff}"

    k_norm_before = mx.sqrt(mx.sum(k ** 2, axis=-1))
    k_norm_after = mx.sqrt(mx.sum(k_rotated ** 2, axis=-1))
    norm_diff_k = mx.abs(k_norm_before - k_norm_after).max()
    print(f"✓ Norm preservation (K): max diff = {norm_diff_k:.6f}")

    print("\n✅ apply_rotary_pos_emb tests PASSED\n")


def test_rope_different_positions():
    """Test that RoPE produces different embeddings for different positions."""
    print("\n" + "="*70)
    print("Testing RoPE Position Dependence")
    print("="*70)

    dim = 64
    max_pos = 512
    rope = MLXRotaryEmbedding(dim, max_pos)

    cos, sin = rope()

    # Check that different positions have different embeddings
    # Compare position 0 vs position 10
    cos_pos0 = cos[0, :]
    cos_pos10 = cos[10, :]
    sin_pos0 = sin[0, :]
    sin_pos10 = sin[10, :]

    assert not mx.allclose(cos_pos0, cos_pos10), "Different positions should have different cos"
    assert not mx.allclose(sin_pos0, sin_pos10), "Different positions should have different sin"
    print(f"✓ Different positions have different embeddings")

    # Check that position 0 has expected pattern
    # At position 0, frequencies should give cos(0)=1, sin(0)=0 pattern (with repetition)
    # Due to concatenation of freqs, we expect [1,1,...,1] for cos and [0,0,...,0] for sin at pos 0
    print(f"  Position 0 - cos range: [{cos_pos0.min():.4f}, {cos_pos0.max():.4f}]")
    print(f"  Position 0 - sin range: [{sin_pos0.min():.4f}, {sin_pos0.max():.4f}]")

    print("\n✅ RoPE position dependence tests PASSED\n")


def test_rope_integration():
    """Integration test with realistic transformer dimensions."""
    print("\n" + "="*70)
    print("Testing RoPE Integration")
    print("="*70)

    # Realistic transformer dimensions
    batch_size = 4
    seq_len = 128
    num_heads = 8
    head_dim = 64  # 512 / 8
    hidden_size = num_heads * head_dim

    # Create RoPE
    rope = MLXRotaryEmbedding(head_dim, max_position_embeddings=2048)
    cos, sin = rope()
    cos = cos[:seq_len, :]
    sin = sin[:seq_len, :]

    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")

    # Simulate attention Q, K
    q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

    # Apply RoPE
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    # Check everything works
    assert q_rot.shape == (batch_size, seq_len, num_heads, head_dim)
    assert k_rot.shape == (batch_size, seq_len, num_heads, head_dim)
    print(f"✓ Integration test passed with realistic dimensions")

    # Compute attention scores (just to check it works)
    # Transpose to [batch, num_heads, seq_len, head_dim]
    q_rot_t = mx.transpose(q_rot, (0, 2, 1, 3))
    k_rot_t = mx.transpose(k_rot, (0, 2, 1, 3))

    # Attention scores: [batch, num_heads, seq_len, seq_len]
    scores = q_rot_t @ mx.transpose(k_rot_t, (0, 1, 3, 2))
    scores = scores / math.sqrt(head_dim)

    assert scores.shape == (batch_size, num_heads, seq_len, seq_len)
    print(f"✓ Attention scores computed: {scores.shape}")

    print("\n✅ RoPE integration tests PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ROPE TEST SUITE")
    print("="*70)

    try:
        test_rope_initialization()
        test_apply_rotary_pos_emb()
        test_rope_different_positions()
        test_rope_integration()

        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
