"""
Test script for loss functions

This script tests:
1. StableMax transformation and log_stablemax
2. Stablemax cross-entropy loss
3. Softmax cross-entropy loss
4. Accuracy computation helper
5. Loss masking with ignore_index
"""

import mlx.core as mx
import sys
sys.path.append('..')

from mlx_utils.losses import (
    stablemax,
    log_stablemax,
    stablemax_cross_entropy,
    softmax_cross_entropy,
    compute_accuracy,
    IGNORE_LABEL_ID
)


def test_stablemax():
    """Test stablemax transformation."""
    print("\n" + "="*70)
    print("Testing StableMax Transformation")
    print("="*70)

    # Test positive values: s(x) = x + 1
    x_pos = mx.array([0.0, 1.0, 2.0, 5.0])
    s_pos = stablemax(x_pos)
    expected_pos = mx.array([1.0, 2.0, 3.0, 6.0])

    assert mx.allclose(s_pos, expected_pos, atol=1e-6), "Positive values incorrect"
    print(f"✓ Positive values: s([0,1,2,5]) = {s_pos}")

    # Test negative values: s(x) = 1/(1-x+eps)
    x_neg = mx.array([-1.0, -2.0, -5.0])
    s_neg = stablemax(x_neg)
    # s(-1) = 1/(1-(-1)) = 1/2 = 0.5
    # s(-2) = 1/(1-(-2)) = 1/3 ≈ 0.333
    # s(-5) = 1/(1-(-5)) = 1/6 ≈ 0.167
    print(f"✓ Negative values: s([-1,-2,-5]) = {s_neg}")
    assert mx.all(s_neg > 0), "Negative values should be positive"
    assert mx.all(s_neg < 1), "Negative transform should be < 1"

    # Test zero
    x_zero = mx.array([0.0])
    s_zero = stablemax(x_zero)
    assert mx.abs(s_zero[0] - 1.0) < 1e-6, "s(0) should be 1.0"
    print(f"✓ Zero: s(0) = {s_zero[0]:.4f}")

    # Test numerical stability with extreme values
    x_extreme = mx.array([-100.0, 100.0])
    s_extreme = stablemax(x_extreme)
    assert mx.all(mx.isfinite(s_extreme)), "Should handle extreme values"
    print(f"✓ Extreme values handled: s([-100,100]) = {s_extreme}")

    print("\n✅ StableMax tests PASSED\n")


def test_log_stablemax():
    """Test log_stablemax function."""
    print("\n" + "="*70)
    print("Testing Log StableMax")
    print("="*70)

    # Test that probabilities sum to 1 after normalization
    x = mx.random.normal((2, 5, 10))  # [batch, seq, vocab]
    logprobs = log_stablemax(x, axis=-1)

    # exp(logprobs) should sum to 1
    probs = mx.exp(logprobs)
    prob_sums = mx.sum(probs, axis=-1)

    assert mx.allclose(prob_sums, mx.ones_like(prob_sums), atol=1e-5), "Probabilities should sum to 1"
    print(f"✓ Probabilities sum to 1: mean={prob_sums.mean():.6f}, std={prob_sums.std():.6f}")

    # Test that all log probabilities are <= 0 (since probs <= 1)
    assert mx.all(logprobs <= 0.0), "Log probabilities should be <= 0"
    print(f"✓ Log probabilities <= 0: max={logprobs.max():.4f}")

    # Test different axes
    x = mx.random.normal((3, 4, 5))
    logprobs_axis0 = log_stablemax(x, axis=0)
    logprobs_axis1 = log_stablemax(x, axis=1)
    logprobs_axis2 = log_stablemax(x, axis=2)

    assert logprobs_axis0.shape == (3, 4, 5)
    assert logprobs_axis1.shape == (3, 4, 5)
    assert logprobs_axis2.shape == (3, 4, 5)
    print(f"✓ Different axes work correctly")

    print("\n✅ Log StableMax tests PASSED\n")


def test_stablemax_cross_entropy():
    """Test stablemax cross-entropy loss."""
    print("\n" + "="*70)
    print("Testing StableMax Cross-Entropy")
    print("="*70)

    batch_size = 2
    seq_len = 4
    vocab_size = 10

    # Create logits and labels
    logits = mx.random.normal((batch_size, seq_len, vocab_size))
    labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))

    # Compute loss
    loss = stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)

    # Check shape
    assert loss.shape == (batch_size, seq_len), f"Expected shape {(batch_size, seq_len)}, got {loss.shape}"
    print(f"✓ Shape correct: {loss.shape}")

    # Check that loss is positive (cross-entropy is always >= 0)
    assert mx.all(loss >= 0), "Loss should be non-negative"
    print(f"✓ Loss is non-negative: min={loss.min():.4f}, max={loss.max():.4f}")

    # Test ignore_index
    labels_with_ignore = mx.array(labels)  # Create new array
    labels_list = labels_with_ignore.tolist()
    labels_list[0][0] = IGNORE_LABEL_ID
    labels_list[1][2] = IGNORE_LABEL_ID
    labels_with_ignore = mx.array(labels_list, dtype=mx.int32)

    loss_with_ignore = stablemax_cross_entropy(logits, labels_with_ignore, ignore_index=IGNORE_LABEL_ID)

    # Check that ignored positions have zero loss
    assert loss_with_ignore[0, 0] == 0.0, "Ignored position should have zero loss"
    assert loss_with_ignore[1, 2] == 0.0, "Ignored position should have zero loss"
    print(f"✓ Ignore index works correctly")

    # Test perfect predictions (logits very high for correct class)
    perfect_logits = mx.zeros((2, 3, 5))
    perfect_labels = mx.array([[0, 1, 2], [3, 4, 0]], dtype=mx.int32)

    # Set logits very high for correct labels
    for b in range(2):
        for s in range(3):
            perfect_logits[b, s, perfect_labels[b, s]] = 10.0

    perfect_loss = stablemax_cross_entropy(perfect_logits, perfect_labels)
    print(f"✓ Near-perfect predictions: mean loss={perfect_loss.mean():.4f}")

    print("\n✅ StableMax Cross-Entropy tests PASSED\n")


def test_softmax_cross_entropy():
    """Test standard softmax cross-entropy loss."""
    print("\n" + "="*70)
    print("Testing Softmax Cross-Entropy")
    print("="*70)

    batch_size = 2
    seq_len = 4
    vocab_size = 10

    # Create logits and labels
    logits = mx.random.normal((batch_size, seq_len, vocab_size))
    labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))

    # Compute loss
    loss = softmax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)

    # Check shape
    assert loss.shape == (batch_size, seq_len), f"Expected shape {(batch_size, seq_len)}, got {loss.shape}"
    print(f"✓ Shape correct: {loss.shape}")

    # Check that loss is positive
    assert mx.all(loss >= 0), "Loss should be non-negative"
    print(f"✓ Loss is non-negative: min={loss.min():.4f}, max={loss.max():.4f}")

    # Test ignore_index
    labels_list = labels.tolist()
    labels_list[0][1] = IGNORE_LABEL_ID
    labels_with_ignore = mx.array(labels_list, dtype=mx.int32)

    loss_with_ignore = softmax_cross_entropy(logits, labels_with_ignore, ignore_index=IGNORE_LABEL_ID)
    assert loss_with_ignore[0, 1] == 0.0, "Ignored position should have zero loss"
    print(f"✓ Ignore index works correctly")

    # Compare with stablemax version (should be similar but not identical)
    stablemax_loss = stablemax_cross_entropy(logits, labels)
    softmax_loss = softmax_cross_entropy(logits, labels)

    diff = mx.abs(stablemax_loss - softmax_loss).mean()
    print(f"✓ Difference from stablemax: mean={diff:.4f}")
    print(f"  (They should be similar but not identical)")

    print("\n✅ Softmax Cross-Entropy tests PASSED\n")


def test_compute_accuracy():
    """Test accuracy computation helper."""
    print("\n" + "="*70)
    print("Testing Accuracy Computation")
    print("="*70)

    batch_size = 3
    seq_len = 5
    vocab_size = 10

    # Create perfect predictions
    logits = mx.zeros((batch_size, seq_len, vocab_size))
    labels = mx.array([
        [1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]
    ], dtype=mx.int32)

    # Set logits high for correct labels
    for b in range(batch_size):
        for s in range(seq_len):
            logits[b, s, labels[b, s]] = 10.0

    token_acc, seq_acc = compute_accuracy(logits, labels)

    # Should be 100% accurate
    assert mx.allclose(token_acc, mx.ones_like(token_acc)), "Perfect predictions should have 100% token accuracy"
    assert mx.allclose(seq_acc, mx.ones_like(seq_acc)), "Perfect predictions should have 100% sequence accuracy"
    print(f"✓ Perfect predictions: token_acc={token_acc.mean():.2%}, seq_acc={seq_acc.mean():.2%}")

    # Create partially correct predictions
    partial_logits = mx.zeros((batch_size, seq_len, vocab_size))
    partial_logits[0, 0, labels[0, 0]] = 10.0  # Correct
    partial_logits[0, 1, labels[0, 1] + 1] = 10.0  # Wrong (shifted by 1)
    partial_logits[0, 2, labels[0, 2]] = 10.0  # Correct
    partial_logits[0, 3, labels[0, 3]] = 10.0  # Correct
    partial_logits[0, 4, labels[0, 4]] = 10.0  # Correct

    # Fill rest with correct predictions
    for b in range(1, batch_size):
        for s in range(seq_len):
            partial_logits[b, s, labels[b, s]] = 10.0

    partial_token_acc, partial_seq_acc = compute_accuracy(partial_logits, labels)

    # First sequence: 4/5 = 80% token accuracy, but 0% sequence accuracy (not all correct)
    assert partial_token_acc[0] == 0.8, f"Expected 0.8, got {partial_token_acc[0]}"
    assert partial_seq_acc[0] == 0.0, "Sequence not fully correct"

    # Other sequences: 100% accurate
    assert mx.allclose(partial_token_acc[1:], mx.ones(2))
    assert mx.allclose(partial_seq_acc[1:], mx.ones(2))
    print(f"✓ Partial accuracy: first seq token_acc={partial_token_acc[0]:.2%}, seq_acc={partial_seq_acc[0]:.2%}")

    # Test with ignore_index
    labels_list = labels.tolist()
    labels_list[0][1] = IGNORE_LABEL_ID  # Ignore wrong prediction
    labels_list[0][2] = IGNORE_LABEL_ID  # Ignore correct prediction
    labels_with_ignore = mx.array(labels_list, dtype=mx.int32)

    ignore_token_acc, ignore_seq_acc = compute_accuracy(partial_logits, labels_with_ignore)

    # Now first sequence has 3/3 valid tokens correct = 100%
    assert ignore_token_acc[0] == 1.0, "After ignoring, should be 100%"
    assert ignore_seq_acc[0] == 1.0, "Sequence should be fully correct"
    print(f"✓ Ignore index: token_acc={ignore_token_acc[0]:.2%}, seq_acc={ignore_seq_acc[0]:.2%}")

    print("\n✅ Accuracy Computation tests PASSED\n")


def test_loss_gradient_flow():
    """Test that gradients flow through loss functions."""
    print("\n" + "="*70)
    print("Testing Gradient Flow")
    print("="*70)

    batch_size = 2
    seq_len = 3
    vocab_size = 5

    logits = mx.random.normal((batch_size, seq_len, vocab_size))
    labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))

    # Test stablemax CE gradient
    def loss_fn_stablemax(logits):
        loss = stablemax_cross_entropy(logits, labels)
        return loss.sum()

    grad_fn = mx.grad(loss_fn_stablemax)
    grads = grad_fn(logits)

    assert grads.shape == logits.shape, "Gradient shape should match logits"
    assert mx.abs(grads).max() > 0, "Gradients should be non-zero"
    print(f"✓ StableMax CE gradients flow: mean={mx.mean(mx.abs(grads)):.4f}")

    # Test softmax CE gradient
    def loss_fn_softmax(logits):
        loss = softmax_cross_entropy(logits, labels)
        return loss.sum()

    grad_fn_softmax = mx.grad(loss_fn_softmax)
    grads_softmax = grad_fn_softmax(logits)

    assert grads_softmax.shape == logits.shape
    assert mx.abs(grads_softmax).max() > 0
    print(f"✓ Softmax CE gradients flow: mean={mx.mean(mx.abs(grads_softmax)):.4f}")

    print("\n✅ Gradient Flow tests PASSED\n")


def test_loss_numerical_stability():
    """Test numerical stability with extreme values."""
    print("\n" + "="*70)
    print("Testing Numerical Stability")
    print("="*70)

    batch_size = 2
    seq_len = 3
    vocab_size = 5

    # Test with very large logits
    large_logits = mx.random.normal((batch_size, seq_len, vocab_size)) * 100
    labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))

    loss_large = stablemax_cross_entropy(large_logits, labels)
    assert mx.all(mx.isfinite(loss_large)), "Should handle large logits"
    print(f"✓ Large logits: loss range=[{loss_large.min():.2f}, {loss_large.max():.2f}]")

    # Test with very small logits
    small_logits = mx.random.normal((batch_size, seq_len, vocab_size)) * 0.001
    loss_small = stablemax_cross_entropy(small_logits, labels)
    assert mx.all(mx.isfinite(loss_small)), "Should handle small logits"
    print(f"✓ Small logits: loss range=[{loss_small.min():.2f}, {loss_small.max():.2f}]")

    # Test with all ignored labels
    all_ignored = mx.full((batch_size, seq_len), IGNORE_LABEL_ID, dtype=mx.int32)
    loss_ignored = stablemax_cross_entropy(large_logits, all_ignored)
    assert mx.all(loss_ignored == 0.0), "All ignored should give zero loss"
    print(f"✓ All ignored labels: loss sum={loss_ignored.sum():.4f}")

    print("\n✅ Numerical Stability tests PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LOSS FUNCTIONS TEST SUITE")
    print("="*70)

    try:
        test_stablemax()
        test_log_stablemax()
        test_stablemax_cross_entropy()
        test_softmax_cross_entropy()
        test_compute_accuracy()
        test_loss_gradient_flow()
        test_loss_numerical_stability()

        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
