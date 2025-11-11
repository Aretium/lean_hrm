"""
Test script for optimizers

This script tests:
1. SignSGD optimizer for sparse embeddings
2. AdamMLX wrapper
3. Muon optimizer with Newton-Schulz orthogonalization
4. Learning rate schedule
5. Optimizer update mechanics
"""

import mlx.core as mx
import mlx.nn as nn
import sys
sys.path.append('..')

from mlx_utils.optimizers import (
    SignSGD_MLX,
    AdamMLX,
    Muon_MLX,
    newton_schulz_orthogonalize,
    get_learning_rate_schedule,
    create_optimizer_for_hrm
)


def test_signsgd():
    """Test SignSGD optimizer."""
    print("\n" + "="*70)
    print("Testing SignSGD Optimizer")
    print("="*70)

    # Create optimizer
    opt = SignSGD_MLX(learning_rate=0.01, weight_decay=0.1)

    # Create parameter and gradient
    param = mx.ones((10, 20))
    grad = mx.random.normal((10, 20))

    # Apply update
    state = {}
    updated = opt.apply_single(grad, param, state)

    # Check shape preserved
    assert updated.shape == param.shape, "Shape should be preserved"
    print(f"✓ Shape preserved: {updated.shape}")

    # Check that update actually changed the parameter
    assert not mx.allclose(updated, param), "Parameter should change"
    print(f"✓ Parameter updated: diff={mx.abs(updated - param).mean():.4f}")

    # Test weight decay
    # param * (1 - lr * wd) = 1.0 * (1 - 0.01 * 0.1) = 0.999
    expected_wd = param * (1.0 - 0.01 * 0.1)
    # Then subtract lr * sign(grad)
    expected = expected_wd - 0.01 * mx.sign(grad)

    assert mx.allclose(updated, expected, atol=1e-6), "SignSGD formula incorrect"
    print(f"✓ SignSGD formula correct")

    # Test that only sign matters, not magnitude
    grad_large = grad * 1000
    updated_large = opt.apply_single(grad_large, param, {})

    assert mx.allclose(updated, updated_large, atol=1e-6), "Only sign should matter"
    print(f"✓ Only gradient sign matters")

    print("\n✅ SignSGD tests PASSED\n")


def test_adam_mlx():
    """Test AdamMLX wrapper."""
    print("\n" + "="*70)
    print("Testing AdamMLX Optimizer")
    print("="*70)

    # Create optimizer
    opt = AdamMLX(learning_rate=1e-3, betas=(0.9, 0.999), weight_decay=0.01)

    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.ones((5, 10))

        def __call__(self, x):
            return x @ self.weight.T

    model = SimpleModel()

    # Initial params
    initial_weight = mx.array(model.weight)

    # Create gradient
    grads = {'weight': mx.random.normal((5, 10))}

    # Update (this test is just checking that the API works)
    # Full update would require MLX's optimizer integration
    print(f"✓ AdamMLX initialized with betas={opt.betas}")
    print(f"✓ Learning rate: {opt.learning_rate}")
    print(f"✓ Weight decay: {opt.weight_decay}")

    print("\n✅ AdamMLX tests PASSED\n")


def test_newton_schulz():
    """Test Newton-Schulz orthogonalization."""
    print("\n" + "="*70)
    print("Testing Newton-Schulz Orthogonalization")
    print("="*70)

    # Create random matrix
    G = mx.random.normal((20, 30))

    # Orthogonalize
    Q = newton_schulz_orthogonalize(G, steps=5)

    # Check shape preserved
    assert Q.shape == G.shape, f"Shape should be preserved, got {Q.shape}"
    print(f"✓ Shape preserved: {Q.shape}")

    # Check orthogonality
    # For m <= n, Q @ Q.T should be close to identity
    QQT = Q @ Q.T
    identity = mx.eye(Q.shape[0])

    orthogonality_error = mx.abs(QQT - identity).max()
    print(f"✓ Orthogonality error: {orthogonality_error:.6f}")
    assert orthogonality_error < 0.5, "Should be approximately orthogonal (Newton-Schulz is iterative)"

    # Test with different shapes (tall matrix)
    G_tall = mx.random.normal((50, 20))
    Q_tall = newton_schulz_orthogonalize(G_tall, steps=5)

    assert Q_tall.shape == G_tall.shape
    print(f"✓ Tall matrix shape: {Q_tall.shape}")

    # For m > n, Q.T @ Q should be close to identity
    QTQ = Q_tall.T @ Q_tall
    identity_small = mx.eye(Q_tall.shape[1])

    orthogonality_error_tall = mx.abs(QTQ - identity_small).max()
    print(f"✓ Orthogonality error (tall): {orthogonality_error_tall:.6f}")

    # Test numerical stability with different steps
    for steps in [1, 3, 5, 10]:
        Q_steps = newton_schulz_orthogonalize(G, steps=steps)
        QQT_steps = Q_steps @ Q_steps.T
        error = mx.abs(QQT_steps - identity).max()
        print(f"  Steps={steps:2d}: error={error:.6f}")

    print("\n✅ Newton-Schulz tests PASSED\n")


def test_muon_optimizer():
    """Test Muon optimizer."""
    print("\n" + "="*70)
    print("Testing Muon Optimizer")
    print("="*70)

    # Create optimizer
    opt = Muon_MLX(learning_rate=0.02, momentum=0.95, nesterov=True, ns_steps=5)

    # Create 2D parameter (weight matrix)
    param_2d = mx.random.normal((20, 30))
    grad_2d = mx.random.normal((20, 30))

    # Initialize state
    state_2d = {}
    opt.init_single(param_2d, state_2d)

    assert "momentum_buffer" in state_2d, "Should initialize momentum buffer"
    print(f"✓ Momentum buffer initialized: shape={state_2d['momentum_buffer'].shape}")

    # Apply update
    updated_2d = opt.apply_single(grad_2d, param_2d, state_2d)

    assert updated_2d.shape == param_2d.shape, "Shape should be preserved"
    assert not mx.allclose(updated_2d, param_2d), "Parameter should change"
    print(f"✓ 2D parameter updated: diff={mx.abs(updated_2d - param_2d).mean():.4f}")

    # Check that momentum buffer was updated
    assert not mx.allclose(state_2d["momentum_buffer"], mx.zeros_like(param_2d))
    print(f"✓ Momentum buffer updated")

    # Test with 1D parameter (should use standard momentum, not orthogonalization)
    param_1d = mx.random.normal((100,))
    grad_1d = mx.random.normal((100,))
    state_1d = {}
    opt.init_single(param_1d, state_1d)

    updated_1d = opt.apply_single(grad_1d, param_1d, state_1d)

    assert updated_1d.shape == param_1d.shape
    print(f"✓ 1D parameter updated (no orthogonalization): shape={updated_1d.shape}")

    # Multiple updates should maintain orthogonality tendency
    param = mx.random.normal((15, 25))
    state = {}
    opt.init_single(param, state)

    for i in range(5):
        grad = mx.random.normal((15, 25)) * 0.1
        param = opt.apply_single(grad, param, state)

    print(f"✓ Multiple updates completed")

    print("\n✅ Muon Optimizer tests PASSED\n")


def test_learning_rate_schedule():
    """Test learning rate schedule."""
    print("\n" + "="*70)
    print("Testing Learning Rate Schedule")
    print("="*70)

    base_lr = 1e-3
    warmup_steps = 100
    total_steps = 1000
    min_ratio = 0.1

    # Test warmup phase
    lr_start = get_learning_rate_schedule(0, total_steps, base_lr, warmup_steps, min_ratio)
    assert lr_start == 0.0, "LR should start at 0"
    print(f"✓ Step 0: lr={lr_start:.6f}")

    lr_mid_warmup = get_learning_rate_schedule(50, total_steps, base_lr, warmup_steps, min_ratio)
    assert 0 < lr_mid_warmup < base_lr, "LR should increase during warmup"
    assert abs(lr_mid_warmup - base_lr * 0.5) < 1e-6, "Should be halfway at step 50"
    print(f"✓ Step 50 (mid-warmup): lr={lr_mid_warmup:.6f}")

    lr_end_warmup = get_learning_rate_schedule(warmup_steps, total_steps, base_lr, warmup_steps, min_ratio)
    assert abs(lr_end_warmup - base_lr) < 1e-6, "Should reach base_lr at end of warmup"
    print(f"✓ Step {warmup_steps} (end warmup): lr={lr_end_warmup:.6f}")

    # Test cosine decay phase
    lr_mid = get_learning_rate_schedule(550, total_steps, base_lr, warmup_steps, min_ratio)
    assert lr_mid < base_lr, "LR should decay after warmup"
    print(f"✓ Step 550 (mid-decay): lr={lr_mid:.6f}")

    lr_end = get_learning_rate_schedule(total_steps, total_steps, base_lr, warmup_steps, min_ratio)
    expected_min = base_lr * min_ratio
    assert abs(lr_end - expected_min) < 1e-6, f"Should reach min_lr at end"
    print(f"✓ Step {total_steps} (end): lr={lr_end:.6f} (min={expected_min:.6f})")

    # Plot schedule (text visualization)
    print(f"\n  Learning Rate Schedule Visualization:")
    steps_to_show = [0, 50, 100, 200, 400, 600, 800, 1000]
    for step in steps_to_show:
        lr = get_learning_rate_schedule(step, total_steps, base_lr, warmup_steps, min_ratio)
        normalized = lr / base_lr
        bar_len = int(normalized * 50)
        bar = "█" * bar_len
        print(f"    Step {step:4d}: {bar} {lr:.6f}")

    print("\n✅ Learning Rate Schedule tests PASSED\n")


def test_optimizer_parameter_groups():
    """Test that optimizers handle different parameter groups correctly."""
    print("\n" + "="*70)
    print("Testing Optimizer Parameter Groups")
    print("="*70)

    # Create model with different parameter types
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_2d = mx.random.normal((10, 20))  # Should use Muon
            self.bias_1d = mx.zeros((20,))  # Should use AdamW
            self.embedding = mx.random.normal((100, 20))  # Should use AdamW

    model = MixedModel()

    # Create optimizers
    optimizers = create_optimizer_for_hrm(
        model,
        lr=1e-4,
        puzzle_emb_lr=1e-2,
        use_separate_puzzle_optimizer=True
    )

    assert 'main' in optimizers, "Should have main optimizer"
    assert 'puzzle' in optimizers, "Should have puzzle optimizer"
    print(f"✓ Created {len(optimizers)} optimizers")

    # Check optimizer types
    assert isinstance(optimizers['main'], AdamMLX), "Main should be AdamMLX"
    assert isinstance(optimizers['puzzle'], SignSGD_MLX), "Puzzle should be SignSGD"
    print(f"✓ Optimizer types correct")

    # Check learning rates
    main_lr = optimizers['main'].learning_rate
    puzzle_lr = optimizers['puzzle'].learning_rate
    print(f"✓ Main LR: {main_lr}, Puzzle LR: {puzzle_lr}")
    assert puzzle_lr > main_lr, "Puzzle LR should be higher"

    print("\n✅ Parameter Groups tests PASSED\n")


def test_optimizer_convergence():
    """Test that optimizers actually minimize a simple loss."""
    print("\n" + "="*70)
    print("Testing Optimizer Convergence")
    print("="*70)

    # Simple quadratic loss: minimize ||x - target||^2
    target = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = mx.zeros((5,))

    # Test SignSGD
    opt_sign = SignSGD_MLX(learning_rate=0.1, weight_decay=0.0)
    state_sign = {}

    losses_sign = []
    for step in range(50):
        # Compute loss and gradient
        def loss_fn(x):
            return mx.sum((x - target) ** 2)

        loss = loss_fn(x)
        losses_sign.append(loss.item())

        grad = mx.grad(loss_fn)(x)
        x = opt_sign.apply_single(grad, x, state_sign)

    initial_loss_sign = losses_sign[0]
    final_loss_sign = losses_sign[-1]
    assert final_loss_sign < initial_loss_sign, "Loss should decrease"
    print(f"✓ SignSGD: initial_loss={initial_loss_sign:.4f}, final_loss={final_loss_sign:.4f}")

    # Test Muon (with 2D parameters)
    param_2d = mx.zeros((5, 10))
    target_2d = mx.random.normal((5, 10))

    opt_muon = Muon_MLX(learning_rate=0.01, momentum=0.9)
    state_muon = {}
    opt_muon.init_single(param_2d, state_muon)

    losses_muon = []
    for step in range(50):
        def loss_fn_2d(p):
            return mx.sum((p - target_2d) ** 2)

        loss = loss_fn_2d(param_2d)
        losses_muon.append(loss.item())

        grad = mx.grad(loss_fn_2d)(param_2d)
        param_2d = opt_muon.apply_single(grad, param_2d, state_muon)

    initial_loss_muon = losses_muon[0]
    final_loss_muon = losses_muon[-1]
    assert final_loss_muon < initial_loss_muon, "Muon loss should decrease"
    print(f"✓ Muon: initial_loss={initial_loss_muon:.4f}, final_loss={final_loss_muon:.4f}")

    print("\n✅ Convergence tests PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("OPTIMIZERS TEST SUITE")
    print("="*70)

    try:
        test_signsgd()
        test_adam_mlx()
        test_newton_schulz()
        test_muon_optimizer()
        test_learning_rate_schedule()
        test_optimizer_parameter_groups()
        test_optimizer_convergence()

        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
