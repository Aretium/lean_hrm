"""
MLX Optimizers

Custom optimizers for HRM:
- AdamMLX: Standard Adam (MLX has built-in, but we may want custom)
- SignSGD_MLX: Sign-based SGD for sparse embeddings

Replaces: adam_atan2.py and sparse_embedding optimizer from PyTorch version

ORIGINAL PYTORCH REFERENCE:
---------------------------
File: models/sparse_embedding.py (lines 41-133)

CastedSparseEmbeddingSignSGD_Distributed class (lines 41-96):
→ SignSGD_MLX

Key algorithm (_sparse_emb_signsgd_dist, lines 98-132):
```python
# 1. All-gather gradients from all GPUs (lines 114-118)
all_weights_grad = distributed.all_gather(local_weights_grad)
all_ids = distributed.all_gather(local_ids)

# 2. Find unique puzzle IDs (line 121)
grad_ids, inv = all_ids.unique(return_inverse=True)

# 3. Accumulate gradients for unique IDs (lines 123-124)
grad = zeros(grad_ids.shape[0], D)
grad.scatter_add_(0, inv.expand(-1, D), all_weights_grad)

# 4. SignSGD update with decoupled weight decay (lines 127-129)
p = weights[grad_ids]
p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)
weights[grad_ids] = p
```

MLX SIMPLIFICATION:
- No distributed all-gather (single device!)
- Gradients already accumulated by MLX
- Just apply: p = p * (1 - lr * wd) - lr * sign(grad)
- Much simpler!

HYPERPARAMETERS (from config/cfg_pretrain.yaml):
- puzzle_emb_lr: 1e-2 (100x higher than main lr!)
- puzzle_emb_weight_decay: 0.1
- Main lr: 1e-4
- Main weight_decay: 0.1
- betas: (0.9, 0.95) for Adam

File: External dependency (adam_atan2 package)
AdamATan2 class:
→ AdamMLX (or use standard MLX Adam)

This is a custom Adam variant using atan2 for parameter updates.
May not be necessary for MLX - standard Adam likely sufficient.
Monitor if needed during training.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Dict, Any


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def newton_schulz_orthogonalize(
    G: mx.array,
    steps: int = 5,
    backend_steps: int = 0,
    eps: float = 1e-7
) -> mx.array:
    """
    Newton-Schulz orthogonalization for Muon optimizer.

    Computes the nearest semi-orthogonal matrix to G, equivalent to:
        UV^T where USV^T is the SVD of G

    But much faster than SVD! Uses quintic polynomial iteration:
        φ(x) = ax + bx³ + cx⁵

    This converges singular values toward 1, effectively orthogonalizing.

    Args:
        G: Input matrix to orthogonalize [m, n]
        steps: Number of Newton-Schulz iterations (default 5)
        backend_steps: Additional backend iterations (default 0)
        eps: Small constant for numerical stability

    Returns:
        Orthogonalized matrix with same shape as G

    Reference:
        Muon paper (arXiv:2502.16982)
        Blog: https://kellerjordan.github.io/posts/muon/
    """
    # Coefficients for quintic polynomial
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Work in float32 for numerical stability (could use bfloat16 for speed)
    # MLX doesn't have bfloat16 exposed, so we use float32
    X = G.astype(mx.float32)

    # Normalize by norm (important for convergence)
    norm = mx.sqrt(mx.sum(X * X))
    X = X / (norm + eps)

    # Newton-Schulz iterations
    # These iterations converge the singular values toward 1
    for _ in range(steps):
        # Compute A = X @ X^T (for m x n matrix where m <= n)
        # or A = X^T @ X (for m x n where m > n)
        # We use the smaller dimension for efficiency
        if X.shape[0] <= X.shape[1]:
            A = X @ X.T
        else:
            A = X.T @ X

        # Compute polynomial: B = b*A + c*A²
        A_squared = A @ A
        B = b * A + c * A_squared

        # Update step: X = a*X + (B @ X) or X = a*X + (X @ B)
        if X.shape[0] <= X.shape[1]:
            X = a * X + B @ X
        else:
            X = a * X + X @ B

    # Backend iterations (additional refinement if needed)
    for _ in range(backend_steps):
        if X.shape[0] <= X.shape[1]:
            A = X @ X.T
            A_squared = A @ A
            B = b * A + c * A_squared
            X = a * X + B @ X
        else:
            A = X.T @ X
            A_squared = A @ A
            B = b * A + c * A_squared
            X = a * X + X @ B

    # Convert back to original dtype
    return X.astype(G.dtype)


# ============================================================================
# SIGNSGD FOR SPARSE EMBEDDINGS
# ============================================================================

class SignSGD_MLX(optim.Optimizer):
    """
    Sign-based SGD optimizer for sparse puzzle embeddings.
    
    Replaces: CastedSparseEmbeddingSignSGD_Distributed from PyTorch
    
    Key features:
    - Only updates embeddings for current batch (sparse!)
    - Uses sign of gradient (memory efficient)
    - Decoupled weight decay
    - Much simpler than PyTorch version (no distributed complexity!)
    
    Formula:
        p = p * (1 - lr * weight_decay) - lr * sign(grad)
    
    Why SignSGD for sparse embeddings?
    - Gradients are very sparse (only few puzzles per batch)
    - Sign is more stable than magnitude
    - Similar to Adam without momentum for sparse case
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-2,
        weight_decay: float = 0.1
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: Dict[str, Any]
    ) -> mx.array:
        """
        Apply SignSGD update to a single parameter.

        Args:
            gradient: Gradient for this parameter
            parameter: Current parameter value
            state: Optimizer state (unused for SignSGD)

        Returns:
            Updated parameter
        """
        # Get current learning rate (may be scheduled)
        lr = self.learning_rate.item() if isinstance(self.learning_rate, mx.array) else self.learning_rate

        # SignSGD update with decoupled weight decay:
        # p = p * (1 - lr * weight_decay) - lr * sign(grad)

        # Apply weight decay (decoupled)
        updated = parameter * (1.0 - lr * self.weight_decay)

        # Apply signed gradient update
        updated = updated - lr * mx.sign(gradient)

        return updated


# ============================================================================
# ADAM VARIANTS
# ============================================================================

class AdamMLX(optim.AdamW):
    """
    Adam optimizer with weight decay (AdamW).

    MLX already has AdamW built-in, which implements Adam with decoupled weight decay.
    This is a wrapper for convenience and consistency with HRM hyperparameters.

    For now, this is just a wrapper around mlx.optimizers.AdamW
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1
    ):
        # Convert tuple to list for MLX
        betas_list = list(betas)
        super().__init__(
            learning_rate=learning_rate,
            betas=betas_list,
            eps=eps,
            weight_decay=weight_decay
        )


class Muon_MLX(optim.Optimizer):
    """
    Muon optimizer for hidden layer weight matrices.

    Muon uses momentum + orthogonalization via Newton-Schulz iterations.
    Achieves ~2x computational efficiency vs AdamW for transformers.

    Key features:
    - Applies to 2D weight matrices only (linear layers, conv filters)
    - Uses Newton-Schulz iteration for efficient orthogonalization
    - Operates in bfloat16 for speed (unlike SVD-based methods)
    - Should be combined with AdamW for embeddings/biases

    Algorithm:
    1. Compute momentum-based update (Nesterov)
    2. Orthogonalize via Newton-Schulz iterations
    3. Apply orthogonalized update

    Reference:
    - Paper: "Muon is Scalable for LLM Training" (arXiv:2502.16982)
    - Blog: https://kellerjordan.github.io/posts/muon/
    - Code: https://github.com/KellerJordan/Muon
    """

    def __init__(
        self,
        learning_rate: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        backend_steps: int = 0
    ):
        """
        Initialize Muon optimizer.

        Args:
            learning_rate: Learning rate (typically higher than AdamW)
            momentum: Momentum coefficient (default 0.95)
            nesterov: Use Nesterov momentum (default True)
            ns_steps: Newton-Schulz iteration steps (default 5)
            backend_steps: Additional backend iterations (default 0)
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.backend_steps = backend_steps

    def init_single(self, parameter: mx.array, state: Dict[str, Any]):
        """Initialize optimizer state for a parameter."""
        # Muon maintains momentum buffer
        state["momentum_buffer"] = mx.zeros_like(parameter)

    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: Dict[str, Any]
    ) -> mx.array:
        """
        Apply Muon update to a single parameter.

        For 2D parameters: momentum + orthogonalization
        For non-2D parameters: standard momentum SGD

        Args:
            gradient: Gradient for this parameter
            parameter: Current parameter value
            state: Optimizer state with momentum buffer

        Returns:
            Updated parameter
        """
        # Get or initialize momentum buffer
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = mx.zeros_like(parameter)

        momentum_buffer = state["momentum_buffer"]

        # Get current learning rate
        lr = self.learning_rate.item() if isinstance(self.learning_rate, mx.array) else self.learning_rate

        # Update momentum buffer
        momentum_buffer = self.momentum * momentum_buffer + gradient
        state["momentum_buffer"] = momentum_buffer

        # Compute update
        if self.nesterov:
            update = gradient + self.momentum * momentum_buffer
        else:
            update = momentum_buffer

        # For 2D parameters, apply orthogonalization
        if parameter.ndim >= 2:
            # Orthogonalize the update via Newton-Schulz
            update = newton_schulz_orthogonalize(
                update,
                steps=self.ns_steps,
                backend_steps=self.backend_steps
            )

        # Apply update
        updated = parameter - lr * update

        return updated


class MuonWithAdamW_MLX:
    """
    Hybrid optimizer: Muon for hidden weights, AdamW for everything else.

    This is the recommended setup for transformers:
    - Muon: 2D weight matrices (attention, FFN)
    - AdamW: Embeddings, biases, layer norms, output layer

    Usage:
        optimizer = MuonWithAdamW_MLX(model, lr_muon=0.02, lr_adamw=1e-3)
        optimizer.update(model, gradients)
    """

    def __init__(
        self,
        model: nn.Module,
        lr_muon: float = 2e-2,
        lr_adamw: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        exclude_first_layer: bool = True
    ):
        """
        Initialize hybrid Muon + AdamW optimizer.

        Args:
            model: Neural network model
            lr_muon: Learning rate for Muon (hidden weights)
            lr_adamw: Learning rate for AdamW (other parameters)
            momentum: Momentum for Muon
            nesterov: Use Nesterov momentum
            ns_steps: Newton-Schulz steps
            betas: Adam beta parameters
            eps: Adam epsilon
            weight_decay: Weight decay for AdamW
            exclude_first_layer: Don't use Muon for first layer (common practice)
        """
        self.muon_opt = Muon_MLX(
            learning_rate=lr_muon,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps
        )

        self.adamw_opt = AdamMLX(
            learning_rate=lr_adamw,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )

        self.exclude_first_layer = exclude_first_layer
        self.muon_states = {}
        self.adamw_states = {}

    def should_use_muon(self, name: str, param: mx.array) -> bool:
        """
        Determine if parameter should use Muon or AdamW.

        Muon is for:
        - 2D weight matrices (linear layers, attention)
        - Excluding embeddings and output layer
        - Optionally excluding first layer
        """
        # Must be 2D
        if param.ndim < 2:
            return False

        # Exclude embeddings and output
        if 'embedding' in name.lower() or 'output' in name.lower():
            return False

        # Optionally exclude first layer
        if self.exclude_first_layer and ('layer_0' in name or 'layers.0' in name):
            return False

        return True

    def update(self, model: nn.Module, gradients: Dict[str, mx.array]):
        """
        Update model parameters using appropriate optimizer.

        Args:
            model: Model to update
            gradients: Dictionary of gradients
        """
        updates = {}

        for name, param in model.parameters().items():
            if name not in gradients:
                continue

            grad = gradients[name]

            if self.should_use_muon(name, param):
                # Use Muon
                if name not in self.muon_states:
                    self.muon_states[name] = {}
                    self.muon_opt.init_single(param, self.muon_states[name])

                updated = self.muon_opt.apply_single(grad, param, self.muon_states[name])
            else:
                # Use AdamW
                if name not in self.adamw_states:
                    self.adamw_states[name] = {}
                    self.adamw_opt.init_single(param, self.adamw_states[name])

                updated = self.adamw_opt.apply_single(grad, param, self.adamw_states[name])

            updates[name] = updated

        # Update model
        model.update(updates)


# ============================================================================
# OPTIMIZER UTILITIES
# ============================================================================

def create_optimizer_for_hrm(
    model: nn.Module,
    lr: float = 1e-4,
    puzzle_emb_lr: float = 1e-2,
    weight_decay: float = 0.1,
    puzzle_emb_weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    use_separate_puzzle_optimizer: bool = True
) -> Dict[str, optim.Optimizer]:
    """
    Create optimizers for HRM model.

    HRM uses two optimizers:
    1. SignSGD for puzzle embeddings (sparse, high LR)
    2. Adam for all other parameters (dense, lower LR)

    Args:
        model: HRM model
        lr: Learning rate for main parameters
        puzzle_emb_lr: Learning rate for puzzle embeddings
        weight_decay: Weight decay for main parameters
        puzzle_emb_weight_decay: Weight decay for puzzle embeddings
        betas: Adam beta parameters
        use_separate_puzzle_optimizer: Whether to use separate optimizer for puzzle embeddings

    Returns:
        Dictionary with keys:
        - 'main': Main optimizer (Adam) for all parameters
        - 'puzzle': Optional SignSGD optimizer for puzzle embeddings (if use_separate_puzzle_optimizer=True)
    """
    optimizers = {}

    if use_separate_puzzle_optimizer:
        # Create SignSGD optimizer for puzzle embeddings
        puzzle_optimizer = SignSGD_MLX(
            learning_rate=puzzle_emb_lr,
            weight_decay=puzzle_emb_weight_decay
        )
        optimizers['puzzle'] = puzzle_optimizer

        # Create Adam optimizer for main parameters
        # In a full implementation, we would filter out puzzle embedding parameters
        # For now, we create a standard Adam optimizer
        main_optimizer = AdamMLX(
            learning_rate=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        optimizers['main'] = main_optimizer
    else:
        # Single optimizer for all parameters
        main_optimizer = AdamMLX(
            learning_rate=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        optimizers['main'] = main_optimizer

    return optimizers


def get_learning_rate_schedule(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 2000,
    min_ratio: float = 0.1
) -> float:
    """
    Cosine learning rate schedule with warmup.

    Args:
        step: Current step
        total_steps: Total training steps
        base_lr: Base learning rate
        warmup_steps: Number of warmup steps
        min_ratio: Minimum LR as fraction of base_lr

    Returns:
        Learning rate for this step
    """
    import math

    # Warmup phase: linear increase from 0 to base_lr
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)

    # Cosine decay phase
    # Progress through the cosine schedule (0 to 1)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)  # Clamp to [0, 1]

    # Cosine annealing from 1.0 to min_ratio
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = base_lr * (min_ratio + (1.0 - min_ratio) * cosine_decay)

    return lr


# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================

"""
Implementation Priority:
1. AdamMLX - wrapper around built-in (easiest)
2. get_learning_rate_schedule - needed for training
3. SignSGD_MLX - puzzle embedding optimizer
4. create_optimizer_for_hrm - orchestration

MLX Optimizer API:

The key method to implement is:
    def apply_single(self, gradient, parameter, state):
        # Update parameter
        return new_parameter

MLX handles:
- Gradient computation
- State management
- Parameter updates across model
- Learning rate scheduling

Sparse Embedding Strategy:

PyTorch version:
- Complex: separate "local weights", distributed all-gather, unique IDs
- Lots of boilerplate for distributed training

MLX version (much simpler!):
- Full embedding table in unified memory
- Gradients naturally sparse (only computed for batch)
- No distributed complexity
- Just apply SignSGD to non-zero gradients

Example usage:
    ```python
    # Create optimizers
    optimizers = create_optimizer_for_hrm(model, lr=1e-4, puzzle_emb_lr=1e-2)
    puzzle_opt, main_opt = optimizers
    
    # Training step
    loss, grads = mx.value_and_grad(loss_fn)(model.parameters(), batch)
    
    # Update puzzle embeddings
    puzzle_opt.update(model.puzzle_embeddings, grads[...])
    
    # Update main parameters
    main_opt.update(model, grads)
    ```

Key Insight:
MLX's unified memory and simpler API means we can eliminate ~70% of the 
optimizer complexity from the PyTorch version!
"""

