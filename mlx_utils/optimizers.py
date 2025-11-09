"""
MLX Optimizers

Custom optimizers for HRM:
- AdamMLX: Standard Adam (MLX has built-in, but we may want custom)
- SignSGD_MLX: Sign-based SGD for sparse embeddings

Replaces: adam_atan2.py and sparse_embedding optimizer from PyTorch version
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Dict, Any


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
        super().__init__(learning_rate=learning_rate)
        # TODO: Implement initialization
        pass
    
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
        # TODO: Implement SignSGD update
        pass


# ============================================================================
# ADAM VARIANTS
# ============================================================================

class AdamMLX(optim.Adam):
    """
    Adam optimizer (standard or custom variant).
    
    MLX already has Adam built-in, but we may want custom behavior:
    - Custom scheduling
    - Gradient clipping
    - Specific hyperparameters
    
    For now, this is just a wrapper around mlx.optimizers.Adam
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1
    ):
        super().__init__(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )


class AdamATan2_MLX(optim.Optimizer):
    """
    Adam with atan2-based parameter updates (if we want to replicate PyTorch version).
    
    Replaces: AdamATan2 from PyTorch (from adam_atan2 package)
    
    Note: This is an experimental variant. May not be necessary for MLX version.
    Standard Adam might work just as well.
    
    TODO: Determine if we actually need this variant or if standard Adam suffices.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1
    ):
        super().__init__(learning_rate=learning_rate)
        # TODO: Implement if needed
        pass


# ============================================================================
# OPTIMIZER UTILITIES
# ============================================================================

def create_optimizer_for_hrm(
    model: nn.Module,
    lr: float = 1e-4,
    puzzle_emb_lr: float = 1e-2,
    weight_decay: float = 0.1,
    puzzle_emb_weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95)
) -> List[optim.Optimizer]:
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
        
    Returns:
        List of [puzzle_emb_optimizer, main_optimizer]
    """
    # TODO: Implement optimizer creation
    pass


def get_learning_rate_schedule(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 2000,
    min_ratio: float = 1.0
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
    # TODO: Implement LR schedule
    pass


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

