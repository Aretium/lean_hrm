"""
MLX Training Loop - Training utilities for HRM

Contains:
- Training step logic
- Evaluation logic
- Checkpointing
- Learning rate scheduling
- Metrics logging

Replaces: pretrain.py from PyTorch version (training loop parts)
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from pathlib import Path


# ============================================================================
# TRAINING STATE
# ============================================================================

class TrainingState:
    """
    Encapsulates all training state.
    
    Replaces: TrainState dataclass from PyTorch version
    
    Simplified from PyTorch version:
    - No distributed state
    - Single optimizer handling
    - Simpler checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizers: List[optim.Optimizer],
        step: int = 0,
        total_steps: int = 100000
    ):
        # TODO: Implement initialization
        pass
    
    def save_checkpoint(self, path: str):
        """Save model and optimizer state."""
        # TODO: Implement checkpoint saving
        pass
    
    def load_checkpoint(self, path: str):
        """Load model and optimizer state."""
        # TODO: Implement checkpoint loading
        pass


# ============================================================================
# TRAINING STEP
# ============================================================================

def train_step(
    model: nn.Module,
    batch: Dict[str, mx.array],
    optimizers: List[optim.Optimizer],
    carry: Any,
    loss_head: nn.Module,
    base_lrs: List[float],
    step: int,
    total_steps: int,
    warmup_steps: int = 2000
) -> Tuple[Any, float, Dict[str, float]]:
    """
    Single training step.

    Process:
    1. Forward pass (with gradient computation)
    2. Compute loss
    3. Backward pass
    4. Update parameters with optimizers
    5. Update learning rates

    Args:
        model: HRM model
        batch: Input batch with "inputs", "targets", "puzzle_identifiers"
        optimizers: List of optimizers
        carry: Current carry state
        loss_head: ACT loss head
        base_lrs: Base learning rates for each optimizer
        step: Current training step
        total_steps: Total training steps
        warmup_steps: Warmup steps for LR schedule

    Returns:
        Tuple of (new_carry, loss_value, metrics_dict)
    """
    # Define loss function for gradient computation
    def loss_fn(model_params):
        # Forward pass through model
        new_carry, outputs = model(carry, batch, training=True)

        # Compute loss using loss head
        loss_dict = loss_head(
            logits=outputs["logits"],
            targets=batch["targets"],
            q_halt_logits=outputs["q_halt_logits"],
            q_continue_logits=outputs["q_continue_logits"],
            target_q_continue=outputs.get("target_q_continue")
        )

        total_loss = loss_dict["total_loss"]

        # Return loss and auxiliary data
        return total_loss, (new_carry, loss_dict)

    # Compute loss and gradients
    (loss_value, (new_carry, loss_dict)), grads = mx.value_and_grad(loss_fn, has_aux=True)(
        model.trainable_parameters()
    )

    # Update model parameters
    for optimizer in optimizers:
        optimizer.update(model, grads)

    # Force evaluation of updates
    mx.eval(model.parameters())
    mx.eval(loss_value)

    # Update learning rates
    update_learning_rates(optimizers, base_lrs, step, total_steps, warmup_steps)

    # Prepare metrics
    metrics = {
        "loss": float(loss_value.item()),
        "lm_loss": float(loss_dict.get("lm_loss", 0).item()),
        "q_loss": float(loss_dict.get("q_loss", 0).item()) if "q_loss" in loss_dict else 0,
        "accuracy": float(loss_dict.get("accuracy", 0).item()) if "accuracy" in loss_dict else 0,
        "lr": float(base_lrs[0] if optimizers else 0)
    }

    return new_carry, float(loss_value.item()), metrics


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(
    model: nn.Module,
    eval_batches: List[Dict[str, mx.array]],
    loss_head: nn.Module,
    max_steps: int = 16
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Process:
    1. Iterate through test data
    2. Run model to completion (until halted)
    3. Compute metrics

    Args:
        model: HRM model
        eval_batches: List of evaluation batches
        loss_head: ACT loss head
        max_steps: Maximum ACT steps

    Returns:
        Dict of evaluation metrics
    """
    total_loss = 0.0
    total_lm_loss = 0.0
    total_accuracy = 0.0
    num_batches = len(eval_batches)

    for batch in eval_batches:
        # Initialize carry for this batch
        carry = model.initial_carry(batch)

        # Run model for max_steps (or until halted)
        for step in range(max_steps):
            carry, outputs = model(carry, batch, training=False)

            # Check if all sequences halted
            if mx.all(carry.halted).item():
                break

        # Compute loss on final outputs
        loss_dict = loss_head(
            logits=outputs["logits"],
            targets=batch["targets"],
            q_halt_logits=outputs["q_halt_logits"],
            q_continue_logits=outputs["q_continue_logits"],
            target_q_continue=None  # No Q-learning in eval
        )

        total_loss += float(loss_dict["total_loss"].item())
        total_lm_loss += float(loss_dict.get("lm_loss", 0).item())
        total_accuracy += float(loss_dict.get("accuracy", 0).item()) if "accuracy" in loss_dict else 0

    # Average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "lm_loss": total_lm_loss / num_batches,
        "accuracy": total_accuracy / num_batches
    }

    return metrics


# ============================================================================
# LEARNING RATE SCHEDULING
# ============================================================================

def cosine_schedule_with_warmup(
    current_step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 2000,
    min_ratio: float = 1.0
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Phase 1 (warmup): Linear from 0 to base_lr
    Phase 2 (cosine): Cosine decay from base_lr to min_ratio * base_lr

    Args:
        current_step: Current training step
        total_steps: Total training steps
        base_lr: Maximum learning rate
        warmup_steps: Number of warmup steps
        min_ratio: Minimum LR as fraction of base_lr

    Returns:
        Learning rate for current step
    """
    import math

    # Phase 1: Linear warmup
    if current_step < warmup_steps:
        return base_lr * (current_step / warmup_steps)

    # Phase 2: Cosine decay
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)  # Clamp to [0, 1]

    # Cosine decay from 1.0 to min_ratio
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr_multiplier = min_ratio + (1.0 - min_ratio) * cosine_decay

    return base_lr * lr_multiplier


def update_learning_rates(
    optimizers: List[optim.Optimizer],
    base_lrs: List[float],
    current_step: int,
    total_steps: int,
    warmup_steps: int = 2000,
    min_ratio: float = 1.0
):
    """
    Update learning rates for all optimizers.

    Args:
        optimizers: List of optimizers
        base_lrs: Base learning rates for each optimizer
        current_step: Current step
        total_steps: Total steps
        warmup_steps: Warmup steps
        min_ratio: Min LR ratio
    """
    for optimizer, base_lr in zip(optimizers, base_lrs):
        new_lr = cosine_schedule_with_warmup(
            current_step, total_steps, base_lr, warmup_steps, min_ratio
        )
        optimizer.learning_rate = new_lr


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizers: List[optim.Optimizer],
    step: int,
    checkpoint_dir: str,
    metadata: Optional[Dict] = None
):
    """
    Save training checkpoint.

    Saves:
    - Model parameters
    - Optimizer states
    - Training step
    - Optional metadata

    Args:
        model: Model to save
        optimizers: Optimizers to save
        step: Current step
        checkpoint_dir: Directory to save to
        metadata: Optional metadata dict
    """
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = checkpoint_dir / f"model_step_{step}.npz"
    model.save_weights(str(model_path))

    # Save metadata (including step and optimizer info)
    metadata = metadata or {}
    metadata["step"] = step

    metadata_path = checkpoint_dir / f"metadata_step_{step}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Checkpoint saved at step {step} to {checkpoint_dir}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizers: Optional[List[optim.Optimizer]] = None
) -> Tuple[int, Dict]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load into
        optimizers: Optional optimizers to load state into

    Returns:
        Tuple of (step, metadata)
    """
    # Load model weights
    model.load_weights(checkpoint_path)

    # Try to load metadata
    metadata_path = checkpoint_path.replace("model_", "metadata_").replace(".npz", ".json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        step = metadata.get("step", 0)
    else:
        # Try to extract step from filename
        try:
            step = int(checkpoint_path.split("step_")[-1].split(".")[0])
        except:
            step = 0
        metadata = {}

    print(f"Checkpoint loaded from step {step}")
    return step, metadata


# ============================================================================
# METRICS LOGGING
# ============================================================================

def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str = "train",
    wandb_run = None
):
    """
    Log metrics to console and W&B.

    Args:
        metrics: Metrics dictionary
        step: Training step
        prefix: Metric prefix (train/eval)
        wandb_run: Optional W&B run object
    """
    # Console logging
    formatted = format_metrics(metrics)
    print(f"[{prefix}] Step {step}: {formatted}")

    # W&B logging (if available)
    if wandb_run is not None:
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb_run.log(prefixed_metrics, step=step)


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics for console display.

    Args:
        metrics: Metrics dict

    Returns:
        Formatted string
    """
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if abs(value) < 0.01 or abs(value) > 1000:
                parts.append(f"{key}={value:.4e}")
            else:
                parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")

    return ", ".join(parts)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(
    model: nn.Module,
    train_batches: List[Dict[str, mx.array]],
    eval_batches: List[Dict[str, mx.array]],
    loss_head: nn.Module,
    optimizers: List[optim.Optimizer],
    base_lrs: List[float],
    total_steps: int,
    warmup_steps: int = 2000,
    eval_interval: int = 1000,
    checkpoint_interval: int = 1000,
    log_interval: int = 100,
    checkpoint_dir: str = "checkpoints",
    wandb_run = None
):
    """
    Main training loop.

    Simplified from PyTorch version:
    - No distributed synchronization
    - No process management
    - Cleaner control flow

    Args:
        model: HRM model
        train_batches: Training batches
        eval_batches: Evaluation batches
        loss_head: ACT loss head
        optimizers: List of optimizers
        base_lrs: Base learning rates
        total_steps: Total training steps
        warmup_steps: Warmup steps for LR schedule
        eval_interval: Steps between evaluations
        checkpoint_interval: Steps between checkpoints
        log_interval: Steps between logging
        checkpoint_dir: Where to save checkpoints
        wandb_run: Optional W&B run
    """
    print(f"\nStarting training for {total_steps} steps...")
    print(f"  - Train batches: {len(train_batches)}")
    print(f"  - Eval batches: {len(eval_batches)}")
    print(f"  - Base LRs: {base_lrs}")
    print(f"  - Warmup steps: {warmup_steps}")
    print(f"  - Eval interval: {eval_interval}")
    print(f"  - Checkpoint interval: {checkpoint_interval}\n")

    # Initialize carry
    carry = model.initial_carry(train_batches[0])

    # Training loop
    batch_idx = 0
    for step in range(1, total_steps + 1):
        # Get next batch (cycle through dataset)
        batch = train_batches[batch_idx % len(train_batches)]
        batch_idx += 1

        # Training step
        carry, loss, metrics = train_step(
            model=model,
            batch=batch,
            optimizers=optimizers,
            carry=carry,
            loss_head=loss_head,
            base_lrs=base_lrs,
            step=step,
            total_steps=total_steps,
            warmup_steps=warmup_steps
        )

        # Log metrics
        if step % log_interval == 0:
            log_metrics(metrics, step, prefix="train", wandb_run=wandb_run)

        # Evaluation
        if step % eval_interval == 0:
            print(f"\nEvaluating at step {step}...")
            eval_metrics = evaluate(
                model=model,
                eval_batches=eval_batches,
                loss_head=loss_head,
                max_steps=model.halt_max_steps
            )
            log_metrics(eval_metrics, step, prefix="eval", wandb_run=wandb_run)
            print()

        # Checkpointing
        if step % checkpoint_interval == 0:
            save_checkpoint(
                model=model,
                optimizers=optimizers,
                step=step,
                checkpoint_dir=checkpoint_dir,
                metadata={"loss": loss, **metrics}
            )

    # Final evaluation
    print(f"\n{'='*80}")
    print("Final Evaluation")
    print(f"{'='*80}")
    eval_metrics = evaluate(
        model=model,
        eval_batches=eval_batches,
        loss_head=loss_head,
        max_steps=model.halt_max_steps
    )
    log_metrics(eval_metrics, total_steps, prefix="eval", wandb_run=wandb_run)

    # Final checkpoint
    save_checkpoint(
        model=model,
        optimizers=optimizers,
        step=total_steps,
        checkpoint_dir=checkpoint_dir,
        metadata={"loss": loss, "final": True, **eval_metrics}
    )

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}\n")


# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================

"""
Implementation Priority:
1. cosine_schedule_with_warmup - needed for training
2. train_step - core training logic
3. evaluate - test performance
4. save_checkpoint, load_checkpoint - persistence
5. train - orchestrates everything

MLX Training Advantages:

PyTorch:
- Complex backward() + optimizer.step()
- Manual gradient scaling
- Distributed all-reduce
- Multiple processes

MLX:
- mx.value_and_grad() for loss + gradients
- Automatic mixed precision
- Single process
- Simpler API

Example training step:
    ```python
    def loss_fn(model, batch, carry):
        carry, loss, metrics, outputs, done = model(carry, batch)
        return loss, (carry, metrics)
    
    # Compute loss and gradients
    (loss, (new_carry, metrics)), grads = mx.value_and_grad(loss_fn, has_aux=True)(
        model, batch, carry
    )
    
    # Update parameters
    optimizer.update(model, grads)
    mx.eval(model.parameters())  # Force evaluation
    ```

Key Simplifications:

1. No distributed:
   ```python
   # PyTorch
   dist.all_reduce(param.grad)
   
   # MLX
   # Not needed!
   ```

2. Simpler checkpointing:
   ```python
   # PyTorch
   torch.save({
       "model": model.state_dict(),
       "optimizer": optimizer.state_dict(),
       ...
   }, path)
   
   # MLX
   model.save_weights(path)  # That's it!
   ```

3. No .cuda(), .to(device):
   ```python
   # PyTorch
   batch = {k: v.cuda() for k, v in batch.items()}
   
   # MLX
   # Already on GPU (unified memory)!
   ```

Training Loop Structure:

1. Initialize model, optimizers, datasets
2. For each step:
   a. Get batch
   b. Forward + backward
   c. Update parameters
   d. Update LR
   e. Log metrics
   f. Evaluate periodically
   g. Checkpoint periodically
3. Final evaluation + checkpoint
"""

