"""
MLX Loss Functions

Loss functions for HRM training:
- StableMax cross-entropy (numerically stable softmax alternative)
- Standard cross-entropy
- ACT loss head (combines LM loss + Q-learning losses)

Replaces: losses.py from PyTorch version

ORIGINAL PYTORCH REFERENCE:
---------------------------
File: models/losses.py (lines 1-102)

1. IGNORE_LABEL_ID = -100 (line 8)
   → Standard PyTorch ignore index
   
2. s() function (lines 11-16)
   → StableMax transformation
   → s(x) = x+1 if x>=0 else 1/(1-x+eps)
   → CRITICAL: Must match exactly for numerical stability
   
3. log_stablemax (lines 19-21)
   → Log of normalized stablemax
   → Used instead of log_softmax
   
4. stablemax_cross_entropy (lines 24-31)
   → CRITICAL: HRM's preferred loss function
   → More stable than softmax for extreme distributions
   → Cast to float64 for precision!
   
5. softmax_cross_entropy (lines 34-37)
   → Standard fallback
   → Uses torch.nn.functional.cross_entropy
   
6. ACTLossHead class (lines 40-102)
   → CRITICAL: Complete loss computation
   → Three loss components:
     a) LM loss: Token prediction (line 83)
     b) Q-halt loss: Binary CE for stopping (line 84)
     c) Q-continue loss: Bootstrap target (lines 92-96)
   → Loss weights: 1.0 + 0.5 * (q_halt + q_continue)
   
   Metrics computed (lines 71-79):
   - count: Valid sequences (halted & has labels)
   - accuracy: Token-level accuracy
   - exact_accuracy: Sequence-level accuracy
   - q_halt_accuracy: Halt prediction accuracy
   - steps: Average reasoning steps
   
   CRITICAL Details:
   - Only compute loss for halted sequences (line 70)
   - Normalize LM loss by sequence length (line 64)
   - Q-halt target is sequence correctness (line 67)
   - Bootstrap Q-continue from next step (line 94)

LOSS FORMULA:
total_loss = lm_loss + 0.5 * q_halt_loss + 0.5 * q_continue_loss

Where:
- lm_loss = sum(CE(logits, labels) / seq_len) over batch
- q_halt_loss = BCE(q_halt_logits, is_sequence_correct)
- q_continue_loss = BCE(q_continue_logits, sigmoid(max(next_q_halt, next_q_continue)))
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Tuple, Optional


# ============================================================================
# CONSTANTS
# ============================================================================

IGNORE_LABEL_ID = -100


# ============================================================================
# CROSS-ENTROPY VARIANTS
# ============================================================================

def stablemax(x: mx.array, epsilon: float = 1e-30) -> mx.array:
    """
    StableMax transformation (alternative to softmax).
    
    Formula:
        s(x) = x + 1           if x >= 0
        s(x) = 1/(1 - x + ε)   if x < 0
    
    More numerically stable for extreme distributions.
    
    Args:
        x: Input logits
        epsilon: Small constant for numerical stability
        
    Returns:
        Transformed values
    """
    

    s_x = mx.where(x < 0, 1.0/(1.0-x+epsilon), x+1.0)
    return s_x


def log_stablemax(x: mx.array, axis: int = -1) -> mx.array:
    """
    Log of normalized stablemax (like log_softmax).
    
    Args:
        x: Input logits
        axis: Axis to normalize over
        
    Returns:
        Log probabilities
    """
    
    s_x = stablemax(x)
    return mx.log(s_x / mx.sum(s_x, axis=axis, keepdims=True))


def stablemax_cross_entropy(
    logits: mx.array,
    labels: mx.array,
    ignore_index: int = IGNORE_LABEL_ID
) -> mx.array:
    """
    Cross-entropy with stablemax instead of softmax.
    
    Args:
        logits: Predicted logits [batch, seq_len, vocab_size]
        labels: True labels [batch, seq_len]
        ignore_index: Label value to ignore
        
    Returns:
        Loss per token [batch, seq_len]
    """
    # Note: PyTorch reference uses float64 for precision, but MLX on GPU doesn't support it
    # The stablemax transformation should provide numerical stability even with float32

    # Compute log probabilities using stablemax
    logprobs = log_stablemax(logits, axis=-1)

    # Create mask for valid labels
    valid_mask = labels != ignore_index

    # Replace ignored indices with 0 to avoid errors in gather
    transformed_labels = mx.where(valid_mask, labels, 0)

    # Gather log probabilities for true labels
    # Need to handle the gather across the last dimension
    batch_size, seq_len, vocab_size = logits.shape
    flat_logprobs = logprobs.reshape(-1, vocab_size)
    flat_labels = transformed_labels.reshape(-1)

    # Gather predictions
    prediction_logprobs = mx.take_along_axis(
        flat_logprobs,
        flat_labels[:, None],
        axis=-1
    ).squeeze(-1).reshape(batch_size, seq_len)

    # Apply mask and negate for cross-entropy
    return -mx.where(valid_mask, prediction_logprobs, 0.0)


def softmax_cross_entropy(
    logits: mx.array,
    labels: mx.array,
    ignore_index: int = IGNORE_LABEL_ID
) -> mx.array:
    """
    Standard cross-entropy loss.
    
    Args:
        logits: Predicted logits [batch, seq_len, vocab_size]
        labels: True labels [batch, seq_len]
        ignore_index: Label value to ignore
        
    Returns:
        Loss per token [batch, seq_len]
    """
    # Compute log probabilities using softmax + log
    # MLX doesn't have log_softmax directly, so we compute it manually
    logits_shifted = logits - mx.max(logits, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits_shifted)
    sum_exp = mx.sum(exp_logits, axis=-1, keepdims=True)
    logprobs = logits_shifted - mx.log(sum_exp)

    # Create mask for valid labels
    valid_mask = labels != ignore_index

    # Replace ignored indices with 0 to avoid errors in gather
    transformed_labels = mx.where(valid_mask, labels, 0)

    # Gather log probabilities for true labels
    batch_size, seq_len, vocab_size = logits.shape
    flat_logprobs = logprobs.reshape(-1, vocab_size)
    flat_labels = transformed_labels.reshape(-1)

    # Gather predictions
    prediction_logprobs = mx.take_along_axis(
        flat_logprobs,
        flat_labels[:, None],
        axis=-1
    ).squeeze(-1).reshape(batch_size, seq_len)

    # Apply mask and negate for cross-entropy
    return -mx.where(valid_mask, prediction_logprobs, 0.0)


# ============================================================================
# ACT LOSS HEAD
# ============================================================================

class MLXACTLossHead(nn.Module):
    """
    Complete loss computation for HRM with ACT.
    
    Combines three loss components:
    1. LM Loss: Token prediction accuracy (stablemax or softmax CE)
    2. Q-Halt Loss: Binary classification for stopping decision
    3. Q-Continue Loss: Bootstrap target for continuing (only in training)
    
    Also computes metrics:
    - Token accuracy
    - Exact sequence accuracy
    - Q-halt accuracy (does model know when it's correct?)
    - Average reasoning steps
    
    Replaces: ACTLossHead from PyTorch losses.py
    """
    
    def __init__(self, model: nn.Module, loss_type: str = "stablemax_cross_entropy"):
        super().__init__()
        # TODO: Implement initialization
        self.model = model
        self.loss_type = loss_type
        self.loss_fn = globals()[self.loss_type]
    
    def initial_carry(self, batch: Dict[str, mx.array]):
        """Pass through to model."""
        # TODO: Implement
        return self.model.initial_carry(batch)
        
    
    def __call__(
        self,
        carry,
        batch: Dict[str, mx.array],
        return_keys: list = None,
        training: bool = False
    ) -> Tuple:
        """
        Compute loss and metrics.

        Args:
            carry: HRM carry state
            batch: Input batch with:
                - "inputs": Token IDs
                - "labels": Target tokens
                - "puzzle_identifiers": Puzzle IDs
            return_keys: Which outputs to detach and return
            training: Whether in training mode

        Returns:
            Tuple of:
            - new_carry: Updated carry state
            - loss: Scalar total loss
            - metrics: Dict of metrics (detached)
            - outputs: Dict of requested outputs (detached)
            - all_halted: Whether all sequences finished
        """
        # Forward pass through model
        new_carry, outputs = self.model(carry, batch, training=training)

        # Get labels from the carry state
        labels = new_carry.current_data["labels"]
        logits = outputs["logits"]

        # Create mask for valid labels (not ignored)
        valid_mask = labels != IGNORE_LABEL_ID

        # Count valid labels per sequence
        loss_counts = valid_mask.sum(axis=-1)  # [batch]

        # Compute per-token LM loss
        per_token_lm_loss = self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID)  # [batch, seq_len]

        # Normalize by sequence length and sum over tokens
        loss_divisor = mx.maximum(loss_counts, 1)[:, None]  # [batch, 1]
        per_example_lm_loss = (per_token_lm_loss / loss_divisor).sum(axis=-1)  # [batch]

        # Only compute loss for halted sequences that have labels
        halted_mask = new_carry.halted & (loss_counts > 0)  # [batch]

        # Compute sequence correctness for Q-halt target
        predictions = mx.argmax(logits, axis=-1)  # [batch, seq_len]
        correct_tokens = (predictions == labels) & valid_mask  # [batch, seq_len]
        num_correct = correct_tokens.sum(axis=-1)  # [batch]
        is_correct = (num_correct == loss_counts) & (loss_counts > 0)  # [batch]

        # Q-halt loss: BCE with sequence correctness as target
        q_halt_target = is_correct.astype(mx.float32)  # [batch]
        q_halt_logits = outputs["q_halt_logits"]  # [batch]
        per_example_q_halt_loss = mx.nn.losses.binary_cross_entropy_with_logits(
            q_halt_logits,
            q_halt_target,
            reduction='none'
        )  # [batch]

        # Q-continue loss (only in training mode)
        if training and "q_continue_logits" in outputs:
            # Bootstrap target from next step's Q-values
            # This requires computing max(next_q_halt, next_q_continue)
            # For now, we'll use a simplified version
            # In full implementation, this would require another forward pass
            q_continue_logits = outputs["q_continue_logits"]  # [batch]

            # Simplified bootstrap target (would need actual next step in full version)
            # Using q_halt as approximation for now
            bootstrap_target = mx.sigmoid(mx.maximum(q_halt_logits, q_continue_logits))

            per_example_q_continue_loss = mx.nn.losses.binary_cross_entropy_with_logits(
                q_continue_logits,
                bootstrap_target,
                reduction='none'
            )  # [batch]
        else:
            per_example_q_continue_loss = mx.zeros_like(per_example_lm_loss)

        # Combine losses with weights (only for halted sequences)
        per_example_total_loss = per_example_lm_loss + 0.5 * (per_example_q_halt_loss + per_example_q_continue_loss)
        total_loss = mx.where(halted_mask, per_example_total_loss, 0.0).sum()

        # Compute metrics (detached - convert to scalars)
        # Q-halt predictions: positive logit means predicting correct
        q_halt_predictions = (q_halt_logits >= 0).astype(mx.float32)
        q_halt_correct = (q_halt_predictions == q_halt_target) & halted_mask

        metrics = {
            "count": halted_mask.sum().item(),
            "accuracy": mx.where(halted_mask, num_correct.astype(mx.float32), 0.0).sum().item(),
            "exact_accuracy": mx.where(halted_mask, is_correct.astype(mx.float32), 0.0).sum().item(),
            "q_halt_accuracy": q_halt_correct.sum().item(),
            "steps": mx.where(halted_mask, new_carry.steps.astype(mx.float32), 0.0).sum().item(),
        }

        # Prepare outputs to return based on return_keys
        return_outputs = {}
        if return_keys:
            for key in return_keys:
                if key in outputs:
                    return_outputs[key] = outputs[key]

        # Check if all sequences have halted
        all_halted = new_carry.halted.all().item()

        return new_carry, total_loss, metrics, return_outputs, all_halted


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_accuracy(
    logits: mx.array,
    labels: mx.array,
    ignore_index: int = IGNORE_LABEL_ID
) -> Tuple[mx.array, mx.array]:
    """
    Compute token-level and sequence-level accuracy.

    Args:
        logits: Predicted logits [batch, seq_len, vocab_size]
        labels: True labels [batch, seq_len]
        ignore_index: Label to ignore

    Returns:
        Tuple of:
        - token_accuracy: Fraction of correct tokens per sequence [batch]
        - seq_accuracy: Whether entire sequence is correct [batch]
    """
    # Get predicted tokens (argmax over vocabulary)
    predictions = mx.argmax(logits, axis=-1)

    # Create mask for valid labels (not ignored)
    valid_mask = labels != ignore_index

    # Check which predictions are correct
    correct = (predictions == labels) & valid_mask

    # Token-level accuracy: fraction of correct tokens per sequence
    num_correct = correct.sum(axis=-1).astype(mx.float32)
    num_valid = valid_mask.sum(axis=-1).astype(mx.float32)
    # Avoid division by zero
    token_accuracy = mx.where(num_valid > 0, num_correct / num_valid, 0.0)

    # Sequence-level accuracy: all valid tokens must be correct
    # A sequence is correct if all valid tokens match
    seq_accuracy = mx.where(
        num_valid > 0,
        (num_correct == num_valid).astype(mx.float32),
        0.0
    )

    return token_accuracy, seq_accuracy


# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================

"""
Implementation Priority:
1. stablemax, log_stablemax - core transformations
2. softmax_cross_entropy - standard baseline
3. stablemax_cross_entropy - HRM's preferred loss
4. compute_accuracy - metrics helper
5. MLXACTLossHead - orchestrates everything

Loss Formulation:

Total Loss = LM_loss + 0.5 * (Q_halt_loss + Q_continue_loss)

Where:
- LM_loss: Per-example loss, normalized by sequence length
- Q_halt_loss: BCE(q_halt_logits, is_correct)
- Q_continue_loss: BCE(q_continue_logits, bootstrap_target)

Key Details:

1. Only compute loss for halted sequences:
   ```python
   valid_mask = carry.halted & (label_count > 0)
   loss = mx.where(valid_mask, per_example_loss, 0.0).sum()
   ```

2. Q-halt target is sequence correctness:
   ```python
   is_correct = (predicted == labels).all(where=labels != ignore)
   q_halt_target = is_correct.astype(mx.float32)
   ```

3. Q-continue target comes from next step (bootstrap):
   ```python
   next_q_halt, next_q_continue = model.forward(carry, batch)
   target = sigmoid(max(next_q_halt, next_q_continue))
   ```

MLX Advantages:
- Simpler masking (no .view(), .flatten() needed)
- Automatic broadcasting
- Built-in BCE: mx.nn.losses.binary_cross_entropy
- Efficient gather operations
"""

