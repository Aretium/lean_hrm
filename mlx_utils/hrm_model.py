"""
MLX HRM Model - Hierarchical Reasoning Model Architecture

This module contains the core HRM model implementation in MLX.

Key components:
- Two-level hierarchy (High/Low)
- Adaptive Computation Time (ACT) with Q-learning
- Recurrent state management (carry)
- Gradient truncation for training stability

Replaces: hrm_act_v1.py from PyTorch version

ORIGINAL PYTORCH REFERENCE:
---------------------------
File: models/hrm/hrm_act_v1.py (lines 1-284)

DATA STRUCTURES:

1. HierarchicalReasoningModel_ACTV1InnerCarry (lines 15-18)
   → HRMInnerCarry
   - z_H: High-level state [batch, seq_len, hidden]
   - z_L: Low-level state [batch, seq_len, hidden]

2. HierarchicalReasoningModel_ACTV1Carry (lines 22-28)
   → HRMCarry
   - inner_carry: HRMInnerCarry
   - steps: Number of reasoning steps [batch]
   - halted: Boolean halt flags [batch]
   - current_data: Dict of batch tensors

CONFIG (lines 31-58):
- H_cycles: 2 (default)
- L_cycles: 2 (default)
- H_layers: 4
- L_layers: 4
- hidden_size: 512
- num_heads: 8
- expansion: 4.0
- halt_max_steps: 16
- halt_exploration_prob: 0.1
- forward_dtype: bfloat16

MODULES:

3. HierarchicalReasoningModel_ACTV1Block (lines 60-83)
   → MLXTransformerBlock
   - Attention + MLP with post-norm (unusual!)
   - RMS norm AFTER each sub-layer

4. HierarchicalReasoningModel_ACTV1ReasoningModule (lines 86-99)
   → MLXReasoningModule
   - Stack of transformer blocks
   - Input injection (adds context)

5. HierarchicalReasoningModel_ACTV1_Inner (lines 102-213)
   → MLXHRMInner
   CRITICAL ALGORITHM (lines 188-213):
   
   ```python
   # Most iterations: NO GRADIENTS (efficient!)
   with torch.no_grad():
       for H_step in range(H_cycles - 1):  # -1 for last
           for L_step in range(L_cycles - 1):
               z_L = L_level(z_L, z_H + input_embeddings)
           z_H = H_level(z_H, z_L)
   
   # Final iteration: WITH GRADIENTS (stable!)
   z_L = L_level(z_L, z_H + input_embeddings)  # ← grad
   z_H = H_level(z_H, z_L)                    # ← grad
   
   # Detach for next iteration
   new_carry = HRMInnerCarry(z_H.detach(), z_L.detach())
   ```
   
   This is THE KEY to HRM's efficiency!
   - Deep computation (H_cycles * L_cycles steps)
   - Only 1 step has gradients
   - Stable training despite depth

6. HierarchicalReasoningModel_ACTV1 (lines 216-284)
   → MLXHRM
   ACT wrapper (lines 240-283):
   
   ```python
   # Update data & carry for halted sequences (lines 242-246)
   new_carry = reset_carry(halted, old_carry)
   new_data = where(halted, new_batch, old_data)
   
   # Forward inner model (line 249)
   new_carry, logits, (q_halt, q_continue) = inner(carry, data)
   
   # Halting logic (lines 257-273):
   if training:
       # Dynamic halting based on Q-values
       halted = (q_halt > q_continue) | is_last_step
       
       # Exploration: random min_steps
       min_steps = random(halt_exploration_prob) * randint(2, max_steps)
       halted = halted & (steps >= min_steps)
       
       # Bootstrap Q-continue target (lines 279-281)
       next_q_halt, next_q_continue = inner(carry, data)
       target_q = sigmoid(max(next_q_halt, next_q_continue))
   else:
       # Inference: always use max_steps (for batching)
       halted = is_last_step
   ```

INITIALIZATION (lines 136-144):
- H_init, L_init: Truncated normal, std=1.0
- Broadcast to [hidden_size]
- Used when resetting carry

Q-HEAD (lines 114, 142-144):
- Linear: hidden_size → 2 (q_halt, q_continue)
- Special init: weights=0, bias=-5 (pessimistic start!)
- Only uses first token: z_H[:, 0]

EMBEDDING DETAILS (lines 146-166):
- Token embeddings scaled by sqrt(hidden_size)
- Puzzle embeddings prepended to sequence
- Position embeddings (learned or RoPE) applied
- Combined with 1/sqrt(2) scaling for learned pos
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# CARRY STATE (Hidden State Management)
# ============================================================================

@dataclass
class HRMInnerCarry:
    """
    Inner recurrent state for H and L modules.
    
    These are the actual "thought" representations that evolve
    as the model reasons through a problem.
    """
    z_H: mx.array  # High-level state [batch, seq_len, hidden_size]
    z_L: mx.array  # Low-level state [batch, seq_len, hidden_size]


@dataclass
class HRMCarry:
    """
    Complete carry state including control flow.
    
    Manages both the reasoning state and the halting mechanism.
    """
    inner_carry: HRMInnerCarry
    
    # ACT (Adaptive Computation Time) state
    steps: mx.array        # Number of reasoning steps taken [batch]
    halted: mx.array       # Whether each example has stopped [batch]
    
    # Current batch data (for multi-step processing)
    current_data: Dict[str, mx.array]


# ============================================================================
# REASONING MODULE (H or L level)
# ============================================================================

class MLXReasoningModule(nn.Module):
    """
    A stack of transformer blocks for either H or L level.

    Key features:
    - Input injection (adds context at input)
    - Multiple transformer layers
    - Used for both high-level and low-level reasoning
    """

    def __init__(self, num_layers: int, **layer_config):
        super().__init__()

        # Import MLXTransformerBlock from layers
        from mlx_utils.layers import MLXTransformerBlock

        # Create stack of transformer blocks
        self.layers = [MLXTransformerBlock(**layer_config) for _ in range(num_layers)]

    def __call__(
        self,
        hidden_states: mx.array,
        input_injection: mx.array,
        cos_sin: Optional[Tuple] = None
    ) -> mx.array:
        """
        Forward pass through reasoning module.

        Args:
            hidden_states: Current state to update
            input_injection: Context to inject (e.g., from other level)
            cos_sin: Optional RoPE embeddings

        Returns:
            Updated hidden states
        """
        # Input injection: add context at the beginning
        hidden_states = hidden_states + input_injection

        # Apply all transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)

        return hidden_states


# ============================================================================
# INNER HRM MODEL (Core Reasoning Engine)
# ============================================================================

class MLXHRMInner(nn.Module):
    """
    Inner HRM model with hierarchical reasoning loops.
    
    This is where the magic happens:
    - H and L modules update alternately
    - Most updates are no-grad (efficient!)
    - Only last update has gradients (stable!)
    - Computes both LM output and Q-values for halting
    
    Replaces: HierarchicalReasoningModel_ACTV1_Inner from PyTorch
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        seq_len: int,
        batch_size: int,

        # Hierarchy config
        H_cycles: int = 2,
        L_cycles: int = 2,
        H_layers: int = 4,
        L_layers: int = 4,

        # Embedding config
        num_puzzle_identifiers: int = 0,
        puzzle_emb_ndim: int = 0,
        pos_encodings: str = "rope",

        # Other
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0
    ):
        super().__init__()

        # Save config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.H_layers = H_layers
        self.L_layers = L_layers
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.puzzle_emb_ndim = puzzle_emb_ndim
        self.pos_encodings = pos_encodings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        # Import required modules
        from mlx_utils.layers import MLXLinear, MLXRotaryEmbedding, trunc_normal_init
        from mlx_utils.embeddings import MLXEmbedding, MLXSparseEmbedding, MLXLearnedPositionEmbedding
        import math

        # Embedding scale: sqrt(hidden_size)
        self.embed_scale = math.sqrt(hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embeddings
        self.embed_tokens = MLXEmbedding(vocab_size, hidden_size, init_std=embed_init_std)

        # LM head (output projection)
        self.lm_head = MLXLinear(hidden_size, vocab_size, bias=False)

        # Q-head for halting (predicts q_halt and q_continue)
        self.q_head = MLXLinear(hidden_size, 2, bias=True)

        # Q-head special initialization: weights=0, bias=-5 (pessimistic start)
        self.q_head.weight = mx.zeros_like(self.q_head.weight)
        self.q_head.bias = mx.full((2,), -5.0)

        # Puzzle embeddings (optional)
        # Calculate puzzle embedding length: ceil(puzzle_emb_ndim / hidden_size)
        self.puzzle_emb_len = -(puzzle_emb_ndim // -hidden_size) if puzzle_emb_ndim > 0 else 0

        if puzzle_emb_ndim > 0:
            self.puzzle_emb = MLXSparseEmbedding(
                num_puzzle_identifiers,
                puzzle_emb_ndim,
                batch_size,
                init_std=0.0  # Zero init for puzzle embeddings
            )
        else:
            self.puzzle_emb = None

        # Position encodings
        max_pos_len = seq_len + self.puzzle_emb_len

        if pos_encodings == "rope":
            self.rotary_emb = MLXRotaryEmbedding(
                dim=hidden_size // num_heads,
                max_position_embeddings=max_pos_len,
                base=rope_theta
            )
        elif pos_encodings == "learned":
            self.embed_pos = MLXLearnedPositionEmbedding(
                max_pos_len,
                hidden_size,
                init_std=embed_init_std
            )
        else:
            raise ValueError(f"Unknown pos_encodings: {pos_encodings}")

        # Transformer block config
        layer_config = {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "expansion": expansion,
            "rms_norm_eps": rms_norm_eps,
            "causal": False  # HRM uses non-causal attention
        }

        # H-level reasoning module
        self.H_level = MLXReasoningModule(H_layers, **layer_config)

        # L-level reasoning module
        self.L_level = MLXReasoningModule(L_layers, **layer_config)

        # Initial states for H and L
        # Initialized with truncated normal, std=1.0
        self.H_init = trunc_normal_init((hidden_size,), std=1.0)
        self.L_init = trunc_normal_init((hidden_size,), std=1.0)

    def _input_embeddings(self, inputs: mx.array, puzzle_identifiers: mx.array) -> mx.array:
        """
        Compute input embeddings from tokens and puzzle IDs.

        Args:
            inputs: Token IDs [batch, seq_len]
            puzzle_identifiers: Puzzle IDs [batch]

        Returns:
            Embeddings [batch, seq_len + puzzle_emb_len, hidden_size]
        """
        # Token embeddings
        embedding = self.embed_tokens(inputs.astype(mx.int32))

        # Add puzzle embeddings if configured
        if self.puzzle_emb is not None:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            # Pad to multiple of hidden_size if needed
            pad_count = self.puzzle_emb_len * self.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = mx.pad(puzzle_embedding, [(0, 0), (0, pad_count)])

            # Reshape and prepend to token embeddings
            puzzle_embedding = puzzle_embedding.reshape(-1, self.puzzle_emb_len, self.hidden_size)
            embedding = mx.concatenate([puzzle_embedding, embedding], axis=1)

        # Add position embeddings if using learned positions
        if self.pos_encodings == "learned":
            # Scale by 1/sqrt(2) to maintain forward variance (as in PyTorch version)
            embedding = 0.707106781 * (embedding + self.embed_pos())

        # Scale by sqrt(hidden_size)
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> HRMInnerCarry:
        """Create empty carry state (will be reset on first use)."""
        # Create empty arrays - these will be filled with H_init/L_init on first reset
        seq_len_with_puzzle = self.seq_len + self.puzzle_emb_len

        return HRMInnerCarry(
            z_H=mx.zeros((batch_size, seq_len_with_puzzle, self.hidden_size)),
            z_L=mx.zeros((batch_size, seq_len_with_puzzle, self.hidden_size))
        )

    def reset_carry(
        self,
        reset_flag: mx.array,
        carry: HRMInnerCarry
    ) -> HRMInnerCarry:
        """
        Reset carry state for halted sequences.

        Args:
            reset_flag: Boolean mask [batch]
            carry: Current carry state

        Returns:
            Carry with reset states where flag is True
        """
        # Reshape reset_flag for broadcasting: [batch] -> [batch, 1, 1]
        reset_flag = mx.expand_dims(mx.expand_dims(reset_flag, axis=1), axis=2)

        # Use mx.where to conditionally reset
        # Where reset_flag is True, use initial state; otherwise, keep current carry
        return HRMInnerCarry(
            z_H=mx.where(reset_flag, self.H_init, carry.z_H),
            z_L=mx.where(reset_flag, self.L_init, carry.z_L)
        )
    
    def __call__(
        self,
        carry: HRMInnerCarry,
        batch: Dict[str, mx.array]
    ) -> Tuple[HRMInnerCarry, mx.array, Tuple[mx.array, mx.array]]:
        """
        One reasoning step.

        Process:
        1. Embed inputs
        2. Run H_cycles x L_cycles iterations (mostly no-grad)
        3. Final update with gradients
        4. Compute LM logits and Q-values

        Args:
            carry: Current state
            batch: Input data with keys:
                - "inputs": Token IDs [batch, seq_len]
                - "puzzle_identifiers": Puzzle IDs [batch]

        Returns:
            - new_carry: Updated state (detached from graph)
            - logits: Language model output [batch, seq_len, vocab_size]
            - (q_halt, q_continue): Q-values for ACT [batch]
        """
        # Prepare sequence info (RoPE if configured)
        cos_sin = self.rotary_emb() if self.pos_encodings == "rope" else None

        # Compute input embeddings
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Initialize H and L states from carry
        z_H = carry.z_H
        z_L = carry.z_L

        # ========================================================================
        # GRADIENT TRUNCATION - THE KEY TO HRM's EFFICIENCY!
        # ========================================================================
        # Most iterations run WITHOUT gradients (efficient!)
        # Only the final iteration computes gradients (stable!)

        # Run H_cycles - 1 full hierarchical iterations without gradients
        for H_step in range(self.H_cycles):
            for L_step in range(self.L_cycles):
                # Skip the very last L update (we'll do it with gradients)
                if (H_step == self.H_cycles - 1) and (L_step == self.L_cycles - 1):
                    continue

                # L-level update (no gradients)
                # Input injection: z_H + input_embeddings
                z_L = mx.stop_gradient(
                    self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
                )

            # Skip the last H update (we'll do it with gradients)
            if H_step == self.H_cycles - 1:
                continue

            # H-level update (no gradients)
            # Input injection: z_L (from L-level)
            z_H = mx.stop_gradient(
                self.H_level(z_H, z_L, cos_sin=cos_sin)
            )

        # ========================================================================
        # FINAL ITERATION WITH GRADIENTS
        # ========================================================================
        # This is the only part that accumulates gradients!

        # Final L-level update WITH gradients
        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)

        # Final H-level update WITH gradients
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        # ========================================================================
        # OUTPUTS
        # ========================================================================

        # Create new carry (detached from computation graph)
        new_carry = HRMInnerCarry(
            z_H=mx.stop_gradient(z_H),
            z_L=mx.stop_gradient(z_L)
        )

        # Language model output (only from actual sequence, not puzzle embeddings)
        logits = self.lm_head(z_H[:, self.puzzle_emb_len:])

        # Q-values for halting (use first token of z_H)
        q_logits = self.q_head(z_H[:, 0])  # [batch, 2]
        q_halt = q_logits[:, 0]           # [batch]
        q_continue = q_logits[:, 1]       # [batch]

        return new_carry, logits, (q_halt, q_continue)


# ============================================================================
# OUTER HRM MODEL (ACT Wrapper)
# ============================================================================

class MLXHRM(nn.Module):
    """
    Complete HRM model with Adaptive Computation Time.
    
    Manages:
    - Multi-step reasoning
    - Halting decisions
    - Training vs inference behavior
    - Carry state management
    
    Replaces: HierarchicalReasoningModel_ACTV1 from PyTorch
    """
    
    def __init__(
        self,
        # Core model config
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        seq_len: int,
        batch_size: int,

        # Hierarchy
        H_cycles: int = 2,
        L_cycles: int = 2,
        H_layers: int = 4,
        L_layers: int = 4,

        # ACT (halting)
        halt_max_steps: int = 16,
        halt_exploration_prob: float = 0.1,

        # Embeddings
        num_puzzle_identifiers: int = 0,
        puzzle_emb_ndim: int = 0,
        pos_encodings: str = "rope",

        # Other
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0
    ):
        super().__init__()

        # Save ACT config
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob

        # Create inner HRM model
        self.inner = MLXHRMInner(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=expansion,
            seq_len=seq_len,
            batch_size=batch_size,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            H_layers=H_layers,
            L_layers=L_layers,
            num_puzzle_identifiers=num_puzzle_identifiers,
            puzzle_emb_ndim=puzzle_emb_ndim,
            pos_encodings=pos_encodings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta
        )

    @property
    def puzzle_emb(self):
        """Access puzzle embeddings (for optimizer)."""
        return self.inner.puzzle_emb
    
    def initial_carry(self, batch: Dict[str, mx.array]) -> HRMCarry:
        """
        Create initial carry state for a new batch.

        Args:
            batch: Input batch (used to infer batch size)

        Returns:
            Initial carry with all sequences marked as halted
        """
        batch_size = batch["inputs"].shape[0]

        return HRMCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=mx.zeros((batch_size,), dtype=mx.int32),
            halted=mx.ones((batch_size,), dtype=mx.bool_),  # All halted initially
            current_data={k: mx.zeros_like(v) for k, v in batch.items()}
        )
    
    def __call__(
        self,
        carry: HRMCarry,
        batch: Dict[str, mx.array],
        training: bool = False
    ) -> Tuple[HRMCarry, Dict[str, mx.array]]:
        """
        One ACT step: reasoning + halting decision.

        Args:
            carry: Current state
            batch: Input data
            training: Whether in training mode

        Returns:
            - new_carry: Updated state
            - outputs: Dictionary with:
                - "logits": LM predictions
                - "q_halt_logits": Q-value for stopping
                - "q_continue_logits": Q-value for continuing
                - "target_q_continue": (training only) Bootstrap target
        """
        # ========================================================================
        # UPDATE CARRY & DATA FOR HALTED SEQUENCES
        # ========================================================================
        # When a sequence is halted, we reset its carry and replace its data

        # Reset inner carry for halted sequences
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        # Reset steps for halted sequences
        new_steps = mx.where(carry.halted, 0, carry.steps)

        # Update current_data: use new batch for halted, keep old data for continuing
        # Need to broadcast halted for each array's dimensions
        new_current_data = {}
        for k, v in batch.items():
            # Reshape halted to broadcast: [batch] -> [batch, 1, 1, ...]
            halted_reshaped = carry.halted
            for _ in range(v.ndim - 1):
                halted_reshaped = mx.expand_dims(halted_reshaped, axis=-1)

            # Use new batch data where halted, keep old data otherwise
            new_current_data[k] = mx.where(halted_reshaped, batch[k], carry.current_data[k])

        # ========================================================================
        # FORWARD INNER MODEL
        # ========================================================================
        # Run one reasoning step

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry,
            new_current_data
        )

        # Prepare outputs
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # ========================================================================
        # HALTING LOGIC
        # ========================================================================

        # Increment steps
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.halt_max_steps

        # Initialize halted flag
        halted = is_last_step

        # Training-specific halting logic
        if training and (self.halt_max_steps > 1):
            # Dynamic halting based on Q-values
            # NOTE: During evaluation, always use max_steps for consistent batching
            halted = halted | (q_halt_logits > q_continue_logits)

            # Exploration: random minimum steps
            # With probability halt_exploration_prob, require a random number of steps
            random_prob = mx.random.uniform(shape=q_halt_logits.shape)
            needs_exploration = random_prob < self.halt_exploration_prob

            # Random min steps between 2 and halt_max_steps
            random_min_steps = mx.random.randint(
                low=2,
                high=self.halt_max_steps + 1,
                shape=new_steps.shape
            )
            min_halt_steps = mx.where(needs_exploration, random_min_steps, 0)

            # Only halt if we've reached the minimum steps
            halted = halted & (new_steps >= min_halt_steps)

            # ====================================================================
            # Q-LEARNING BOOTSTRAP TARGET
            # ====================================================================
            # Compute next Q-values for bootstrapping (no gradients)
            # This is like a "what if we continue?" prediction

            # Run inner model again to get next Q-values (without gradients)
            _, _, (next_q_halt, next_q_continue) = self.inner(new_inner_carry, new_current_data)

            # Target Q-continue: sigmoid of max(next_q_halt, next_q_continue)
            # If it's the last step, use next_q_halt (should stop)
            # Otherwise, use the max (best future action)
            target_q_logits = mx.where(is_last_step, next_q_halt, mx.maximum(next_q_halt, next_q_continue))
            outputs["target_q_continue"] = mx.sigmoid(target_q_logits)

        # ========================================================================
        # CREATE NEW CARRY
        # ========================================================================

        new_carry = HRMCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data
        )

        return new_carry, outputs


# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================

"""
Implementation Priority:
1. HRMInnerCarry, HRMCarry - data structures
2. MLXReasoningModule - builds on transformer blocks
3. MLXHRMInner - core reasoning logic (most complex!)
4. MLXHRM - ACT wrapper (orchestration)

Key Implementation Details:

1. Gradient Truncation (Critical!):
   ```python
   # Most iterations no-grad
   with mx.stop_gradient():
       for most_cycles:
           update_states()
   
   # Only last iteration has gradients
   z_L = L_level(...)  # ← gradients flow here
   z_H = H_level(...)  # ← and here
   ```

2. Halting Logic:
   - Training: Dynamic halting based on Q-values
   - Inference: Always use max_steps (for consistent batching)
   - Exploration: Random min_steps to encourage learning

3. Carry Management:
   - Reset carry when halted=True
   - Preserve carry when halted=False
   - Use mx.where for conditional updates

4. Q-Learning Bootstrap:
   - Compute next Q-values without gradients
   - Use as targets for current Q-values
   - No replay buffer needed (many parallel envs)

MLX-Specific Optimizations:
- Use @mx.compile on inner loops
- Minimize Python control flow
- Batch all operations
- Let MLX handle memory efficiently
"""

