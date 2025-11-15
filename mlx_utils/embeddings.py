"""
MLX Embeddings - Token, Position, and Puzzle Embeddings

This module contains MLX implementations of embedding layers:
- Token embeddings
- Learned position embeddings
- Sparse puzzle embeddings (unique to HRM)

Key differences from PyTorch version:
- No manual dtype casting
- Simpler sparse embedding implementation
- Unified memory management

ORIGINAL PYTORCH REFERENCE:
---------------------------
File: models/layers.py
- CastedEmbedding (lines 62-78)
  → MLXEmbedding
  → Truncated normal init, cast to forward dtype

File: models/sparse_embedding.py (lines 1-133)
- CastedSparseEmbedding (lines 11-39)
  → MLXSparseEmbedding
  → CRITICAL: Unique per-puzzle embeddings
  → Original: Complex local_weights buffer + distributed all-gather
  → MLX: Simplified with unified memory
  
- CastedSparseEmbeddingSignSGD_Distributed (lines 41-96)
  → See optimizers.py for SignSGD_MLX
  → Original: All-gather → unique → sign(grad) update
  → MLX: Much simpler (no distributed!)

File: models/common.py (lines 1-33)
- trunc_normal_init_ (lines 7-32)
  → trunc_normal_init in MLX
  → CRITICAL: JAX-style truncated normal (not PyTorch's!)
  → Mathematically correct std dev
  
File: models/hrm/hrm_act_v1.py
- Puzzle embedding usage (lines 116-121, 146-166)
  → puzzle_emb_len calculation (ceil div)
  → Concatenate before token embeddings
  → Position embeddings cover both

INITIALIZATION NOTES:
- Token embeddings: init_std = 1.0 / sqrt(hidden_size)
- Puzzle embeddings: init_std = 0.0 (zero init!)
- Position embeddings: init_std = 1.0 / sqrt(hidden_size)
- Scale factor: sqrt(hidden_size) applied to combined embeddings
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import math


# ============================================================================
# STANDARD EMBEDDINGS
# ============================================================================

class MLXEmbedding(nn.Module):
    """
    Standard embedding layer with custom initialization.

    Replaces: CastedEmbedding from PyTorch layers.py

    Note: No need for explicit casting in MLX!
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float = 1.0
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize with truncated normal (matches PyTorch version)
        # Import from layers.py to avoid duplication
        from mlx_utils.layers import trunc_normal_init
        self.weight = trunc_normal_init(
            (num_embeddings, embedding_dim),
            std=init_std
        )

    def __call__(self, indices: mx.array) -> mx.array:
        """
        Look up embeddings.

        Args:
            indices: Integer indices [batch, seq_len] or [batch, ...]

        Returns:
            Embeddings [batch, seq_len, embedding_dim] or [batch, ..., embedding_dim]
        """
        # Simple indexing in MLX - no need for F.embedding
        return self.weight[indices]


# ============================================================================
# SPARSE EMBEDDINGS (HRM-specific)
# ============================================================================

class MLXSparseEmbedding(nn.Module):
    """
    Sparse embedding layer for puzzle-specific representations.

    Replaces: CastedSparseEmbedding from PyTorch sparse_embedding.py

    Key features:
    - Each training example gets unique embedding
    - Only stores embeddings for current batch (memory efficient)
    - Updated via custom SignSGD optimizer

    Implementation details:
    - Main weight table: persistent, stores all puzzle embeddings
    - Local weights: batch-only, trainable copy for current batch
    - Local IDs: tracks which puzzles are in current batch
    - Custom optimizer uses local_weights gradients + local_ids to update main weights
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        batch_size: int,
        init_std: float = 0.0
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self._training = True  # Track training mode manually

        # Main embedding table (persistent, not trainable directly)
        from mlx_utils.layers import trunc_normal_init
        self.weight = trunc_normal_init(
            (num_embeddings, embedding_dim),
            std=init_std
        )

        # Local weights and IDs for training
        # These are used to track which embeddings are in the current batch
        # and accumulate gradients only for those embeddings
        self.local_weights = mx.zeros((batch_size, embedding_dim))
        self.local_ids = mx.zeros((batch_size,), dtype=mx.int32)

    def train(self, mode: bool = True):
        """Set training mode."""
        self._training = mode

    def eval(self):
        """Set evaluation mode."""
        self._training = False

    def __call__(self, puzzle_ids: mx.array) -> mx.array:
        """
        Look up puzzle embeddings.

        Args:
            puzzle_ids: Puzzle indices [batch]

        Returns:
            Puzzle embeddings [batch, embedding_dim]
        """
        if not self._training:
            # Test mode: direct lookup, no gradient tracking
            return self.weight[puzzle_ids]

        # Training mode: use local_weights buffer
        # This allows the optimizer to track which embeddings were used
        # and accumulate gradients only for those embeddings

        # Copy current weights for this batch (stop gradients on the copy operation)
        # This is similar to PyTorch's with torch.no_grad(): local_weights.copy_()
        batch_weights = self.weight[puzzle_ids]

        # Update local buffers
        # In MLX, we create new arrays rather than in-place updates
        # The local_weights will be the trainable version
        self.local_weights = batch_weights
        self.local_ids = puzzle_ids

        # Return local_weights which will accumulate gradients during backprop
        return self.local_weights


# ============================================================================
# INITIALIZATION UTILITIES
# ============================================================================

# NOTE: trunc_normal_init is implemented in mlx_utils/layers.py
# Import from there to avoid duplication


# ============================================================================
# POSITION EMBEDDINGS
# ============================================================================

class MLXLearnedPositionEmbedding(nn.Module):
    """
    Learned position embeddings (alternative to RoPE).

    Used in some HRM configurations instead of rotary embeddings.
    """

    def __init__(
        self,
        max_position_embeddings: int,
        embedding_dim: int,
        init_std: float = 1.0
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.embedding_dim = embedding_dim

        # Initialize with truncated normal
        from mlx_utils.layers import trunc_normal_init
        self.weight = trunc_normal_init(
            (max_position_embeddings, embedding_dim),
            std=init_std
        )

    def __call__(self) -> mx.array:
        """
        Get all position embeddings.

        Returns:
            Position embeddings [max_position_embeddings, embedding_dim]
        """
        return self.weight


# ============================================================================
# COMBINED EMBEDDING LAYER
# ============================================================================

class MLXCombinedEmbeddings(nn.Module):
    """
    Combines token, position, and puzzle embeddings.

    This is what the HRM model actually uses at input.

    Process:
    1. Token embeddings from vocabulary
    2. Optional puzzle embeddings (prepended to sequence)
    3. Position embeddings (learned or RoPE)
    4. Scaling by sqrt(hidden_size)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_seq_len: int,
        num_puzzle_identifiers: int = 0,
        puzzle_emb_ndim: int = 0,
        pos_encoding_type: str = "rope",  # "rope" or "learned"
        batch_size: int = 1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.puzzle_emb_ndim = puzzle_emb_ndim
        self.pos_encoding_type = pos_encoding_type

        # Embedding scale: sqrt(hidden_size)
        self.embed_scale = math.sqrt(hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embeddings
        self.token_embeddings = MLXEmbedding(
            vocab_size,
            hidden_size,
            init_std=embed_init_std
        )

        # Puzzle embeddings (optional)
        # Calculate how many embedding positions we need for puzzle embeddings
        # ceil division: puzzle_emb_len = ceil(puzzle_emb_ndim / hidden_size)
        self.puzzle_emb_len = -(puzzle_emb_ndim // -hidden_size) if puzzle_emb_ndim > 0 else 0

        if puzzle_emb_ndim > 0:
            self.puzzle_embeddings = MLXSparseEmbedding(
                num_puzzle_identifiers,
                puzzle_emb_ndim,
                batch_size,
                init_std=0.0  # Zero init for puzzle embeddings
            )
        else:
            self.puzzle_embeddings = None

        # Position embeddings (only if using learned positional encodings)
        # RoPE is handled separately in the attention layer
        if pos_encoding_type == "learned":
            self.position_embeddings = MLXLearnedPositionEmbedding(
                max_seq_len + self.puzzle_emb_len,
                hidden_size,
                init_std=embed_init_std
            )
        else:
            self.position_embeddings = None

    def __call__(
        self,
        input_ids: mx.array,
        puzzle_ids: Optional[mx.array] = None
    ) -> mx.array:
        """
        Compute combined embeddings.

        Args:
            input_ids: Token indices [batch, seq_len]
            puzzle_ids: Optional puzzle indices [batch]

        Returns:
            Combined embeddings [batch, seq_len + puzzle_len, hidden_size]
        """
        # 1. Get token embeddings
        embeddings = self.token_embeddings(input_ids)

        # 2. Prepend puzzle embeddings if present
        if self.puzzle_embeddings is not None and puzzle_ids is not None:
            # Get puzzle embeddings: [batch, puzzle_emb_ndim]
            puzzle_emb = self.puzzle_embeddings(puzzle_ids)

            # Pad to multiple of hidden_size if needed
            pad_count = self.puzzle_emb_len * self.hidden_size - self.puzzle_emb_ndim
            if pad_count > 0:
                # Pad along the last dimension
                puzzle_emb = mx.pad(puzzle_emb, [(0, 0), (0, pad_count)])

            # Reshape to [batch, puzzle_emb_len, hidden_size]
            batch_size = puzzle_emb.shape[0]
            puzzle_emb = puzzle_emb.reshape(batch_size, self.puzzle_emb_len, self.hidden_size)

            # Concatenate puzzle embeddings before token embeddings
            embeddings = mx.concatenate([puzzle_emb, embeddings], axis=1)

        # 3. Add position embeddings if using learned positions
        if self.position_embeddings is not None:
            # Get position embeddings for the full sequence length
            pos_emb = self.position_embeddings()
            seq_len = embeddings.shape[1]
            # Slice to match sequence length and add
            # Scale by 1/sqrt(2) to maintain forward variance (as in PyTorch version)
            embeddings = 0.707106781 * (embeddings + pos_emb[:seq_len, :])

        # 4. Scale embeddings by sqrt(hidden_size)
        return self.embed_scale * embeddings


# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================

"""
Implementation Priority:
1. trunc_normal_init - needed by all embeddings
2. MLXEmbedding - basic building block
3. MLXLearnedPositionEmbedding - simple extension
4. MLXSparseEmbedding - HRM-specific, more complex
5. MLXCombinedEmbeddings - orchestrates everything

MLX Advantages for Embeddings:
- No need for separate weight and buffer management
- Unified memory means no CPU<->GPU transfers
- Simpler gradient handling
- Auto-differentiable gather operations

Sparse Embedding Strategy:
Instead of PyTorch's complex "local weights" system:
1. Keep full embedding table
2. Use mx.gather for batch lookup
3. Custom gradient update via optimizer
4. Much simpler than CUDA version!
"""

