"""
MLX Dataset - Data loading for HRM training

Handles loading and batching of puzzle datasets:
- ARC (Abstraction and Reasoning Corpus)
- Sudoku puzzles
- Maze navigation

Replaces: puzzle_dataset.py from PyTorch version

Key differences:
- Uses mlx-data instead of torch DataLoader
- Simpler (no distributed training complexity)
- No worker processes needed (unified memory!)
"""

import mlx.core as mx
import mlx.data as dx
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# CONSTANTS
# ============================================================================

IGNORE_LABEL_ID = -100


# ============================================================================
# METADATA
# ============================================================================

@dataclass
class DatasetMetadata:
    """
    Metadata for a puzzle dataset.
    
    Contains information about:
    - Vocabulary size
    - Sequence length
    - Number of puzzles
    - Special token IDs
    """
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    pad_id: int
    blank_identifier_id: int
    ignore_label_id: Optional[int]
    
    # Dataset organization
    sets: List[str]  # e.g., ["train", "test"]
    total_groups: int
    mean_puzzle_examples: float


# ============================================================================
# DATASET CLASS
# ============================================================================

class MLXPuzzleDataset:
    """
    Puzzle dataset loader using mlx-data.
    
    Replaces: PuzzleDataset from PyTorch version
    
    Loads preprocessed puzzle data:
    - inputs.npy: Token sequences
    - labels.npy: Target sequences
    - puzzle_identifiers.npy: Which puzzle each example belongs to
    - puzzle_indices.npy: Start index of each puzzle
    - group_indices.npy: Grouping information
    
    Key simplifications from PyTorch version:
    - No rank/world_size (single device)
    - No worker processes (unified memory)
    - Simpler batching logic
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",  # "train" or "test"
        batch_size: int = 768,
        test_mode: bool = False,
        seed: int = 0
    ):
        # TODO: Implement initialization
        pass
    
    def _load_metadata(self) -> DatasetMetadata:
        """Load dataset metadata from JSON."""
        # TODO: Implement metadata loading
        pass
    
    def _load_data_files(self):
        """Load numpy arrays for this split."""
        # TODO: Implement data loading
        pass
    
    def _create_train_stream(self):
        """
        Create training data stream.
        
        Features:
        - Random shuffling of puzzle groups
        - Multiple epochs per iteration
        - Batch packing (multiple puzzles per batch)
        """
        # TODO: Implement training stream
        pass
    
    def _create_test_stream(self):
        """
        Create test data stream.
        
        Features:
        - Sequential processing
        - Complete puzzle evaluation
        - No shuffling
        """
        # TODO: Implement test stream
        pass
    
    def batch(self, num_batches: Optional[int] = None):
        """
        Iterate over batches.
        
        Args:
            num_batches: Optional limit on number of batches
            
        Yields:
            Tuples of (set_name, batch_dict, batch_size)
        """
        # TODO: Implement batch iteration
        pass


# ============================================================================
# BATCH UTILITIES
# ============================================================================

def collate_batch(
    examples: List[Dict[str, np.ndarray]],
    batch_size: int,
    pad_id: int,
    blank_identifier_id: int
) -> Dict[str, mx.array]:
    """
    Collate list of examples into a batch.
    
    Handles:
    - Stacking examples
    - Padding to batch_size
    - Converting to MLX arrays
    
    Args:
        examples: List of example dicts
        batch_size: Target batch size
        pad_id: Token to use for padding
        blank_identifier_id: Puzzle ID for padding
        
    Returns:
        Batch dictionary with MLX arrays
    """
    # TODO: Implement batch collation
    pass


def sample_batch_from_groups(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    inputs: np.ndarray,
    labels: np.ndarray,
    puzzle_identifiers: np.ndarray,
    start_index: int,
    batch_size: int
) -> Tuple[int, Dict[str, np.ndarray]]:
    """
    Sample a batch from puzzle groups (training mode).
    
    Process:
    1. Pick puzzle groups in order
    2. Sample examples from each group
    3. Pack into batch up to batch_size
    
    Args:
        rng: Random number generator
        group_order: Shuffled group indices
        puzzle_indices: Start index of each puzzle
        group_indices: Start puzzle of each group
        inputs: Input tokens
        labels: Label tokens
        puzzle_identifiers: Puzzle IDs
        start_index: Where to start in group_order
        batch_size: Target batch size
        
    Returns:
        Tuple of (new_start_index, batch_dict)
    """
    # TODO: Implement batch sampling
    pass


# ============================================================================
# DATA AUGMENTATION (Optional)
# ============================================================================

def augment_puzzle(
    inputs: mx.array,
    labels: mx.array,
    vocab_size: int
) -> Tuple[mx.array, mx.array]:
    """
    Apply data augmentation to puzzles.
    
    Possible augmentations:
    - Token remapping (permute vocabulary)
    - Grid rotations (for spatial puzzles)
    - Sequence reversal
    
    Note: HRM datasets are already augmented during preprocessing.
    This is for additional online augmentation if desired.
    
    Args:
        inputs: Input tokens
        labels: Label tokens
        vocab_size: Size of vocabulary
        
    Returns:
        Augmented (inputs, labels)
    """
    # TODO: Implement augmentation (optional)
    pass


# ============================================================================
# DATASET BUILDERS (Optional utilities)
# ============================================================================

def load_arc_dataset(data_dir: str) -> MLXPuzzleDataset:
    """Load ARC dataset."""
    # TODO: Implement
    pass


def load_sudoku_dataset(data_dir: str) -> MLXPuzzleDataset:
    """Load Sudoku dataset."""
    # TODO: Implement
    pass


def load_maze_dataset(data_dir: str) -> MLXPuzzleDataset:
    """Load Maze dataset."""
    # TODO: Implement
    pass


# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================

"""
Implementation Priority:
1. DatasetMetadata - data structure
2. _load_metadata - JSON parsing
3. _load_data_files - numpy loading
4. collate_batch - batching logic
5. _create_test_stream - simpler (sequential)
6. _create_train_stream - more complex (shuffling)
7. batch - main iteration method

MLX-Data Advantages:

PyTorch DataLoader:
- Multi-process workers
- Complex prefetching
- IPC overhead
- Manual memory pinning

MLX-Data:
- Single-process (unified memory!)
- Simpler pipeline
- Automatic batching
- Lazy loading

Example usage:
    ```python
    # Create dataset
    train_data = MLXPuzzleDataset(
        "data/arc-aug-1000",
        split="train",
        batch_size=768,
        test_mode=False
    )
    
    # Iterate
    for set_name, batch, effective_size in train_data.batch():
        # batch = {
        #     "inputs": [batch_size, seq_len],
        #     "labels": [batch_size, seq_len],
        #     "puzzle_identifiers": [batch_size]
        # }
        outputs = model(batch)
    ```

Key Simplifications:

1. No distributed:
   - No rank/world_size
   - No all-gather
   - No per-rank batching

2. No worker processes:
   - Unified memory = fast access
   - No IPC overhead
   - Simpler debugging

3. Cleaner API:
   - mlx-data uses functional pipeline
   - Compose transforms easily
   - Better for experimentation
"""

