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
        seed: int = 0,
        epochs_per_iter: int = 1  # Batch X epochs in an iteration
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.batch_size = batch_size
        self.test_mode = test_mode
        self.seed = seed
        self.epochs_per_iter = epochs_per_iter

        # Load metadata
        self.metadata = self._load_metadata()

        # Load data files
        self._data = {}
        self._load_data_files()

        # Iteration counter (for reproducible shuffling)
        self._iters = 0

    def _load_metadata(self) -> DatasetMetadata:
        """Load dataset metadata from JSON."""
        metadata_path = os.path.join(self.dataset_path, self.split, "dataset.json")
        with open(metadata_path, "r") as f:
            data = json.load(f)

        return DatasetMetadata(
            vocab_size=data["vocab_size"],
            seq_len=data["seq_len"],
            num_puzzle_identifiers=data["num_puzzle_identifiers"],
            pad_id=data["pad_id"],
            blank_identifier_id=data["blank_identifier_id"],
            ignore_label_id=data.get("ignore_label_id"),
            sets=data["sets"],
            total_groups=data["total_groups"],
            mean_puzzle_examples=data["mean_puzzle_examples"]
        )

    def _load_data_files(self):
        """Load numpy arrays for this split."""
        # Field names and their memory mapping modes
        field_mmap_modes = {
            "inputs": "r",  # Memory-mapped for large files
            "labels": "r",  # Memory-mapped for large files
            # Keep indices in memory (small)
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data for each set (e.g., "train", "test")
        for set_name in self.metadata.sets:
            self._data[set_name] = {}
            for field_name, mmap_mode in field_mmap_modes.items():
                file_path = os.path.join(
                    self.dataset_path,
                    self.split,
                    f"{set_name}__{field_name}.npy"
                )
                self._data[set_name][field_name] = np.load(file_path, mmap_mode=mmap_mode)
    
    def _create_train_stream(self):
        """
        Create training data stream.

        Features:
        - Random shuffling of puzzle groups
        - Multiple epochs per iteration
        - Batch packing (multiple puzzles per batch)
        """
        for set_name, dataset in self._data.items():
            # Increment iteration counter for reproducible shuffling
            self._iters += 1

            # Create RNG for this iteration
            rng = np.random.Generator(np.random.Philox(seed=self.seed + self._iters))

            # Shuffle puzzle groups and repeat for multiple epochs
            num_groups = dataset["group_indices"].size - 1
            group_order = np.concatenate([
                rng.permutation(num_groups)
                for _ in range(self.epochs_per_iter)
            ])

            start_index = 0
            while start_index < group_order.size:
                # Sample a batch from puzzle groups
                start_index, batch_dict, effective_size = sample_batch_from_groups(
                    rng=rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    inputs=dataset["inputs"],
                    labels=dataset["labels"],
                    puzzle_identifiers=dataset["puzzle_identifiers"],
                    start_index=start_index,
                    batch_size=self.batch_size
                )

                # Drop last incomplete batch
                if effective_size < self.batch_size:
                    break

                # Collate and yield
                batch = collate_batch(
                    batch_dict,
                    self.batch_size,
                    self.metadata.pad_id,
                    self.metadata.blank_identifier_id,
                    self.metadata.ignore_label_id
                )

                yield set_name, batch, effective_size
    
    def _create_test_stream(self):
        """
        Create test data stream.

        Features:
        - Sequential processing
        - Complete puzzle evaluation
        - No shuffling
        """
        for set_name, dataset in self._data.items():
            total_examples = len(dataset["inputs"])

            # Process examples sequentially in batches
            start_index = 0
            while start_index < total_examples:
                end_index = min(total_examples, start_index + self.batch_size)

                # Get puzzle IDs for this batch
                # Need to find which puzzle each example belongs to
                puzzle_indices_list = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], start_index, side="right") - 1

                for i in range(start_index, end_index):
                    # Advance puzzle_index if we've moved to the next puzzle
                    while (puzzle_index + 1 < len(dataset["puzzle_indices"]) and
                           i >= dataset["puzzle_indices"][puzzle_index + 1]):
                        puzzle_index += 1
                    puzzle_indices_list.append(puzzle_index)

                # Create batch
                batch_dict = {
                    "inputs": dataset["inputs"][start_index:end_index],
                    "labels": dataset["labels"][start_index:end_index],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices_list]
                }

                # Collate and yield
                batch = collate_batch(
                    batch_dict,
                    self.batch_size,
                    self.metadata.pad_id,
                    self.metadata.blank_identifier_id,
                    self.metadata.ignore_label_id
                )

                yield set_name, batch, end_index - start_index

                # Advance to next batch
                start_index += self.batch_size
    
    def batch(self, num_batches: Optional[int] = None):
        """
        Iterate over batches.

        Args:
            num_batches: Optional limit on number of batches

        Yields:
            Tuples of (set_name, batch_dict, effective_batch_size)
        """
        # Choose stream based on mode
        if self.test_mode:
            stream = self._create_test_stream()
        else:
            stream = self._create_train_stream()

        # Iterate with optional limit
        count = 0
        for set_name, batch, effective_size in stream:
            yield set_name, batch, effective_size

            count += 1
            if num_batches is not None and count >= num_batches:
                break


# ============================================================================
# BATCH UTILITIES
# ============================================================================

def collate_batch(
    batch: Dict[str, np.ndarray],
    batch_size: int,
    pad_id: int,
    blank_identifier_id: int,
    ignore_label_id: Optional[int] = None
) -> Dict[str, mx.array]:
    """
    Collate numpy arrays into a batch.

    Handles:
    - Converting dtypes to int32
    - Replacing ignore labels
    - Padding to batch_size
    - Converting to MLX arrays

    Args:
        batch: Dictionary with numpy arrays
        batch_size: Target batch size
        pad_id: Token to use for padding
        blank_identifier_id: Puzzle ID for padding
        ignore_label_id: Optional ignore label to replace

    Returns:
        Batch dictionary with MLX arrays
    """
    # Convert dtype to int32
    batch = {k: v.astype(np.int32) for k, v in batch.items()}

    # Convert ignore label IDs
    if ignore_label_id is not None:
        batch["labels"][batch["labels"] == ignore_label_id] = IGNORE_LABEL_ID

    # Pad to batch_size if needed
    current_size = batch["puzzle_identifiers"].size
    if current_size < batch_size:
        pad_size = batch_size - current_size

        pad_values = {
            "inputs": pad_id,
            "labels": IGNORE_LABEL_ID,
            "puzzle_identifiers": blank_identifier_id
        }

        # Pad each field
        batch = {
            k: np.pad(
                v,
                ((0, pad_size),) + ((0, 0),) * (v.ndim - 1),
                constant_values=pad_values[k]
            )
            for k, v in batch.items()
        }

    # Convert to MLX arrays
    return {k: mx.array(v) for k, v in batch.items()}


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
) -> Tuple[int, Dict[str, np.ndarray], int]:
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
        Tuple of (new_start_index, batch_dict, effective_batch_size)
    """
    # Pack examples into a full batch
    batch_indices = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        # How many examples to take from this puzzle
        append_size = min(puzzle_size, batch_size - current_size)

        # Randomly sample examples from this puzzle (without replacement)
        sampled_indices = puzzle_start + rng.choice(puzzle_size, append_size, replace=False)

        # Add to batch
        batch_indices.append(sampled_indices)
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))

        current_size += append_size

    # Concatenate all indices
    batch_indices = np.concatenate(batch_indices)
    batch_puzzle_indices = np.concatenate(batch_puzzle_indices)

    # Create batch dictionary
    batch = {
        "inputs": inputs[batch_indices],
        "labels": labels[batch_indices],
        "puzzle_identifiers": puzzle_identifiers[batch_puzzle_indices]
    }

    return start_index, batch, current_size


# ============================================================================
# DATA AUGMENTATION (Optional - not implemented yet)
# ============================================================================

# Note: HRM datasets are already augmented during preprocessing using
# dihedral transforms. Additional online augmentation can be added here
# if needed in the future.


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

