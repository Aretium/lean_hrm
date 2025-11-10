# Hierarchical Reasoning Model - MLX Implementation

![HRM Architecture](./assets/hrm.png)

**Work in Progress** - This is an MLX (Apple Silicon) implementation of the Hierarchical Reasoning Model, designed to work efficiently on systems with as little as 16GB of unified memory.

## Overview

This repository contains an ongoing MLX implementation of the [Hierarchical Reasoning Model (HRM)](https://arxiv.org/abs/2506.21734), a novel recurrent architecture that achieves exceptional performance on complex reasoning tasks with only 27 million parameters.

### Key Features

- **MLX Native**: Optimized for Apple Silicon using MLX framework
- **Memory Efficient**: Designed to run on 16GB systems through innovative memory optimizations
- **Full Model Capacity**: Maintains the full 27M parameter model architecture
- **Performance Optimized**: Includes various improvements to make training and inference faster

## Current Status

🚧 **Work in Progress** - This implementation is actively under development. The core architecture is being ported from PyTorch to MLX with memory-efficient optimizations.

### Planned Improvements

- Memory-efficient attention mechanisms (replacing FlashAttention with MLX-native alternatives)
- Gradient checkpointing strategies
- Activation quantization and compression
- Reversible/residual reversible architectures
- Optimized ACT (Adaptive Computation Time) halting
- Leveraging MLX unified memory architecture

## Installation

```bash
# Install MLX and dependencies
pip install -r requirements_mlx.txt
```

## Usage

*Coming soon* - Training and evaluation scripts are being developed.

## Original PyTorch Implementation

For the original PyTorch/CUDA implementation, see the parent [HRM repository](../HRM/).

## Citation

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

