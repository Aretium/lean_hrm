# MLX-HRM Layer Tests

This directory contains comprehensive tests for all MLX layer implementations.

## Test Structure

- **test_basic_layers.py** - Tests for MLXLinear and rms_norm
- **test_rope.py** - Tests for Rotary Position Embeddings (RoPE)
- **test_swiglu.py** - Tests for SwiGLU activation function
- **test_attention.py** - Tests for Multi-Head Attention mechanism
- **test_transformer_block.py** - Tests for complete TransformerBlock
- **run_all_tests.py** - Master test runner

## Running Tests

### Run All Tests
```bash
cd tests
python run_all_tests.py
```

### Run Individual Tests
```bash
cd tests
python test_basic_layers.py
python test_rope.py
python test_swiglu.py
python test_attention.py
python test_transformer_block.py
```

## What's Tested

### Basic Layers
- MLXLinear initialization with truncated normal weights
- Bias initialization (zero init)
- Shape preservation
- RMS normalization correctness
- Numerical stability

### RoPE (Rotary Position Embeddings)
- Cos/sin cache generation
- Position-dependent embeddings
- Norm preservation during rotation
- Integration with attention

### SwiGLU
- Expansion factor calculation
- Gate-up projection splitting
- SiLU activation
- Gradient flow
- Various expansion configurations

### Attention
- QKV projection
- Scaled dot-product attention
- Causal masking
- RoPE integration
- Different sequence lengths
- Grouped-query attention support

### Transformer Block
- Post-norm architecture
- Residual connections
- Stacked blocks
- Causal variants
- Gradient flow
- Various configurations

## Test Coverage

All tests verify:
- ✓ Shape correctness
- ✓ Numerical stability
- ✓ Gradient flow
- ✓ Edge cases (zeros, small values)
- ✓ Integration between components
- ✓ PyTorch reference matching

## Expected Output

When all tests pass, you should see:

```
================================================================================
TEST SUMMARY
================================================================================
✅ PASSED     - Basic Layers (Linear, RMSNorm)
✅ PASSED     - Rotary Position Embeddings (RoPE)
✅ PASSED     - SwiGLU Activation
✅ PASSED     - Multi-Head Attention
✅ PASSED     - Transformer Block
================================================================================
Total: 5/5 test suites passed

🎉 ALL TEST SUITES PASSED! 🎉
```
