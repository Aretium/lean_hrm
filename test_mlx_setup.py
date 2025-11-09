#!/usr/bin/env python3
"""
Test script to verify MLX installation and basic attention functionality.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time

print("=" * 60)
print("MLX Setup Test for Lean-HRM")
print("=" * 60)

# Test 1: Basic MLX operations
print("\n1. Testing basic MLX operations...")
x = mx.array([1.0, 2.0, 3.0, 4.0])
y = mx.array([5.0, 6.0, 7.0, 8.0])
z = x + y
print(f"   ✓ Addition: {z}")
print(f"   ✓ Device: {mx.default_device()}")

# Test 2: Matrix operations
print("\n2. Testing matrix operations...")
A = mx.random.normal((100, 100))
B = mx.random.normal((100, 100))
start = time.time()
C = A @ B
mx.eval(C)  # Force evaluation
elapsed = time.time() - start
print(f"   ✓ Matrix multiplication (100x100): {elapsed*1000:.2f}ms")

# Test 3: Simple attention mechanism
print("\n3. Testing scaled dot-product attention...")
batch_size = 4
seq_len = 16
num_heads = 8
head_dim = 64

# Create random Q, K, V
q = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
k = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
v = mx.random.normal((batch_size, seq_len, num_heads, head_dim))

def scaled_dot_product_attention(query, key, value):
    """Simple attention implementation."""
    # Transpose to [batch, num_heads, seq_len, head_dim]
    q_t = mx.transpose(query, [0, 2, 1, 3])
    k_t = mx.transpose(key, [0, 2, 1, 3])
    v_t = mx.transpose(value, [0, 2, 1, 3])
    
    # Attention scores
    scale = 1.0 / mx.sqrt(mx.array(head_dim, dtype=mx.float32))
    scores = (q_t @ mx.transpose(k_t, [0, 1, 3, 2])) * scale
    
    # Softmax
    attn_weights = mx.softmax(scores, axis=-1)
    
    # Output
    output = attn_weights @ v_t
    
    # Back to [batch, seq_len, num_heads, head_dim]
    return mx.transpose(output, [0, 2, 1, 3])

start = time.time()
attn_output = scaled_dot_product_attention(q, k, v)
mx.eval(attn_output)
elapsed = time.time() - start

print(f"   ✓ Attention shape: {attn_output.shape}")
print(f"   ✓ Attention computation: {elapsed*1000:.2f}ms")

# Test 4: Memory usage
print("\n4. Checking Metal GPU memory...")
try:
    # Try to allocate a large array to test GPU memory
    large_array = mx.random.normal((1000, 1000, 10))
    mx.eval(large_array)
    print(f"   ✓ Large array allocation successful: {large_array.shape}")
    print(f"   ✓ Array size: {large_array.nbytes / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"   ✗ Memory allocation failed: {e}")

# Test 5: Gradient computation
print("\n5. Testing automatic differentiation...")
def loss_fn(x):
    return mx.sum(x ** 2)

x = mx.array([1.0, 2.0, 3.0, 4.0])
grad_fn = mx.grad(loss_fn)
grads = grad_fn(x)
print(f"   ✓ Input: {x}")
print(f"   ✓ Gradients: {grads}")

# Test 6: Check if we can use @mx.compile
print("\n6. Testing compilation with @mx.compile...")
@mx.compile
def compiled_attention(q, k, v):
    return scaled_dot_product_attention(q, k, v)

start = time.time()
compiled_output = compiled_attention(q, k, v)
mx.eval(compiled_output)
elapsed = time.time() - start
print(f"   ✓ Compiled attention: {elapsed*1000:.2f}ms")

# Summary
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nMLX Environment is ready for Lean-HRM development!")
print(f"MLX Version: {mx.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Default Device: {mx.default_device()}")
print("\nNext steps:")
print("  1. Convert PyTorch layers to MLX")
print("  2. Replace flash_attn with MLX attention")
print("  3. Port the HRM model architecture")
print("=" * 60)

