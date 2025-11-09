# PyTorch Reference Guide for MLX Implementation

This guide maps the original PyTorch HRM implementation to the new MLX structure.

---

## 📁 File Mapping

| Original PyTorch | MLX Equivalent | Notes |
|-----------------|----------------|-------|
| `models/layers.py` | `mlx_utils/layers.py` | Attention, SwiGLU, RoPE, RMS norm |
| `models/sparse_embedding.py` | `mlx_utils/embeddings.py` | Sparse puzzle embeddings |
| `models/common.py` | `mlx_utils/embeddings.py` | trunc_normal_init |
| `models/losses.py` | `mlx_utils/losses.py` | StableMax CE, ACT loss |
| `models/hrm/hrm_act_v1.py` | `mlx_utils/hrm_model.py` | Main HRM architecture |
| `pretrain.py` | `mlx_utils/training.py` | Training loop |
| `puzzle_dataset.py` | `mlx_utils/dataset.py` | Data loading |
| `adam_atan2` (external) | `mlx_utils/optimizers.py` | Adam optimizer |

---

## 🔍 Component-by-Component Reference

### 1. **Attention** (`models/layers.py:98-136`)

**Original (PyTorch):**
```python
from flash_attn import flash_attn_func

class Attention(nn.Module):
    def forward(self, cos_sin, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        query, key, value = split_qkv(qkv)
        query, key = apply_rotary_pos_emb(query, key, cos_sin)
        attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
        return self.o_proj(attn_output)
```

**MLX (`mlx_utils/layers.py`):**
```python
class MLXAttention(nn.Module):
    def __call__(self, hidden_states, cos_sin):
        # Same structure, replace flash_attn_func with:
        # scaled_dot_product_attention(q, k, v)
        pass
```

**Key differences:**
- No FlashAttention dependency
- Native MLX attention (Metal-optimized)
- Same numerical output expected

---

### 2. **RMS Norm** (`models/layers.py:151-158`)

**Original (PyTorch):**
```python
def rms_norm(hidden_states, variance_epsilon=1e-5):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
```

**MLX (`mlx_utils/layers.py`):**
```python
def rms_norm(hidden_states, variance_epsilon=1e-5):
    # CRITICAL: Must cast to float32 for precision (like PyTorch!)
    variance = mx.mean(hidden_states.astype(mx.float32) ** 2, axis=-1, keepdims=True)
    normalized = hidden_states.astype(mx.float32) * mx.rsqrt(variance + variance_epsilon)
    return normalized.astype(hidden_states.dtype)
```

**Validation:**
Use `reference_rms_norm()` in `mlx_utils/reference.py` to verify!

---

### 3. **StableMax Loss** (`models/losses.py:11-31`)

**Original (PyTorch):**
```python
def s(x, epsilon=1e-30):
    return torch.where(x<0, 1/(1-x+epsilon), x+1)

def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))

def stablemax_cross_entropy(logits, labels, ignore_index=-100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    # ... gather and return loss
```

**MLX (`mlx_utils/losses.py`):**
```python
def stablemax(x, epsilon=1e-30):
    return mx.where(x < 0, 1.0/(1.0-x+epsilon), x+1.0)

def log_stablemax(x, axis=-1):
    s_x = stablemax(x)
    return mx.log(s_x / mx.sum(s_x, axis=axis, keepdims=True))
```

**Critical:**
- Must match PyTorch exactly (numerical stability!)
- Use float64 for logprobs
- Validate with `reference_stablemax()` in `mlx_utils/reference.py`

---

### 4. **Sparse Puzzle Embeddings** (`models/sparse_embedding.py:11-39`)

**Original (PyTorch):**
```python
class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, batch_size, init_std, cast_to):
        self.weights = nn.Buffer(init(num_embeddings, embedding_dim))
        self.local_weights = nn.Buffer(zeros(batch_size, embedding_dim), requires_grad=True)
        self.local_ids = nn.Buffer(zeros(batch_size, dtype=int32))
    
    def forward(self, inputs):
        if self.training:
            with torch.no_grad():
                self.local_weights.copy_(self.weights[inputs])
                self.local_ids.copy_(inputs)
            return self.local_weights.to(self.cast_to)
        else:
            return self.weights[inputs].to(self.cast_to)
```

**MLX (`mlx_utils/embeddings.py`):**
```python
class MLXSparseEmbedding(nn.Module):
    # MUCH SIMPLER! No local_weights buffer needed
    def __init__(self, num_embeddings, embedding_dim, batch_size, init_std):
        self.weights = mx.zeros((num_embeddings, embedding_dim))
        # Unified memory = no need for separate local buffer!
    
    def __call__(self, inputs):
        return self.weights[inputs]  # That's it!
```

**Key difference:**
- MLX's unified memory eliminates complex buffer management
- Gradients flow naturally
- ~70% less code!

---

### 5. **SignSGD Optimizer** (`models/sparse_embedding.py:98-132`)

**Original (PyTorch):**
```python
def _sparse_emb_signsgd_dist(local_weights_grad, local_ids, weights, lr, weight_decay, world_size):
    # 1. All-gather from all GPUs
    all_weights_grad = dist.all_gather(local_weights_grad)
    all_ids = dist.all_gather(local_ids)
    
    # 2. Find unique IDs
    grad_ids, inv = all_ids.unique(return_inverse=True)
    
    # 3. Accumulate gradients
    grad = zeros(grad_ids.shape[0], D)
    grad.scatter_add_(0, inv.expand(-1, D), all_weights_grad)
    
    # 4. Update
    p = weights[grad_ids]
    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)
    weights[grad_ids] = p
```

**MLX (`mlx_utils/optimizers.py`):**
```python
class SignSGD_MLX(optim.Optimizer):
    def apply_single(self, gradient, parameter, state):
        # No distributed all-gather needed!
        # MLX already has accumulated gradients
        
        # Update: p = p * (1 - lr * wd) - lr * sign(grad)
        return parameter * (1 - self.lr * self.weight_decay) - self.lr * mx.sign(gradient)
```

**Key difference:**
- No distributed training complexity
- Single device = simpler logic
- Same algorithm, cleaner implementation

---

### 6. **HRM Hierarchical Loop** (`models/hrm/hrm_act_v1.py:188-213`)

**Original (PyTorch):**
```python
# CRITICAL: Gradient truncation for stable training
with torch.no_grad():
    z_H, z_L = carry.z_H, carry.z_L
    
    for _H_step in range(H_cycles):
        for _L_step in range(L_cycles):
            if not ((_H_step == H_cycles - 1) and (_L_step == L_cycles - 1)):
                z_L = self.L_level(z_L, z_H + input_embeddings)
        
        if not (_H_step == H_cycles - 1):
            z_H = self.H_level(z_H, z_L)

# Final step: WITH GRADIENTS
z_L = self.L_level(z_L, z_H + input_embeddings)
z_H = self.H_level(z_H, z_L)

new_carry = HRMInnerCarry(z_H.detach(), z_L.detach())
```

**MLX (`mlx_utils/hrm_model.py`):**
```python
# Same algorithm! Use mx.stop_gradient()
z_H, z_L = carry.z_H, carry.z_L

# Most iterations: no gradients
for h_step in range(H_cycles):
    for l_step in range(L_cycles):
        is_last_step = (h_step == H_cycles - 1) and (l_step == L_cycles - 1)
        
        if is_last_step:
            # Final step: WITH gradients
            z_L = self.L_level(z_L, z_H + input_embeddings)
        else:
            # Earlier steps: NO gradients
            z_L = mx.stop_gradient(self.L_level(z_L, z_H + input_embeddings))
    
    if h_step < H_cycles - 1:
        z_H = mx.stop_gradient(self.H_level(z_H, z_L))
    else:
        z_H = self.H_level(z_H, z_L)

new_carry = HRMInnerCarry(mx.stop_gradient(z_H), mx.stop_gradient(z_L))
```

**This is THE KEY to HRM's efficiency!**

---

## 📊 Hyperparameters Reference

From `config/cfg_pretrain.yaml`:

```yaml
Architecture:
  H_cycles: 2
  L_cycles: 2
  H_layers: 4
  L_layers: 4
  hidden_size: 512
  num_heads: 8
  expansion: 4.0
  
Training:
  global_batch_size: 768
  lr: 1e-4
  puzzle_emb_lr: 1e-2  # 100x higher!
  lr_warmup_steps: 2000
  lr_min_ratio: 1.0
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1
  
ACT:
  halt_max_steps: 16
  halt_exploration_prob: 0.1
```

All available in: `mlx_utils.reference.OriginalSpecs`

---

## 🧪 Numerical Validation

Always validate MLX implementations match PyTorch!

```python
from mlx_utils.reference import NumericalValidator

validator = NumericalValidator()

# Test RMS norm
validator.test_rms_norm(your_rms_norm_fn)

# Test StableMax
validator.test_stablemax(your_stablemax_fn)

# Print results
validator.print_summary()
```

Or manual comparison:

```python
from mlx_utils.reference import compare_outputs, print_comparison

mlx_output = mlx_model(batch)
pytorch_output = pytorch_model(batch)  # Run separately

result = compare_outputs(mlx_output, pytorch_output, "model_output")
print_comparison(result)
```

---

## ⚠️ Critical Implementation Notes

### 1. **RMS Norm Precision**
```python
# MUST cast to float32 for computation (like PyTorch!)
x_f32 = x.astype(mx.float32)
variance = mx.mean(x_f32 ** 2, axis=-1, keepdims=True)
normalized = x_f32 * mx.rsqrt(variance + eps)
return normalized.astype(x.dtype)
```

### 2. **StableMax Float64**
```python
# MUST use float64 for log computation
logits_f64 = logits.astype(mx.float64)
logprobs = log_stablemax(logits_f64, axis=-1)
```

### 3. **Gradient Truncation**
```python
# Most H/L updates: NO gradients
z = mx.stop_gradient(layer(z, context))

# Final update: WITH gradients
z = layer(z, context)
```

### 4. **Puzzle Embedding Init**
```python
# Zero initialization!
puzzle_emb = mx.zeros((num_puzzles, emb_dim))
```

### 5. **Q-Head Init**
```python
# Pessimistic initialization
q_head.weight = mx.zeros(...)
q_head.bias = mx.full(..., -5.0)  # Not 0!
```

---

## 📚 Quick Reference: Where to Find Things

| Need to know... | Look at... |
|----------------|------------|
| Exact architecture specs | `mlx_utils.reference.OriginalSpecs` |
| Original PyTorch code | Comments in each `mlx_utils/*.py` file |
| How to validate | `mlx_utils/reference.py` → NumericalValidator |
| Test data generation | `mlx_utils.reference.generate_test_batch()` |
| Loss formula | `mlx_utils/losses.py` docstring |
| Training hyperparams | `mlx_utils.reference.OriginalSpecs.TRAINING` |
| Optimizer settings | `mlx_utils.reference.OriginalSpecs.TRAINING` |

---

## 🎯 Implementation Checklist

For each component you implement:

- [ ] Read PyTorch reference (line numbers in docstring)
- [ ] Understand algorithm (not just transcribe!)
- [ ] Implement in MLX
- [ ] Write unit test
- [ ] Validate numerically against reference
- [ ] Check gradients flow correctly
- [ ] Verify performance (speed & memory)
- [ ] Document any deviations

---

## 💡 Tips for Implementation

1. **Start simple:** Implement reference versions first (exact PyTorch port)
2. **Validate early:** Test each function independently
3. **Compare outputs:** Use `compare_outputs()` liberally
4. **Check gradients:** Not just forward pass!
5. **Profile:** MLX might be faster in unexpected places
6. **Document:** Note any MLX-specific optimizations

---

**See also:**
- `mlx_utils/README.md` - Implementation plan
- `SETUP_MLX.md` - Environment details
- `SCAFFOLDING_COMPLETE.md` - Project status

---

**Ready to implement with confidence!** 🚀

Every MLX component has its PyTorch reference documented.
Use the validation utilities to ensure correctness.

