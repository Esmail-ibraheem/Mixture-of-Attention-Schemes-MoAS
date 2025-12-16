# Mixture of Attention Schemes (MoAS) - Implementation Walkthrough

## Overview

This project implements a **Mixture of Attention Schemes (MoAS)** - a novel Transformer architecture that combines different KV-sharing schemes (MHA, GQA, MQA) with learned per-token routing.

### Research Question
> Can a learned mixture of attention schemes (MHA/GQA/MQA) achieve better quality-efficiency trade-offs than any single scheme?

---

## Architecture

### Three Attention Schemes

#### 1. **MHA (Multi-Head Attention)**
- `num_q_heads = num_kv_heads = H` (e.g., 6 heads)
- Best quality, largest KV cache, slowest decoding
- Standard Transformer attention

#### 2. **GQA (Grouped-Query Attention)**
- `num_q_heads = H, num_kv_heads = G` (e.g., 6 Q heads, 2 KV heads)
- Middle ground: fewer KV heads than queries
- Each KV head shared across multiple Q heads

#### 3. **MQA (Multi-Query Attention)**
- `num_q_heads = H, num_kv_heads = 1`
- Fastest decoding, smallest KV cache
- Single KV head shared across all Q heads

### Mixing Strategies

#### Static MoAS
```python
y = (O_MHA + O_GQA + O_MQA) / 3
```
Simple average of all three attention outputs.

#### Dynamic MoAS (Routed)
```python
# Per-token routing
r_i = W_2 · GELU(W_1 · x_i)  # Router logits
g_i = softmax(r_i)            # Gate weights [g_MHA, g_GQA, g_MQA]

# Weighted combination
y_i = g_i[MHA] * O_MHA[i] + g_i[GQA] * O_GQA[i] + g_i[MQA] * O_MQA[i]
```

**Load Balancing Loss**: Encourages balanced usage of all three schemes
```python
L_balance = MSE(avg_gates, [1/3, 1/3, 1/3])
L_total = L_task + λ * L_balance
```

---

## Implementation

### File Structure

```
TransformersFactory/
├── moht_components.py    # Attention scheme implementations
│   ├── MHAAttention
│   ├── GQAAttention
│   ├── MQAAttention
│   ├── StaticMoASAttention
│   └── MoASAttention (with router)
├── moht_gpt.py          # GPT model with MoAS support
├── train.py             # Training script
├── test_moas.py         # Unit tests
└── compare_moas.py      # Comparison training script
```

### Key Components

#### [moht_components.py](file:///f:/TransformersFactory/moht_components.py)

Implements all attention schemes:
- **MHAAttention**: Standard multi-head attention
- **GQAAttention**: Grouped-query with configurable `num_kv_heads`
- **MQAAttention**: Single shared KV head
- **StaticMoASAttention**: Averages all three
- **MoASAttention**: Learned routing with load balancing

#### [moht_gpt.py](file:///f:/TransformersFactory/moht_gpt.py)

GPT model supporting three attention types via config:
```python
config = GPTConfig(
    attention_type='moas',  # 'baseline', 'static_moas', 'moas'
    n_layer=6,
    n_head=6,
    n_embd=384,
    ...
)
```

---

## Test Results

### Unit Tests ✓

All tests passed successfully:

```
Testing Baseline MHA (baseline)
✓ Logits shape: torch.Size([2, 64, 100])
✓ Loss: 4.6175
✓ Gradients computed successfully

Testing Static MoAS (averaged) (static_moas)
✓ Logits shape: torch.Size([2, 64, 100])
✓ Loss: 4.7045
✓ Gradients computed successfully

Testing Dynamic MoAS (routed) (moas)
✓ Logits shape: torch.Size([2, 64, 100])
✓ Loss: 4.6731
✓ Load balancing loss: 0.000015
✓ Gradients computed successfully
```

### Model Statistics

| Attention Type | Parameters | Notes |
|---------------|-----------|-------|
| Baseline MHA | 1.68M | Standard attention |
| Static MoAS | 5.05M | 3x attention modules |
| Dynamic MoAS | 5.13M | 3x attention + router |

---

## Experimental Results

We ran a comparison experiment on WikiText-2 (500 iterations) training Baseline MHA, Static MoAS, and Dynamic MoAS.

| Model | Parameters | Final Val Loss | Notes |
|-------|------------|----------------|-------|
| Baseline MHA | 7.19M | **2.2940** | Best performance, standard architecture |
| Static MoAS | 10.14M | 2.3093 | Averaging attention schemes slightly hurts performance |
| Dynamic MoAS | 10.29M | 2.3074 | Outperforms Static MoAS, close to Baseline |

**Analysis**:
- **Baseline Dominance**: For this small scale and short training duration (500 steps), the standard MHA baseline performed best. This is expected as MHA has valid strong inductive biases and doesn't need to learn how to route.
- **Routing Helps**: Dynamic MoAS (learned routing) obtained a lower loss (2.3074) than Static MoAS (2.3093), suggesting the router is learning a non-trivial combination of schemes better than a simple average.
- **Efficiency Potential**: While Dynamic MoAS has more parameters due to having MHA+GQA+MQA parallel branches (in this implementation), in a real deployment, we would conditionally execute only top-k branches or use the routing to select a sparse expert, potentially saving massive compute for "easy" tokens (using MQA) while keeping MHA for "hard" tokens.

---

## Usage

### 1. Test Implementation

```bash
python test_moas.py
```

### 2. Train Baseline Model

```bash
# Edit train.py
attention_type = 'baseline'

python train.py
```

### 3. Train Static MoAS

```bash
# Edit train.py
attention_type = 'static_moas'

python train.py
```

### 4. Train Dynamic MoAS

```bash
# Edit train.py
attention_type = 'moas'
load_balance_weight = 0.01

python train.py
```

### 5. Compare All Variants

```bash
python compare_moas.py
```

This will train all three variants and save comparison results.

---

## Next Steps

### Immediate
1. ✓ Implement all attention schemes (MHA, GQA, MQA)
2. ✓ Implement static mixture
3. ✓ Implement router with load balancing
4. ✓ Create test suite
5. ⏳ Run comparison experiments

### Research Directions
1. **Routing Analysis**: Visualize which tokens prefer which schemes
2. **Efficiency Metrics**: Measure actual KV cache size and inference speed
3. **Ablations**:
   - Different `num_kv_heads` for GQA (1, 2, 3, 4)
   - Router architecture variations
   - Load balancing weight tuning
4. **Scaling**: Test on larger models and datasets

---

## Key Insights

### Why This Matters

1. **Quality-Efficiency Trade-off**: MHA is slow but accurate, MQA is fast but may lose quality. MoAS learns to use the right scheme for each token.

2. **Adaptive Computation**: Different tokens may benefit from different attention patterns. The router learns this automatically.

3. **Practical Impact**: KV cache size is a major bottleneck in LLM inference. MoAS could reduce cache size while maintaining quality.

### Expected Behavior

- **Easy tokens** (e.g., common words): Router should prefer MQA (fast)
- **Hard tokens** (e.g., rare words, complex context): Router should prefer MHA (accurate)
- **Middle ground**: GQA provides a balance

---

## Code Highlights

### Router Implementation

```python
class MoASAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Three attention branches
        self.mha = MHAAttention(config)
        self.gqa = GQAAttention(config, num_kv_heads=2)
        self.mqa = MQAAttention(config)
        
        # Router: 2-layer MLP
        self.router = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 3)
        )
    
    def forward(self, x):
        # Compute all attention outputs
        o_mha = self.mha(x)
        o_gqa = self.gqa(x)
        o_mqa = self.mqa(x)
        
        # Stack and route
        outputs = torch.stack([o_mha, o_gqa, o_mqa], dim=2)
        gates = F.softmax(self.router(x), dim=-1)
        
        # Mix per token
        y = (gates.unsqueeze(-1) * outputs).sum(dim=2)
        return y
```

---

## Conclusion

This implementation provides a complete framework for researching mixture-of-attention-schemes. All core components are tested and working. The next step is to run comprehensive experiments and analyze the routing behavior.

**Status**: ✓ Implementation complete, ready for experiments
