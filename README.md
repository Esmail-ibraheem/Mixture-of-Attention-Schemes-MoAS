# Mixture of Attention Schemes (MoAS) - README

## Overview

This repository implements **Mixture of Attention Schemes (MoAS)**, a novel Transformer architecture that combines different KV-sharing schemes (MHA, GQA, MQA) with learned per-token routing.

### Research Question
> Can a learned mixture of attention schemes (MHA/GQA/MQA) achieve better quality-efficiency trade-offs than any single scheme?

---

## Quick Start

### 1. Test the implementation
```bash
python test_moas.py
```

### 2. Train a model
```bash
# Edit train.py to set attention_type
python train.py
```

### 3. Compare all variants
```bash
python compare_moas.py
```

---

## Architecture

### Attention Schemes

1. **MHA (Multi-Head Attention)**: `num_q_heads = num_kv_heads = 6`
   - Best quality, largest KV cache

2. **GQA (Grouped-Query Attention)**: `num_q_heads = 6, num_kv_heads = 2`
   - Middle ground: balanced quality/efficiency

3. **MQA (Multi-Query Attention)**: `num_q_heads = 6, num_kv_heads = 1`
   - Fastest, smallest KV cache

### Mixing Strategies

- **Baseline**: Standard MHA only
- **Static MoAS**: Average of MHA + GQA + MQA
- **Dynamic MoAS**: Learned per-token routing

---

## Files

- `moht_components.py` - Attention scheme implementations
- `moht_gpt.py` - GPT model with MoAS support
- `train.py` - Training script
- `test_moas.py` - Unit tests
- `compare_moas.py` - Comparison experiments

---

## Configuration

Edit `train.py`:

```python
# Choose attention type
attention_type = 'moas'  # 'baseline', 'static_moas', 'moas'

# Model size
n_layer = 6
n_head = 6
n_embd = 384

# MoAS specific
load_balance_weight = 0.01  # For 'moas' only
```

---

## Results

All tests passing ✓

See [walkthrough.md](file:///C:/Users/Esmail/.gemini/antigravity/brain/2f4ebd9a-0cb7-4035-b336-a950e70e8112/walkthrough.md) for detailed results and analysis.

---

## Next Steps

1. Run comparison experiments
2. Analyze routing behavior
3. Measure KV cache efficiency
4. Scale to larger models

---

## Citation

If you use this code, please cite:

```
@misc{moas2025,
  title={Mixture of Attention Schemes: Learning to Route Between MHA, GQA, and MQA},
  author={Esmail Gumaan},
  year={2025}
}
```
