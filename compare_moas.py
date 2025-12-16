"""
Compare training runs for Baseline, Static MoAS, and Dynamic MoAS
"""
import os
import time
import math
import numpy as np
import torch
from moht_gpt import GPT, GPTConfig

# Configuration
out_dir = 'out_comparison'
os.makedirs(out_dir, exist_ok=True)

# Data
dataset = 'wikitext-2'
batch_size = 12
block_size = 256

# Model
n_layer = 4
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# Training
learning_rate = 3e-4
max_iters = 500
eval_interval = 50
eval_iters = 20
log_interval = 10

# MoAS specific
load_balance_weight = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load data
def get_data():
    data_dir = os.path.join('data', dataset)
    input_file_path = os.path.join(data_dir, 'input.txt')
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Character-level tokenization
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    
    train_ids = encode(data)
    n = len(train_ids)
    train_data = np.array(train_ids[:int(n*0.9)], dtype=np.uint16)
    val_data = np.array(train_ids[int(n*0.9):], dtype=np.uint16)
    return train_data, val_data, vocab_size

train_data, val_data, vocab_size = get_data()

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(attention_type, name):
    print(f"\n{'='*70}")
    print(f"Training {name} ({attention_type})")
    print('='*70)
    
    # Create model
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
        attention_type=attention_type
    )
    
    model = GPT(config)
    model.to(device)
    
    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=learning_rate,
        betas=(0.9, 0.99),
        device_type=device
    )
    
    # Training loop
    results = {
        'iters': [],
        'train_loss': [],
        'val_loss': []
    }
    
    X, Y = get_batch('train')
    t0 = time.time()
    
    for iter_num in range(max_iters + 1):
        # Evaluation
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            results['iters'].append(iter_num)
            results['train_loss'].append(losses['train'].item())
            results['val_loss'].append(losses['val'].item())
        
        if iter_num == max_iters:
            break
        
        # Forward
        logits, loss = model(X, Y)
        
        # Add load balancing loss for MoAS
        if attention_type == 'moas':
            lb_loss = model.get_load_balancing_loss(X)
            loss = loss + load_balance_weight * lb_loss
        
        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Fetch next batch
        X, Y = get_batch('train')
        
        # Logging
        if iter_num % log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            lossf = loss.item()
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    
    print(f"\n✓ {name} training completed!")
    return results

# Train all three variants
results_baseline = train_model('baseline', 'Baseline MHA')
results_static = train_model('static_moas', 'Static MoAS')
results_moas = train_model('moas', 'Dynamic MoAS (Routed)')

# Save results
import pickle
with open(os.path.join(out_dir, 'comparison_results.pkl'), 'wb') as f:
    pickle.dump({
        'baseline': results_baseline,
        'static_moas': results_static,
        'moas': results_moas
    }, f)

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Baseline MHA    - Final val loss: {results_baseline['val_loss'][-1]:.4f}")
print(f"Static MoAS     - Final val loss: {results_static['val_loss'][-1]:.4f}")
print(f"Dynamic MoAS    - Final val loss: {results_moas['val_loss'][-1]:.4f}")
print("="*70)
