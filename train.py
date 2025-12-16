import os
import time
import math
import pickle
import contextlib
import numpy as np
import torch
import tiktoken
from moht_gpt import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
out_dir = 'out'
eval_interval = 200
log_interval = 10
eval_iters = 20
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
# data
dataset = 'wikitext-2'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256 # context of up to 256 tokens
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False # do we use bias inside LayerNorm and Linear layers?
attention_type = 'baseline' # 'baseline', 'static_moas', 'moas'
load_balance_weight = 0.01 # weight for load balancing loss (only for 'moas')
# adamw optimizer
learning_rate = 1e-3 # max learning rate
max_iters = 2000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 2000 # should be ~= max_iters per Chinchilla
min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def get_data():
    # simple data loader for wikitext-2
    # we will download it if it doesn't exist
    data_dir = os.path.join('data', dataset)
    os.makedirs(data_dir, exist_ok=True)
    input_file_path = os.path.join(data_dir, 'input.txt')
    if not os.path.exists(input_file_path):
        import requests
        print("Downloading WikiText-2...")
        data_url = 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt'
        try:
            with open(input_file_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)
        except Exception as e:
            print(f"Failed to download data: {e}")
            print("Creating dummy data instead.")
            with open(input_file_path, 'w', encoding='utf-8') as f:
                f.write("Hello world " * 10000)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # tokenize (character level)
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
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
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Training Setup
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys} # will be useful for logging

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = contextlib.nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout, attention_type=attention_type)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    # TODO: implement resume
    pass

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model # unwrap DDP container if needed
running_mfu = -1.0

print(f"Training on {device}...")

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(local_iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if local_iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {local_iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if always_save_checkpoint:
            if local_iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': local_iter_num,
                    'best_val_loss': losses['val'],
                    'config': config_keys,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    if local_iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            
            # Add load balancing loss for MoAS
            if attention_type == 'moas':
                lb_loss = model.get_load_balancing_loss(X)
                loss = loss + load_balance_weight * lb_loss
            
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        # scaler.scale(loss).backward()
        loss.backward()
    
    # clip the gradient
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # step the optimizer and scaler if training in fp16
    optimizer.step()
    # scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if local_iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {local_iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    
    local_iter_num += 1

    # termination conditions
    if local_iter_num > max_iters:
        break

print("Training finished!")
