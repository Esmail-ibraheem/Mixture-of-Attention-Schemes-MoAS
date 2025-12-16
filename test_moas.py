"""
Test script for MoAS attention schemes
"""
import torch
from moht_gpt import GPT, GPTConfig

def test_attention_type(attention_type, name):
    print(f"\n{'='*60}")
    print(f"Testing {name} ({attention_type})")
    print('='*60)
    
    config = GPTConfig(
        block_size=256,
        vocab_size=100,
        n_layer=2,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=False,
        attention_type=attention_type
    )
    
    model = GPT(config)
    
    # Create dummy input
    batch_size = 2
    seq_len = 64
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    print("\nForward pass...")
    logits, loss = model(idx, targets)
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Loss: {loss.item():.4f}")
    
    # Test load balancing loss for MoAS
    if attention_type == 'moas':
        print("\nTesting load balancing loss...")
        lb_loss = model.get_load_balancing_loss(idx)
        print(f"✓ Load balancing loss: {lb_loss.item():.6f}")
    
    # Backward pass
    print("\nBackward pass...")
    loss.backward()
    print("✓ Gradients computed successfully")
    
    print(f"\n✓ {name} test passed!")

def main():
    print("Testing MoAS Implementation")
    print("="*60)
    
    # Test each attention type
    test_attention_type('baseline', 'Baseline MHA')
    test_attention_type('static_moas', 'Static MoAS (averaged)')
    test_attention_type('moas', 'Dynamic MoAS (routed)')
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)

if __name__ == '__main__':
    main()
