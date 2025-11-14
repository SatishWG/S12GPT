import os
import math
import torch
from myGPT import GPT, GPTConfig, DataLoaderLite
import time
import argparse

def train_gpt(restart: bool = False, ckpt_path: str = "final_gpt_model.pth", num_epochs: int = 100):
    # Set device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Initialize model with config
    config = GPTConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=12,  # Smaller model for faster training
        n_head=12,
        n_embd=768
    )
    model = GPT(config)
    model.to(device)

    # Initialize data loader
    train_loader = DataLoaderLite(B=4, T=32)  # Batch size=4, Sequence length=32

    # Learning rate schedule parameters
    max_lr = 6e-3   # 0.006
    min_lr = 3e-3   # 0.003
    steps_per_epoch = 100  # matches the inner loop in training
    total_steps = max(1, num_epochs * steps_per_epoch)
    warmup_steps = max(1, int(total_steps * 0.03))  # 3% warmup (at least 1 step)

    # Training parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

    # LambdaLR that implements: warmup from min_lr->max_lr then cosine anneal back to min_lr
    min_over_max = min_lr / max_lr
    def lr_lambda(step: int):
        if step < warmup_steps:
            # linear warmup of the multiplicative factor from min_over_max -> 1
            progress = step / float(warmup_steps)
            return min_over_max + (1.0 - min_over_max) * progress
        else:
            # cosine annealing from factor=1 down to min_over_max
            denom = float(max(1, total_steps - warmup_steps))
            progress = (step - warmup_steps) / denom
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_over_max + (1.0 - min_over_max) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # set scheduler to step 0 to initialize the optimizer lr to min_lr before training updates
    scheduler.step(0)

    best_loss = float('inf')
    start_time = time.time()

    # resume from checkpoint if available (unless restart is requested)
    start_epoch = 0
    if (not restart) and os.path.exists(ckpt_path):
        print(f"Found checkpoint '{ckpt_path}', loading...")
        ckpt = torch.load(ckpt_path, map_location=device)
        # support both a raw state_dict and a full checkpoint dict
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception:
                    print("Warning: failed to restore optimizer state (incompatible).")
            start_epoch = ckpt.get('epoch', 0) + 1
        else:
            # assume ckpt is just a state_dict
            model.load_state_dict(ckpt)
            start_epoch = 0
        print(f"Resuming training from epoch {start_epoch}")
    else:
        if restart and os.path.exists(ckpt_path):
            print(f"--restart specified, ignoring existing checkpoint '{ckpt_path}' and starting from scratch.")
        else:
            print("No checkpoint loaded, starting from scratch.")

    # Training loop
    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for i in range(steps_per_epoch):  # Process steps_per_epoch batches per epoch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits, loss = model(x, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Add gradient clipping
            optimizer.step()

            # advance scheduler (per optimization step)
            global_step += 1
            scheduler.step(global_step)

            total_loss += loss.item()
            num_batches += 1

            if i % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Batch {i:3d}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, "best_gpt_model.pth")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s!")
    print(f"Best loss achieved: {best_loss:.4f}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("--restart", action="store_true", help="Restart training from scratch and ignore final_gpt_model.pth if present")
    parser.add_argument("--ckpt", type=str, default="final_gpt_model.pth", help="Path to checkpoint file to resume from")
    parser.add_argument("--epochs", "-e", type=int, default=1000, help="Number of epochs to train")
    args = parser.parse_args()

    print("Starting GPT training...")
    try:
        model = train_gpt(restart=args.restart, ckpt_path=args.ckpt, num_epochs=args.epochs)
        torch.save(model.state_dict(), args.ckpt)
        print(f"Final model saved to {args.ckpt}")
        # generate_text(model, prompt="In a distant future", max_tokens=50)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")

if __name__ == "__main__":
    main()
