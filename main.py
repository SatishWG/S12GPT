import torch
from myGPT import GPT, GPTConfig, DataLoaderLite
import time

def train_gpt():
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

    # Training parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    num_epochs = 50  # Increased from 5 to 10
    best_loss = float('inf')
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for i in range(100):  # Process 100 batches per epoch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits, loss = model(x, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Add gradient clipping
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if i % 10 == 0:
                print(f"Batch {i:3d}, Loss: {loss.item():.4f}")

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

# def generate_text(model, prompt="Once upon a time", max_tokens=50):
#     # STOP
#     num_return_sequences = 5
#     max_length = 30
#     torch.manual_seed(42)
#     torch.cuda.manual_seed(42)
#     model.eval()
#     while x.size(1) < max_length:
#     # forward the model to get the logits
#         with torch.no_grad():
#             # TODO: Implement text generation using the trained model
#             # This would require tokenizing the prompt and implementing the generation logic
#             # pass
#             logits = model(x)[0] # (B, T, vocab_size)
#             # take the logits at the last position
#             logits = logits[:, -1, :] # (B, vocab_size)
#             # get the probabilities
#             probs = F.softmax(logits, dim=-1)
#             # do top-k sampling of 50 (huggingface pipeline default)
#             # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#             topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#             # select a token from the top-k probabilities
#             # note: multinomial does not demand the input to sum to 1
#             ix = torch.multinomial(topk_probs, 1) # (B, 1)
#             # gather the corresponding indices
#             xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#             # append to the sequence
#             x = torch.cat((x, xcol), dim=1)

def main():
    print("Starting GPT training...")
    try:
        model = train_gpt()
        torch.save(model.state_dict(), "final_gpt_model.pth")
        print("Final model saved to final_gpt_model.pth")
        # generate_text(model, prompt="In a distant future", max_tokens=50)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")

if __name__ == "__main__":
    main()
