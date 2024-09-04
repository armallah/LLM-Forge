import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from model import GPTLanguageModel, encode, decode, device

def get_random_chunk(split, block_size, batch_size, Q=1):
    filename = f"openwebtext/output_{split}.txt"
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - Q*block_size*batch_size)
            mm.seek(start_pos)
            block = mm.read(Q*block_size*batch_size-1)
            decoded_block = block.decode("utf-8", errors="ignore").replace("\r", "")
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data

def get_batch(split, block_size, batch_size):
    data = get_random_chunk(split, block_size, batch_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, block_size, batch_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model, batch_size, block_size, max_iters, learning_rate, eval_iters):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_iters == 0:
            losses = estimate_loss(model, eval_iters, block_size, batch_size)
            print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
        xb, yb = get_batch('train', block_size, batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(f"Final loss: {loss.item()}")

def save_model(model, filename='model-01.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully")

def load_model(filename='model-01.pkl'):
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        return model.to(device)
    except:
        print("No model found, creating new model")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT-Language Model Training')
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()

    batch_size = int(args.batch_size)
    block_size = 128
    n_embd = 384
    n_layer = 2
    n_head = 2
    learning_rate = 3e-4
    max_iters = 200
    eval_iters = 100

    print(f"Using device: {device}")

    with open("openwebtext/vocab.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)

    model = load_model()
    if model is None:
        model = GPTLanguageModel(vocab_size).to(device)

    train(model, batch_size, block_size, max_iters, learning_rate, eval_iters)
    save_model(model)