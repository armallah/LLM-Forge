import argparse
from src.training import train, save_model
from src.chatbot import chat_loop
from src.model import GPTLanguageModel, device, vocab_size, load_model

def main():
    parser = argparse.ArgumentParser(description='GPT Language Model')
    parser.add_argument('mode', choices=['train', 'chat'], help='Mode to run the model in')
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size for training')
    args = parser.parse_args()

    if args.mode == 'train':
        print("Training mode")
        batch_size = int(args.batch_size)
        block_size = 128
        n_embd = 384
        n_layer = 2
        n_head = 2
        
        learning_rate = 3e-4
        max_iters = 500
        eval_iters = 100

        print(f"Using device: {device}")

        model = load_model()
        if model is None:
            model = GPTLanguageModel(vocab_size).to(device)

        train(model, batch_size, block_size, max_iters, learning_rate, eval_iters)
        save_model(model)

    elif args.mode == 'chat':
        print("Chat mode")
        model = load_model()
        if model is None:
            print("Error: No model found. Please train the model first.")
        else:
            chat_loop(model)

if __name__ == "__main__":
    main()