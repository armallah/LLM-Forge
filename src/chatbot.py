import torch
from model import GPTLanguageModel, encode, decode, device, load_model

def generate_text(model, prompt, max_new_tokens=150):
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(model.generate(context.unsqueeze(0), max_new_tokens=max_new_tokens)[0].tolist())
    return generated_chars

def chat_loop(model):
    print("Using chatbot... ['quit' to exit].")
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'quit':
            break
        response = generate_text(model, prompt)
        print(f"Bot: {response}")

if __name__ == "__main__":
    model = load_model()
    if model is None:
        print("Error: No model found. Please train the model first.")
    else:
        chat_loop(model)