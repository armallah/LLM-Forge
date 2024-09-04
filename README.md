# Large Language Model

This project implements a simple GPT (Generative Pre-trained Transformer) language model for text generation. It includes functionality for training the model and using it in a chatbot-like interface.

## Project Structure

The project consists of four main Python files:

1. `main.py`: The entry point of the application, handling command-line arguments and orchestrating the training and chat modes.
2. `training.py`: Contains functions for training the model, including data loading and loss estimation.
3. `chatbot.py`: Implements the interactive chat loop for text generation using the trained model.
4. `model.py`: Defines the GPT language model architecture and utility functions for encoding/decoding text.

## Requirements

- Python 3.7+
- PyTorch
- tqdm (for progress bars during training)

You can install the required packages using pip:

```
pip install torch tqdm
```

## Usage

### Training the Model

To train the model, run:

```
python main.py train
```

You can optionally specify the batch size:

```
python main.py train -batch_size 64
```

### Chatting with the Model

To interact with the trained model in a chat-like interface, run:

```
python main.py chat
```

## Important Note

This repository does not include the training data or a pre-trained model due to size restrictions. To use this project, you'll need to provide:

1. Training data:
   - Place the training data in a directory named `openwebtext` in the project root.
   - Required files: `vocab.txt`, `output_train.txt`, and `output_val.txt`.

2. Pre-trained model (for chat mode):
   - After training, the model will be saved as `model-01.pkl` in the project root.
   - If you have a pre-trained model, place it in the project root with this filename.

Without these files, the project will not function as intended. Make sure to prepare your data and train the model before attempting to use the chat feature.

## Customization

You can adjust various hyperparameters in the `main.py` file, such as:

- `batch_size`
- `block_size`
- `n_embd` (embedding dimension)
- `n_layer` (number of transformer layers)
- `n_head` (number of attention heads)
- `learning_rate`
- `max_iters` (maximum training iterations)
- `eval_iters` (evaluation frequency)

## License

[MIT License](https://opensource.org/licenses/MIT)




