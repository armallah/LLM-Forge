import os
import lzma
from tqdm import tqdm

def xz_files_in_dir(directory):
    files = []
    for file in os.listdir(directory):
        if file.endswith(".xz") and os.path.isfile(os.path.join(directory, file)):
            files.append(file)
    return files

# Get the current directory
current_dir = os.getcwd()

# Create 'openwebtext' directory in the current directory if it doesn't exist
output_dir = os.path.join(current_dir, "openwebtext")
os.makedirs(output_dir, exist_ok=True)

folder_path = "D:/Users/armal/LLM/openwebtext/openwebtext"  # Keep the original input path
output_file_train = os.path.join(output_dir, "output_train.txt")
output_file_val = os.path.join(output_dir, "output_val.txt")
vocab_file = os.path.join(output_dir, "vocab.txt")

files = xz_files_in_dir(folder_path)
total_files = len(files)

split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()

def process_files(file_list, output_file):
    with open(output_file, "w", encoding="utf-8") as output:
        for filename in tqdm(file_list, total=len(file_list)):
            file_path = os.path.join(folder_path, filename)
            with lzma.open(file_path, "rt", encoding="utf-8") as infile:
                text = infile.read()
                output.write(text)
                characters = set(text)
                vocab.update(characters)

print("Processing training files...")
process_files(files_train, output_file_train)

print("Processing validation files...")
process_files(files_val, output_file_val)

print("Writing vocabulary file...")
with open(vocab_file, "w", encoding="utf-8") as vocab_output:
    for char in sorted(vocab):
        vocab_output.write(char + "\n")

print(f"Processing complete. Output files are in: {output_dir}")
print(f"Created files: {os.listdir(output_dir)}")
