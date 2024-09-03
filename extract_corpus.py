import os
import lzma
from tqdm import tqdm

def xz_files_in_dir(directory):
    files = []
    for file in os.listdir(directory):
        if file.endswith(".xz") and os.path.isfile(os.path.join(directory, file)):
            files.append(file)
    return files

folder_path =  "D:/Users/armal/LLM/openwebtext/openwebtext"
# output_fule = "output{}.txt"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file= "vocab.txt"
# split_files = int(input("How many files to split into?"))

files = xz_files_in_dir(folder_path)
total_files = len(files)

split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

# max_count = total_files // split_files if split_files != 0 else total_files

vocab = set()

with open(output_file_train, "w", encoding="utf-8") as output:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            output.write(text)
            charachters = set(text)
            vocab.update(charachters)

with open(output_file_val, "w", encoding="utf-8") as output:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            output.write(text)
            charachters = set(text)
            vocab.update(charachters)

with open(vocab_file, "w", encoding="utf-8") as vocab_output:
    for char in vocab:
        vocab_output.write(char)
        vocab_output.write("\n")