# import torch
# model = torch.load("pytorch_model.bin", map_location="cpu")
# with open("model.bin", "wb") as f:
#     for name, param in model.items():
#         f.write(f"{name}\n".encode())
#         dims = param.shape
#         f.write(f"{len(dims)} {' '.join(map(str, dims))}\n".encode())
#         f.write(param.detach().cpu().numpy().astype("float32").tobytes())

import torch
import os
import shutil
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer

# Create Model directory if it doesn't exist
os.makedirs("Model/GPT2", exist_ok=True)

# Download GPT-2 model and tokenizer (using smallest variant for example)
model_name = "gpt2"  # You can change to "gpt2-medium", "gpt2-large", etc.
config = GPT2Config.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name, config=config)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Get the state dict
state_dict = model.state_dict()

# Save converted model to Model folder
output_path = os.path.join("Model/GPT2", "model.bin")
with open(output_path, "wb") as f:
    for name, param in state_dict.items():
        f.write(f"{name}\n".encode())
        dims = param.shape
        f.write(f"{len(dims)} {' '.join(map(str, dims))}\n".encode())
        f.write(param.detach().cpu().numpy().astype("float32").tobytes())

# Copy vocab.json and merges.txt to Model folder
temp_dir = "temp_tokenizer"
tokenizer.save_pretrained(temp_dir)  # Save tokenizer files temporarily
shutil.copy(os.path.join(temp_dir, "vocab.json"), os.path.join("Model", "vocab.json"))
shutil.copy(os.path.join(temp_dir, "merges.txt"), os.path.join("Model", "merges.txt"))

# Remove the temporary tokenizer directory
shutil.rmtree(temp_dir)

# Remove the cached GPT-2 model and tokenizer files
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        print(f"Removed cached model files from {cache_dir}")
    except Exception as e:
        print(f"Error removing cache: {e}")
else:
    print("No cache directory found")

print(f"Model converted and saved to {output_path}")
print(f"Tokenizer files copied to Model/vocab.json and Model/merges.txt")
