import torch
model = torch.load("pytorch_model.bin", map_location="cpu")
with open("model.bin", "wb") as f:
    for name, param in model.items():
        f.write(f"{name}\n".encode())
        dims = param.shape
        f.write(f"{len(dims)} {' '.join(map(str, dims))}\n".encode())
        f.write(param.detach().cpu().numpy().astype("float32").tobytes())

