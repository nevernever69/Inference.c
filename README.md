# 🧠 LLM(GPT-2) Inference in C

A simple and efficient C-based inference engine.
### [this repo is only for educational purposes]

---

## ✅ Implemented 

- [x] Single-threaded inference
- [x] Key-Value (KV) cache for fast decoding
- [x] GPT-2 model support only (converted via `convert.py`)

---

## ⚙️ Setup & Build Instructions (via `make`)

### 🛠️ 1. Install System Dependencies

```bash
make setup
```
### 🛠️ 2. Download the model and convert it

```bash
make download_gpt2
```
### 🛠️ 2. Build the Project

```bash
make compile 
```
### 🛠️ 3. Run the Project

```bash
make run
```

