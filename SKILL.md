---
name: huggingface
description: |
  Helps developers interact with HuggingFace Hub for machine learning workflows.
  Supports authentication, model/dataset operations, inference, and fine-tuning.
  Activate with: huggingface, hugging face, hf, transformers, pipeline, inference,
  model hub, dataset hub, AutoModel, AutoTokenizer, fine-tune, deploy model
license: Apache-2.0
compatibility: Requires Python 3.8+, internet access for Hub operations
metadata:
  author: Claude Agent
  version: "1.0.0"
allowed-tools: Bash Read Write Edit WebFetch
---

# HuggingFace Hub Integration

Interact with the HuggingFace ecosystem for model management, inference, and training workflows.

## Overview

This skill helps you work with HuggingFace Hub - authenticate, download/upload models and datasets, run inference using pipelines or AutoModel patterns, and fine-tune models. It supports both CLI (`hf` command) and Python SDK approaches.

## Instructions

### 1. Authentication Setup

```bash
# Login interactively (recommended)
hf auth login

# Or set token via environment variable
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Verify authentication
hf auth whoami
```

### 2. Download Models or Datasets

```bash
# Download entire model repository
hf download meta-llama/Llama-2-7b-hf

# Download specific files
hf download gpt2 config.json pytorch_model.bin

# Download to specific directory
hf download gpt2 --local-dir ./models/gpt2
```

### 3. Run Inference

```python
# Quick inference with pipeline
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)
print(result[0]["generated_text"])
```

### 4. Upload Models to Hub

```bash
# Create a new repository
hf repo create my-fine-tuned-model --type model

# Upload files
hf upload my-username/my-fine-tuned-model ./output/
```

### 5. Fine-tune a Model

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load model and dataset
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
dataset = load_dataset("imdb")

# Configure training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    push_to_hub=True,
)

# Train
trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"])
trainer.train()
trainer.push_to_hub()
```

## Usage Examples

### Example 1: Authenticate and Download a Model

**Input**: "I need to download the Mistral-7B model from HuggingFace"

**Output**: First authenticate, then download the model:

```bash
# Authenticate (required for gated models like Mistral)
hf auth login

# Download the model
hf download mistralai/Mistral-7B-v0.1 --local-dir ./models/mistral-7b
```

### Example 2: Run Text Generation Inference

**Input**: "Run inference on GPT-2 to generate text"

**Output**: Use the pipeline API for simple inference:

```python
from transformers import pipeline

# Create generator pipeline
generator = pipeline("text-generation", model="gpt2", device="cpu")

# Generate text
outputs = generator(
    "The future of AI is",
    max_new_tokens=100,
    num_return_sequences=1,
    temperature=0.7
)

for output in outputs:
    print(output["generated_text"])
```

### Example 3: Upload a Fine-tuned Model

**Input**: "Upload my trained model to HuggingFace Hub"

**Output**: Create repository and upload:

```bash
# Create the repository
hf repo create my-bert-classifier --type model

# Upload all files from your output directory
hf upload your-username/my-bert-classifier ./fine-tuned-model/

# Or upload with Python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./fine-tuned-model",
    repo_id="your-username/my-bert-classifier",
    repo_type="model"
)
```

## Guidelines

- **Always authenticate first** for private or gated models (Llama, Mistral, etc.)
- **Use cache effectively**: Models are cached in `~/.cache/huggingface/hub` by default
- **Check model task type**: Match pipeline task to model (text-generation, text-classification, etc.)
- **Manage tokens securely**: Use `hf auth login` instead of hardcoding tokens in scripts
- **Specify device explicitly**: Use `device="cuda"` for GPU or `device="cpu"` for CPU inference
- **Use revision parameter**: Pin model versions with `revision="v1.0"` for reproducibility

## Common Patterns

### CLI Pattern: Quick Download and Cache Check

```bash
# Check what's in cache
hf cache ls

# Download model
hf download facebook/opt-350m

# Remove old cached models
hf cache rm facebook/opt-125m
```

### Python Pattern: AutoModel for Full Control

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Tokenize and generate
inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Python Pattern: InferenceClient for API Access

```python
from huggingface_hub import InferenceClient

# Use serverless inference API
client = InferenceClient(token="hf_xxxx")

# Text generation
response = client.text_generation(
    "Explain quantum computing:",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens=200
)
print(response)
```

## Edge Cases

- **Private/Gated Models**: Require authentication and accepting model license on HF website first
- **Large Models (>10GB)**: Use `hf download --resume` for interrupted downloads; consider quantized versions
- **GPU Memory Errors**: Use `device_map="auto"` for automatic model sharding across devices
- **Organization Repos**: Use format `org-name/model-name` for organization repositories
- **Offline Usage**: Set `HF_HUB_OFFLINE=1` to use only cached models

## References

For detailed technical reference, see:
- [REFERENCE.md](references/REFERENCE.md) - Complete CLI commands, Python APIs, and troubleshooting

## Limitations

- Requires internet connection for downloading models and Hub operations
- Large models need significant disk space (some exceed 100GB)
- GPU inference requires CUDA-compatible hardware and drivers
- Some models require accepting license terms on the HuggingFace website
- Rate limits apply to serverless Inference API (use dedicated endpoints for production)
