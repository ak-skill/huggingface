# HuggingFace Hub Skill for Claude Code

A comprehensive Claude Agent skill for seamless interaction with the HuggingFace ecosystem. This skill enables developers to authenticate, download/upload models and datasets, run inference, and fine-tune machine learning models directly through Claude Code.

## Overview

This skill provides first-class support for HuggingFace Hub workflows, making it easy to:

- **Authenticate** with HuggingFace Hub using CLI or environment tokens
- **Download** models and datasets from the Hub with smart caching
- **Upload** trained models and datasets to your HuggingFace repositories
- **Run Inference** using pipelines, AutoModel classes, or serverless APIs
- **Fine-tune** models with the Trainer API and push to Hub
- **Manage** repositories, cache, and compute jobs

## Installation

### Prerequisites

- Python 3.8 or higher
- Internet access for Hub operations
- (Optional) CUDA-compatible GPU for accelerated inference and training

### Quick Start

1. **Clone or download this skill** into your Claude Code skills directory
2. **Install required Python packages**:

```bash
pip install transformers huggingface_hub datasets torch
```

3. **Authenticate with HuggingFace Hub**:

```bash
hf auth login
```

The skill will be automatically activated when you mention keywords like `huggingface`, `hf`, `transformers`, `model hub`, or `fine-tune` in your conversation with Claude.

## Features

### 1. Authentication & Setup

Securely authenticate with HuggingFace Hub to access private and gated models:

```bash
# Interactive login with browser
hf auth login

# Or use environment variable
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Verify your authentication
hf auth whoami
```

### 2. Model & Dataset Management

Download and cache models efficiently:

```bash
# Download entire model repository
hf download gpt2 --local-dir ./models/gpt2

# Download specific files only
hf download meta-llama/Llama-2-7b-hf config.json tokenizer.json

# Download datasets
hf download --repo-type dataset imdb
```

Upload your trained models:

```bash
# Create repository
hf repo create my-username/my-model --type model

# Upload model files
hf upload my-username/my-model ./output-dir/
```

### 3. Inference Workflows

**Quick inference with pipelines**:

```python
from transformers import pipeline

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)
print(result[0]["generated_text"])

# Sentiment analysis
classifier = pipeline("text-classification")
result = classifier("I love HuggingFace!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

**Advanced control with AutoModel**:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

inputs = tokenizer("The future of AI is", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**Serverless API inference**:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="hf_xxxx")
response = client.text_generation(
    "Explain quantum computing:",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens=200
)
print(response)
```

### 4. Fine-tuning

Train models on custom datasets and push to Hub:

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
    hub_model_id="my-username/my-imdb-classifier"
)

# Train and push to Hub
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)
trainer.train()
trainer.push_to_hub()
```

### 5. Cache Management

Manage disk space and model versions:

```bash
# List cached models
hf cache ls

# Remove old models
hf cache rm facebook/opt-125m

# Clear entire cache
hf cache rm --all
```

## Skill Activation

This skill automatically activates when Claude detects HuggingFace-related keywords in your conversation:

- `huggingface`, `hugging face`, `hf`
- `transformers`, `pipeline`, `inference`
- `model hub`, `dataset hub`
- `AutoModel`, `AutoTokenizer`
- `fine-tune`, `deploy model`

No manual activation required - just start talking about HuggingFace workflows!

## Usage Examples

### Example 1: Download and Run GPT-2

```
User: "I want to download GPT-2 and generate some text"

Claude will:
1. Download the GPT-2 model using `hf download gpt2`
2. Create a Python script with pipeline API
3. Run inference and show results
```

### Example 2: Fine-tune BERT on Custom Dataset

```
User: "Help me fine-tune BERT on the IMDB dataset for sentiment analysis"

Claude will:
1. Load BERT model and IMDB dataset
2. Set up Trainer with appropriate arguments
3. Fine-tune the model
4. Push the trained model to HuggingFace Hub
```

### Example 3: Access Gated Model

```
User: "I need to use Llama 2 7B for text generation"

Claude will:
1. Guide you through authentication
2. Help accept the model license on HuggingFace
3. Download the model with proper credentials
4. Set up inference code
```

## Advanced Features

### Large Model Support

Handle models that don't fit in GPU memory:

```python
# Automatic device mapping and sharding
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16
)

# 4-bit quantization for reduced memory
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)
```

### Offline Mode

Work with cached models without internet:

```bash
export HF_HUB_OFFLINE=1
# Now use any cached models
```

### Fast Downloads

Speed up downloads for large models:

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download meta-llama/Llama-2-7b-hf
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Authentication token | None |
| `HF_HOME` | Cache directory | `~/.cache/huggingface` |
| `HF_HUB_OFFLINE` | Offline mode | `0` |
| `HF_HUB_ENABLE_HF_TRANSFER` | Fast downloads | `0` |

### Recommended Setup

```bash
# Set up your environment
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
export HF_HOME="/path/to/large/disk/huggingface-cache"

# Enable fast downloads
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## Documentation

- **[SKILL.md](SKILL.md)** - Complete skill instructions and guidelines
- **[REFERENCE.md](references/REFERENCE.md)** - Detailed CLI commands, Python APIs, and troubleshooting

## Common Use Cases

1. **Research & Experimentation**: Quickly download and test state-of-the-art models
2. **Production Deployment**: Download production-ready models with version pinning
3. **Model Training**: Fine-tune models on custom datasets and share with the community
4. **Dataset Management**: Access and upload datasets for machine learning projects
5. **API Integration**: Use serverless inference APIs for scalable applications

## Limitations

- Requires internet connection for downloading models and Hub operations
- Large models need significant disk space (some exceed 100GB)
- GPU inference requires CUDA-compatible hardware and drivers
- Some models require accepting license terms on the HuggingFace website
- Rate limits apply to serverless Inference API (use dedicated endpoints for production)

## Troubleshooting

### Authentication Issues

```bash
# Re-authenticate
hf auth logout
hf auth login

# Verify
hf auth whoami
```

### Gated Model Access

1. Visit the model page on huggingface.co
2. Read and accept the license
3. Wait a few minutes for access to propagate
4. Retry download

### Out of Memory Errors

Use device mapping and quantization:

```python
model = AutoModelForCausalLM.from_pretrained(
    "large-model",
    device_map="auto",
    load_in_4bit=True
)
```

### Slow Downloads

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

For more troubleshooting, see [REFERENCE.md](references/REFERENCE.md#troubleshooting).

## License

Apache-2.0

## Contributing

This skill is part of the Claude Code Agent Skills ecosystem. Contributions, bug reports, and feature requests are welcome!

## Support

- Check the [REFERENCE.md](references/REFERENCE.md) for detailed technical documentation
- Visit [HuggingFace Documentation](https://huggingface.co/docs) for library-specific help
- Report issues in the skill repository

## Version

1.0.0

---

**Made for Claude Code** - Empowering developers with seamless HuggingFace integration
