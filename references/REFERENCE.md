# HuggingFace Technical Reference

Complete technical documentation for HuggingFace Hub CLI and Python libraries.

## CLI Command Reference

### Authentication Commands

| Command | Description | Example |
|---------|-------------|---------|
| `hf auth login` | Interactive login with browser or token | `hf auth login` |
| `hf auth logout` | Remove stored credentials | `hf auth logout` |
| `hf auth whoami` | Display current authenticated user | `hf auth whoami` |
| `hf auth token` | Print current access token | `hf auth token` |

```bash
# Login with token directly (non-interactive)
hf auth login --token hf_xxxxxxxxxxxxxxxxxxxx

# Login and add token to git credentials
hf auth login --add-to-git-credential
```

### Download Commands

| Command | Options | Description |
|---------|---------|-------------|
| `hf download <repo_id>` | `--local-dir`, `--revision`, `--include`, `--exclude` | Download model/dataset |

```bash
# Download entire repository
hf download gpt2

# Download to specific directory
hf download gpt2 --local-dir ./my-models/gpt2

# Download specific files only
hf download gpt2 config.json tokenizer.json

# Download specific revision/branch
hf download gpt2 --revision v1.0

# Download with file filters
hf download bigscience/bloom --include "*.json" --exclude "*.bin"

# Resume interrupted download
hf download meta-llama/Llama-2-7b-hf --resume

# Download dataset
hf download --repo-type dataset imdb
```

### Upload Commands

| Command | Options | Description |
|---------|---------|-------------|
| `hf upload <repo_id> <path>` | `--repo-type`, `--commit-message` | Upload files to Hub |

```bash
# Upload single file
hf upload my-username/my-model ./model.safetensors

# Upload directory
hf upload my-username/my-model ./output-dir/

# Upload to specific path in repo
hf upload my-username/my-model ./local-file.bin models/file.bin

# Upload with commit message
hf upload my-username/my-model ./output/ --commit-message "Add trained weights"

# Upload dataset
hf upload my-username/my-dataset ./data/ --repo-type dataset
```

### Repository Management

| Command | Options | Description |
|---------|---------|-------------|
| `hf repo create <name>` | `--type`, `--private`, `--organization` | Create new repository |
| `hf repo settings <repo_id>` | Various | Manage repository settings |

```bash
# Create public model repository
hf repo create my-bert-model --type model

# Create private repository
hf repo create my-private-model --type model --private

# Create in organization
hf repo create my-model --type model --organization my-org

# Create dataset repository
hf repo create my-dataset --type dataset

# Create Space
hf repo create my-app --type space --space-sdk gradio
```

### Cache Management

| Command | Description |
|---------|-------------|
| `hf cache ls` | List cached repositories |
| `hf cache rm <pattern>` | Remove cached files |

```bash
# List all cached models
hf cache ls

# Show cache size details
hf cache ls --verbose

# Remove specific model from cache
hf cache rm gpt2

# Remove by pattern
hf cache rm "facebook/*"

# Clear entire cache (use with caution)
hf cache rm --all
```

### Jobs Commands (Compute Workloads)

```bash
# List running jobs
hf jobs list

# Submit a training job
hf jobs run my-training-script.py --hardware gpu.a10g.small

# View job logs
hf jobs logs <job-id>

# Cancel a job
hf jobs cancel <job-id>
```

## Python Library Reference

### huggingface_hub Library

#### HfApi - Repository Operations

```python
from huggingface_hub import HfApi

api = HfApi()

# Create repository
api.create_repo(repo_id="my-username/my-model", repo_type="model", private=False)

# Upload file
api.upload_file(
    path_or_fileobj="./model.safetensors",
    path_in_repo="model.safetensors",
    repo_id="my-username/my-model",
    repo_type="model"
)

# Upload folder
api.upload_folder(
    folder_path="./output",
    repo_id="my-username/my-model",
    repo_type="model"
)

# Delete repository
api.delete_repo(repo_id="my-username/my-model", repo_type="model")

# List models
models = api.list_models(author="facebook", search="opt")
for model in models:
    print(model.id)

# Get model info
model_info = api.model_info("gpt2")
print(model_info.downloads)
```

#### hf_hub_download - File Downloads

```python
from huggingface_hub import hf_hub_download

# Download single file
file_path = hf_hub_download(
    repo_id="gpt2",
    filename="config.json"
)

# Download with revision
file_path = hf_hub_download(
    repo_id="gpt2",
    filename="pytorch_model.bin",
    revision="v1.0"
)

# Download to specific cache directory
file_path = hf_hub_download(
    repo_id="gpt2",
    filename="config.json",
    cache_dir="./my-cache"
)

# Force re-download
file_path = hf_hub_download(
    repo_id="gpt2",
    filename="config.json",
    force_download=True
)
```

#### InferenceClient - Serverless API

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="hf_xxxx")

# Text Generation
response = client.text_generation(
    "The answer to the universe is",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95
)

# Chat Completion
messages = [
    {"role": "user", "content": "What is machine learning?"}
]
response = client.chat_completion(
    messages=messages,
    model="mistralai/Mistral-7B-Instruct-v0.1",
    max_tokens=500
)
print(response.choices[0].message.content)

# Text Classification
result = client.text_classification("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}]

# Image Classification
result = client.image_classification("cat.jpg")

# Image Generation
image = client.text_to_image("A cat wearing a hat")
image.save("output.png")

# Embeddings
embeddings = client.feature_extraction("Hello world")
```

### transformers Library

#### Pipeline - High-Level API

| Task | Pipeline Name | Example Model |
|------|--------------|---------------|
| Text Generation | `text-generation` | gpt2, meta-llama/Llama-2-7b |
| Text Classification | `text-classification` | distilbert-base-uncased-finetuned-sst-2-english |
| Question Answering | `question-answering` | distilbert-base-cased-distilled-squad |
| Summarization | `summarization` | facebook/bart-large-cnn |
| Translation | `translation` | Helsinki-NLP/opus-mt-en-de |
| Fill Mask | `fill-mask` | bert-base-uncased |
| Image Classification | `image-classification` | google/vit-base-patch16-224 |
| Object Detection | `object-detection` | facebook/detr-resnet-50 |

```python
from transformers import pipeline

# Text Generation
generator = pipeline("text-generation", model="gpt2", device=0)  # device=0 for GPU
result = generator("Once upon a time", max_new_tokens=50, do_sample=True, temperature=0.7)

# Sentiment Analysis
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love HuggingFace!")  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Question Answering
qa = pipeline("question-answering")
result = qa(question="What is the capital of France?", context="France is a country. Paris is its capital.")

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
result = summarizer(long_text, max_length=130, min_length=30)

# Zero-shot Classification
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a tutorial about machine learning",
    candidate_labels=["education", "politics", "business"]
)
```

#### AutoModel Classes

| Class | Use Case |
|-------|----------|
| `AutoModel` | Base model without head |
| `AutoModelForCausalLM` | Text generation (GPT-style) |
| `AutoModelForSeq2SeqLM` | Encoder-decoder (T5, BART) |
| `AutoModelForSequenceClassification` | Text classification |
| `AutoModelForTokenClassification` | NER, POS tagging |
| `AutoModelForQuestionAnswering` | Extractive QA |
| `AutoModelForMaskedLM` | Fill-mask (BERT-style) |

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Generate text
inputs = tokenizer("The future of AI is", return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Loading Large Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load with automatic device mapping (shards across GPUs)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load with 4-bit quantization (reduced memory)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)
```

### datasets Library

```python
from datasets import load_dataset, Dataset

# Load from Hub
dataset = load_dataset("imdb")
print(dataset["train"][0])

# Load specific split
train_data = load_dataset("imdb", split="train")

# Load specific configuration
dataset = load_dataset("glue", "mrpc")

# Stream large datasets
dataset = load_dataset("c4", "en", streaming=True)
for example in dataset["train"]:
    print(example)
    break

# Create from pandas
import pandas as pd
df = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})
dataset = Dataset.from_pandas(df)

# Push dataset to Hub
dataset.push_to_hub("my-username/my-dataset")
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Authentication token | None |
| `HF_HOME` | Cache directory | `~/.cache/huggingface` |
| `HF_HUB_CACHE` | Hub cache subdirectory | `$HF_HOME/hub` |
| `HF_HUB_OFFLINE` | Offline mode (use cache only) | `0` |
| `HF_HUB_DISABLE_TELEMETRY` | Disable usage tracking | `0` |
| `TRANSFORMERS_CACHE` | Transformers cache (deprecated) | `$HF_HOME` |
| `HF_HUB_ENABLE_HF_TRANSFER` | Use hf_transfer for fast downloads | `0` |

```bash
# Set cache directory
export HF_HOME=/data/huggingface-cache

# Enable offline mode
export HF_HUB_OFFLINE=1

# Enable fast downloads (requires hf_transfer package)
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### Model Configuration Files

| File | Purpose |
|------|---------|
| `config.json` | Model architecture configuration |
| `tokenizer_config.json` | Tokenizer settings |
| `generation_config.json` | Default generation parameters |
| `pytorch_model.bin` | PyTorch weights (legacy) |
| `model.safetensors` | SafeTensors weights (recommended) |

## Detailed Examples

### Complete Fine-tuning Workflow

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# 1. Load dataset
dataset = load_dataset("imdb")

# 2. Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 5. Configure training
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_model_id="my-username/my-imdb-classifier"
)

# 6. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

# 7. Train and push
trainer.train()
trainer.push_to_hub()
```

### Batch Inference Processing

```python
from transformers import pipeline
import torch
from tqdm import tqdm

# Initialize pipeline with batching
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0 if torch.cuda.is_available() else -1,
    batch_size=8
)

# Process texts in batches
texts = [
    "The future of technology",
    "In a world where",
    "Scientists discovered",
    "The most important thing"
] * 100  # 400 texts

results = []
for i in tqdm(range(0, len(texts), 8)):
    batch = texts[i:i+8]
    outputs = generator(batch, max_new_tokens=50, do_sample=True)
    results.extend([out[0]["generated_text"] for out in outputs])

print(f"Processed {len(results)} texts")
```

### Custom Model Upload with Card

```python
from huggingface_hub import HfApi, ModelCard, ModelCardData

# Save model locally
model.save_pretrained("./my-model")
tokenizer.save_pretrained("./my-model")

# Create model card
card_data = ModelCardData(
    language="en",
    license="mit",
    library_name="transformers",
    tags=["text-classification", "sentiment-analysis"],
    datasets=["imdb"],
    metrics=[{"type": "accuracy", "value": 0.92}]
)

card = ModelCard.from_template(
    card_data,
    model_id="my-username/my-model",
    model_description="A fine-tuned model for sentiment analysis",
    training_procedure="Fine-tuned on IMDB dataset for 3 epochs"
)
card.save("./my-model/README.md")

# Upload everything
api = HfApi()
api.create_repo("my-username/my-model", exist_ok=True)
api.upload_folder(folder_path="./my-model", repo_id="my-username/my-model")
```

## Troubleshooting

### Authentication Failures

**Symptom**: `401 Unauthorized` or `Access denied` errors

**Cause**: Invalid or expired token, or missing permissions

**Solution**:
```bash
# Re-authenticate
hf auth logout
hf auth login

# Verify authentication
hf auth whoami

# Check token permissions at https://huggingface.co/settings/tokens
# Ensure "Read" permission for downloads, "Write" for uploads
```

### Gated Model Access Denied

**Symptom**: `Access to model is restricted` error

**Cause**: Model requires license acceptance

**Solution**:
1. Visit the model page on huggingface.co
2. Read and accept the license/terms
3. Wait a few minutes for access to propagate
4. Re-run your download command

### Download/Network Errors

**Symptom**: `ConnectionError`, `Timeout`, or incomplete downloads

**Cause**: Network issues or large file size

**Solution**:
```bash
# Resume interrupted download
hf download <repo_id> --resume

# Enable fast transfer (for large files)
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download <repo_id>

# Use Python with retry logic
from huggingface_hub import hf_hub_download
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def download_with_retry(repo_id, filename):
    return hf_hub_download(repo_id, filename)
```

### Out of Memory (OOM) Errors

**Symptom**: `CUDA out of memory` or `RuntimeError: CUDA error`

**Cause**: Model too large for GPU memory

**Solution**:
```python
# Use device_map for automatic sharding
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16  # Use half precision
)

# Or use quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)

# Reduce batch size in training
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # Reduce from default
    gradient_accumulation_steps=4   # Accumulate gradients
)
```

### Model Compatibility Issues

**Symptom**: `KeyError`, `AttributeError`, or unexpected model behavior

**Cause**: Version mismatch between model and library

**Solution**:
```bash
# Update transformers to latest
pip install --upgrade transformers

# Or install specific version from model card
pip install transformers==4.35.0

# Check model's required library version in config.json or README
```

### Slow Downloads

**Symptom**: Very slow download speeds

**Cause**: Default download method not optimized

**Solution**:
```bash
# Install and enable hf_transfer
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download specific files only (not entire repo)
hf download gpt2 pytorch_model.bin config.json tokenizer.json
```

### Cache Issues

**Symptom**: Disk full, or using wrong model version

**Cause**: Cache not managed properly

**Solution**:
```bash
# Check cache size
hf cache ls

# Remove specific models
hf cache rm old-model-name

# Clear all cache (careful!)
hf cache rm --all

# Use custom cache location
export HF_HOME=/path/with/more/space
```
