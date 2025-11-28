# Fine-Tuning vs. From-Scratch Training for Text Classification

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Prepare dataset (AG News - 4-class news classification)
uv run python scripts/prepare_data.py --dataset ag_news

# 3. Train from-scratch model (recommended: run in background on GPU)
uv run python scripts/train_from_scratch.py --dataset ag_news --model-size medium --epochs 50

# 4. Fine-tune pre-trained model (GPT-2 with LoRA)
uv run python scripts/train_fine_tune.py --dataset ag_news --model gpt2 --use-lora --epochs 10

# 5. Evaluate and compare models (to be implemented)
# uv run python scripts/compare_models.py
```

---

## **Course Task: Fine-Tuning vs. From-Scratch Training for Text Classification**

### **Objective**
The goal of this assignment is to compare two approaches to training decoder-only language models for **full-text classification**:

1. **From-scratch training**  
   - A small decoder-only model created, initialized, and trained entirely by the student.  
   - Designed to avoid overfitting by limiting parameter count.  

2. **Fine-tuning a pre-trained model**  
   - Select a suitable pre-trained decoder-only model from Hugging Face.  
   - Fine-tune it on the same downstream dataset.  

Students should analyze the differences in training behavior, generalization, stability, and performance.

---

### **Dataset Selection**
Choose **one text classification dataset** from Hugging Face with at least a few thousand examples.

**Recommended datasets** (tested and working):

- **PolyAI/banking77** - 77 intent classes, financial domain, ~10K samples (RECOMMENDED - challenging multi-class)
- **stanfordnlp/imdb** - Binary sentiment, movie reviews, 25K train/test (long texts, nuanced)
- **yelp_polarity** - Binary sentiment, Yelp reviews, 560K train (large dataset)
- **ag_news** - 4 news categories, 120K train (simple, may be too easy)

**Other options:**
- Tweet Eval (sentiment or hate speech)
- Toxic Comment Classification
- [Polish youth slang](https://huggingface.co/datasets/jziebura/polish_youth_slang_classification)

Important requirements:
- The dataset must involve **full text classification** (not token classification).
- Choose a dataset with enough complexity to show the value of fine-tuning vs from-scratch.
- The dataset must be split into **train/validation/test** (use provided splits or create them).

---

### **Models to Train**

#### **1. From-Scratch Model**
- A **decoder-only architecture** (e.g., small GPT-like model). NOTE FOR LLM: The transformer in the decoder-only-transformer is huge (50M params), it needs to be changed to make it 0.5M-10M.
- The model must be **significantly smaller** than the fine-tuned model to avoid overfitting.  
  - Typical size: 0.5M–10M parameters.  
  - Shallow depth, small embedding dimension.  
- Add a **classification head**:
  - Either take the final hidden state of the last token and feed it into an MLP classifier, **or**
  - Use mean/max pooling over hidden states followed by an MLP classifier.
- Training considerations:
  - Use a tokenizer of your choice (trained or pre-trained).  
  - Limit sequence length based on your GPU memory.  
  - Expect the from-scratch model to require longer training and achieve weaker results.

#### **2. Fine-Tuned Pre-trained Model**
- Select a **decoder-only model available on Hugging Face** (e.g., GPT-2, GPT-Neo, GPT-J, Pythia, Phi-1.5 or any smaller model depending on your GPU).  
- Attach a **classification head**:
  - Add an MLP on top of the final transformer block.  
  - Or use an existing Hugging Face classification architecture if compatible.
- Fine-tune the entire model or choose a technique such as:
  - LoRA  
  - Freeze-then-unfreeze  
  - Adapter modules  
- Expect this model to train **much faster**, converge better, and achieve higher performance.

---

### **Evaluation Metrics**

Each model must be evaluated using:
- **Classification accuracy**  
- **F1 score** (macro or weighted)  
- **Training time** (total and per epoch)  
- **Inference time** on the test set  
- **Model size** (parameter count)

You should also:
- Inspect whether the from-scratch model overfits (validation curves).  
- Discuss stability and convergence speed.

---

### **General Plan for the Experiments**

#### **1. Dataset Preparation & Verification**
- Load dataset splits and inspect several examples.  
- Verify class distribution; apply stratification if generating custom splits.  
- Clean data if needed (remove empty texts, fix encoding issues).  
- Tokenize the corpus:
  - For the fine-tuned model: use its pre-trained tokenizer.  
  - For the from-scratch model: you may reuse a previous tokenizer or train a small one.  
- Set up a reasonable maximum sequence length (e.g., 128–512 tokens).

#### **2. Running the From-Scratch Experiment**
- Initialize the small decoder-only model.  
- Add a classification head.  
- Train with:
  - AdamW optimizer  
  - Learning rate warmup  
  - Gradient clipping  
  - Regularization (dropout, smaller model size) to reduce overfitting  
- Save model checkpoints and record training curves.  

#### **3. Running the Fine-Tuning Experiment**
- Load the selected pre-trained model and corresponding tokenizer.  
- Attach a classification head or use an existing one (e.g., GPT2ForSequenceClassification).  
- Fine-tune:
  - Much lower learning rate than from-scratch  
  - Fewer epochs  
  - Monitor validation accuracy/f1 regularly  
- Compare training curves with the from-scratch run.

#### **4. Running Evaluation**
- Compute performance metrics on the held-out test set.  
- Measure inference time (e.g., classify 1,000 examples).  
- Analyze training logs and learning curves.

---

### **Deliverables**

#### **1. Code**
Provide:
- Model definitions for both approaches  
- Training scripts  
- Evaluation scripts  
- Tokenization/configuration code  

#### **2. Report (4–6 pages)**
Your report must include:

##### **Model Descriptions**
- Size, parameter count, architecture outline  
- Description of the classification head  
- Tokenizer type and vocabulary size  

##### **Dataset Description**
- Basic statistics  
- Examples  
- Preprocessing steps  

##### **Training & Evaluation**
- Training curves for both models  
- Classification metrics (accuracy, f1)  
- Training and inference time  
- Observations of convergence, stability, overfitting  

##### **Comparative Analysis**
Discuss:
- Why fine-tuning outperforms from-scratch  
- When training from scratch might be preferred  
- Sensitivity to model size and dataset size  
- Practical insights from implementation

--- 

### Literature

* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805), Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova
* [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever

---

### **Tips & Hints**

- **Expect the from-scratch model to struggle**, unless it is very small and the task is easy.  
- **Do not oversize the from-scratch model** — it will overfit immediately.  
- **The fine-tuned model may require careful LR scheduling** to avoid catastrophic forgetting.  
- Try using **gradient accumulation** if GPU memory is limited.  
- Use **early stopping** for both models to get clean comparisons.  
- If models diverge:
  - Lower the learning rate  
  - Reduce sequence length  
  - Increase batch size (or accumulation)  

---

### **Summary**
- Select a moderate-sized classification dataset.
- Train **two decoder-only models**:
  1. A **small from-scratch model**
  2. A **fine-tuned pre-trained model**
- Compare performance, training behavior, parameter count, and computational efficiency.
- Submit code and a detailed comparative report.

---

## Quick Start (Banking77 Dataset)

If you want to get started quickly with the recommended Banking77 dataset:

```bash
# 1. Prepare data
uv run python scripts/prepare_data.py --dataset PolyAI/banking77

# 2. Train from-scratch model (tiny size to avoid overfitting)
uv run python scripts/train_from_scratch.py \
    --dataset banking77 \
    --model-size tiny \
    --epochs 50 \
    --early-stopping-patience 10 \
    --max-length 64

# 3. Fine-tune GPT-2 with LoRA
uv run python scripts/train_fine_tune.py \
    --dataset banking77 \
    --model gpt2 \
    --use-lora \
    --epochs 10 \
    --max-length 64

# 4. Compare results in results/ folder
```

**Why Banking77?**
- 77 intent classes (challenging multi-class classification)
- Financial domain with nuanced, similar intents
- Short texts (typical user queries)
- Better demonstrates the value of pre-training vs from-scratch
- ~10K training samples (right size for comparing approaches)

---

## Implementation Guide

### 1. Data Preparation

**Prepare a classification dataset:**

```bash
# Banking77 (77-class intent classification) - RECOMMENDED FOR CHALLENGING COMPARISON
uv run python scripts/prepare_data.py --dataset PolyAI/banking77

# AG News (4-class news: World, Sports, Business, Sci/Tech)
uv run python scripts/prepare_data.py --dataset ag_news

# IMDB (binary sentiment)
uv run python scripts/prepare_data.py --dataset stanfordnlp/imdb

# Yelp Polarity (binary sentiment)
uv run python scripts/prepare_data.py --dataset yelp_polarity

# Custom output directory
uv run python scripts/prepare_data.py --dataset PolyAI/banking77 --output data/processed/my_dataset
```

**What this does:**
- Downloads dataset from Hugging Face
- Creates train/val/test splits (90%/10% from train, uses existing test)
- Saves as JSONL format (one JSON per line)
- Generates metadata.json with class names and statistics

**Output structure:**
```
data/processed/ag_news/
├── train.jsonl       # Training data
├── val.jsonl         # Validation data
├── test.jsonl        # Test data
└── metadata.json     # Dataset metadata
```

### 2. From-Scratch Training

**Train a small transformer from random initialization:**

```bash
# Basic training with Banking77 (recommended for challenging comparison)
uv run python scripts/train_from_scratch.py --dataset banking77 --model-size tiny

# Train different model sizes
uv run python scripts/train_from_scratch.py --dataset banking77 --model-size tiny    # ~1.7M params
uv run python scripts/train_from_scratch.py --dataset banking77 --model-size small   # ~3.3M params
uv run python scripts/train_from_scratch.py --dataset banking77 --model-size medium  # ~4.9M params (default)
uv run python scripts/train_from_scratch.py --dataset banking77 --model-size large   # ~8.1M params

# AG News (if using AG News instead)
uv run python scripts/train_from_scratch.py --dataset ag_news --model-size tiny

# Custom hyperparameters
uv run python scripts/train_from_scratch.py \
    --dataset banking77 \
    --model-size tiny \
    --epochs 100 \
    --batch-size 64 \
    --lr 5e-4 \
    --max-length 64

# Enable early stopping (RECOMMENDED - stops if no improvement for N epochs)
uv run python scripts/train_from_scratch.py \
    --dataset banking77 \
    --model-size tiny \
    --epochs 50 \
    --early-stopping-patience 10

# Resume from checkpoint
uv run python scripts/train_from_scratch.py \
    --dataset banking77 \
    --resume checkpoints/from_scratch_banking77_tiny/checkpoint_epoch_50.pt
```

**Training features:**
- Auto-detects GPU (CUDA/MPS) and optimizes batch size
- Mixed precision training (AMP) on CUDA
- OneCycleLR scheduler with 10% warmup
- Gradient clipping for stability
- **Early stopping** to prevent overfitting (monitors validation F1)
- Saves best model (based on validation F1)
- Saves checkpoints every 10 epochs
- Generates training curves plot
- Tracks: loss, accuracy, F1 (macro/weighted)

**Running in background (recommended for long training):**

```bash
# Start training in background
nohup uv run python scripts/train_from_scratch.py \
    --dataset ag_news \
    --model-size medium \
    --epochs 50 > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Or use tmux/screen for better control
tmux new -s training
uv run python scripts/train_from_scratch.py --dataset ag_news --model-size medium --epochs 50
# Ctrl+B, then D to detach
# tmux attach -t training  # to reattach
```

**Output files:**
```
checkpoints/from_scratch_ag_news_medium/
├── best_model.pt                 # Best model (highest validation F1)
├── checkpoint_epoch_10.pt        # Periodic checkpoints
├── checkpoint_epoch_20.pt
└── ...

results/from_scratch_ag_news_medium/
├── training_history.json         # All metrics per epoch
├── training_curves.png           # Loss, accuracy, F1, LR plots
└── final_results.json            # Final test set performance
```

### 3. Fine-Tuning Pre-Trained Models

**Train a pre-trained model:**

```bash
# Basic fine-tuning with Banking77 (recommended)
uv run python scripts/train_fine_tune.py --dataset banking77 --model gpt2 --use-lora

# With LoRA (efficient fine-tuning, only ~0.8M trainable params - RECOMMENDED)
uv run python scripts/train_fine_tune.py --dataset banking77 --model gpt2 --use-lora

# Full fine-tuning (all 124M params trainable)
uv run python scripts/train_fine_tune.py --dataset banking77 --model gpt2

# Freeze base model (only train classification head, ~77K trainable params for 77 classes)
uv run python scripts/train_fine_tune.py --dataset banking77 --model gpt2 --freeze-base

# AG News (if using AG News instead)
uv run python scripts/train_fine_tune.py --dataset ag_news --model gpt2 --use-lora

# Different pre-trained models
uv run python scripts/train_fine_tune.py --dataset banking77 --model gpt2-medium --use-lora         # 355M params
uv run python scripts/train_fine_tune.py --dataset banking77 --model EleutherAI/gpt-neo-125m --use-lora  # 125M params
uv run python scripts/train_fine_tune.py --dataset banking77 --model openai-community/gpt2-large --use-lora  # 774M params

# Custom hyperparameters
uv run python scripts/train_fine_tune.py \
    --dataset banking77 \
    --model gpt2 \
    --epochs 10 \
    --batch-size 32 \
    --lr 2e-5 \
    --max-length 64

# LoRA with custom parameters
uv run python scripts/train_fine_tune.py \
    --dataset banking77 \
    --model gpt2 \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32
```

**Training features:**
- Auto-detects and uses model's pre-trained tokenizer
- Supports full fine-tuning, LoRA, or frozen base
- Lower learning rate (2e-5) for stable fine-tuning
- Fewer epochs (10) than from-scratch
- Same evaluation metrics as from-scratch
- Saves best model and checkpoints

**Running in background:**
```bash
nohup uv run python scripts/train_fine_tune.py \
    --dataset ag_news \
    --model gpt2 \
    --use-lora > fine_tuning.log 2>&1 &

tail -f fine_tuning.log
```

**Output files:**
```
checkpoints/fine_tuned_ag_news_gpt2_lora/  # or _full, _frozen
├── best_model.pt                          # Best model checkpoint
├── checkpoint_epoch_5.pt
└── ...

results/fine_tuned_ag_news_gpt2_lora/
├── training_history.json
├── training_curves.png
└── final_results.json
```

### 4. Evaluation and Comparison (To Be Implemented)

**Evaluate a trained model:**

```bash
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/from_scratch_ag_news_medium/best_model.pt \
    --dataset ag_news
```

**Compare both models:**

```bash
uv run python scripts/compare_models.py \
    --from-scratch checkpoints/from_scratch_ag_news_medium/best_model.pt \
    --fine-tuned checkpoints/fine_tuned_ag_news_gpt2/best_model.pt
```

---

## Project Structure

```
.
├── models/                            # Model implementations
│   ├── __init__.py
│   └── small_transformer_classifier.py  # Small transformer (1.7M-8.1M params)
│
├── utils/                             # Utility modules
│   ├── config.py                      # SmallTransformerConfig and FineTuneConfig
│   ├── classification_dataset.py     # Dataset loading for classification
│   └── metrics.py                     # Evaluation metrics (accuracy, F1, etc.)
│
├── scripts/                           # Training and evaluation scripts
│   ├── prepare_data.py                # Download and split datasets
│   ├── train_from_scratch.py          # Train small transformer from scratch
│   ├── train_fine_tune.py             # Fine-tune pre-trained model (to be implemented)
│   ├── evaluate.py                    # Evaluate models (to be implemented)
│   └── compare_models.py              # Compare both approaches (to be implemented)
│
├── data/                              # Data directory
│   ├── raw/                           # Raw datasets
│   └── processed/                     # Preprocessed datasets (JSONL)
│
├── checkpoints/                       # Model checkpoints
│   ├── from_scratch_ag_news_medium/   # From-scratch model checkpoints
│   └── fine_tuned_ag_news_gpt2/       # Fine-tuned model checkpoints
│
├── results/                           # Results and plots
│   ├── from_scratch_ag_news_medium/   # Training curves, metrics
│   └── fine_tuned_ag_news_gpt2/       # Training curves, metrics
│
├── README.md                          # This file
├── CLAUDE.md                          # Guide for Claude Code
└── pyproject.toml                     # Dependencies (uv)
```

---

## Model Architecture

### Tokenization

Both models use the **GPT-2 tokenizer** from Hugging Face:
- Pre-trained BPE (Byte-Pair Encoding) tokenizer
- Vocabulary size: 50,257 tokens
- Handles any text without OOV issues
- Same tokenizer for both from-scratch and fine-tuned models (ensures fair comparison)

This simplifies the implementation and ensures the only difference between models is the architecture and training approach, not the tokenization.

### Small Transformer Classifier (From-Scratch)

**Available sizes:**

| Size   | Layers | Embedding Dim | FFN Dim | Heads | Parameters |
|--------|--------|---------------|---------|-------|------------|
| tiny   | 2      | 128           | 512     | 4     | ~1.7M      |
| small  | 3      | 192           | 768     | 4     | ~3.3M      |
| medium | 3      | 256           | 1024    | 4     | ~4.9M      |
| large  | 4      | 320           | 1280    | 4     | ~8.1M      |

**Architecture features:**
- Decoder-only transformer (GPT-style)
- Sinusoidal positional encoding
- Causal attention masking
- Pre-LayerNorm for stable training
- Classification: uses last token hidden state → Linear(num_classes)

**Test model sizes:**
```bash
uv run python models/small_transformer_classifier.py
```

---

## Configuration

The config automatically optimizes settings based on your environment:

**Local GPU (CUDA):**
- Batch size: 32 (adjustable)
- Mixed precision: Enabled
- Workers: 4

**Cloud GPU (RunPod/Colab):**
- Batch size: 192 (larger for more VRAM)
- Gradient accumulation: 2
- Workers: 8

**Apple Silicon (MPS):**
- Batch size: 32
- Mixed precision: Disabled
- Workers: 0

**Test configuration:**
```bash
uv run python utils/config.py
```

---

## Tips and Troubleshooting

### Training Tips

**From-scratch model:**
- Start with `medium` size (~4.9M params)
- Train for 50+ epochs (converges slowly)
- Use learning rate 3e-4 to 5e-4
- Monitor validation F1, not just accuracy
- Expect ~70-85% accuracy on AG News

**Fine-tuned model:**
- Use learning rate 1e-5 to 5e-5 (much lower!)
- Train for 5-10 epochs (converges fast)
- Monitor for catastrophic forgetting
- Expect ~90-95% accuracy on AG News

### Common Issues

**Out of memory:**
```bash
# Reduce batch size
--batch-size 16

# Reduce sequence length
--max-length 64

# Use gradient accumulation (effective batch = batch_size * accumulation)
# (to be added to training script)
```

**Training diverges (loss → NaN):**
- Lower learning rate by 2-10x
- Check gradient clipping is enabled
- Reduce batch size
- Try different model size

**Slow training:**
- Ensure GPU is being used (check logs for device: cuda/mps)
- Increase batch size if you have VRAM
- Use mixed precision (AMP) - enabled by default on CUDA
- Run in background on cloud GPU

**Model underfitting:**
- Increase model size (try `large`)
- Train for more epochs
- Increase learning rate slightly
- Reduce dropout

**Model overfitting:**
- Decrease model size (try `small` or `tiny`)
- Add more dropout
- Use label smoothing (already enabled)
- Get more training data

---

## Expected Results

**AG News (4-class news classification):**

| Model              | Params | Accuracy | F1 (Macro) | Training Time* |
|--------------------|--------|----------|------------|----------------|
| From-scratch tiny  | 1.7M   | 75-80%   | 0.75-0.80  | 1-2 hours      |
| From-scratch medium| 4.9M   | 80-85%   | 0.80-0.85  | 2-4 hours      |
| GPT-2 (fine-tuned) | 124M   | 90-93%   | 0.90-0.93  | 20-40 min      |

*On NVIDIA RTX 3090 or similar

---

## Next Steps

1. ✅ **Data preparation** - Completed
2. ✅ **From-scratch training** - Completed
3. ✅ **Fine-tuning script** - Completed
4. ⏳ **Evaluation script** - To be implemented
5. ⏳ **Comparison script** - To be implemented
6. ⏳ **Report writing** - After training completes

---

## Literature

* [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/pdf/1810.04805), Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova
* [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever