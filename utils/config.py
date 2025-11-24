"""
Configuration file for classification models.
Supports GPU (CUDA), Apple Silicon (MPS), and CPU training.
Includes both from-scratch small transformer and fine-tuned pre-trained models.
"""

import torch
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Global configuration for the project."""

    # Dataset name (will be set from preprocessing/training)
    dataset_name: str = "default"

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    data_wikipedia_dir: Path = project_root / "data" / "wikipedia"
    checkpoints_dir: Path = project_root / "checkpoints"
    results_dir: Path = project_root / "results"

    # Device configuration (GPU -> MPS -> CPU hierarchy)
    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Training environment (auto-detect or set via env variable)
    training_env: str = os.environ.get("TRAINING_ENV", "auto")  # "auto", "local", "cloud"

    # DataLoader optimization
    num_workers: int = 0  # Will be set based on environment
    pin_memory: bool = False  # Will be set based on device

    # Data processing
    vocab_size: int = 10000  # For whitespace and sentencepiece tokenizers (GPT-2 uses 50,257)
    max_seq_length: int = 256  # Increased from 128 for better context (RTX 5090 can handle it)
    train_split: float = 0.85
    val_split: float = 0.10
    test_split: float = 0.05

    # Tokenizer
    min_frequency: int = 2  # Minimum token frequency to include in vocab

    # Training hyperparameters
    batch_size: int = 32  # Will be optimized based on GPU
    num_epochs: int = 50  # More epochs for from-scratch training
    learning_rate: float = 3e-4  # Will be adjusted for batch size and model type
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Gradient accumulation (for effective larger batch sizes)
    gradient_accumulation_steps: int = 1  # 1 = no accumulation

    # Mixed precision training (for faster GPU training)
    use_amp: bool = False  # Will be set based on device

    # Learning rate warmup
    warmup_steps: int = 2000  # Longer warmup for large batches

    # Label smoothing for better generalization
    label_smoothing: float = 0.1

    # Model saving
    save_every_n_epochs: int = 2

    # Plotting
    plot_every_n_epochs: int = 1  # Generate plots after every N epochs

    # Evaluation
    eval_batch_size: int = 64

    # Generation
    max_gen_length: int = 100
    temperature: float = 1.0
    top_k: int = 50

    def __post_init__(self):
        """Create directories if they don't exist and optimize settings based on device."""
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect training environment
        if self.training_env == "auto":
            # Check if running in a cloud environment (common indicators)
            is_cloud = (
                os.path.exists("/.dockerenv") or  # Docker container
                os.environ.get("RUNPOD_POD_ID") is not None or  # RunPod
                os.environ.get("COLAB_GPU") is not None or  # Google Colab
                os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None  # Kaggle
            )
            self.training_env = "cloud" if is_cloud else "local"

        # Optimize settings based on device and environment
        if self.device == "cuda":
            self.pin_memory = True
            self.use_amp = True  # Enable mixed precision for NVIDIA GPUs

            if self.training_env == "cloud":
                # Cloud GPU optimization (RunPod, Colab, etc.)
                self.num_workers = 8  # More workers for cloud GPUs
                if self.batch_size <= 64:  # Only increase if using default/small batch
                    # Optimized batch size to maximize VRAM without OOM
                    self.batch_size = 192  # Sweet spot for 50M model on RTX 5090
                    self.gradient_accumulation_steps = 2  # Effective batch = 192 * 2 = 384

                    # Scale learning rate with effective batch size
                    # Effective batch = 192 * 2 = 384
                    if self.learning_rate == 3e-4:
                        effective_batch = self.batch_size * self.gradient_accumulation_steps
                        self.learning_rate = 3e-4 * (effective_batch / 32)
                        self.learning_rate = min(self.learning_rate, 1e-3)  # Cap at 1e-3
            else:
                # Local GPU optimization
                self.num_workers = 4
        elif self.device == "mps":
            # Apple Silicon optimization
            self.pin_memory = False
            self.num_workers = 0  # MPS works better with single process
            self.use_amp = False  # AMP not fully supported on MPS
        else:
            # CPU fallback
            self.pin_memory = False
            self.num_workers = 4
            self.use_amp = False


@dataclass
class SmallTransformerConfig(Config):
    """Configuration for small from-scratch transformer classifier (0.5M-10M params)."""

    # Model architecture (small for from-scratch training)
    embedding_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 1024
    dropout: float = 0.3
    model_size: str = "medium"  # Options: "tiny" (~0.5M), "small" (~2M), "medium" (~5M), "large" (~10M)

    # Classification specific
    num_classes: int = 4  # Will be set from dataset

    # Training (from-scratch requires more epochs)
    num_epochs: int = 50

    # Learning rate scheduler
    scheduler_patience: int = 3

    # Estimated parameters: ~2-5M depending on vocab size

    def __repr__(self):
        return (f"SmallTransformer(layers={self.num_layers}, heads={self.num_heads}, "
                f"emb={self.embedding_dim}, ff={self.ff_dim})")


@dataclass
class FineTuneConfig(Config):
    """Configuration for fine-tuning pre-trained models."""

    # Pre-trained model to use
    pretrained_model: str = "gpt2"  # Options: gpt2, gpt2-medium, gpt-neo-125m, etc.

    # Classification specific
    num_classes: int = 4  # Will be set from dataset

    # Fine-tuning (fewer epochs, lower LR)
    num_epochs: int = 10
    learning_rate: float = 2e-5  # Much lower for fine-tuning
    warmup_steps: int = 500  # Shorter warmup

    # LoRA/adapter settings (optional)
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # Freeze strategy
    freeze_base: bool = False  # If True, only train classification head

    def __repr__(self):
        return f"FineTune(model={self.pretrained_model}, freeze_base={self.freeze_base})"


def get_config(model_type: str = "small_transformer") -> Config:
    """
    Get configuration for specified model type.

    Args:
        model_type: One of "small_transformer" (from-scratch) or "fine_tune" (pre-trained)

    Returns:
        Config object
    """
    if model_type.lower() == "small_transformer":
        return SmallTransformerConfig()
    elif model_type.lower() == "fine_tune":
        return FineTuneConfig()
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose 'small_transformer' or 'fine_tune'.")


if __name__ == "__main__":
    # Test configuration
    print("=" * 80)
    print("CONFIGURATION TEST")
    print("=" * 80)

    small_config = get_config("small_transformer")
    print("\nSmall Transformer Config (from-scratch):")
    print(f"  Device: {small_config.device}")
    print(f"  Training env: {small_config.training_env}")
    print(f"  Batch size: {small_config.batch_size}")
    print(f"  Learning rate: {small_config.learning_rate}")
    print(f"  Epochs: {small_config.num_epochs}")
    print(f"  Model: {small_config}")

    finetune_config = get_config("fine_tune")
    print("\nFine-tune Config:")
    print(f"  Device: {finetune_config.device}")
    print(f"  Training env: {finetune_config.training_env}")
    print(f"  Batch size: {finetune_config.batch_size}")
    print(f"  Learning rate: {finetune_config.learning_rate}")
    print(f"  Epochs: {finetune_config.num_epochs}")
    print(f"  Model: {finetune_config}")

    print("\n" + "=" * 80)
