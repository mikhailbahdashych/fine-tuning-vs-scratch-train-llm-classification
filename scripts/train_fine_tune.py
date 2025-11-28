"""
Fine-tuning script for pre-trained decoder-only models.

Usage:
    uv run python scripts/train_fine_tune.py --dataset banking77 --model gpt2 --use-lora
    uv run python scripts/train_fine_tune.py --dataset ag_news --model gpt2 --use-lora
    uv run python scripts/train_fine_tune.py --dataset banking77 --model gpt2 --freeze-base
"""

import argparse
import json
import time
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.classification_dataset import create_dataloaders
from utils.metrics import compute_classification_metrics


def config_to_dict(config):
    """Convert config to JSON-serializable dictionary."""
    config_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
        else:
            config_dict[key] = value
    return config_dict


def setup_lora(model, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """
    Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning.

    Args:
        model: The model to apply LoRA to
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout

    Returns:
        Model with LoRA applied
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        print(f"\nApplying LoRA:")
        print(f"  Rank (r): {lora_r}")
        print(f"  Alpha: {lora_alpha}")
        print(f"  Dropout: {lora_dropout}")

        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn", "c_proj"]  # GPT-2 attention modules
        )

        # Apply LoRA
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model

    except ImportError:
        print("\nWarning: peft library not found. Install with: pip install peft")
        print("Continuing without LoRA...")
        return model


def freeze_base_model(model):
    """
    Freeze all parameters except the classification head.

    Args:
        model: The model to freeze
    """
    print("\nFreezing base model parameters...")

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classification head
    for param in model.score.parameters():
        param.requires_grad = True

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"  Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"  Frozen parameters: {total-trainable:,} ({(total-trainable)/total*100:.2f}%)")


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Calculate accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # Compute metrics
    metrics = compute_classification_metrics(all_labels, all_predictions)
    metrics['loss'] = avg_loss

    return metrics


def plot_training_curves(history, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 0].plot(history['val_f1'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score (Macro)')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(history['learning_rate'])
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained model for classification")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., banking77, ag_news)")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Pre-trained model (gpt2, gpt2-medium, EleutherAI/gpt-neo-125m, etc.)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: from config)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--freeze-base", action="store_true", help="Freeze base model (only train head)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Load config
    config = get_config("fine_tune")

    # Override config with args
    if args.batch_size:
        config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.num_epochs = args.epochs
    config.pretrained_model = args.model
    config.use_lora = args.use_lora
    config.freeze_base = args.freeze_base

    print("=" * 80)
    print("FINE-TUNING PRE-TRAINED MODEL FOR CLASSIFICATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Pre-trained model: {args.model}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Use LoRA: {args.use_lora}")
    print(f"  Freeze base: {args.freeze_base}")
    print(f"  Mixed precision: {config.use_amp}")

    # Prepare paths
    data_dir = Path("data/processed") / args.dataset
    model_suffix = "lora" if args.use_lora else ("frozen" if args.freeze_base else "full")
    model_name_safe = args.model.replace("/", "_")
    checkpoint_dir = Path("checkpoints") / f"fine_tuned_{args.dataset}_{model_name_safe}_{model_suffix}"
    results_dir = Path("results") / f"fine_tuned_{args.dataset}_{model_name_safe}_{model_suffix}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Set padding token (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    print(f"  Vocabulary size: {vocab_size:,}")

    # Load data
    print(f"\nLoading data from {data_dir}...")
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    num_classes = metadata['num_classes']
    class_names = metadata['class_names']
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {class_names}")

    # Adjust hyperparameters based on number of classes
    if num_classes > 20 and not args.lr:  # User didn't specify LR
        # For multi-class problems (>20 classes), use higher learning rate
        original_lr = config.learning_rate
        config.learning_rate = 5e-5  # 2.5x higher than default 2e-5
        print(f"\n  ℹ️  Auto-adjusted learning rate: {original_lr:.2e} → {config.learning_rate:.2e}")
        print(f"     (Higher LR recommended for {num_classes} classes)")

    if num_classes > 20:
        # Increase warmup steps for harder tasks
        original_warmup = config.warmup_steps
        config.warmup_steps = min(2000, len(train_loader) * 2)  # 2 epochs or 2000 steps
        if config.warmup_steps != original_warmup:
            print(f"  ℹ️  Auto-adjusted warmup steps: {original_warmup} → {config.warmup_steps}")
            print(f"     (More warmup needed for {num_classes} classes)")

    # Load pre-trained model
    print(f"\nLoading pre-trained model: {args.model}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_classes,
        pad_token_id=tokenizer.pad_token_id
    )

    # Apply LoRA if requested
    if args.use_lora:
        model = setup_lora(model, args.lora_r, args.lora_alpha)

    # Freeze base model if requested
    if args.freeze_base and not args.use_lora:
        freeze_base_model(model)

    model = model.to(config.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # Training setup
    # Use model's internal loss (already includes label smoothing if configured)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * config.num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rate': []
    }

    best_val_f1 = 0
    start_time = time.time()

    print(f"\n{'=' * 80}")
    print("STARTING TRAINING")
    print('=' * 80)

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, None, config.device, scaler
        )

        # Validate
        val_metrics = evaluate(model, val_loader, config.device)

        epoch_time = time.time() - epoch_start

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])

        # Record learning rate
        current_lr = scheduler.get_last_lr()[0]
        history['learning_rate'].append(current_lr)

        # Print metrics
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
        print(f"  Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Epoch Time: {epoch_time:.1f}s")

        # Save best model
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            checkpoint_path = checkpoint_dir / "best_model.pt"

            # Save model based on type
            if args.use_lora:
                # Save LoRA adapters
                model.save_pretrained(checkpoint_dir / "best_model_lora")
            else:
                # Save full model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_f1': best_val_f1,
                    'config': config_to_dict(config),
                    'metadata': metadata,
                    'model_name': args.model
                }, checkpoint_path)

            print(f"    Saved new best model (F1: {best_val_f1:.4f})")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            if args.use_lora:
                model.save_pretrained(checkpoint_dir / f"checkpoint_epoch_{epoch+1}_lora")
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                    'config': config_to_dict(config),
                    'metadata': metadata,
                    'model_name': args.model
                }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    total_time = time.time() - start_time

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print('=' * 80)
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best validation F1: {best_val_f1:.4f}")

    # Save training history
    history_path = results_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Saved training history to {history_path}")

    # Plot training curves
    plot_path = results_dir / "training_curves.png"
    plot_training_curves(history, plot_path)

    # Final evaluation on test set
    print(f"\n{'=' * 80}")
    print("FINAL EVALUATION ON TEST SET")
    print('=' * 80)

    # Load best model
    if args.use_lora:
        # Load LoRA adapters
        from peft import PeftModel
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=num_classes,
            pad_token_id=tokenizer.pad_token_id
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir / "best_model_lora")
        model = model.to(config.device)
    else:
        checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, config.device)

    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"  Precision (Macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {test_metrics['recall_macro']:.4f}")

    # Save final results
    final_results = {
        'model_name': args.model,
        'use_lora': args.use_lora,
        'freeze_base': args.freeze_base,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'dataset': args.dataset,
        'num_classes': num_classes,
        'training_time_minutes': total_time / 60,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'config': config_to_dict(config)
    }

    results_path = results_dir / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n  Saved final results to {results_path}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
