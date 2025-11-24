"""
Training script for from-scratch small transformer classifier.

Usage:
    uv run python scripts/train_from_scratch.py --dataset ag_news
    uv run python scripts/train_from_scratch.py --dataset ag_news --model-size large --epochs 100
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
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models import create_small_classifier
from utils.config import get_config
from utils.classification_dataset import create_dataloaders
from utils.metrics import compute_classification_metrics


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
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

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
    parser = argparse.ArgumentParser(description="Train from-scratch transformer classifier")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., ag_news)")
    parser.add_argument("--model-size", type=str, default="medium", choices=["tiny", "small", "medium", "large"],
                        help="Model size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: from config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: from config)")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Load config
    config = get_config("small_transformer")

    # Override config with args
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    config.num_epochs = args.epochs
    config.model_size = args.model_size

    print("=" * 80)
    print("TRAINING FROM-SCRATCH TRANSFORMER CLASSIFIER")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model size: {args.model_size}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Mixed precision: {config.use_amp}")

    # Prepare paths
    data_dir = Path("data/processed") / args.dataset
    checkpoint_dir = Path("checkpoints") / f"from_scratch_{args.dataset}_{args.model_size}"
    results_dir = Path("results") / f"from_scratch_{args.dataset}_{args.model_size}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (using GPT-2 tokenizer for simplicity)
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
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

    # Create model
    print(f"\nCreating {args.model_size} model...")
    model = create_small_classifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        size=args.model_size
    )
    model = model.to(config.device)

    num_params = model.get_num_parameters()
    print(f"  Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = AdamW(
        model.parameters(),
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
            model, train_loader, optimizer, scheduler, criterion, config.device, scaler
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, config.device)

        epoch_time = time.time() - epoch_start

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])

        # Record learning rate (sample from scheduler)
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': best_val_f1,
                'config': vars(config),
                'metadata': metadata
            }, checkpoint_path)
            print(f"    Saved new best model (F1: {best_val_f1:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'config': vars(config),
                'metadata': metadata
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
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, config.device)

    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"  Precision (Macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {test_metrics['recall_macro']:.4f}")

    # Save final results
    final_results = {
        'model_size': args.model_size,
        'num_parameters': num_params,
        'dataset': args.dataset,
        'num_classes': num_classes,
        'training_time_minutes': total_time / 60,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'config': vars(config)
    }

    results_path = results_dir / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n  Saved final results to {results_path}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
