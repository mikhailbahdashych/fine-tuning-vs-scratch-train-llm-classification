"""
Dataset utilities for text classification.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class ClassificationDataset(Dataset):
    """PyTorch dataset for text classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128
    ):
        """
        Initialize classification dataset.

        Args:
            texts: List of text samples
            labels: List of integer labels
            tokenizer: Tokenizer (Hugging Face or custom)
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        if hasattr(self.tokenizer, 'encode_plus'):
            # Hugging Face tokenizer
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
        else:
            # Custom tokenizer (for from-scratch model)
            tokens = self.tokenizer.encode(text)

            # Truncate or pad
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

            input_ids = torch.tensor(tokens, dtype=torch.long)
            attention_mask = torch.tensor([1 if t != self.tokenizer.pad_token_id else 0 for t in tokens], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_jsonl_data(file_path: Path) -> Tuple[List[str], List[int]]:
    """
    Load data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
            labels.append(data['label'])

    return texts, labels


def load_metadata(data_dir: Path) -> Dict:
    """
    Load dataset metadata.

    Args:
        data_dir: Directory containing metadata.json

    Returns:
        Metadata dictionary
    """
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def create_dataloaders(
    data_dir: Path,
    tokenizer,
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create dataloaders for train/val/test splits.

    Args:
        data_dir: Directory containing train.jsonl, val.jsonl, test.jsonl
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory

    Returns:
        Tuple of (train_loader, val_loader, test_loader, metadata)
    """
    data_dir = Path(data_dir)

    # Load metadata
    metadata = load_metadata(data_dir)

    # Load data
    train_texts, train_labels = load_jsonl_data(data_dir / "train.jsonl")
    val_texts, val_labels = load_jsonl_data(data_dir / "val.jsonl")
    test_texts, test_labels = load_jsonl_data(data_dir / "test.jsonl")

    print(f"Loaded data:")
    print(f"  Train: {len(train_texts):,} examples")
    print(f"  Val: {len(val_texts):,} examples")
    print(f"  Test: {len(test_texts):,} examples")

    # Create datasets
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = ClassificationDataset(test_texts, test_labels, tokenizer, max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    # Test dataset loading
    from transformers import AutoTokenizer

    print("=" * 80)
    print("TESTING CLASSIFICATION DATASET")
    print("=" * 80)

    # Check if data exists
    data_dir = Path("data/processed/ag_news")
    if not data_dir.exists():
        print(f"\nError: Dataset not found at {data_dir}")
        print("Run: uv run python scripts/prepare_data.py --dataset ag_news")
        exit(1)

    # Load with GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataloaders
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=128,
        batch_size=4,
        num_workers=0,
        pin_memory=False
    )

    print(f"\nMetadata:")
    print(f"  Dataset: {metadata['dataset_name']}")
    print(f"  Classes: {metadata['num_classes']}")
    print(f"  Class names: {metadata['class_names']}")

    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Test a batch
    print(f"\nTesting batch loading...")
    batch = next(iter(train_loader))
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Sample input IDs: {batch['input_ids'][0][:20]}")
    print(f"  Sample labels: {batch['labels']}")

    print("\n" + "=" * 80)
    print("  Dataset loading test successful!")
    print("=" * 80)
