"""
Data preparation script for text classification.

Downloads and preprocesses datasets from Hugging Face, splits into train/val/test,
and saves in a format ready for both from-scratch and fine-tuned training.

Usage:
    python scripts/prepare_data.py --dataset ag_news
    python scripts/prepare_data.py --dataset imdb
    python scripts/prepare_data.py --dataset yelp_polarity
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from collections import Counter
import json


def download_and_split_dataset(dataset_name: str, output_dir: Path, val_split_ratio: float = 0.1):
    """
    Download dataset from Hugging Face and create train/val/test splits.

    Args:
        dataset_name: Name of the dataset on Hugging Face
        output_dir: Directory to save the processed dataset
        val_split_ratio: Ratio of training data to use for validation
    """
    print("=" * 80)
    print(f"PREPARING DATASET: {dataset_name}")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from Hugging Face...")
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    print(f"  Dataset loaded successfully!")
    print(f"  Available splits: {list(dataset.keys())}")

    # Check what splits we have
    has_train = "train" in dataset
    has_test = "test" in dataset
    has_validation = "validation" in dataset or "val" in dataset

    if not has_train:
        print("Error: Dataset must have a 'train' split!")
        return False

    # Get train split
    train_data = dataset["train"]

    # Get test split or create one from train
    if has_test:
        test_data = dataset["test"]
        print(f"\n  Using existing test split: {len(test_data):,} examples")
    else:
        # Split 10% from train for testing
        split_dataset = train_data.train_test_split(test_size=0.1, seed=42)
        train_data = split_dataset["train"]
        test_data = split_dataset["test"]
        print(f"\n  Created test split from train: {len(test_data):,} examples")

    # Get validation split or create one from train
    if has_validation:
        val_key = "validation" if "validation" in dataset else "val"
        val_data = dataset[val_key]
        print(f"  Using existing validation split: {len(val_data):,} examples")
    else:
        # Split from train for validation
        split_dataset = train_data.train_test_split(test_size=val_split_ratio, seed=42)
        train_data = split_dataset["train"]
        val_data = split_dataset["test"]
        print(f"  Created validation split from train: {len(val_data):,} examples")

    print(f"\nFinal split sizes:")
    print(f"  Train: {len(train_data):,} examples")
    print(f"  Validation: {len(val_data):,} examples")
    print(f"  Test: {len(test_data):,} examples")

    # Analyze dataset structure
    print(f"\nDataset structure:")
    print(f"  Features: {train_data.features}")
    print(f"  Columns: {train_data.column_names}")

    # Identify text and label columns
    text_column = None
    label_column = None

    # Common text column names
    for col in ["text", "sentence", "review", "content", "document"]:
        if col in train_data.column_names:
            text_column = col
            break

    # Common label column names
    for col in ["label", "labels", "class", "sentiment", "category"]:
        if col in train_data.column_names:
            label_column = col
            break

    if not text_column or not label_column:
        print(f"\nError: Could not identify text and label columns!")
        print(f"  Available columns: {train_data.column_names}")
        print(f"  Please specify them manually in the script.")
        return False

    print(f"\nIdentified columns:")
    print(f"  Text column: '{text_column}'")
    print(f"  Label column: '{label_column}'")

    # Get label information
    label_feature = train_data.features[label_column]
    if hasattr(label_feature, 'names'):
        # ClassLabel feature
        num_classes = label_feature.num_classes
        class_names = label_feature.names
    else:
        # Integer labels - infer from data
        all_labels = set(train_data[label_column])
        num_classes = len(all_labels)
        class_names = [f"class_{i}" for i in range(num_classes)]

    print(f"\nLabel information:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {class_names}")

    # Analyze class distribution
    for split_name, split_data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        label_counts = Counter(split_data[label_column])
        print(f"\n{split_name} class distribution:")
        for label_idx in sorted(label_counts.keys()):
            count = label_counts[label_idx]
            percentage = count / len(split_data) * 100
            label_name = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
            print(f"  {label_name}: {count:,} ({percentage:.1f}%)")

    # Analyze text lengths
    print(f"\nAnalyzing text lengths...")
    train_lengths = [len(text.split()) for text in train_data[text_column][:1000]]  # Sample 1000
    avg_length = sum(train_lengths) / len(train_lengths)
    max_length = max(train_lengths)
    min_length = min(train_lengths)

    print(f"  Average words per text: {avg_length:.1f}")
    print(f"  Min words: {min_length}")
    print(f"  Max words: {max_length}")
    print(f"  Recommended max_seq_length: {min(512, int(avg_length * 2))}")

    # Show sample examples
    print(f"\nSample examples:")
    for i in range(min(3, len(train_data))):
        text = train_data[text_column][i]
        label = train_data[label_column][i]
        label_name = class_names[label] if label < len(class_names) else f"class_{label}"

        # Truncate long texts
        display_text = text[:200] + "..." if len(text) > 200 else text
        print(f"\n  Example {i+1}:")
        print(f"    Label: {label_name} ({label})")
        print(f"    Text: {display_text}")

    # Save splits to disk
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"SAVING DATASET TO: {output_dir}")
    print('=' * 80)

    # Save as JSON Lines format (one JSON object per line)
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_file = output_dir / f"{split_name}.jsonl"
        print(f"\nSaving {split_name} split to {output_file}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in split_data:
                # Create a simplified format
                json_obj = {
                    "text": example[text_column],
                    "label": int(example[label_column])
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

        print(f"    Saved {len(split_data):,} examples")

    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "text_column": text_column,
        "label_column": label_column,
        "splits": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        },
        "text_stats": {
            "avg_words": avg_length,
            "min_words": min_length,
            "max_words": max_length,
            "recommended_max_seq_length": min(512, int(avg_length * 2))
        }
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved metadata to {metadata_file}")

    print(f"\n{'=' * 80}")
    print("DATASET PREPARATION COMPLETE!")
    print('=' * 80)
    print(f"\nDataset saved to: {output_dir}")
    print(f"Files created:")
    print(f"  - train.jsonl ({len(train_data):,} examples)")
    print(f"  - val.jsonl ({len(val_data):,} examples)")
    print(f"  - test.jsonl ({len(test_data):,} examples)")
    print(f"  - metadata.json")
    print(f"\nNext steps:")
    print(f"  1. Train from-scratch model:")
    print(f"     uv run python scripts/train_from_scratch.py --dataset {dataset_name}")
    print(f"  2. Fine-tune pre-trained model:")
    print(f"     uv run python scripts/train_fine_tune.py --dataset {dataset_name}")
    print('=' * 80)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare text classification dataset from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare AG News dataset (4-class news classification)
  python scripts/prepare_data.py --dataset ag_news

  # Prepare IMDB dataset (binary sentiment classification)
  python scripts/prepare_data.py --dataset imdb

  # Prepare Yelp Polarity dataset (binary sentiment)
  python scripts/prepare_data.py --dataset yelp_polarity

  # Custom output directory
  python scripts/prepare_data.py --dataset ag_news --output data/processed/my_dataset
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset on Hugging Face (e.g., ag_news, imdb, yelp_polarity)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/processed/<dataset_name>)"
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1 = 10%%)"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output is None:
        output_dir = Path("data") / "processed" / args.dataset
    else:
        output_dir = Path(args.output)

    # Download and split dataset
    success = download_and_split_dataset(
        dataset_name=args.dataset,
        output_dir=output_dir,
        val_split_ratio=args.val_split
    )

    if not success:
        print("\n✗ Dataset preparation failed!")
        exit(1)
    else:
        print("\n  Dataset preparation succeeded!")
        exit(0)


if __name__ == "__main__":
    main()
