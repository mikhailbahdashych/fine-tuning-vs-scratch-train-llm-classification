"""
Quick script to explore available classification datasets.
"""

from datasets import load_dataset

# Try multiple datasets
datasets_to_try = [
    ("ag_news", None),  # News classification (4 classes)
    ("yelp_polarity", None),  # Sentiment (2 classes)
    ("imdb", None),  # Movie reviews (2 classes)
]

print("=" * 80)
print("EXPLORING CLASSIFICATION DATASETS")
print("=" * 80)

for dataset_name, config in datasets_to_try:
    try:
        print(f"\n{'=' * 80}")
        print(f"DATASET: {dataset_name}")
        print('=' * 80)

        # Load a small sample first
        dataset = load_dataset(dataset_name, config, split="train[:100]")

        print(f"\nDataset structure:")
        print(f"  Features: {dataset.features}")
        print(f"\nSample entry:")
        print(dataset[0])

        print(f"\nDataset columns: {dataset.column_names}")

        # Check full dataset
        full_dataset = load_dataset(dataset_name, config)
        print(f"\nAvailable splits: {list(full_dataset.keys())}")

        for split_name, split_data in full_dataset.items():
            print(f"\n{split_name.upper()} split: {len(split_data):,} examples")

        print(f"\n  {dataset_name} loaded successfully!")
        break  # Use the first one that works

    except Exception as e:
        print(f"\n✗ Failed to load {dataset_name}: {e}")
        continue

# Try the defunct one with trust_remote_code
try:
    print(f"\n{'=' * 80}")
    print("Trying defunct-datasets/amazon_reviews_multi with trust_remote_code...")
    print('=' * 80)
    dataset = load_dataset("defunct-datasets/amazon_reviews_multi", "en", split="train[:100]", trust_remote_code=True)

    print(f"\nDataset structure:")
    print(f"  Features: {dataset.features}")
    print(f"\nSample entry:")
    print(dataset[0])

    print(f"\n  amazon_reviews_multi loaded successfully with trust_remote_code!")

except Exception as e:
    print(f"\n✗ Failed to load amazon_reviews_multi: {e}")

print("\n" + "=" * 80)
