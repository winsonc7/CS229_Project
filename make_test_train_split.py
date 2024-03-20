import json
import random
import sys

TRAIN_RATIO = 0.8

def main(path):
    # Read the JSON file
    with open(path, 'r') as f:
        data = json.load(f)

    # Shuffle the samples
    random.shuffle(data)

    # Split the data into train and test sets
    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # Write train and test sets to new JSON files
    train_path = f"{dataset_path[:-5]}_train.json"
    test_path = f"{dataset_path[:-5]}_test.json"

    with open(train_path, 'w') as train_file:
        json.dump(train_data, train_file, indent=4)
    with open(test_path, 'w') as test_file:
        json.dump(test_data, test_file, indent=4)

# Run: python make_test_train_split.py hf_data/all_data.json
if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        sys.exit(1)

    # Get the dataset path and the output directory path from the command-line arguments
    dataset_path = sys.argv[1]
    main(dataset_path)