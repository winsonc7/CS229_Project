import json
import sys

def filter_subjects(original_data, excluded_subjects):
    # Read the JSON file
    with open(original_data, 'r') as f:
        data = json.load(f)

    # Filter out samples with the excluded subjects
    filtered_data = [sample for sample in data if sample['subject'] not in excluded_subjects]

    # Write the filtered samples to a new JSON file
    output_file = original_data.replace('.json', '_filtered.json')
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)

# Usage: python rm_class.py stem_data/stem.json chem math
if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) < 3:
        print("Usage: python rm_subjects.py <json_file> <excluded_subject1> <excluded_subject2> ...")
        sys.exit(1)

    # Get the JSON file path and excluded subjects from the command-line arguments
    original = sys.argv[1]
    excluded_subjects = sys.argv[2:]

    # Filter out samples with the excluded subjects
    filter_subjects(original, excluded_subjects)
