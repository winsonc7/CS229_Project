import json
import sys

def main(json_files):
    combined_dict = {}

    # Iterate over each JSON file
    for file in json_files:
        # Read the JSON file
        with open(file, 'r') as f:
            data = json.load(f)

        # Merge dictionaries
        combined_dict.update(data)

    # Write combined dictionary to a new JSON file
    output_file = "combined.json"
    with open(output_file, 'w') as f:
        json.dump(combined_dict, f, indent=4)

# Run: python combine_features.py path1 path2 path3
if __name__ == "__main__":
    vecs = sys.argv[1:]
    main(vecs)
