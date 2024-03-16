import json
import csv
import sys
import re

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

def main(data_file, feature_file, class_dict):
    # Load data and features
    with open(data_file, 'r') as f:
        data = json.load(f)
    with open(feature_file, 'r') as f:
        features = json.load(f).keys()

    # Determine column names
    columns = ['y'] + ['x' + str(i) for i in range(len(features))]

    # Open CSV file for writing
    csv_file = f"{data_file[:-5]}_{len(columns) - 1}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write column headers
        writer.writerow(columns)

        # Write data rows
        for sample in data:
            # Initialize feature count dictionary for the sample
            feature_count = {feature: 0 for feature in features}

            # Update feature count dictionary based on sample text
            question = sample['question']
            cleaned_question = clean_text(question)
            for word in cleaned_question.split():
                if word in feature_count:
                    feature_count[word] += 1

            # Determine y value
            y_value = class_dict[sample['subject']]

            # Write row to CSV file
            writer.writerow([y_value] + [feature_count[feature] for feature in features])

# Usage: python convert_to_csv.py data_path feature_path class0 class1
# python convert_to_csv.py stem_data/stem_test.json stem_data/stem_train_freq_vec_300.json math chem phy
if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) < 4:
        sys.exit(1)

    # Get the JSON file path and excluded subjects from the command-line arguments
    data_path = sys.argv[1]
    feature_path = sys.argv[2]
    classes = sys.argv[3:]
    class_dict = {}
    for label in classes:
        class_dict[label] = classes.index(label)
        
    main(data_path, feature_path, class_dict)