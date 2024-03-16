import json
import re
from collections import Counter
from nltk.corpus import stopwords
import sys

def create_subject_files(dataset_path):
    # Read the JSON dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Initialize a dictionary to store samples by subject
    samples_by_subject = {}

    # Organize samples by subject
    for sample in dataset:
        subject = sample['subject']
        if subject not in samples_by_subject:
            samples_by_subject[subject] = []
        samples_by_subject[subject].append(sample)

    # Write samples to separate JSON files for each subject
    file_names = []
    for subject, samples in samples_by_subject.items():
        file_name = f"{dataset_path[:-5]}_{subject}.json"
        file_names.append(file_name)
        with open(file_name, 'w') as f:
            json.dump(samples, f, indent=4)
    
    return file_names, len(samples_by_subject)

# Function to clean text by removing non-alphabetic characters
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

def make_vec(dataset_path, feature_size):
    # Read the JSON file
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Initialize an empty dictionary to store word frequencies
    overall_word_freq = Counter()

    stop_words = set(stopwords.words('english'))

    # Iterate over each sample
    for sample in data:
        # Clean the question text
        question = sample['question']
        cleaned_question = clean_text(question)

        # Tokenize the cleaned question text
        words = cleaned_question.split()
        words = [word for word in words if word not in stop_words]

        # Create a word frequency vector for the sample
        word_freq = Counter(words)

        # Update the overall word frequency vector
        overall_word_freq.update(word_freq)

    # Convert Counter object to a dictionary
    subject_word_freq_dict = dict(overall_word_freq)
    if len(subject_word_freq_dict) < feature_size:
        print("ERROR NOT ENOUGH FEATURES")
    while len(subject_word_freq_dict) > feature_size:
        for i in range(len(subject_word_freq_dict) - feature_size):
            min_value_key = min(subject_word_freq_dict, key=subject_word_freq_dict.get)
            del subject_word_freq_dict[min_value_key]
    

    return subject_word_freq_dict

def main(dataset_path, feature_size):
    subject_paths, num_subs = create_subject_files(dataset_path)
    overall_word_freq_dict = {}
    size_per_sub = feature_size // num_subs

    for path in subject_paths:
      subject_dict = make_vec(path, size_per_sub)
      for key, value in subject_dict.items():
          overall_word_freq_dict[key] = value

    # Store the overall word frequency vector in a JSON file
    save_path = f"{dataset_path[:-5]}_freq_{feature_size}_between_{len(subject_paths)}_subjects.json"
    with open(save_path, 'w') as f:
        json.dump(overall_word_freq_dict, f)

# Run: python datafeatures_balanced.py dataset_path feature_size
# python datafeatures.py stem_data/stem_train.json 300
if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        sys.exit(1)

    # Get the dataset path and the output directory path from the command-line arguments
    feature_size = int(sys.argv[2])
    dataset_path = sys.argv[1]

    main(dataset_path, feature_size)
