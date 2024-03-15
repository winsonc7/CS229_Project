import json
import re
from collections import Counter
from nltk.corpus import stopwords
import sys

LOWER = 0.01
UPPER = 1
DELTA = 0.01

# Function to clean text by removing non-alphabetic characters
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

def main(dataset_path, feature_size):
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
    overall_word_freq_dict = dict(overall_word_freq)

    max_count = max(overall_word_freq_dict.values())

    lower_lim = LOWER
    upper_lim = UPPER
    while len(overall_word_freq_dict) > feature_size:
        bottom_threshold = max_count * lower_lim
        top_threshold = max_count * upper_lim
        overall_word_freq_dict = {word: count for word, count in overall_word_freq_dict.items() if bottom_threshold <= count <= top_threshold}
        lower_lim += DELTA

    # Store the overall word frequency vector in a JSON file
    save_path = f"{dataset_path[:-5]}_freq_vec_max_{feature_size}.json"
    with open(save_path, 'w') as f:
        json.dump(overall_word_freq_dict, f)

# Run: python datafeatures.py dataset_path feature_size
if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        sys.exit(1)

    # Get the dataset path and the output directory path from the command-line arguments
    feature_size = int(sys.argv[2])
    dataset_path = sys.argv[1]

    main(dataset_path, feature_size)


