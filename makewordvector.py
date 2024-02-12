from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import json

LOWER = 0.05
UPPER = 0.95

# The extract_text function is not robust for certain pdfs
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Convert tokens to lowercase and remove non-alphanumeric characters
    tokens = [token.lower() for token in tokens if token.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    return filtered_tokens

def save_word_frequency_vector(word_freq, filename):
    with open(filename, 'w') as file:
        json.dump(word_freq, file)

def create_word_frequency_vector(tokens):
    word_freq = Counter(tokens)
    
    # Calculate the total number of words
    total_words = max(word_freq.values())
    
    # Calculate the threshold for the bottom and top 10%
    bottom_threshold = int(total_words * LOWER)
    top_threshold = int(total_words * UPPER)
    
    # Filter out words below the bottom threshold and above the top threshold
    filtered_word_freq = {word: count for word, count in word_freq.items() if bottom_threshold < count < top_threshold}
    
    return filtered_word_freq


def main(pdf_path, output_filename):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Preprocess the text
    tokens = preprocess_text(text)
    
    # Create word frequency vector
    word_freq = create_word_frequency_vector(tokens)

    save_word_frequency_vector(word_freq, output_filename)

if __name__ == "__main__":
    elem_path = "elementary_chapter.pdf"
    hs_path = "high_school_chapter.pdf"
    main(elem_path, "elem_word_vector.json")
    main(hs_path, "hs_word_vector.json")
