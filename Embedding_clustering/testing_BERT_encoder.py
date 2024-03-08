
# importing libraries
import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

#This file runs a quick and dirty couple of tests of the BERT encoder with cosine similarity to test.
#There were three tests with types of problems with labels: physics physics, physics chem, physics math
#the physics physics was best as excected which is encouraging

#Code mostly taken from https://www.geeksforgeeks.org/how-to-generate-word-embedding-using-bert/



def main():

    # Set a random seed

    # Example sentence for similarity comparison #Similarity Score : 0.956
    # text = "GeeksforGeeks is a computer science portal"
    # example_sentence = "GeeksforGeeks is a technology website"

    # Example sentence for similarity comparison Physics, physics #Similarity Score : 0.862
    # text = "The diameter of a cylinder is measured using a vernier callipers with no zero error. It is found that the zero of the vernier scale lies between $5.10 cm$ and $5.15 cm$ of the main scale. The vernier scale has 50 division equivalent to $2.45 cm$. The $24^{\\text {th }}$ division of the vernier scale exactly coincides with one of the main scale divisions. The diameter of the cylinder is:\n\n(A) $5.112 cm$\n(B) $5.124 cm$\n(C) $5.136 cm$\n(D) $5.148 cm$"
    # example_sentence = "A horizontal stretched string, fixed at two ends, is vibrating in its fifth harmonic according to the equation, $y(x$, $\\mathrm{t})=(0.01 \\mathrm{~m}) \\sin \\left[\\left(62.8 \\mathrm{~m}^{-1}\\right) \\mathrm{x}\\right] \\cos \\left[\\left(628 \\mathrm{~s}^{-1}\\right) \\mathrm{t}\\right]$. Assuming $\\pi=3.14$, the correct statement(s) is (are):\n\n(A) The number of nodes is 5.\n(B) The length of the string is $0.25 \\mathrm{~m}$.\n(C) The maximum displacement of the midpoint of the string its equilibrium position is $0.01 \\mathrm{~m}$.\n(D) The fundamental frequency is $100 \\mathrm{~Hz}$."


    #Physics Chemistry : 0.756
    # text = "The diameter of a cylinder is measured using a vernier callipers with no zero error. It is found that the zero of the vernier scale lies between $5.10 cm$ and $5.15 cm$ of the main scale. The vernier scale has 50 division equivalent to $2.45 cm$. The $24^{\\text {th }}$ division of the vernier scale exactly coincides with one of the main scale divisions. The diameter of the cylinder is:\n\n(A) $5.112 cm$\n(B) $5.124 cm$\n(C) $5.136 cm$\n(D) $5.148 cm$"
    # example_sentence = "Concentrated nitric acid, upon long standing, turns yellow-brown due to the formation of\n\n(A) NO\n(B) $\\mathrm{NO}_{2}$\n(C) $\\mathrm{N}_{2} \\mathrm{O}$\n(D) $\\mathrm{N}_{2} \\mathrm{O}_{4}$"

    #Physics Math : 0.758
    text = "The diameter of a cylinder is measured using a vernier callipers with no zero error. It is found that the zero of the vernier scale lies between $5.10 cm$ and $5.15 cm$ of the main scale. The vernier scale has 50 division equivalent to $2.45 cm$. The $24^{\\text {th }}$ division of the vernier scale exactly coincides with one of the main scale divisions. The diameter of the cylinder is:\n\n(A) $5.112 cm$\n(B) $5.124 cm$\n(C) $5.136 cm$\n(D) $5.148 cm$"
    example_sentence = "Perpendiculars are drawn from points on the line $\\frac{x+2}{2}=\\frac{y+1}{-1}=\\frac{z}{3}$ to the plane $x+y+z=3$. The feet of perpendiculars lie on the line\n\n(A) $\\frac{x}{5}=\\frac{y-1}{8}=\\frac{z-2}{-13}$\n(B) $\\frac{x}{2}=\\frac{y-1}{3}=\\frac{z-2}{-5}$\n(C) $\\frac{x}{4}=\\frac{y-1}{3}=\\frac{z-2}{-7}$\n(D) $\\frac{x}{2}=\\frac{y-1}{-7}=\\frac{z-2}{5}$"


    random_seed = 42
    random.seed(random_seed)

    # Set a random seed for PyTorch (for GPU as well)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Input text


    # Tokenize and encode text using batch_encode_plus
    # The function returns a dictionary containing the token IDs and attention masks
    encoding = tokenizer.batch_encode_plus(
        # List of input texts
        [text],
        padding = True,  # Pad to the maximum sequence length
        truncation = True,  # Truncate to the maximum sequence length if necessary
        return_tensors = 'pt',  # Return PyTorch tensors
        add_special_tokens = True  # Add special tokens CLS and SEP
        )

    input_ids = encoding['input_ids']  # Token IDs
    # print input IDs
    print(f"Input ID: {input_ids}")
    attention_mask = encoding['attention_mask']  # Attention mask
    # print attention mask
    print(f"Attention mask: {attention_mask}")

    # Generate embeddings using BERT model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state  # This contains the embeddings

    # Output the shape of word embeddings
    print(f"Shape of Word Embeddings: {word_embeddings.shape}")

    # Decode the token IDs back to text
    decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # print decoded text
    print(f"Decoded Text: {decoded_text}")
    # Tokenize the text again for reference
    tokenized_text = tokenizer.tokenize(decoded_text)
    # print tokenized text
    print(f"tokenized Text: {tokenized_text}")
    # Encode the text
    encoded_text = tokenizer.encode(text, return_tensors='pt')  # Returns a tensor
    # Print encoded text
    print(f"Encoded Text: {encoded_text}")

    # Print word embeddings for each token
    for token, embedding in zip(tokenized_text, word_embeddings[0]):
        # print(f"Token: {token}")
        print(f"Embedding: {embedding}")
        print("\n")

    # Compute the average of word embeddings to get the sentence embedding
    sentence_embedding = word_embeddings.mean(dim=1)  # Average pooling along the sequence length dimension

    # Print the sentence embedding
    print("Sentence Embedding:")
    print(sentence_embedding)

    # Output the shape of the sentence embedding
    print(f"Shape of Sentence Embedding: {sentence_embedding.shape}")


    # Tokenize and encode the example sentence
    example_encoding = tokenizer.batch_encode_plus(
        [example_sentence],
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    example_input_ids = example_encoding['input_ids']
    example_attention_mask = example_encoding['attention_mask']

    # Generate embeddings for the example sentence
    with torch.no_grad():
        example_outputs = model(example_input_ids, attention_mask=example_attention_mask)
        example_sentence_embedding = example_outputs.last_hidden_state.mean(dim=1)

    # Compute cosine similarity between the original sentence embedding and the example sentence embedding
    similarity_score = cosine_similarity(sentence_embedding, example_sentence_embedding)

    # Print the similarity score
    print("Cosine Similarity Score:", similarity_score[0][0])

if __name__ == '__main__':
    main()

