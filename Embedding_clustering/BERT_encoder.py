
# importing libraries
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

def main():

    # Example usage
    text_list = [
        "This is the first sentence.",
        "This is the second sentence.",
        "And this is the third sentence."
    ]

    text_list_2 = ["GeeksforGeeks is a computer science portal",
                    "GeeksforGeeks is a technology website"]

    encoder = BERT_encoder()
    problem_embeddings = encoder.generate_embeddings(text_list)

    # print(type(problem_embeddings))
    # print(problem_embeddings.shape)
    print(problem_embeddings)

    cosine_similarity = encoder.calc_cosine_similarity(text_list[0], text_list[1])
    print("Cosine similarity Score for first example is : ", cosine_similarity)

    cosine_similarity = encoder.calc_cosine_similarity(text_list_2[0], text_list_2[1])
    print("Cosine similarity Score for second example is : ", cosine_similarity)


class BERT_encoder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def generate_embeddings(self, problem_strings):
        #input parameter problem_strings is a list where list[i] = "problem text here"
        #if len(problem_strings) = n, output is a numpy array of shape n x embedding_length

        problem_embeddings = []

        for problem in problem_strings:
            # Tokenize and encode text
            encoding = self.tokenizer.encode_plus(
                problem,
                padding=True,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            # Generate embeddings using BERT model
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                prob_embedding = outputs.last_hidden_state.mean(dim=1)  # Average pooling along the sequence length dimension
                problem_embeddings.append(prob_embedding)

        problem_embeddings = torch.cat(problem_embeddings, dim=0)
        return problem_embeddings.numpy()

    def calc_cosine_similarity(self, string_1, string_2):
        #Function for sanity check + debugging : takes in two strings outputs cosine similarity of their embeddings
        list = [string_1, string_2]
        embeddings = self.generate_embeddings(list)
        print("embeddings " , embeddings.shape)
        similarity_score = cosine_similarity(embeddings[0].reshape(1,-1), embeddings[1].reshape(1,-1))
        return similarity_score

if __name__ == '__main__':
    main()

