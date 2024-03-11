
# importing libraries
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from JEEBench_reader import JEEBench_reader
from BERT_encoder import BERT_encoder
from sklearn.cluster import KMeans
import numpy as np
from itertools import permutations
from Sentence_Transformer import Sentence_Transformer

def main():

    filename = "JEEBench_data/dataset.json"

    #Run this line once then comment it out (took my computer about 2.5 minutes):
    #generate_BERT_embeddings_from_data(filename)

    #This line is fast but no need to fun over and over
    #generate_Sentence_embeddings_from_data(filename)


    embeddings = np.load("Sentence_embeddings.npy")
    print("Embeddings info: ")
    print(type(embeddings))
    print(embeddings.shape)


    data_reader = JEEBench_reader(filename)
    data_reader.read_data()
    data_reader.read_data_to_list()
    problem_labels = data_reader.prob_labels_numbered
    actual_labels = np.array(problem_labels)

    k = 3 #number of clusters
    predicted_labels = run_K_means(embeddings, k, metric='euclidean')

    #print("actual labels :  ", problem_labels)
    accuracy = calculate_accuracy(predicted_labels, actual_labels)
    print("accuracy for euclidean metric is : ", accuracy)


    k = 3 #number of clusters
    predicted_labels = run_K_means(embeddings, k, metric= calc_cosine_similarity)

    #print("actual labels :  ", problem_labels)
    accuracy = calculate_accuracy(predicted_labels, actual_labels)
    print("accuracy for cosine similarity metric is : ", accuracy)


def generate_BERT_embeddings_from_data(filename):
    #Uses the class JEEBench_reader defined in JEEBench_reader.py to read in data
    #Uses the class BERT_encoder defined in BERT_encoder.py to generate the embeddings
    #saves the embeddings to a numpy matrix of dimensions n x embedding length -- for JEEBench data is 515 x 768

    data_reader = JEEBench_reader(filename)
    data_reader.read_data()
    data_reader.read_data_to_list()

    encoder = BERT_encoder()
    problem_embeddings = encoder.generate_embeddings(data_reader.problem_list)
    np.save("problem_embeddings.npy", problem_embeddings)

def generate_Sentence_embeddings_from_data(filename):
    #Uses the class JEEBench_reader defined in JEEBench_reader.py to read in data
    #Uses the class BERT_encoder defined in BERT_encoder.py to generate the embeddings
    #saves the embeddings to a numpy matrix of dimensions n x embedding length -- for JEEBench data is 515 x 768

    data_reader = JEEBench_reader(filename)
    data_reader.read_data()
    data_reader.read_data_to_list()

    encoder = Sentence_Transformer()
    problem_embeddings = encoder.generate_embeddings(data_reader.problem_list)
    np.save("Sentence_embeddings.npy", problem_embeddings)



def calculate_accuracy(predicted_labels, actual_labels):
    #We don't know the mapping of labels outputted by k means and the actual labels given in the dataset
    #This function checks all the possible mappings and outputs the accuracy for the optimal mapping

    accuracies = []
    map = [0,1,2]
    all_mappings = list(permutations(map))
    for i in range(len(all_mappings)):
        accuracy = calc_accuracy_for_mapping(predicted_labels, actual_labels, all_mappings[i])
        accuracies.append(accuracy)

    return max(accuracies)


def calc_accuracy_for_mapping(predicted_labels, actual_labels, mapping):
    #Mapping is a tuple . For example, (1,0,2) means a zero in predicted labels maps to 1 in actual labels,
    #1 in predicted labels maps to 0 in actual labels, etc...
    n = predicted_labels.shape[0]
    corrected_predicted_labels = []
    for i in range(n):
        if predicted_labels[i] == 0:
            corrected_predicted_labels.append(mapping[0])
        elif predicted_labels[i] == 1:
            corrected_predicted_labels.append(mapping[1])
        else:
            corrected_predicted_labels.append(mapping[2])

    num_correct = 0

    for i in range(n):
        if corrected_predicted_labels[i] == actual_labels[i]:
            num_correct += 1
    return num_correct / n

def run_K_means(embeddings, k, metric='euclidean'):
    # Create KMeans instance

    kmeans = KMeans(n_clusters=k, random_state=42)
    if metric != 'euclidean':
        kmeans.euclidean_distances = calc_cosine_similarity
        print("cos distances")

    # Fit the data to the model
    kmeans.fit(embeddings)

    # Get the centroids of the clusters
    centroids = kmeans.cluster_centers_

    # Get the cluster assignments for each data point
    predicted_labels = kmeans.labels_

    return predicted_labels

def calc_cosine_similarity(v1, v2):
    return 1 - cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))




if __name__ == '__main__':
    main()

