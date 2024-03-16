

from JEEBench_reader import JEEBench_reader
from BERT_encoder import BERT_encoder
from sklearn.cluster import KMeans
import numpy as np
from itertools import permutations
from Sentence_Transformer import Sentence_Transformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from util import *
from sklearn.metrics import confusion_matrix
#from nn_keras import implement_nn_keras

def main():

    filename = "JEEBench_data/dataset.json"
    train_split = .8
    k = 3
    # BIRCH params
    branch = 12
    thresh = 0.5
    random_seed = 42


    data_reader = JEEBench_reader(filename, use_random_seed=True, random_seed=random_seed)
    data_reader.read_data()
    data_reader.read_data_to_list()
    train, test, train_labels, test_labels = data_reader.split_train_test(train_percent=train_split, verbose=True)

    #This line is fast but no need to run over and over -- functions from util.py
    generate_Sentence_embeddings_from_list(train, "sentence_train_emb.npy")
    generate_Sentence_embeddings_from_list(test, "sentence_test_emb.npy")
    # generate_BERT_embeddings_from_list(train, "bert_train_emb.npy")
    # generate_BERT_embeddings_from_list(test, "bert_test_emb.npy")

    # #The only line you have to change to switch from BERT to Sentence is this file that's being loaded
    train_embeddings = np.load("sentence_train_emb.npy") # 410 x 384
    test_embeddings = np.load("sentence_test_emb.npy") # 105 x 384
    # train_embeddings = np.load("bert_train_emb.npy") # 410 x 384
    # test_embeddings = np.load("bert_test_emb.npy") # 105 x 384


    #This line takes a while, do ones and remember params
    # branch, thresh = compute_BIRCH_params(train_embeddings,k,train_labels)
    # print("Best branching parameter is : ", branch, "Best threshold parameter is : ", thresh)
    #

    #train_clust_labels = run_BIRCH(train_embeddings,k,branch,thresh, use_random_seed=True, random_seed=random_seed)

    train_clust_labels = run_Hierarchical(train_embeddings,k, use_random_seed=True, random_seed=random_seed)

    accuracy, corrected_train_clust_labels = calculate_accuracy(train_clust_labels,train_labels)
    print("Accuracy of Initial clustering step is : " , accuracy)

    #Switch Between these two lines to switch from basic nn to nn keras
    predicted_test_labels = implement_basic_nn(train_embeddings, corrected_train_clust_labels, test_embeddings)
    #predicted_test_labels = implement_nn_keras(train_embeddings,corrected_train_clust_labels, test_embeddings)

    nn_accuracy = calc_nn_acc(predicted_test_labels, test_labels)
    print("Overall neural network accuracy is : ", nn_accuracy)

    cm = confusion_matrix(test_labels, predicted_test_labels)
    print("confusion matrix is : ")
    print(cm)
if __name__ == '__main__':
    main()
