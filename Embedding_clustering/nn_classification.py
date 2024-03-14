





#https://www.educative.io/answers/implement-neural-network-for-classification-using-scikit-learn


import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score



x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.20, random_state=4)

NN = MLPClassifier()
NN.fit(x_train, y_train)

y_pred = NN.predict(x_test)

