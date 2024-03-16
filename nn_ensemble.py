import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from ensemble_util import ensemble_neural_network
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

# Load the data
train_path='stem_data/stem_train_168feat_norm.csv'
test_path='stem_data/stem_test_168feat_norm.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Split features and target variable
x_train = train_data.drop(columns=['y'])
y_train = train_data['y']
x_test = test_data.drop(columns=['y'])
y_test = test_data['y']

ensemble_predictor = ensemble_neural_network(x_train, y_train, B=1)
y_pred = ensemble_predictor(x_train)
print(classification_report(y_train, y_pred))
y_pred = ensemble_predictor(x_test)
print(classification_report(y_test, y_pred))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

classes = [i for i in range(3)]
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))  # Assuming you have a list of class labels
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
  

# TRAIN STATS
# y_pred = ensemble_predictor(x_train)

# Calculate ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(y_train, y_pred)
# roc_auc = auc(fpr, tpr)

# Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# TEST STATS
# y_pred = ensemble_predictor(x_test)

# Calculate ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)