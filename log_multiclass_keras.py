
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import itertools
from tensorflow.keras import regularizers
from sklearn.linear_model import LogisticRegression


# Load the data
train_path='stem_data/stem_train_300.csv'
test_path='stem_data/stem_test_300.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Split features and target variable
x_train = train_data.drop(columns=['y'])
y_train = train_data['y']
x_test = test_data.drop(columns=['y'])
y_test = test_data['y']

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# Fit the model to the training data
model.fit(x_train, y_train)

y_pred_probs = model.predict_proba(x_train)
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate classification report
print(classification_report(y_train, y_pred))

y_pred_probs = model.predict_proba(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate classification report
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

# Get the coefficients (weights) and intercepts
coefficients = model.coef_
classdict = ["math", "chem", "phy"]
associations = []
print(np.mean(coefficients, axis=1))
aligns = np.argmax(coefficients, axis=0)
for i in range(coefficients.shape[1]):
    associations.append(f"{i}={classdict[aligns[i]]}")

print(associations)
intercepts = model.intercept_

# Print the coefficients and intercepts
print('Coefficients (Weights):')
print('Intercepts:')
print(intercepts)