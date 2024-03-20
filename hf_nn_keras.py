
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from matplotlib import pyplot as plt
import itertools
from tensorflow.keras import regularizers
from scipy.spatial.distance import cosine


# Load the data
train_path='hf_data/all_data_500.csv'
train_data = pd.read_csv(train_path)


# Split features and target variable
x_train = train_data.drop(columns=['y'])
y_train = train_data['y']

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=12, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("checkpoint")
# Train the model
model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.2)
model.save("neural_mmlu_500.h5")

# Make predictions
y_pred_probs = model.predict(x_train)
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate classification report
print(classification_report(y_train, y_pred))

# Compute the confusion matrix
cm = confusion_matrix(y_train, y_pred)

classes = ["college_bio", "college_chem", "college_cs", "college_math", "college_phy", "elem_math", "hs_bio", "hs_chem", "hs_cs", "hs_math", "hs_phy", "hs_stats"]
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'NN Confusion Matrix, Feat=100, Acc={round(accuracy_score(y_train, y_pred), 4)}')
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