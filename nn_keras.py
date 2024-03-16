
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the data
train_path='stem_data/stem_train_168feat.csv'
test_path='stem_data/stem_test_168feat.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
print("checkpoint")

# Split features and target variable
x_train = train_data.drop(columns=['y'])
y_train = train_data['y']
x_test = test_data.drop(columns=['y'])
y_test = test_data['y']

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
print("checkpoint")
# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')  # 3 classes, so use softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("checkpoint")
# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_split=0.2)
print("checkpoint")

# Evaluate the model
loss, accuracy = model.evaluate(X_train_scaled, y_train)
print(f'Test Accuracy: {accuracy}')

# Make predictions
y_pred_probs = model.predict(X_train_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate classification report
print(classification_report(y_train, y_pred))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {accuracy}')

# Make predictions
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate classification report
print(classification_report(y_test, y_pred))