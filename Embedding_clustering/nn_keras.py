
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def implement_nn_keras(x_train, y_train, x_test):
    # Define the neural network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=16, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='softmax')  # 3 classes, so use softmax activation
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_train, y_train)
    print(f'Test Accuracy: {accuracy}')

    # Make predictions
    y_pred_probs = model.predict(x_train)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Generate classification report
    print(classification_report(y_train, y_pred))



    # # Evaluate the model
    # loss, accuracy = model.evaluate(x_test, y_test)
    # print(f'Test Accuracy: {accuracy}')

    # Make predictions
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # # Generate classification report
    # print(classification_report(y_test, y_pred))

    return y_pred