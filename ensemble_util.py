import numpy as np
from sklearn.utils import resample
import tensorflow as tf

def ensemble_neural_network(X_train, y_train, B):
    ensemble_models = []
    
    # Generate B bootstrap samples and train separate neural networks
    for b in range(B):
        print(b)
        # Create a bootstrap sample
        X_boot, y_boot = resample(X_train, y_train, replace=True)
        # Create a new neural network model
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(units=3, activation='softmax')  # 3 classes, so use softmax activation
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train the model on the bootstrap sample
        model.fit(X_boot, y_boot, epochs=30, batch_size=64, validation_split=0.2, verbose=0)
        
        # Add the trained model to the ensemble
        ensemble_models.append(model)
    
    # Combine predictions of individual models by averaging
    def ensemble_predict(X):
        predictions = np.zeros((X.shape[0], len(ensemble_models)))
        for i, model in enumerate(ensemble_models):
            predictions[:, i] = np.argmax(model.predict(X), axis=1)
        ensemble_prediction = np.mean(predictions, axis=1).astype(int)
        return ensemble_prediction
    
    return ensemble_predict

"""
ensemble_predictor = ensemble_neural_network(X_train, y_train, B=10)
ensemble_predictions = ensemble_predictor(X_test)
"""