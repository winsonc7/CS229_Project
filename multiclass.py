import numpy as np
import util

def softmax(z):
    # Compute softmax probabilities
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtracting max for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    # Compute cross-entropy loss
    y_pred = np.clip(y_pred, 1e-5, 1 - 1e-5)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def initialize_parameters(input_size, num_classes):
    # Initialize model parameters (weights and biases) randomly
    np.random.seed(0)
    W = np.random.randn(input_size, num_classes)
    b = np.zeros((1, num_classes))
    return W, b

def forward_propagation(X, W, b):
    # Forward propagation
    Z = np.dot(X, W) + b
    A = softmax(Z)
    return A

def backward_propagation(X, A, y_true):
    # Backward propagation
    dZ = A - y_true
    dW = np.dot(X.T, dZ)
    db = np.sum(dZ, axis=0, keepdims=True)
    return dW, db

def update_parameters(W, b, dW, db, learning_rate):
    # Update model parameters using gradient descent
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b

def train(X_train, y_train, num_classes, num_epochs, learning_rate, reg):
    # Initialize model parameters
    input_size = X_train.shape[1]
    W, b = initialize_parameters(input_size, num_classes)

    # One-hot encode class labels
    y_train_int = np.array(y_train, dtype=int)
    y_one_hot = np.eye(num_classes)[y_train_int]
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward propagation
        A = forward_propagation(X_train, W, b)

        # Compute loss
        loss = cross_entropy_loss(y_one_hot, A)

        # Backward propagation
        dW, db = backward_propagation(X_train, A, y_one_hot)
        dW += 2*reg*W

        # Update parameters
        W, b = update_parameters(W, b, dW, db, learning_rate)

        # Print loss every few epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")

    return W, b


def get_accuracy(pred, truth):
    correct = 0
    for i in range(truth.shape[0]):
        if truth[i] == pred[i]:
            correct += 1
    return correct / truth.shape[0]

def metrics(x, y):
    pred = np.argmax(x, axis=1)
    print(f'Accuracy = {get_accuracy(pred, y)}')

def main(train_path, test_path, num_classes, num_epochs, lr, reg):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)
    x_test, y_test = util.load_csv(test_path, add_intercept=True)

    W, b = train(x_train, y_train, num_classes, num_epochs, lr, reg)

    pred_probs = forward_propagation(x_train, W, b)
    print("Training Accuracy:")
    metrics(pred_probs, y_train)

    pred_probs = forward_propagation(x_test, W, b)
    print("Test Accuracy:")
    metrics(pred_probs, y_test)
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='stem_data/stem_train_168feat.csv',
         test_path='stem_data/stem_test_168feat.csv',
         num_classes=3,
         num_epochs=200,
         lr=0.001,
         reg=0.00)
    