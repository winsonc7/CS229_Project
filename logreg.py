import numpy as np
import util
import matplotlib.pyplot as plt

def get_accuracy(pred, truth):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(truth.shape[0]):
        if truth[i] == 1:
            if pred[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred[i] == 1:
                fp += 1
            else:
                tn += 1
    return (tp, fp, tn, fn)


def main(train_path, test_path, save_path):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)
    x_test, y_test = util.load_csv(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    model = LogisticRegression()
    model.fit(x_train, y_train)
    pred_probs = model.predict(x_test)
    for i in range(pred_probs.shape[0]):
        if pred_probs[i] > 0.5:
            pred_probs[i] = 1
        else:
            pred_probs[i] = 0
    tp, fp, tn, fn = get_accuracy(pred_probs, y_test)
    print(f'Accuracy = {(tp + tn) / pred_probs.shape[0]}')
    a_0 = tn / (tn + fp)
    a_1 = tp / (tp + fn)
    print(f'A_0 = {a_0}')
    print(f'A_1 = {a_1}')
    print(f'Balanced Accuracy = {0.5 * (a_0 + a_1)}')
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=1, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

        # *** START CODE HERE ***
        self.lamb = 0.001
        # *** END CODE HERE ***

    def fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(x.shape[1])

        freq = 1000
        avg_loss = 0

        def loss(label, sig):
            if label == 1:
                return(np.log(sig + self.eps))
            else:
                return(np.log(1 - sig + self.eps))

        def sigmoid(num):
            return 1 / (1 + np.exp(-num))

        halt = False
        iter = 0
        while not halt:
            avg_loss = 0
            iter += 1
            update = np.zeros(self.theta.shape[0])
            for i in range(x.shape[0]):
                eta = self.theta @ x[i]
                diff = y[i] - sigmoid(eta)
                update += diff * x[i]
                avg_loss += loss(y[i], sigmoid(eta))
            update *= -1 / x.shape[0]
            update += self.lamb * self.theta
            new_theta = self.theta - (self.learning_rate * update)
            if np.linalg.norm(self.theta - new_theta, ord=1) < self.eps or iter > self.max_iter:
                halt = True
                if iter > self.max_iter:
                    print("Did not converge")
            if self.verbose and iter % freq == 0:
                print(f'Avg loss = {-1 * avg_loss / x.shape[0]}')
                print(f'Theta = {self.theta}')
                print(f'On iteration {iter}, norm is {np.linalg.norm(self.theta - new_theta, ord=1):.10f}')
            self.theta = new_theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        pred = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            pred[i] = 1 / (1 + np.exp(-1 * self.theta @ x[i]))
        return pred
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='stem_data/mathchem_train_34feat.csv',
         test_path='stem_data/mathchem_test_34feat.csv',
         save_path='logreg_pred_b.png')