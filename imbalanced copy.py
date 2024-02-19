import numpy as np
import util
import sys
from random import random

sys.path.append('../logreg_stability')

### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
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

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(validation_path, add_intercept=True)

    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()

    model = LogisticRegression()
    model.fit(x_train, y_train)
    pred_probs = model.predict(x_eval)
    np.savetxt(output_path_naive, pred_probs)
    for i in range(pred_probs.shape[0]):
        if pred_probs[i] > 0.5:
            pred_probs[i] = 1
        else:
            pred_probs[i] = 0
    tp, fp, tn, fn = get_accuracy(pred_probs, y_eval)
    print(f'Accuracy = {(tp + tn) / pred_probs.shape[0]}')
    a_0 = tn / (tn + fp)
    a_1 = tp / (tp + fn)
    print(f'A_0 = {a_0}')
    print(f'A_1 = {a_1}')
    print(f'Balanced Accuracy = {0.5 * (a_0 + a_1)}')
    util.plot(x_eval, y_eval, model.theta, 'logreg_pred_a.png')

    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times

    def newdata(x, y):
        idxs = np.where(y == 1)[0]
        repeat_idxs = np.repeat(idxs, (1 / kappa) - 1)
        newx = np.vstack((x, x[repeat_idxs]))
        newy = np.concatenate((y, np.ones(len(repeat_idxs))))
        return newx, newy

    new_x, new_y = newdata(x_train, y_train)
    model.fit(new_x, new_y)
    pred_probs = model.predict(x_eval)
    np.savetxt(output_path_upsampling, pred_probs)
    for i in range(pred_probs.shape[0]):
        if pred_probs[i] > 0.5:
            pred_probs[i] = 1
        else:
            pred_probs[i] = 0
    tp, fp, tn, fn = get_accuracy(pred_probs, y_eval)
    print(f'Accuracy = {(tp + tn) / pred_probs.shape[0]}')
    a_0 = tn / (tn + fp)
    a_1 = tp / (tp + fn)
    print(f'A_0 = {a_0}')
    print(f'A_1 = {a_1}')
    print(f'Balanced Accuracy = {0.5 * (a_0 + a_1)}')
    util.plot(x_eval, y_eval, model.theta, 'logreg_pred_b.png')
    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
