import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import multivariate_normal

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    mu = np.zeros((K, x.shape[1]))
    sigma = np.zeros((K, x.shape[1], x.shape[1]))
    groups = {i: [] for i in range(K)}
    for i in range(x_all.shape[0]):
        assign = np.random.randint(K)
        groups[assign].append(x_all[i])
    for i in range(K):
        groups[i] = np.array(groups[i])
    for i in range(K):
        mu[i] = np.mean(groups[i], axis=0)
        sigma[i] = np.cov(groups[i], rowvar=False)

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones(K) / K

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (n, K)
    n = x.shape[0]
    w = np.ones((x.shape[0], K)) / K
    # w = np.ones((x_all.shape[0], K)) / K

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = None
    prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE
        it += 1
        freq = 5
        prev_ll = ll

        # (1) E-step: Update your estimates in w
        for i in range(x.shape[0]):
            for j in range(w.shape[1]):
                distribution = multivariate_normal(mean=mu[j], cov=sigma[j])
                w[i,j] = distribution.pdf(x[i]) * phi[j]
        w /= w.sum(axis=1, keepdims=True)

        # (2) M-step: Update the model parameters phi, mu, and sigma

        #phi
        phi = w.sum(axis=0, keepdims=True).reshape(-1) / x.shape[0]

        #mu
        mu = w.T @ x
        mu /= w.sum(axis=0, keepdims=True).T

        #sigma
        for j in range(sigma.shape[0]):
            temp = np.zeros((x.shape[1], x.shape[1]))
            for i in range(x.shape[0]):
                centered_x = x[i] - mu[j]
                temp += w[i, j] * np.outer(centered_x, centered_x)
            temp /= w[:, j].sum() 
            sigma[j] = temp

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        pdf_values = np.zeros((x.shape[0], K))
        for j in range(K):
            distribution = multivariate_normal(mean=mu[j], cov=sigma[j])
            pdf_values[:, j] = distribution.pdf(x)

        weighted_sum = pdf_values @ phi
        ll = np.sum(np.log(weighted_sum + 1e-5))

        if it % freq == 0:
            print(ll - prev_ll)
        # *** END CODE HERE ***
    
    print(it)
    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = None
    prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        
        # *** START CODE HERE ***
        it += 1
        freq = 5
        prev_ll = ll

        # (1) E-step: Update your estimates in w
        for i in range(x.shape[0]):
            for j in range(w.shape[1]):
                distribution = multivariate_normal(mean=mu[j], cov=sigma[j])
                w[i,j] = distribution.pdf(x[i]) * phi[j]
        w /= w.sum(axis=1, keepdims=True)

        # (2) M-step: Update the model parameters phi, mu, and sigma

        # phi
        unsup_sum = w.sum(axis=0, keepdims=True).reshape(-1)
        z_one_hot = np.eye(K)[z_tilde.astype(int).ravel()]
        sup_sum = z_one_hot.sum(axis=0, keepdims=True).reshape(-1)
        phi = (unsup_sum + alpha * sup_sum) / (x.shape[0] + alpha * x_tilde.shape[0])
 
        # mu
        unlabeled_x_sums = w.T @ x
        label_x_sums = z_one_hot.T @ x_tilde
        mu = (unlabeled_x_sums + alpha * label_x_sums) / (unsup_sum[:, np.newaxis] + alpha * sup_sum[:, np.newaxis])

        # sigma
        for j in range(sigma.shape[0]):
            temp = np.zeros((x.shape[1], x.shape[1]))
            for i in range(x.shape[0]):
                centered_x = x[i] - mu[j]
                temp += w[i, j] * np.outer(centered_x, centered_x)
            for i in range(x_tilde.shape[0]):
                if z_tilde[i] == j:
                    centered_tilde = x_tilde[i] - mu[j]
                    temp += alpha * np.outer(centered_tilde, centered_tilde)
            temp /= (w[:, j].sum() + alpha * sup_sum[j])
            sigma[j] = temp

        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        unsup_pdf_values = np.zeros((x.shape[0], K))
        sup_pdf_values = np.zeros((x_tilde.shape[0], K))

        for j in range(K):
            distribution = multivariate_normal(mean=mu[j], cov=sigma[j])
            unsup_pdf_values[:, j] = distribution.pdf(x)
            sup_pdf_values[:, j] = distribution.pdf(x_tilde)

        unsup_sum = unsup_pdf_values @ phi
        ll = np.sum(np.log(unsup_sum + 1e-5))

        sup_sum = sup_pdf_values @ phi
        ll += alpha * np.sum(np.log(sup_sum + 1e-5))

        if it % freq == 0:
            print(ll - prev_ll)

        # *** END CODE HERE ***

    print(it)
    return w


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.

        # main(is_semi_supervised=True, trial_num=t)

        # *** END CODE HERE ***
