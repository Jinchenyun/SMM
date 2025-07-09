import numpy as np
import cvxpy as cp
from sklearn.metrics import accuracy_score
import random


class SMM:
    """
    Support Matrix Machine (SMM) classifier.

    Parameters:
    -----------
    c : float, default=1.0
        Regularization parameter.
    p : float, default=2.0
        ADMM penalty parameter.
    random_seed : int, default=42
        Random seed for reproducibility.
    tao : float, default=1.0
        Thresholding parameter for singular values.
    epochs : int, default=100
        Number of training epochs.
    """

    def __init__(self, c=1.0, p=2.0, random_seed=42, tau=1.0, epochs=100):

        # Set random seed for reproducibility
        np.random.seed(int(random_seed))
        random.seed(int(random_seed))
        self.W = None
        self.b = random.random()
        self.S = None
        self.A = None
        self.c = c
        self.p = p
        self.epochs = epochs
        self.tau = tau

    def fit(self, X, y):
        """
        Fit the SMM model to the training data.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, p, q)
            Training input samples as 3D matrices.
        y : ndarray of shape (n_samples,)
            Target values (class labels) as -1 or 1.
        """
        self.X = X
        self.y = y
        self.b = random.random()
        self.W = np.random.uniform(0, 1, size=self.X.shape[1:3])
        self.S = np.random.uniform(0, 1, size=self.X.shape[1:3])
        self.A = np.random.uniform(0, 1, size=self.X.shape[1:3])

        # Run ADMM optimization
        out = self._ADMM(self.X, self.y, self.W, self.b, self.c, self.p, self.S, self.A, self.tau, self.epochs)

        # Update model parameters
        self.W = out['W']
        self.b = out['b']
        self.S = out['S']
        self.A = out['A']

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, p, q)
            Input samples as 3D matrices.

        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (-1 or 1).
        """
        return np.sign(np.trace(self.W.T @ X, axis1=1, axis2=2) + self.b)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, p, q)
            Test samples.
        y : ndarray of shape (n_samples,)
            True labels for X.

        Returns:
        --------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        return accuracy_score(y, self.predict(X))

    def _ADMM(self, X, y, W, b, c, p, S, A, tau, epochs):
        """
            ADMM optimization algorithm for SMM.

            This is an internal method and should not be called directly.
        """
        n = self.X.shape[0]
        traces = np.einsum('ipq,jpq->ij', X, X)
        K = np.outer(y, y) * traces / (p + 1)

        print('Training SMM model...')
        for epoch in range(epochs):
            q = np.ones(n) - (y * np.trace((A + p * S).T @ X, axis1=1, axis2=2)) / (p + 1)
            a = cp.Variable(n)
            objective = cp.Minimize(0.5 * cp.quad_form(a, K) - q.T @ a)
            constraints = [
                a >= 0,
                a <= c,
                y.T @ a == 0
            ]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP, verbose=False)
            a = a.value

            W_temp = (A + p * S + np.einsum('i, i, ipq -> pq', a, y, X))
            b_temp = np.sum(y - np.trace(W_temp.T @ X, axis1=1, axis2=2)) / len(a)

            U, s, Vh = np.linalg.svd(p * W_temp - A, full_matrices=False)
            St = np.diag(np.maximum(s - tau, 0))
            Dt = U @ St @ Vh
            S_temp = Dt / p
            A_temp = self.A - p * (W_temp - S_temp)

            acc1 = accuracy_score(y, np.sign(np.trace(W.T @ self.X, axis1=1, axis2=2) + b))
            acc2 = accuracy_score(y, np.sign(np.trace(W_temp.T @ self.X, axis1=1, axis2=2) + b_temp))
            diff = acc1 - acc2

            W = W_temp
            S = S_temp
            b = b_temp
            A = A_temp

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, train accuracy: {acc2}')

            if abs(diff) < 1e-6:
                break

        return {'W': W, 'S': S, 'b': b, 'A': A}
