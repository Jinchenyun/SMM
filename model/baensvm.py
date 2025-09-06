import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from model.utils import clip_dcd_optimizer


class BAENSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, lambda_val=1.0, tau=1.0, theta=1.0, sigma=1.0,
                 kernel='linear', fit_intercept=True, eps=1e-5,
                 eps_cccp=1e-5, max_steps=4000, cccp_steps=100):
        """
        BEN-SVM (Bounded Elastic Net SVM) implementation

        Parameters:
        C : float, regularization parameter
        lambda_val : float, loss function parameter
        tau : float, asymmetry parameter (0 < tau <= 1)
        theta : float, elastic net mixing parameter
        sigma : float, bandwidth for RBF kernel
        kernel : {'linear', 'rbf'}, kernel type
        fit_intercept : bool, whether to add intercept term
        eps : float, optimization tolerance
        eps_cccp : float, CCCP convergence tolerance
        max_steps : int, maximum optimization iterations
        cccp_steps : int, maximum CCCP iterations
        """
        self.C = C
        self.lambda_val = lambda_val
        self.tau = tau
        self.theta = theta
        self.sigma = sigma
        self.kernel = kernel
        self.fit_intercept = fit_intercept
        self.eps = eps
        self.eps_cccp = eps_cccp
        self.max_steps = max_steps
        self.cccp_steps = cccp_steps

    def _linear_kernel(self, X1, X2):
        """Linear kernel function"""
        return X1 @ X2.T

    def _rbf_kernel(self, X1, X2, sigma):
        """RBF kernel function"""
        # Vectorized RBF kernel implementation
        gamma = 1.0 / (2 * sigma ** 2)
        sq_dists = np.sum(X1 ** 2, axis=1)[:, np.newaxis] + np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-gamma * sq_dists)

    def _kernel_function(self, X1, X2):
        """Select kernel based on initialization"""
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2, self.sigma)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def _update_deltak_en(self, f, lambda_val, theta, tau):
        """
        Compute delta_k for BEN-SVM

        Parameters:
        f : array, decision function values
        lambda_val : float, loss function parameter
        theta : float, elastic net mixing parameter
        tau : float, asymmetry parameter

        Returns:
        delta_k : array, computed values
        """
        delta_k = np.zeros_like(f)
        # For positive f
        pos_idx = f >= 0
        if np.any(pos_idx):
            f_pos = f[pos_idx]
            term = (1 - 1 / (1 + lambda_val * f_pos) ** 2)
            delta_k[pos_idx] = lambda_val * term * (theta * (f_pos - 1) - 1)

        # For negative f
        neg_idx = f < 0
        if np.any(neg_idx):
            f_neg = f[neg_idx]
            term = (1 - 1 / (1 + tau * lambda_val * f_neg) ** 2)
            delta_k[neg_idx] = -tau * lambda_val * term * (1 - theta * (f_neg + 1))

        return delta_k

    def fit(self, X, y):
        """
        Train BEN-SVM model using CCCP algorithm

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels (±1)
        """
        self.classes_ = unique_labels(y)

        if self.fit_intercept:
            X = np.column_stack((X, np.ones(X.shape[0])))

        self.X_ = X.copy()
        self.y_ = y.copy()

        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        if self.fit_intercept:
            X = np.hstack([X, np.ones((n_samples, 1))])

        K = self._kernel_function(X, X)

        H = np.diag(y.flatten()) @ K @ np.diag(y.flatten())
        e = np.ones((n_samples, 1))

        if self.tau == 0:
            u0 = np.zeros((n_samples, 1))
        else:
            u0 = np.full((n_samples, 1), (1 + self.tau) * self.lambda_val * self.C / 2)

        delta_k = np.zeros((n_samples, 1))

        for i in range(self.cccp_steps):
            lb = -self.C * delta_k
            ub = -self.C * delta_k + self.lambda_val * self.C * (self.tau + 1) * e

            u0 = np.clip(u0, lb, ub)

            solver_info = clip_dcd_optimizer(
                H, e.flatten(), lb.flatten(), ub.flatten(),
                self.eps, self.max_steps, u0.flatten()
            )
            u = solver_info['x'].reshape(-1, 1)

            if np.linalg.norm(u - u0) < self.eps_cccp:
                break
            else:
                u0 = u

            f = 1 - H @ u0
            delta_k = self._update_deltak_en(f, self.lambda_val, self.theta, self.tau)

        self.coef_ = np.diag(y.flatten()) @ u0

        return self

    def decision_function(self, X):
        """
        Compute decision function values

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Test samples

        Returns:
        decision : array, shape (n_samples,)
            Decision function values
        """
        check_is_fitted(self)
        X = check_array(X)

        if self.fit_intercept and X.shape[1] == self.X_.shape[1]:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        elif self.fit_intercept and X.shape[1] == self.X_.shape[1] - 1:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        K = self._kernel_function(X, self.X_)
        return K @ self.coef_

    def predict(self, X):
        """
        Predict using trained model

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Test samples

        Returns:
        pred : array, shape (n_samples,)
            Predicted labels (±1)
        """
        decision = self.decision_function(X)
        return np.sign(decision).flatten()

    def set_params(self, **parameters):
        """
        设置模型参数
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
