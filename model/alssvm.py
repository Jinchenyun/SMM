import numpy as np
from model.utils import clip_dcd_optimizer
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class ALSSVM():
    def __init__(self, C=1.0, p=0.5, sigma=1.0, kernel='linear',
                 fit_intercept=True, eps=1e-5, max_steps=1000):
        """
        Asymmetric Least Squares SVM (ALSSVM) implementation

        Parameters:
        C : float, regularization parameter
        p : float, asymmetry parameter (0 < p < 1)
        sigma : float, bandwidth for RBF kernel
        kernel : {'linear', 'rbf'}, kernel type
        fit_intercept : bool, whether to add intercept term
        eps : float, optimization tolerance
        max_steps : int, maximum optimization iterations
        """
        self.C = C
        self.p = p
        self.sigma = sigma
        self.kernel = kernel
        self.fit_intercept = fit_intercept
        self.eps = eps
        self.max_steps = max_steps

    def _linear_kernel(self, X1, X2):
        """Linear kernel function"""
        return X1 @ X2.T

    def _rbf_kernel(self, X1, X2, sigma):
        """RBF kernel function"""
        gamma = 1.0 / (2 * sigma ** 2)
        K = np.exp(-gamma * np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2))
        return K

    def _kernel_function(self, X1, X2):
        """Select kernel based on initialization"""
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2, self.sigma)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def fit(self, X, y):
        """
        Train ALSSVM model

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
        H = np.block([[H, -H], [-H, H]])

        I1 = np.diag(1 / self.p * np.ones(n_samples))
        I12 = np.zeros((n_samples, n_samples))
        I4 = np.diag(1 / (1 - self.p) * np.ones(n_samples))
        In = np.block([[I1, I12], [I12, I4]])

        H = H + In / self.C
        q = -np.ones(2 * n_samples)
        q[:n_samples] = 1

        lb = np.zeros(2 * n_samples)
        ub = np.full(2 * n_samples, np.inf)
        u0 = lb.copy()

        solver_info = clip_dcd_optimizer(H, q, lb, ub, self.eps, self.max_steps, u0)
        u = solver_info['x'].flatten()
        alpha = u[:n_samples] - u[n_samples:]
        self.coef_ = np.diag(y.flatten()) @ alpha

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
