import numpy as np
from scipy.linalg import svd
from sklearn.metrics import accuracy_score
from model.utils import clip_dcd_optimizer
import scipy.sparse as sp
import osqp


class TPinSMM:
    def __init__(self, C=1, lambda_=1, tau=1, s=1, mu=10, epsilon=1e-4, max_iter=500):
        self.C = C
        self.lambda_ = lambda_
        self.tau = tau
        self.s = s
        self.mu = mu
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.W = None
        self.b = None

    def fit(self, X, y, verbose=False):
        self.X = X
        self.y = y

        n, p, q = X.shape

        Beta_old = np.ones((p, q))
        V_old = np.ones((p, q))
        W_old = V_old.copy()
        b_old = 0

        X_flat = X.reshape(n, -1)
        K = X_flat @ X_flat.T
        Q = (y.reshape(-1, 1) @ y.reshape(1, -1)) * K / (1 + self.mu)
        Q = sp.csc_matrix(Q)
        A = np.vstack([
            y.reshape(1, n),
            np.eye(n)
        ])
        A = sp.csc_matrix(A)

        prob = osqp.OSQP()
        prob.setup(P=Q,
                   q=None,
                   A=A,
                   l=np.zeros(n+1),
                   u=np.ones(n+1),
                   polish=False,
                   verbose=False,
                   warm_start=True,
                   eps_abs=1e-4,
                   eps_rel=1e-4,
                   scaling=True)

        c = float('inf')
        alpha = 1
        for i in range(self.max_iter):
            obj_old = self.obj(W_old, b_old, X, y)
            sigma = self.compute_sigam(X, y, W_old, b_old)
            lb = -1 * sigma
            lb = np.concatenate([np.array([0]), lb])
            ub = (1 + self.tau) * self.C - sigma
            ub = np.concatenate([np.array([0]), ub])
            f = 1 - y * np.einsum('pq, ipq -> i', (Beta_old + V_old * self.mu), X) / (1 + self.mu)

            prob.update(q=-f,
                        l=lb,
                        u=ub)
            res = prob.solve()
            z = res.x

            W_new = (Beta_old + self.mu * V_old + np.einsum('i, i, ipq -> pq', z, y, X)) / (1 + self.mu)
            mask = (z > -1 * sigma) & (z < (1 + self.tau) * self.C - sigma)
            b_new = np.mean(y[mask] - np.einsum('pq, ipq -> i', W_new, X[mask]))

            V_new = self.svt(self.mu * W_new - Beta_old, self.lambda_) / self.mu
            Beta_new = Beta_old + self.mu * (V_new - W_new)

            c_new = np.linalg.norm(Beta_new - Beta_old, ord='fro') ** 2 / self.mu + \
                     np.linalg.norm(V_new - V_old, ord='fro') ** 2 * self.mu
            if c_new < 0.999 * c:
                alpha_new = (1 + np.sqrt(1 + 4 * alpha**2)) / 2
                V_new = V_new + (alpha - 1) / alpha_new * (V_new - V_old)
                Beta_new = Beta_new + (alpha - 1) / alpha_new * (Beta_new - Beta_old)
                c, alpha = c_new, alpha_new
            else:
                c, alpha = c_new / 0.999, 1
                V_new, Beta_new= V_old.copy(), Beta_old.copy()

            obj_new = self.obj(W_new, b_new, X, y)
            if verbose:
                print(f'CCCP: {i}, obj value: {obj_old:.4f} -> {obj_new:.4f}')

            W_old, b_old = W_new.copy(), b_new
            V_old, Beta_old = V_new.copy(), Beta_new.copy()

            if np.abs(obj_new - obj_old) < self.epsilon:
                break

        self.W = W_old.copy()
        self.b = b_old

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
        out = np.sign(np.einsum('pq, ipq->i', self.W, X) + self.b)
        return out

    def obj(self, W, b, X, y):
        u = 1 - y * (np.einsum('pq, ipq -> i', W, X) + b)
        out1 = 0.5 * np.trace(W.T @ W)
        out2 = self.lambda_ * np.sum(np.linalg.svd(W, full_matrices=False, compute_uv=False))
        out3 = self.C * np.sum(self.tpin(u))
        out = out1 + out2 + out3
        return out


    def tpin(self, u):
        out1 = np.maximum(0, (1 + self.tau) * u)
        out2 = np.maximum(0, self.tau*(u + self.s))
        out = out1 - (out2 - self.tau * self.s)
        return out

    def compute_sigam(self, X, y, W, b):
        u = 1 - y * (np.einsum('pq, ipq -> i', W, X) + b)
        out = np.where(
            u >= -1 * self.s,
            self.C * self.tau,
            0
        )
        return out

    def svt(self, M, Tau):
        """奇异值阈值化算子"""
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        s_thresh = np.maximum(s - Tau, 0)
        D = U @ np.diag(s_thresh) @ Vh
        return D

    def set_params(self, **parameters):
        """
        设置模型参数
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
