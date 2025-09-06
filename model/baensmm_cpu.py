import time
import numpy as np
import scipy
from scipy.linalg import eigvalsh
from sklearn.metrics import accuracy_score
import osqp
import scipy.sparse as sp

class BaenSMM():
    """
    Support Matrix Machine (SMM) Based on Bounded Asymmetric Elastic Net Loss.

    Parameters:
    -----------
    gamma : float, default=1.0, range: be > 0
        Penalty parameter of nuclear norm
    c : float, default=1.0, range: be > 0
        Regularization parameter.
    eta : float, default=1.2, range: be > 0
        Parameter of bounded asymmetric elastic net loss
    lambda : float, defauit=5
        Parameter of bounded asymmetric elastic net loss
    tau : float, default=0.8, range: (0.0, 1.0)
        Parameter of asymmetric elastic net loss and pinball loss
    theta: float, default=0.5. range: (0.0, 1.0)
        Parameter of asymmetric elastic net loss
    random_seed : int, default=42
        Random seed for reproducibility
    cccp_epochs : int, default=100
        iterations of cccp
    admm_epochs : int default=100
        iterations of admm
    epsilon : float default=1e-6
    """

    def __init__(self, gamma=0.1, c=1, eta=1.2, lambda_=5, tau=0.8, theta=0.5, rho=0.01,
                 cccp_epochs=100, admm_epochs=100, epsilon=1e-6, random_seed=42):

        self.random_seed = random_seed
        self.gamma = gamma
        self.c = c
        self.eta = eta
        self.lambda_ = lambda_
        self.tau = tau
        self.theta = theta
        self.cccp_epochs = cccp_epochs
        self.admm_epochs = admm_epochs
        self.epsilon = epsilon
        self.rho = rho
        self.b = None
        self.W = None  # Initialize W from S

    def fit(self, X, y, verbose=False):
        """
        Fit the SMM model to the training data by CCCP-ADMM

        Parameters:
        -----------
        X : ndarray of shape (n_samples, p, q)
            Training input samples as 3D matrices.
        y : ndarray of shape (n_samples,)
            Target values (class labels) as -1 or 1.
        """

        self.A = self.lambda_ * self.theta / (self.tau * self.eta) + 1
        self.B = self.lambda_ * (1 - self.theta) / self.eta

        self.X = X
        self.y = y
        self.verbose = verbose

        # n, p, q = self.X.shape
        np.random.seed(int(self.random_seed))
        self.b = np.array(0)
        self.W = np.random.uniform(-0.01, 0.01, size=self.X.shape[1:])

        out = self.cccp_admm()

        self.W = out['W']
        self.b = out['b']

    def decision(self, X):
        out = np.einsum('pq, ipq->i', self.W, np.asarray(X)) + self.b
        return out

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

    def cccp_admm(self):
        n, p, q = self.X.shape
        D = np.diag(self.y)
        e = np.ones(n)
        I = np.eye(n)

        X_flatten = self.X.reshape(n, p * q, order='F').T
        XD = X_flatten @ D
        DXXD = XD.T @ XD

        f21 = -1 * self.c * self.B * e.reshape(n, 1)
        f2 = np.vstack([f21, f21 / self.tau])
        f3 = np.vstack([e.reshape(n, 1), -e.reshape(n, 1)])

        H1 = np.vstack([np.hstack([DXXD, -DXXD]),
                        np.hstack([-DXXD, DXXD])]) / (1 + self.rho)
        H2 = np.vstack([np.hstack([I, I / self.tau]),
                        np.hstack([I / self.tau, I / (self.tau ** 2)])]) / (2 * self.A * self.c)
        H = H1 + H2
        H = sp.csc_matrix(H)

        A_eq = np.hstack([self.y.reshape(1, -1),
                          -self.y.reshape(1, -1)])
        A_ineq = np.vstack([np.hstack([I, I / self.tau]),
                            np.eye(2 * n)])
        A = np.vstack([A_eq, A_ineq])
        A = sp.csc_matrix(A)

        prob = osqp.OSQP()
        prob.setup(P=H,
                   q=None,
                   A=A,
                   l=np.zeros(3 * n + 1),
                   u=np.ones(3 * n + 1),
                   polish=False,
                   verbose=False,
                   warm_start=True,
                   eps_abs=1e-4,
                   eps_rel=1e-4,
                   scaling=True)

        W_cccp_old, b_cccp_old = self.W.copy(), self.b.copy()

        for k in range(self.cccp_epochs):
            obj_old = self.obj(W_cccp_old, b_cccp_old)
            u = 1 - self.y * (np.tensordot(W_cccp_old, self.X, axes=([0, 1], [1, 2])) + b_cccp_old)
            sigma = -1 * self.h_prime(u)

            eq = self.c * (sigma.reshape(1, -1) @ self.y.reshape(-1, 1))
            eq = float(eq.squeeze())
            l = np.concatenate([np.array([eq]),
                                self.c * self.B * e,
                                np.zeros(2 * n)])
            u = np.concatenate([np.array([eq]),
                                np.inf * np.ones(3 * n)])
            prob.update(l=l,
                        u=u)

            c1 = 1
            c2 = float('inf')
            W_admm_old, b_admm_old = W_cccp_old.copy(), b_cccp_old.copy()
            Lambda_admm_old, S_admm_old = np.ones((p, q)), np.zeros((p, q))
            for l in range(self.admm_epochs):
                ao_old = self.admm_obj(W_admm_old, b_admm_old, sigma)

                v = -1 * (Lambda_admm_old + self.rho * S_admm_old).reshape(p * q, 1, order='F')
                f11 = -self.c * (DXXD @ sigma.reshape(n, 1)) - XD.T @ v
                f1 = np.vstack([f11, -f11])
                f = f1 / (1 + self.rho) + f2 / (2 * self.A * self.c) - f3

                prob.update(q=f)
                res = prob.solve()
                z = res.x

                # 更新W
                alpha_gpu = np.asarray(z[:n])
                beta_gpu = np.asarray(z[n:])
                w = -1 / (1 + self.rho) * (XD @ ((self.c * sigma - alpha_gpu + beta_gpu).reshape(-1, 1)) + v)
                W_admm_new = w.reshape(p, q, order='F')
                # 更新b
                mask = (alpha_gpu + beta_gpu / self.tau) > (self.c * self.B)
                if np.sum(mask) > 1:
                    b_mask = self.y[mask] - np.tensordot(W_admm_new, self.X[mask], axes=([0, 1], [1, 2]))
                    b_admm_new = np.sum(b_mask) / np.sum(mask)
                else:
                    b_mask = self.y - np.tensordot(W_admm_new, self.X, axes=([0, 1], [1, 2]))
                    b_admm_new = np.mean(b_mask)

                # 更新S和Lambda
                S_admm_new = self.SVT(self.rho * W_admm_new - Lambda_admm_old, self.gamma) / self.rho
                Lambda_admm_new = Lambda_admm_old + self.rho * (S_admm_new - W_admm_new)

                ao_new = self.admm_obj(W_admm_new, b_admm_new, sigma)

                c2_new = np.linalg.norm(Lambda_admm_new - Lambda_admm_old, ord='fro') ** 2 / self.rho + \
                         np.linalg.norm(S_admm_new - S_admm_old, ord='fro') ** 2 * self.rho
                if c2_new <= 0.999 * c2:
                    c1_new = (1 + np.sqrt(1 + 4 * c1 ** 2)) / 2
                    S_admm_new = S_admm_new + (c1 - 1) / c1_new * (S_admm_new - S_admm_old)
                    Lambda_admm_new = Lambda_admm_new + (c1 - 1) / c1_new * (Lambda_admm_new - Lambda_admm_old)
                    c1, c2 = c1_new, c2_new
                else:
                    S_admm_new, Lambda_admm_new = S_admm_old.copy(), Lambda_admm_old.copy()
                    c1, c2 = 1, c2_new / 0.999

                if self.verbose and (l + 1) % 10 == 0:
                    acc = accuracy_score(
                        self.y,
                        np.sign(np.tensordot(W_admm_new, self.X, axes=([0, 1], [1, 2])) + b_admm_new)
                    )
                    print(f'ADMM/CCCP: {l + 1}/{k + 1}, accuracy: {acc:.4f}, '
                          f'admm objection: {ao_old:.4f} -> {ao_new:.4f}')

                W_admm_old, b_admm_old = W_admm_new.copy(), b_admm_new.copy()
                S_admm_old, Lambda_admm_old = S_admm_new.copy(), Lambda_admm_new.copy()

                if np.abs(ao_new - ao_old) / ao_old < self.epsilon:
                    break

            W_cccp_new, b_cccp_new = W_admm_old.copy(), b_admm_old.copy()
            obj_new = self.obj(W_cccp_new, b_cccp_new)

            if self.verbose:
                print(f'CCCP: {k + 1}, object value: {obj_old:.4f} -> {obj_new:.4f}')

            W_cccp_old, b_cccp_old= W_cccp_new.copy(), b_cccp_new.copy()
            if np.abs(obj_new - obj_old) / obj_old < self.epsilon:
                break

        out = {'W': W_cccp_old, 'b': b_cccp_old}
        return out

    def SVT(self, M, gamma):
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        St = np.diag(np.maximum(s - gamma, 0))
        Dt = U @ St @ Vh

        return Dt

    def h_prime(self, u):
        d1 = self.lambda_ * (self.theta * u**2 + (1 - self.theta) * u)
        d2 = self.lambda_ * self.tau * (self.theta * u**2 - (1 - self.theta) * u)
        out = np.where(u >= 0,
                       self.lambda_ * (2 * self.theta * u + 1 - self.theta) / (self.eta * (1 + d1)**2) - 2 * self.A * u - self.B,
                       self.lambda_ * self.tau * (2 * self.theta * u - 1 + self.theta) / (self.eta * (1 + d2)**2) - \
                       2 * self.A * self.tau**2 * u + self.B * self.tau)
        return out

    def pin(self, u):
        out = np.where(
            u >= 0,
            u,
            -1 * self.tau * u
        )
        return out

    def admm_obj(self, W, b, sigma):
        u = 1 - self.y * (np.einsum('pq, ipq -> i', W, self.X) + b)
        p = self.pin(u)
        out1 = 0.5 * np.trace(W.T @ W)
        s = np.linalg.svd(W, full_matrices=False, compute_uv=False)
        out2 = self.gamma * np.sum(s)
        out3 = self.A * self.c * np.sum(p**2)
        out4 = self.B * self.c * np.sum(p)
        out5 = self.c * np.sum(sigma * self.y * (np.einsum('pq, ipq -> i', W, self.X) + b))
        out = out1 + out2 + out3 + out4 + out5
        return out

    def obj(self, W, b):
        out1 = 0.5 * np.trace(W.T @ W)
        s = np.linalg.svd(W, full_matrices=False, compute_uv=False)
        out2 = self.gamma * np.sum(s)
        u = 1 - self.y * (np.einsum('pq, ipq -> i', W, self.X) + b)
        out3 = self.c * np.sum(self.Baen(u))
        return out1 + out2 + out3

    def Baen(self, u):
        out = 1 / self.eta * (1 - 1 / (1 + self.lambda_ * self.aen(u)))
        return out

    def aen(self, u):
        out = np.where(u >= 0,
                       self.theta * u**2 + (1 - self.theta) * u,
                       self.tau * (self.theta * u**2 - (1 - self.theta) * u))
        return out

    def set_params(self, **parameters):
        """
        设置模型参数
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=False):
        """
        获取模型参数
        """
        return {
            'gamma': self.gamma,
            'c': self.c,
            'eta': self.eta,
            'lambda_': self.lambda_,
            'tau': self.tau,
            'theta': self.theta,
            'rho': self.rho
        }