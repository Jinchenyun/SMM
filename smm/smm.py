import numpy as np
import cvxpy as cp
from sklearn.metrics import accuracy_score
import random
from smm.utils import g, Baen, h_baen_diff, h_bsp_diff, Bsp


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
        out = self.ADMM(self.X, self.y, self.W, self.b, self.c, self.p, self.S, self.A, self.tau, self.epochs)

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

    def ADMM(self, X, y, W, b, c, p, S, A, tau, epochs):
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

            W_temp = (A + p * S + np.einsum('i, i, ipq -> pq', a, y, X)) / (1 + p)
            s = np.where((0 < a) & (a < c))
            b_temp = np.sum(y[s] * np.trace((A + p * S).T @ X[s], axis1=1, axis2=2)) / s[0].shape[0]

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


class RobustSMM():
    """
    Support Matrix Machine (SMM) Based on Bounded Loss.

    Parameters:
    -----------
    gamma : float, default=1.0, range: be > 0
        Penalty parameter of nuclear norm
    c : float, default=1.0, range: be > 0
        Regularization parameter.
    eta : float, default=2.0, range: be > 0
        Parameter of bounded asymmetric elastic net loss
    lambda : float, defauit=1.0
        Parameter of bounded asymmetric elastic net loss
    tau : float, default=0.5, range: (0.0, 1.0)
        Parameter of asymmetric elastic net loss and pinball loss
    theta: float, default=1.0
        Parameter of asymmetric elastic net loss
    rho : float, default=2.0
        ADMM penalty parameter.
    random_seed : int, default=42
        Random seed for reproducibility..
    cccp_epochs : int, default=100
        Number of iterations in CCCP of CCCP-ADMM
    admm_epochs : int default=100
        Number of iiterations in ADMM of CCCP-ADMM
    loss : str default='baen'
        Name of Loss function
    epsilon : float default=1e-6
    """

    def __init__(self, gamma=0.1, c=1, eta=2, lambda_=1, tau=0.5, theta=1, cccp_epochs=100, rho=2, admm_epochs=100,
                 epsilon=1e-3,
                 loss='baen', random_seed=42):
        np.random.seed(int(random_seed))
        random.seed(int(random_seed))

        self.gamma = gamma
        self.c = c
        self.eta = eta
        self.lambda_ = lambda_
        self.tau = tau
        self.theta = theta
        self.rho = rho
        self.cccp_epochs = cccp_epochs
        self.admm_epochs = admm_epochs
        self.epsilon = epsilon
        self.loss = loss
        self.W = None
        self.b = None
        self.S = None
        self.A = None

    def fit(self, X, y, verbose=False):
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

        out = self.cccp_admm(self.X, self.y, self.W, self.b, self.S, self.A, self.eta, self.gamma, self.c, self.lambda_,
                             self.tau,
                             self.theta, self.rho, self.cccp_epochs, self.admm_epochs,
                             self.epsilon, self.loss, verbose)
        self.W = out['W']
        self.b = out['b']

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

    def cccp_admm(self, X, y, W, b, S, A, eta, gamma, c, lambda_, tau, theta, rho, cccp_epochs, admm_epochs, epsilon,
                  loss, verbose):
        """
            CCCP-ADMM optimization algorithm for RobustSMM.
        """
        n = self.X.shape[0]
        traces = np.einsum('ipq,jpq->ij', X, X)
        K = np.outer(y, y) * traces / (rho + 1)
        K += 1e-10 * np.eye(K.shape[0])

        for cccp_epoch in range(cccp_epochs):
            W_old = W
            b_old = b
            sigma = self.caculate_sigma(X, y, W, b, eta, lambda_, theta, tau, loss)

            for admm_epoch in range(admm_epochs):
                alpha = 1
                beta = 1

                # update W and b
                q = np.ones(n) - (y * np.trace((A + rho * S).T @ X, axis1=1, axis2=2)) / (rho + 1)
                a = cp.Variable(n)
                objective = cp.Minimize(0.5 * cp.quad_form(a, K) - q.T @ a)
                l = -1 * c * lambda_ * tau - c * sigma
                u = c * lambda_ - c * sigma
                constraints = [
                    a >= l,
                    a <= u,
                    y.T @ a == 0
                ]
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.OSQP, verbose=False)
                a = a.value
                W_new = (A + rho * S + np.einsum('i, i, ipq -> pq', a, y, X)) / (1 + rho)
                s = np.where((l < a) & (a < u))
                b_new = np.sum(y[s] * np.trace((A + rho * S).T @ X[s], axis1=1, axis2=2)) / s[0].shape[0]

                # update S and A
                S_new = self.SVT(rho * W_new - A, gamma) / rho
                A_new = A + rho * (S_new - W_new)

                g_old = self.caculate_g(X, y, W, b, eta, lambda_, tau, theta, gamma, loss)
                g_new = self.caculate_g(X, y, W_new, b_new, eta, lambda_, tau, theta, gamma, loss)
                if np.abs((g_new - g_old) / g_old) <= epsilon:
                    W = W_new
                    b = b_new
                    S = S_new
                    A = A_new
                    break
                else:
                    beta_new = (np.linalg.norm(A_new - A, ord='fro') ** 2 / rho +
                                rho * np.linalg.norm(S_new - S, ord='fro') ** 2)
                    if beta_new < 0.99 * beta:
                        # 加速求解
                        alpha_new = (1 + np.sqrt(1 + 4 * alpha ** 2)) / 2
                        S_new = S_new + (alpha - 1) / alpha_new * (S_new - S)
                        A_new = A_new + (alpha - 1) / alpha_new * (A_new - S)
                    else:
                        # 回退
                        alpha_new = 1
                        S_new = S
                        A_new = A
                        beta_new = beta / 0.99

                    W = W_new
                    b = b_new
                    S = S_new
                    A = A_new
                    alpha = alpha_new
                    beta = beta_new

                    if verbose:
                        if admm_epoch % 10 == 0:
                            pre = np.sign(np.trace(W.T @ X, axis1=1, axis2=2) + b)
                            acc = accuracy_score(y, pre)
                            print(f'Epoch{admm_epoch}/{cccp_epoch}/{cccp_epochs}, accuracy of train: {acc}')

            f_old = (0.5 * np.trace(W.T @ W) + np.sum(np.linalg.svd(W, compute_uv=False)) +
                     c * self.caculate_f(X, y, W_old, b_old, eta, lambda_, tau, theta, loss))
            f_new = (0.5 * np.trace(W.T @ W) + np.sum(np.linalg.svd(W, compute_uv=False)) +
                     c * self.caculate_f(X, y, W, b, eta, lambda_, tau, theta, loss))
            if np.abs((f_new - f_old) / f_old) <= epsilon:
                break
        return {'W': W, 'b': b, 'S': S, 'A': A}

    def caculate_sigma(self, X, y, W, b, eta, lambda_, theta, tau, loss):
        sigma = np.zeros((y.shape[0]))
        if loss == 'baen':
            u = 1 - y * (np.trace(W.T @ X, axis1=1, axis2=2) + b)
            u_sq = u ** 2
            denominator_base = 1 + lambda_ * (theta / 2 * u_sq + (1 - theta) * np.abs(u))
            denominator = eta * denominator_base ** 2

            mask_pos = u >= 0
            numerator_pos = theta * (u[mask_pos] - 1) + 1
            sigma[mask_pos] = lambda_ * (1 - numerator_pos / denominator[mask_pos])

            mask_neg = ~mask_pos
            numerator_neg = theta * (u[mask_neg] + 1) - 1
            sigma[mask_neg] = -lambda_ * tau * (1 + numerator_neg / denominator[mask_neg])
        elif loss == 'bsp':
            u = 1 - y * (np.trace(W.T @ X, axis1=1, axis2=2) + b)
            denominator = (1 + lambda_ * u ** 2) ** 2

            mask_pos = u >= 0
            sigma[mask_pos] = lambda_ * (1 - 2 * u[mask_pos] / denominator[mask_pos])

            mask_neg = ~mask_pos
            sigma[mask_neg] = -lambda_ * tau * (1 + 2 * u[mask_neg] / denominator[mask_neg])

        return sigma

    def SVT(self, M, gamma):
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        St = np.diag(np.maximum(s - gamma, 0))
        Dt = U @ St @ Vh
        return Dt

    def caculate_g(self, X, y, W, b, eta, lambda_, tau, theta, gamma, loss):
        if loss == 'baen':
            trace_terms = np.trace(W.T@X, axis1=1, axis2=2)
            u = 1 - y * (trace_terms + b)
            g1 = np.sum(g(u, lambda_, tau))
            g2 = np.sum(-h_baen_diff(u, eta, lambda_, tau, theta) * y * (trace_terms + b))
            out = 0.5 * np.trace(W.T @ W) + gamma * np.sum(np.linalg.svd(W, compute_uv=False)) + g1 + g2
            return out
        elif loss == 'bsp':
            trace_terms = np.trace(W.T @ X, axis1=1, axis2=2)
            u = 1 - y * (trace_terms + b)
            g1 = np.sum(g(u, lambda_, tau))
            g2 = np.sum(-h_bsp_diff(u, eta, lambda_, tau, theta) * y * (trace_terms + b))
            out = 0.5 * np.trace(W.T @ W) + gamma * np.sum(np.linalg.svd(W, compute_uv=False)) + g1 + g2
            return out



    def caculate_f(self, X, y, W, b, eta, lambda_, tau, theta, loss):
        if loss == 'baen':
            u = 1 - y * (np.trace(W.T @ X, axis1=1, axis2=2) + b)
            f = np.sum(Baen(u, eta, lambda_, tau, theta))
            return f
        elif loss == 'bsp':
            u = 1 - y * (np.trace(W.T @ X, axis1=1, axis2=2) + b)
            f = Bsp(u, eta, lambda_, tau)
            return f
