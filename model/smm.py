import numpy as np
from scipy.linalg import svd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from model.utils import clip_dcd_optimizer
import scipy.sparse as sp
import osqp


class SMM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, Tau=0.1, max_steps=500, eps=1e-4,
                 max_iter=500, rho=10, eta=0.999, fit_intercept=True):

        self.C = C
        self.Tau = Tau
        self.max_steps = max_steps
        self.eps = eps
        self.max_iter = max_iter
        self.rho = rho
        self.eta = eta
        self.fit_intercept = fit_intercept

    def _objective_value(self, W, b, X, y):
        """计算目标函数值"""
        n_samples, p, q = X.shape
        X_flat = X.reshape(n_samples, -1)
        w_flat = W.reshape(-1)

        # 计算决策函数
        f = 1 - y * (X_flat @ w_flat + b)
        pos_idx = f >= 0

        # 计算核范数
        _, s, _ = svd(W, full_matrices=False)
        nuclear_norm = np.sum(s)

        obj = 0.5 * np.sum(w_flat ** 2) + self.Tau * nuclear_norm
        obj += self.C * np.sum(f[pos_idx])
        return obj, 0.5 * np.sum(w_flat ** 2) + self.Tau * nuclear_norm, self.C * np.sum(f[pos_idx])

    def _svt(self, X, Tau):
        """奇异值阈值化算子"""
        U, s, Vh = svd(X, full_matrices=False)
        s_thresh = np.maximum(s - Tau, 0)
        rank = np.sum(s_thresh > 0)
        nuclear_norm = np.sum(s_thresh)
        D = U @ np.diag(s_thresh) @ Vh
        return D, nuclear_norm, rank

    def fit(self, X, y):
        """训练SMM模型"""
        y = np.array(y).astype(float)
        n_samples, p, q = X.shape

        X_flat = X.reshape(n_samples, -1)
        n_features = p * q

        yyT = np.outer(y, y)
        H = (X_flat @ X_flat.T) * yyT / (self.rho + 1)
        A = np.vstack([y.reshape(1, -1), np.eye(n_samples)])
        prob = osqp.OSQP()

        v_km1 = np.zeros((n_features, 1))
        v_hatk = v_km1.copy()
        Lambda_km1 = np.ones((n_features, 1))
        Lambda_hatk = Lambda_km1.copy()
        t_k = 1
        c_km1 = 0

        recent_obj = np.zeros(50)
        recent_idx = 0

        for k in range(self.max_iter):
            l_term = 1 - (X_flat @ (Lambda_hatk.flatten() + self.rho * v_hatk.flatten())) * y / (self.rho + 1)
            lb = np.zeros(n_samples)  # 下界为0
            lb = np.concatenate([np.array([0]), lb])
            ub = np.full(n_samples, self.C)  # 上界为C
            ub = np.concatenate([np.array([0]), ub])

            prob.setup(P=sp.csc_matrix(H),
                       q=-l_term,
                       A=sp.csc_matrix(A),
                       l=lb,
                       u=ub,
                       polish=False,
                       verbose=False,
                       warm_start=True,
                       eps_abs=1e-4,
                       eps_rel=1e-4,
                       scaling=True)
            res = prob.solve()
            alpha = res.x
            alpha = alpha.reshape(-1, 1)

            w_flat = (Lambda_hatk + self.rho * v_hatk + X_flat.T @ (alpha * y.reshape(-1, 1))) / (self.rho + 1)
            W_k = w_flat.reshape(p, q)

            support_idx = (alpha.flatten() > 0) & (alpha.flatten() < self.C)
            if np.any(support_idx):
                b = np.mean(y.reshape(-1, 1)[support_idx] - X_flat[support_idx] @ w_flat)
            else:
                b = np.mean(y - X_flat @ w_flat)

            Lambda_mat = Lambda_hatk.reshape(p, q)
            V_mat, _, _ = self._svt(self.rho * W_k - Lambda_mat, self.Tau)
            v_k = V_mat.ravel().reshape(-1, 1) / self.rho

            Lambda_k = Lambda_hatk + self.rho * (v_k - w_flat)

            delta_Lambda = Lambda_k - Lambda_hatk
            delta_v = v_k - v_hatk
            c_k = (np.sum(delta_Lambda ** 2) / self.rho +
                   self.rho * np.sum(delta_v ** 2))

            if c_k < self.eta * c_km1:
                t_k1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k ** 2))
                v_hatk1 = v_k + (t_k - 1) / t_k1 * (v_k - v_km1)
                Lambda_hatk1 = Lambda_k + (t_k - 1) / t_k1 * (Lambda_k - Lambda_km1)
            else:
                t_k1 = 1
                v_hatk1 = v_km1.copy()
                Lambda_hatk1 = Lambda_km1.copy()
                c_k = c_km1 / self.eta

            v_hatk = v_hatk1
            Lambda_hatk = Lambda_hatk1
            v_km1 = v_k.copy()
            Lambda_km1 = Lambda_k.copy()
            c_km1 = c_k
            t_k = t_k1

            current_obj, w_value, loss = self._objective_value(W_k, b, X, y)
            recent_obj[recent_idx] = current_obj
            recent_idx = (recent_idx + 1) % len(recent_obj)

            if k >= len(recent_obj):
                avg_obj = np.mean(recent_obj)
                rel_change = np.abs(current_obj - avg_obj) / (np.abs(avg_obj) + 1e-10)
                if rel_change < self.eps:
                    break

        self.coef_ = W_k
        self.intercept_ = b
        self.n_iter_ = k + 1
        self.support_vectors_ = X
        self.dual_coef_ = alpha * y.reshape(-1, 1)
        return self

    def decision_function(self, X):
        """计算决策函数值"""
        n_samples, p, q = X.shape
        X_flat = X.reshape(n_samples, -1)
        w_flat = self.coef_.reshape(-1)
        return X_flat @ w_flat + self.intercept_

    def predict(self, X):
        """预测类别标签"""
        return np.sign(self.decision_function(X))

    def set_params(self, **parameters):
        """
        设置模型参数
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self