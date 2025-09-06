import numpy as np
import osqp
from scipy.linalg import svd
import scipy.sparse as sp
from .utils import clip_dcd_optimizer


class PinSMM:
    def __init__(self, C=1.0, tau=0.5, Tau=1.0, max_steps=500, eps=1e-4,
                 max_iter=500, rho=10, eta=0.999):
        """
        PinSMM 分类器

        参数:
        C : float, 正则化参数
        tau : float, 分位数损失参数 (0 < tau < 1)
        Tau : float, 核范数正则化参数
        max_steps : int, 坐标下降最大迭代次数
        eps : float, 收敛阈值
        max_iter : int, ADMM最大迭代次数
        rho : float, ADMM惩罚参数
        eta : float, 重启条件参数
        """
        self.C = C
        self.tau = tau
        self.Tau = Tau
        self.max_steps = max_steps
        self.eps = eps
        self.max_iter = max_iter
        self.rho = rho
        self.eta = eta

    def _objective_value(self, w, b, X, y):
        """计算目标函数值"""
        p, q = self.p, self.q
        W = w.reshape(p, q)
        f = 1 - y * (X.dot(w) + b)

        idx = f >= 0
        obj_val = (0.5 * np.sum(w ** 2) +
                   self.Tau * np.sum(svd(W, compute_uv=False)) +
                   self.C * np.sum(f[idx]) -
                   self.C * self.tau * np.sum(f[~idx]))

        return obj_val

    def _svt(self, X, Tau):
        """奇异值阈值操作"""
        U, s, Vt = svd(X, full_matrices=False)
        s_shrink = np.maximum(0, s - Tau)
        D = U @ np.diag(s_shrink) @ Vt
        nuc_norm = np.sum(s_shrink)
        rank = np.sum(s_shrink > 0)
        return D, nuc_norm, rank

    def fit(self, X, y):
        """
        训练PinSMM模型

        参数:
        X : 三维数组, 形状 (n, p, q)
        y : 一维数组, 形状 (n,), 包含±1标签
        """
        # 验证输入
        n, p, q = X.shape
        self.p, self.q = p, q

        # 将X重塑为2D数组 (n, p*q)
        X_flat = X.reshape(n, p * q)

        # 确保y是数值型数组
        y = np.array(y, dtype=np.float64)
        y[y == 0] = -1  # 将0标签转为-1

        # 计算二次型矩阵
        H = (X_flat @ X_flat.T) * np.outer(y, y) / (self.rho + 1)
        A = np.vstack([y.reshape(1, -1), np.eye(n)])
        prob = osqp.OSQP()

        # 初始化变量
        v_km1 = np.zeros(p * q)
        v_hatk = v_km1.copy()
        Lambda_km1 = np.ones(p * q)
        Lambda_hatk = Lambda_km1.copy()
        t_k = 1
        c_km1 = 0

        # 初始化alpha
        if self.tau == 0:
            u0 = np.zeros(n)
        else:
            u0 = np.full(n, (1 - self.tau) * self.C / 2)

        # 设置提前终止迭代指标
        recent_number = 50
        recent_idx = 0
        obj_recent = np.zeros(recent_number)

        # ADMM主循环
        for k in range(self.max_iter):
            # 更新alpha
            l = 1 - (X_flat @ (Lambda_hatk + self.rho * v_hatk)) * y / (self.rho + 1)
            lb = np.full(n, -self.tau * self.C)
            lb = np.concatenate([np.array([0]), lb])
            ub = np.full(n, self.C)
            ub = np.concatenate([np.array([0]), ub])

            prob.setup(P=sp.csc_matrix(H),
                       q=-l,
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

            # result = clip_dcd_optimizer(H, -1*l, lb, ub, self.eps, self.max_steps, u0)
            alpha = res.x.flatten()
            u0 = alpha.copy()

            # 更新w_k和b
            w_k = (Lambda_hatk + self.rho * v_hatk + X_flat.T @ (alpha * y)) / (self.rho + 1)

            # 计算bias项b
            idex = (alpha > 0) & (alpha < self.C)
            if np.any(idex):
                b = np.mean(y[idex] - X_flat[idex] @ w_k)
            else:
                b = 0.0

            # 更新v_k
            W_k = w_k.reshape(p, q)
            Lambda_mat = Lambda_hatk.reshape(p, q)
            V, _, _ = self._svt(self.rho * W_k - Lambda_mat, self.Tau)
            v_k = V.ravel() / self.rho

            # 更新Lambda_k
            Lambda_k = Lambda_hatk + self.rho * (v_k - w_k)

            # 计算收敛指标
            c_k = (np.sum((Lambda_k - Lambda_hatk) ** 2) / self.rho) + \
                  self.rho * np.sum((v_k - v_hatk) ** 2)

            # 重启检查
            if c_k < self.eta * c_km1:
                t_k1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k ** 2))
                v_hatk1 = v_k + (t_k - 1) / t_k1 * (v_k - v_km1)
                Lambda_hatk1 = Lambda_k + (t_k - 1) / t_k1 * (Lambda_k - Lambda_km1)
                restart = False
            else:
                t_k1 = 1
                v_hatk1 = v_km1.copy()
                Lambda_hatk1 = Lambda_km1.copy()
                c_k = c_km1 / self.eta
                restart = True

            # 更新迭代参数
            v_hatk = v_hatk1
            Lambda_hatk = Lambda_hatk1
            v_km1 = v_k.copy()
            Lambda_km1 = Lambda_k.copy()
            c_km1 = c_k
            t_k = t_k1

            # 计算目标函数值
            obj_k = self._objective_value(w_k, b, X_flat, y)
            recent_idx = (recent_idx + 1) % recent_number
            obj_recent[recent_idx] = obj_k

            # 检查收敛性
            if k > recent_number:
                mean_obj = np.mean(obj_recent)
                if abs(obj_k - mean_obj) / abs(mean_obj) < self.eps:
                    break

        # 保存模型参数
        self.w = w_k
        self.b = b
        self.stop_iter = k + 1
        self.n_iter = k + 1
        return self

    def predict(self, X):
        """
        使用训练好的模型进行预测

        参数:
        X : 三维数组, 形状 (m, p, q)

        返回:
        pred : 预测标签 (±1)
        """
        m, p, q = X.shape
        if p != self.p or q != self.q:
            raise ValueError(f"输入数据的形状({p},{q})与训练数据({self.p},{self.q})不匹配")

        # 将X重塑为2D数组
        X_flat = X.reshape(m, self.p * self.q)

        # 计算预测值并返回符号
        pred = X_flat @ self.w + self.b
        return np.sign(pred)

    def set_params(self, **parameters):
        """
        设置模型参数
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self