import numpy as np
from scipy.sparse import diags
from model.utils import clip_dcd_optimizer


class ENSVM:
    def __init__(self, C1=1, C2=1, kernel='linear', gamma=None, degree=3, coef0=0,
                 eps=1e-5, max_steps=4000, fit_intercept=True):
        self.C1 = C1
        self.C2 = C2
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.eps = eps
        self.max_steps = max_steps
        self.fit_intercept = fit_intercept
        self.coef = None
        self.X_train = None
        self.y_train = None
        self.class_set = None

    def _calculate_kernel(self, X1, X2=None):
        """计算核矩阵"""
        if X2 is None:
            X2 = X1
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = 1.0 / X1.shape[1]
            dist = np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
            return np.exp(-self.gamma * dist)
        elif self.kernel == 'poly':
            return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        else:
            raise ValueError("Unsupported kernel")

    def _calculate_svm_H(self, K, y):
        """计算Hessian矩阵"""
        return np.outer(y, y) * K

    def fit(self, X, y):
        """训练模型"""
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        # 标签编码处理
        self.class_set = np.unique(y)
        y_encoded = np.where(y == self.class_set[0], 1, -1)

        # 添加截距项
        if self.fit_intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # 设置默认gamma
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]

        # 计算核矩阵
        K = self._calculate_kernel(X)
        n = K.shape[0]

        # 构建二次规划问题
        H0 = self._calculate_svm_H(K, y_encoded)
        np.fill_diagonal(H0, H0.diagonal() + 1 / self.C1)
        IC1 = diags([1 / self.C1], [0], shape=(n, n)).toarray()

        dualH = np.block([[H0, IC1], [IC1, IC1]])
        q = np.zeros(2 * n)
        q[:n] = 1
        lb = np.full(2 * n, -self.C2)
        lb[:n] = 0
        ub = np.full(2 * n, np.inf)
        u0 = lb.copy()

        # 优化求解
        solver_info = clip_dcd_optimizer(
            H=dualH, q=q, lb=lb, ub=ub, u0=u0,
            eps=self.eps, max_steps=self.max_steps
        )

        # 获取系数
        alphas = solver_info['x'].flatten()
        self.coef = y_encoded.flatten() * alphas[:n]
        self.X_train = X
        self.y_train = y_encoded

    def predict(self, X):
        """预测新样本"""
        if self.coef is None:
            raise RuntimeError("模型未训练")

        if self.fit_intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        K_test = self._calculate_kernel(X, self.X_train)
        decision = K_test @ self.coef.reshape(-1, 1)
        return np.where(decision >= 0, self.class_set[0], self.class_set[1])

    def set_params(self, **parameters):
        """
        设置模型参数
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
