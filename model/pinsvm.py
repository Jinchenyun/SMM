import numpy as np
from numpy.linalg import norm

from model.utils import clip_dcd_optimizer


class PinSVM:
    def __init__(self, C=1, kernel='linear', tau=0.5, gamma=None,
                 degree=3, coef0=0, eps=1e-5, max_steps=4000,
                 fit_intercept=True):
        """
        参数:
        C: 正则化参数
        kernel: 核函数 ('linear', 'rbf', 'poly')
        tau: pinball损失参数
        gamma: 核函数参数 (默认 1/n_features)
        degree: 多项式核阶数
        coef0: 多项式核常数项
        eps: 优化精度
        max_steps: 最大迭代次数
        fit_intercept: 是否添加偏置项
        """
        self.C = C
        self.kernel = kernel
        self.tau = tau
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.eps = eps
        self.max_steps = max_steps
        self.fit_intercept = fit_intercept
        self.class_set = None
        self.coef = None
        self.X_train = None

    def _kernel_matrix(self, X1, X2):
        """计算核矩阵"""
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            gamma = self.gamma if self.gamma else 1.0 / X1.shape[1]
            dist_sq = np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
            return np.exp(-gamma * np.abs(dist_sq))
        elif self.kernel == 'poly':
            gamma = self.gamma if self.gamma else 1.0 / X1.shape[1]
            return (gamma * (X1 @ X2.T) + self.coef0) ** self.degree

    def _calculate_svm_H(self, K, y):
        """计算SVM对偶问题的H矩阵"""
        return np.outer(y, y) * K

    def fit(self, X, y):
        """训练模型"""
        # 预处理标签
        self.class_set = np.unique(y)

        y_trans = np.where(y == self.class_set[0], 1, -1).astype(np.float64)
        y_trans = y_trans.reshape(-1, 1)

        # 添加偏置项
        if self.fit_intercept:
            X = np.column_stack((X, np.ones(X.shape[0])))

        # 设置gamma默认值
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]

        self.X_train = X.copy()

        # 计算核矩阵
        K = self._kernel_matrix(X, X)

        # 准备对偶问题参数
        n = K.shape[0]
        H = self._calculate_svm_H(K, y_trans)
        q = np.ones((n, 1))
        lb = np.full((n, 1), -self.tau * self.C)
        ub = np.full((n, 1), self.C)
        u0 = np.zeros((n, 1)) if self.tau == 0 else (lb + ub) / 2

        # 优化求解
        solver_result = clip_dcd_optimizer(
            H=H, q=q, lb=lb, ub=ub, u0=u0,
            max_steps=self.max_steps, eps=self.eps
        )

        # 保存模型系数
        alphas = solver_result['x'].reshape(-1, 1)
        self.coef = y_trans * alphas

    def decision_function(self, X):
        """计算决策函数值"""
        if self.fit_intercept:
            X = np.column_stack((X, np.ones(X.shape[0])))
        K_test = self._kernel_matrix(X, self.X_train)
        return K_test @ self.coef

    def predict(self, X):
        """预测类别"""
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.class_set[0], self.class_set[1])

    def set_params(self, **parameters):
        """
        设置模型参数
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
