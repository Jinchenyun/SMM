import time
import numpy as np
from sklearn.model_selection import train_test_split
from model.baensmm_gpu import BaenSMM
from data.Dataset import INRIA
from sklearn.metrics import accuracy_score

params = {'gamma': 100, 'c': 4, 'eta': 1, 'lambda_': 0.1, 'tau': 0.3, 'theta': 0, 'rho': 10}
data = INRIA(standardize=True, test_size=0.3, noise=None)
# params = {'Tau': 10, 'C': 0.1}
X, y = data.get_all()
# X_train, y_train = data.get_train()-
# X_test, y_test = data.get_test()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=int(9), shuffle=True)

print(np.unique(y_train, return_counts=True))
s = time.time()
print('start')
model = BaenSMM(cccp_epochs=200, admm_epochs=2, epsilon=1e-4)
# model = SMM()
model.set_params(**params)
model.fit(X_train, y_train, verbose=True)
# model.fit(X_train, y_train)
e = time.time()
print(f'\n{(e-s)/60}min')
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, model.predict(X_test))
print(f'\n测试集准确率：{acc}')

# W = cp.asnumpy(model.W)
# b = cp.asnumpy(model.b)
#
# print(f'\n0.5*tr(W^T*W): {0.5 * np.trace(W.T @ W):.4f}')
# print(f'Γ||W||_*: {params['gamma'] * np.sum(np.linalg.svd(W, full_matrices=False, compute_uv=False))}')
# print(f'C*sum(L_Baen(u)): {params['c'] * np.sum(Baen(1-y_train*(np.einsum('pq, ipq -> i', W, X_train)+b),
#                                                      params['eta'], params['lambda_'], params['tau'], params['theta'])):.4f}')
# #
# train_out = model.decision(X_train)
# # test_out = model.decision(X_test)
# # print(test_out)
# print(f'mean of tr(W^T*X)+b: {np.mean(train_out)}')
# plt.hist(1-y_train*train_out, bins=50)
# plt.title("Distribution of u = 1 - y(wx + b)")
# plt.xlabel("u")
# plt.ylabel("Frequency")
# plt.show()


#
#
# cm = confusion_matrix(y_test, y_pred)
#
# # 绘制混淆矩阵
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap='Blues', values_format='d')  # 'd'表示显示整数
# plt.title('Confusion Matrix')
# plt.show()