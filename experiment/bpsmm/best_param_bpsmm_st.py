import itertools
import random
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from data.Dataset import INRIA, Caltech, BCI_IV_IIa
from sklearn.metrics import accuracy_score
from model.baensmm_cpu import BaenSMM
from tqdm import tqdm


param_grid = {
    'gamma': [10 ** i for i in range(-1, 5)],
    'c': [2 ** i for i in range(-3, 4)],
    'eta': [1],
    'lambda_': [0.01, 0.1, 1, 10],
    'tau': [0.1, 0.3, 0.5, 0.7, 1],
    'theta': [0],
    'rho': [10]
}

print('finding valid combinations...')
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print('Load dataset...')
# data = Caltech(standardize=True, test_size=0.3, classes='cup_vs_butterfly')
data = BCI_IV_IIa(standardize=True, test_size=0.3, classes='left_vs_right', subject='A01T')
X, y = data.get_all()
print(X.shape)
print(np.unique(y, return_counts=True))

print('start training...')
best_score = 0
best_params = None
best_std = 0
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for params in tqdm(param_combinations, desc="超参数调优进度", unit="组"):
    fold_scores = []
    i = 1
    for train_index, val_index in kf.split(X):
        print(f'{i} | {params}')
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = BaenSMM(cccp_epochs=500, admm_epochs=10, epsilon=1e-4)
        model.set_params(**params)
        model.fit(X_train, y_train, verbose=False)
        score = accuracy_score(y_val, model.predict(X_val))
        fold_scores.append(score)
        i += 1

    mean_score = np.mean(fold_scores)
    tqdm.write(f"   当前分数! 分数: {mean_score:.4f}, 标准差: {np.std(np.array(fold_scores)):.4f}")
    tqdm.write(f'   当前超参数：{params}')

    if mean_score > best_score:
        best_score = mean_score
        best_params = params
        best_std = np.std(np.array(fold_scores))
        tqdm.write(f"🔥 发现最佳分数! 分数: {mean_score:.4f}, 标准差: {np.std(np.array(fold_scores)):.4f}")
        tqdm.write(f'🔥 发现最佳超参数：{params}')
    elif np.std(np.array(fold_scores)) < best_std and mean_score == best_score:
        best_score = mean_score
        best_params = params
        best_std = np.std(np.array(fold_scores))
        tqdm.write(f"🔥 发现最佳分数! 分数: {mean_score:.4f}, 标准差: {np.std(np.array(fold_scores)):.4f}")
        tqdm.write(f'🔥 发现最佳超参数：{params}')
    else:
        tqdm.write(f"🔥 当前最佳分数! 分数: {best_score:.4f}, 标准差: {best_std:.4f}")
        tqdm.write(f'🔥 当前最佳超参数：{best_params}')

print('best parameters: ', best_params)
print(f'average accuracy: {np.mean(np.array(best_score)):.4f} ± {best_std:.4f}')
