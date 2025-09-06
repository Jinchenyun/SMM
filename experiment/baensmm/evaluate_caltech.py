import itertools
import random
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from data.Dataset import INRIA, Caltech, BCI_IV_IIa
from sklearn.metrics import accuracy_score
from model.baensmm_cpu import BaenSMM
from tqdm import tqdm


cccp_epochs = 200
admm_epochs = 10


def evaluate_params(params, X, y, epoch):
    X_train_val_clean, X_test_clean, y_train_val_clean, y_test_clean = train_test_split(
        X, y, test_size=0.3, random_state=int(2 * epoch + 1)
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    try:
        for train_index, val_index in kf.split(X_train_val_clean):
            X_train, X_val = X_train_val_clean[train_index], X_train_val_clean[val_index]
            y_train, y_val = y_train_val_clean[train_index], y_train_val_clean[val_index]
            model = BaenSMM(cccp_epochs=cccp_epochs, admm_epochs=admm_epochs, epsilon=1e-4)
            model.set_params(**params)
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            accuracies.append(acc)

        mean_score = np.mean(accuracies)
        std_score = np.std(accuracies)
    except:
        mean_score = 0
        std_score = 0

    return {
        'params': params,
        'mean_accuracy': mean_score,
        'std_accuracy': std_score,
        'fold_scores': accuracies
    }


classes = 'dolphin_vs_cup'
data_clean = Caltech(standardize=True, test_size=0.3, classes=classes, noise=None)
data_gn = Caltech(standardize=True, test_size=0.3, classes=classes, noise='gaussian')
data_spn = Caltech(standardize=True, test_size=0.3, classes=classes, noise='spn')

X_clean, y_clean = data_clean.get_all()
X_gn, y_gn = data_gn.get_all()
X_spn, y_spn = data_spn.get_all()

param_grid = {
    'gamma': [10 ** i for i in range(-1, 5)],
    'c': [2 ** i for i in range(-3, 4)],
    'eta': [1],
    'lambda_': [0.01, 0.1, 1, 10],
    'tau': [0.1, 0.3, 0.5, 0.7, 1],
    'theta': [0.1, 0.3, 0.5, 0.7, 0.9],
    'rho': [50]
}


best_params = []
evaluate_result_clean = []
evaluate_result_gn = []
evaluate_result_spn = []

# 外层进度条
outer_pbar = tqdm(range(10), desc="模型评估进度", unit="run", position=0)

for epoch in outer_pbar:
    n_jobs = 15
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(params, X_clean, y_clean, epoch) for params in ParameterGrid(param_grid)
    )

    # 按平均准确率排序结果
    sorted_results = sorted(results, key=lambda x: x['mean_accuracy'], reverse=True)
    best_param = sorted_results[0]['params']
    best_params.append(best_param)

    # 无噪声数据集评估
    X_train_val_clean, X_test_clean, y_train_val_clean, y_test_clean = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=int(2 * epoch + 1)
    )

    model = BaenSMM(cccp_epochs=cccp_epochs, admm_epochs=admm_epochs, epsilon=1e-4)
    model.set_params(**best_param)
    model.fit(X_train_val_clean, y_train_val_clean, verbose=False)
    y_pred = model.predict(X_test_clean)
    acc_clean = accuracy_score(y_test_clean, y_pred)
    evaluate_result_clean.append(acc_clean)

    # 高斯噪声数据集评估
    X_train_val_gn, X_test_gn, y_train_val_gn, y_test_gn = train_test_split(
        X_gn, y_clean, test_size=0.3, random_state=int(2 * epoch + 1)
    )
    model = BaenSMM(cccp_epochs=cccp_epochs, admm_epochs=admm_epochs, epsilon=1e-4)
    model.set_params(**best_param)
    model.fit(X_train_val_gn, y_train_val_gn, verbose=False)
    y_pred = model.predict(X_test_gn)
    acc_gn = accuracy_score(y_test_gn, y_pred)
    evaluate_result_gn.append(acc_gn)

    # 椒盐噪声数据集评估
    X_train_val_spn, X_test_spn, y_train_val_spn, y_test_spn = train_test_split(
        X_spn, y_spn, test_size=0.3, random_state=int(2 * epoch + 1)
    )
    model = BaenSMM(cccp_epochs=cccp_epochs, admm_epochs=admm_epochs, epsilon=1e-4)
    model.set_params(**best_param)
    model.fit(X_train_val_spn, y_train_val_spn, verbose=False)
    y_pred = model.predict(X_test_spn)
    acc_spn = accuracy_score(y_test_spn, y_pred)
    evaluate_result_spn.append(acc_spn)

# 关闭外层进度条
outer_pbar.close()

evaluate_result_clean = np.array(evaluate_result_clean)
evaluate_result_gn = np.array(evaluate_result_gn)
evaluate_result_spn = np.array(evaluate_result_spn)

acc_mean_clean = np.mean(evaluate_result_clean)
acc_std_clean = np.std(evaluate_result_clean)

acc_mean_gn = np.mean(evaluate_result_gn)
acc_std_gn = np.std(evaluate_result_gn)

acc_mean_spn = np.mean(evaluate_result_spn)
acc_std_spn = np.std(evaluate_result_spn)

print(best_params)
print(f'模型评估结果：\n'
      f'无噪声数据集评估结果：{acc_mean_clean * 100:.2f} ± {acc_std_clean * 100:.2f}\n'
      f'高斯噪声数据集评估结果：{acc_mean_gn * 100:.2f} ± {acc_std_gn * 100:.2f}\n'
      f'椒盐噪声数据集评估结果：{acc_mean_spn * 100:.2f} ± {acc_std_spn * 100:.2f}')
