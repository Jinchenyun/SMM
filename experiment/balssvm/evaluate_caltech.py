import itertools
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from tqdm import tqdm
from data.Dataset import INRIA, Caltech, BCI_IV_IIa
from model.baensvm import BAENSVM
from sklearn.metrics import accuracy_score
import numpy as np


param_grid = {
    'C': [2**i for i in range(-3, 4)],
    'lambda_val': [0.01, 0.1, 1, 10],
    'tau': [0.1, 0.3, 0.5, 0.7, 1],
    'theta': [1],
    'kernel': ['linear']
}
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

classes = 'camera_vs_butterfly'
data_clean = Caltech(standardize=True, test_size=0.3, classes=classes, noise=None)
data_gn = Caltech(standardize=True, test_size=0.3, classes=classes, noise='gaussian')
data_spn = Caltech(standardize=True, test_size=0.3, classes=classes, noise='spn')

X_clean, y_clean = data_clean.get_all()
X_gn, y_gn = data_gn.get_all()
X_spn, y_spn = data_spn.get_all()

n = X_clean.shape[0]
X_clean = X_clean.reshape(n, -1, order='F')
X_gn = X_gn.reshape(n, -1, order='F')
X_spn = X_spn.reshape(n, -1, order='F')

evaluate_result_clean = []
evaluate_result_gn = []
evaluate_result_spn = []

# 外层进度条
outer_pbar = tqdm(range(10), desc="模型评估进度", unit="run", position=0)

for epoch in outer_pbar:
    results = []

    # k折交叉验证+网格搜索
    X_train_val_clean, X_test_clean, y_train_val_clean, y_test_clean = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=int(2 * epoch + 1)
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for params in ParameterGrid(param_grid):
        accuracies = []

        for train_index, val_index in kf.split(X_train_val_clean):
            X_train, X_val = X_train_val_clean[train_index], X_train_val_clean[val_index]
            y_train, y_val = y_train_val_clean[train_index], y_train_val_clean[val_index]
            model = BAENSVM()
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        # 存储结果
        results.append({
            'params': params,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc
        })

    # 按平均准确率排序结果
    sorted_results = sorted(results, key=lambda x: x['mean_accuracy'], reverse=True)

    # 无噪声数据集评估
    best_param = sorted_results[0]['params']
    model = BAENSVM()
    model.set_params(**best_param)
    model.fit(X_train_val_clean, y_train_val_clean)
    y_pred = model.predict(X_test_clean)
    acc_clean = accuracy_score(y_test_clean, y_pred)
    evaluate_result_clean.append(acc_clean)

    # 高斯噪声数据集评估
    X_train_val_gn, X_test_gn, y_train_val_gn, y_test_gn = train_test_split(
        X_gn, y_clean, test_size=0.3, random_state=int(2 * epoch + 1)
    )
    model = BAENSVM()
    model.set_params(**best_param)
    model.fit(X_train_val_gn, y_train_val_gn)
    y_pred = model.predict(X_test_gn)
    acc_gn = accuracy_score(y_test_gn, y_pred)
    evaluate_result_gn.append(acc_gn)

    # 椒盐噪声数据集评估
    X_train_val_spn, X_test_spn, y_train_val_spn, y_test_spn = train_test_split(
        X_spn, y_spn, test_size=0.3, random_state=int(2 * epoch + 1)
    )
    model = BAENSVM()
    model.set_params(**best_param)
    model.fit(X_train_val_spn, y_train_val_spn)
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

print(f'模型评估结果：\n'
      f'无噪声数据集评估结果：{acc_mean_clean * 100:.2f} ± {acc_std_clean * 100:.2f}\n'
      f'高斯噪声数据集评估结果：{acc_mean_gn * 100:.2f} ± {acc_std_gn * 100:.2f}\n'
      f'椒盐噪声数据集评估结果：{acc_mean_spn * 100:.2f} ± {acc_std_spn * 100:.2f}')
