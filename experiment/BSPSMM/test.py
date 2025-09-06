import time
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from model.baensmm_cpu import BaenSMM
from data.Dataset import INRIA, Caltech, BCI_IV_IIa
from sklearn.metrics import accuracy_score

params = [{'c': 0.125, 'eta': 1, 'gamma': 10, 'lambda_': 10, 'rho': 50, 'tau': 0.3, 'theta': 1},
          {'c': 4, 'eta': 1, 'gamma': 100, 'lambda_': 1, 'rho': 50, 'tau': 0.1, 'theta': 1},
          {'c': 0.25, 'eta': 1, 'gamma': 10, 'lambda_': 10, 'rho': 50, 'tau': 1, 'theta': 1},
          {'c': 2, 'eta': 1, 'gamma': 100, 'lambda_': 1, 'rho': 50, 'tau': 0.3, 'theta': 1},
          {'c': 0.125, 'eta': 1, 'gamma': 1, 'lambda_': 0.1, 'rho': 50, 'tau': 0.1, 'theta': 1},
          {'c': 1, 'eta': 1, 'gamma': 1000, 'lambda_': 10, 'rho': 50, 'tau': 1, 'theta': 1},
          {'c': 2, 'eta': 1, 'gamma': 10000, 'lambda_': 10, 'rho': 50, 'tau': 1, 'theta': 1},
          {'c': 2, 'eta': 1, 'gamma': 100, 'lambda_': 1, 'rho': 50, 'tau': 0.1, 'theta': 1},
          {'c': 0.5, 'eta': 1, 'gamma': 1000, 'lambda_': 10, 'rho': 50, 'tau': 0.3, 'theta': 1},
          {'c': 0.125, 'eta': 1, 'gamma': 10, 'lambda_': 1, 'rho': 50, 'tau': 0.1, 'theta': 1}]

classes = 'dolphin_vs_camera'
data_clean = Caltech(standardize=True, test_size=0.3, classes=classes, noise=None)
data_gn = Caltech(standardize=True, test_size=0.3, classes=classes, noise='gaussian')
data_spn = Caltech(standardize=True, test_size=0.3, classes=classes, noise='spn')

X_clean, y_clean = data_clean.get_all()
X_gn, y_gn = data_gn.get_all()
X_spn, y_spn = data_spn.get_all()

evaluate_result_clean = []
evaluate_result_gn = []
evaluate_result_spn = []

epoch = 0
for best_param in params:
    print(best_param)
    # 无噪声数据集评估
    X_train_val_clean, X_test_clean, y_train_val_clean, y_test_clean = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=int(2 * epoch + 1)
    )

    model = BaenSMM(cccp_epochs=200, admm_epochs=10, epsilon=1e-4)
    model.set_params(**best_param)
    model.fit(X_train_val_clean, y_train_val_clean, verbose=False)
    y_pred = model.predict(X_test_clean)
    acc_clean = accuracy_score(y_test_clean, y_pred)
    evaluate_result_clean.append(acc_clean)

    # 高斯噪声数据集评估
    X_train_val_gn, X_test_gn, y_train_val_gn, y_test_gn = train_test_split(
        X_gn, y_clean, test_size=0.3, random_state=int(2 * epoch + 1)
    )
    model = BaenSMM(cccp_epochs=200, admm_epochs=10, epsilon=1e-4)
    model.set_params(**best_param)
    model.fit(X_train_val_gn, y_train_val_gn, verbose=False)
    y_pred = model.predict(X_test_gn)
    acc_gn = accuracy_score(y_test_gn, y_pred)
    evaluate_result_gn.append(acc_gn)

    # 椒盐噪声数据集评估
    X_train_val_spn, X_test_spn, y_train_val_spn, y_test_spn = train_test_split(
        X_spn, y_spn, test_size=0.3, random_state=int(2 * epoch + 1)
    )
    model = BaenSMM(cccp_epochs=200, admm_epochs=10, epsilon=1e-4)
    model.set_params(**best_param)
    model.fit(X_train_val_spn, y_train_val_spn, verbose=False)
    y_pred = model.predict(X_test_spn)
    acc_spn = accuracy_score(y_test_spn, y_pred)
    evaluate_result_spn.append(acc_spn)

    epoch += 1

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
