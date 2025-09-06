import time
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from model.baensmm_cpu import BaenSMM
from data.Dataset import INRIA, Caltech
from sklearn.metrics import accuracy_score

params = {'gamma': 1, 'c': 0.25, 'eta': 1, 'lambda_': 0.1, 'tau': 0.1, 'theta': 0.3, 'rho': 50}
data = Caltech(standardize=True, test_size=0.3, classes='cup_vs_butterfly', noise='spn')
X, y = data.get_all()
n, p, q = X.shape

fold_scores = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
s = time.time()
i = 1
for train_index, val_index in kf.split(X):
    print(f'{i} | {params}')
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    model = BaenSMM(cccp_epochs=200, admm_epochs=30, epsilon=1e-4)
    model.set_params(**params)

    model.fit(X_train, y_train, verbose=True)
    score = accuracy_score(y_val, model.predict(X_val))
    fold_scores.append(score)
    i += 1

print(f'平均准确率:{np.mean(np.array(fold_scores))*100:.2f}±{np.std(np.array(fold_scores))*100:.2f}')
print(f'{(time.time()-s)/60}min')
