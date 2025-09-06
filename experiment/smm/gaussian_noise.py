import numpy as np
from sklearn.model_selection import train_test_split, KFold
from data.Dataset import INRIA, Caltech
from sklearn.metrics import accuracy_score
from model.smm import SMM

params = {'Tau': 100, 'C': 1}
data = Caltech(standardize=True, test_size=0.5, classes='cup_vs_butterfly', noise='gaussian')
X, y = data.get_all()
print(X.shape)
n, p, q = X.shape

fold_scores = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
i = 1
for train_index, val_index in kf.split(X):
    print(f'{i} | {params}')
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = SMM()
    model.set_params(**params)
    model.fit(X_train, y_train)
    score = accuracy_score(y_val, model.predict(X_val))
    fold_scores.append(score)

    i += 1
mean = np.mean(fold_scores)
std = np.std(np.array(fold_scores))

print(f"平均准确率: {mean*100:.2f}±{std*100:.2f}")
