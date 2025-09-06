import numpy as np
from sklearn.model_selection import train_test_split, KFold
from data.Dataset import INRIA, Caltech
from sklearn.metrics import accuracy_score
from model.alssvm import ALSSVM

params = {'C': 0.125, 'p': 0.1, 'kernel': 'linear'}
data = Caltech(standardize=True, test_size=0.5, classes='cup_vs_butterfly', noise='spn')
X, y = data.get_all()
n, p, q = X.shape
X = X.reshape(n, -1, order='F')

print('start training...')
accuracies = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
i = 1
for train_index, val_index in kf.split(X):
    print(f'{i} | {params}')
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    model = ALSSVM(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)
    i += 1

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f"平均准确率: {mean_acc*100:.2f}±{std_acc*100:.2f}")