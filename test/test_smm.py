import numpy as np
import os
import gzip
import pandas as pd
from io import StringIO
from smm.smm import SMM


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(project_root, "test", "test_data", "SMNI_CMI_TRAIN")
test_path = os.path.join(project_root, "test", "test_data", "SMNI_CMI_TEST")

# 存储数据和标签
X_train = []
y_train = []
X_test = []
y_test = []

print('Reading train dataset...')
for subdir in os.listdir(train_path):
    subdir_path = os.path.join(train_path, subdir)

    if not os.path.isdir(subdir_path):
        continue

    label = 1 if subdir[3] == 'a' else -1
    for file in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file)
        with gzip.open(file_path, 'rt') as f:
            content = f.read()
            df = pd.read_csv(StringIO(content),
                             sep=r'\s+',  # 空格分隔
                             comment='#',  # 忽略#开头的行
                             header=None,
                             names=['trial', 'channel', 'sample', 'voltage'])
            d = df.pivot(index='sample', columns='channel', values='voltage')
        X_train.append(d)
        y_train.append(label)

print('Reading test dataset...')
for subdir in os.listdir(test_path):
    subdir_path = os.path.join(test_path, subdir)

    if not os.path.isdir(subdir_path):
        continue

    label = 1 if subdir[3] == 'a' else -1
    for file in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file)
        with gzip.open(file_path, 'rt') as f:
            content = f.read()
            df = pd.read_csv(StringIO(content),
                             sep=r'\s+',  # 空格分隔
                             comment='#',  # 忽略#开头的行
                             header=None,
                             names=['trial', 'channel', 'sample', 'voltage'])
            d = df.pivot(index='sample', columns='channel', values='voltage')
        X_test.append(d)
        y_test.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

m = np.mean(np.vstack((X_train, X_test)), axis=0)
s = np.std(np.vstack((X_train, X_test)), axis=0)
X_train = (X_train - m) / s
X_test = (X_test - m) / s

smm = SMM()
smm.fit(X_train, y_train)

print(f'Accuracy of train dataset: {smm.score(X_train, y_train)}')
print(f'Accuracy of test dataset: {smm.score(X_test, y_test)}')
