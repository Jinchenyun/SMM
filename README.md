# Support Matrix Machine (SMM) Classifier

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![PyPI Version](https://img.shields.io/pypi/v/smm-classifier)]() <!-- 发布后添加 -->

A Python implementation of Support Matrix Machine classifier for matrix-form data, using ADMM optimization.


## Installation

### 通过PyPI安装 (推荐)

```bash
pip install smm-classifier
```

### 从源码安装

```bash
git clone https://github.com/yourusername/SMM.git
cd SMM
pip install -e .
```

## 快速开始

```python
import numpy as np
from smm import SMM

# 生成随机数据
X_train = np.random.randn(100, 10, 10)  # 100个10x10矩阵样本
y_train = np.sign(np.random.randn(100)) # 二分类标签(-1或1)

# 初始化模型
model = SMM(c=1.0, p=2.0, epochs=100)

# 训练
model.fit(X_train, y_train)

# 预测
X_test = np.random.randn(20, 10, 10)
y_pred = model.predict(X_test)

# 评估
accuracy = model.score(X_test, y_test)
```

## 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|---------|-------------|
| `c` | float | 1.0 | 正则化系数 |
| `p` | float | 2.0 | ADMM惩罚参数 |
| `tao` | float | 1.0 | 奇异值阈值 |
| `epochs` | int | 100 | 最大训练轮数 |
| `random_seed` | int | 42 | 随机种子 |

