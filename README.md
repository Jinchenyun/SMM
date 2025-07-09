# Support Matrix Machine (SMM) Classifier

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![PyPI Version](https://img.shields.io/pypi/v/smm-classifier)]() <!-- 发布后添加 -->

A Python implementation of Support Matrix Machine classifier for matrix-form data, using ADMM optimization.

## Features

- ✅ 支持三维矩阵输入 (n_samples × dim1 × dim2)
- ✅ 基于ADMM的高效优化算法
- ✅ 内置早停机制和训练过程可视化
- ✅ 兼容scikit-learn的API设计

## Installation

### 通过PyPI安装 (推荐)

```bash
pip install smm-classifier
```

### 从源码安装

```bash
git clone https://github.com/yourusername/smm-classifier.git
cd smm-classifier
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

## 进阶功能

### 自定义核函数

```python
class CustomSMM(SMM):
    def _compute_kernel(self, X):
        # 实现自定义核计算逻辑
        return custom_kernel_matrix
```

### 训练过程监控

```python
history = model.fit(X_train, y_train, verbose=True)
# 返回包含各epoch精度和损失的字典
```

## 示例案例

见 [examples/](examples/) 目录：
- `image_classification.ipynb` - 图像分类应用
- `time_series_analysis.ipynb` - 时间序列分析

## 开发指南

### 运行测试

```bash
pytest tests/
```

### 贡献代码

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/your-feature`)
3. 提交修改 (`git commit -am 'Add some feature'`)
4. 推送分支 (`git push origin feature/your-feature`)
5. 创建Pull Request

## 引用

如果您在研究中使用了本库，请引用：

```bibtex
@software{smm_classifier,
  author = {Your Name},
  title = {SMM Classifier},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/smm-classifier}}
```

## 许可证

[MIT License](LICENSE) © 2023 Your Name
