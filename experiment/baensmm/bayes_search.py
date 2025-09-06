import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, make_scorer
import warnings
from model.baensmm_cpu import BaenSMM
from data.Dataset import INRIA, Caltech
warnings.filterwarnings('ignore')


param_space = {
    'gamma': Categorical([0.1, 1, 10, 100, 1000, 10000]),
    'c': Categorical([0.125, 0.25, 0.5, 1, 2, 4, 8]),
    'eta': Categorical([1]),
    'lambda_': Categorical([0.01, 0.1, 1, 10]),
    'tau': Categorical([0.1, 0.3, 0.5, 0.7, 1]),
    'theta': Categorical([0.1, 0.3, 0.5, 0.7, 0.9]),
    'rho': Categorical([100]),
    'cccp_epochs': Categorical([200]),
    'admm_epochs': Categorical([10]),
}

model = BaenSMM(epsilon=1e-4)

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    n_iter=50,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring=make_scorer(accuracy_score),
    n_jobs=6,
    verbose=2,
    random_state=42,
    n_points=6
)

data = Caltech(standardize=True, test_size=0.3, classes='dolphin_vs_butterfly')
X, y = data.get_all()
bayes_search.fit(X, y)

# 输出最佳参数和最佳分数
print("Best parameters found: ", bayes_search.best_params_)
print("Best cross-validation score: ", bayes_search.best_score_)

best_index = bayes_search.best_index_

# 从cv_results_中提取每一折的分数
cv_scores = []
for i in range(bayes_search.cv.n_splits):
    split_key = f'split{i}_test_score'
    if split_key in bayes_search.cv_results_:
        score = bayes_search.cv_results_[split_key][best_index]
        cv_scores.append(score)
        print(f"Fold {i+1} score: {score:.4f}")

# 计算平均分数和标准差
mean_score = np.mean(cv_scores)
std_score = np.std(cv_scores)
print(f"\nAverage CV score: {mean_score:.4f} (±{std_score:.4f})")
