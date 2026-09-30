# TIT-LR-005｜逻辑回归（Logistic Regression）第 5 次：四项基础特征 + FamilySize

[返回全部实验](../EXPERIMENTS.md) · 旧编号 `T05` · 任务：Titanic 生还二分类（binary classification）

## 配置（Configuration）

| 项目 | 内容 |
| --- | --- |
| 算法 | `logistic_regression` |
| 算法内尝试序号 | 5 |
| 输入原始列 | `Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, `Parch` |
| 进入模型的特征 | `Pclass`, `Sex`, `Age`, `Fare`, `FamilySize` |
| 超参数（hyperparameters） | `{"max_iter": 1000, "random_state": 42}` |
| 数据 SHA256 | `bb1bda464cc6819d412b41d34be69fd89d26b372dc24c09421c3dbca1b0dbe9f` |

预处理（preprocessing）：数值缺失值用训练折中位数填充并缩放；Sex 缺失值用训练折众数填充，再做独热编码（one-hot encoding）。 FamilySize 在每折流程内由 SibSp + Parch + 1 构造。

## 本地验证（Local CV）

- 分层五折，打乱，随机种子 42；准确率（accuracy）均分 **79.23%**，标准差 `0.011000`。
- 五折分数：`0.810056`、`0.780899`、`0.792135`、`0.780899`、`0.797753`。
- 种子 42–46 的五次五折均分平均为 **79.37%**；各次均分：`42: 0.792348`、`43: 0.794614`、`44: 0.795738`、`45: 0.794602`、`46: 0.791262`。

## Kaggle 云端结果（Public leaderboard）

- 未提交；云端分数与名次均未知。

## 复现与判断

本地运行：`.local/venv/bin/python titanic_poc/run.py --experiment-id TIT-LR-005`。结果与预测文件分别保存在 `.local/titanic/experiments/TIT-LR-005/results.json` 和 `submission.csv`；脚本本身不会上传 Kaggle。

将两项家庭计数合为 FamilySize，未加入 Cabin；未提交。

同一训练集被多次用于方案比较，较高的本地分数可能受到选择偏差影响；不要把交叉验证分数当作公榜分数。
