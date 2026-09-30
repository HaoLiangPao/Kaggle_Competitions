# TIT-BL-001｜多数类基线（Majority Baseline）第 1 次：永远预测训练折的多数类

[返回全部实验](../EXPERIMENTS.md) · 旧编号 `T00` · 任务：Titanic 生还二分类（binary classification）

## 配置（Configuration）

| 项目 | 内容 |
| --- | --- |
| 算法 | `majority` |
| 算法内尝试序号 | 1 |
| 输入原始列 | `Pclass`, `Sex`, `Age`, `Fare` |
| 进入模型的特征 | 无；多数类模型忽略乘客特征 |
| 超参数（hyperparameters） | `{"strategy": "most_frequent"}` |
| 数据 SHA256 | `bb1bda464cc6819d412b41d34be69fd89d26b372dc24c09421c3dbca1b0dbe9f` |

多数类基线忽略所有输入特征，只在每个训练折统计标签中的多数类。

## 本地验证（Local CV）

- 分层五折，打乱，随机种子 42；准确率（accuracy）均分 **61.62%**，标准差 `0.002325`。
- 五折分数：`0.614525`、`0.617978`、`0.617978`、`0.617978`、`0.612360`。
- 尚未完成五组随机种子的重复交叉验证。

## Kaggle 云端结果（Public leaderboard）

- 未提交；云端分数与名次均未知。

## 复现与判断

本地运行：`.local/venv/bin/python titanic_poc/run.py --experiment-id TIT-BL-001`。结果与预测文件分别保存在 `.local/titanic/experiments/TIT-BL-001/results.json` 和 `submission.csv`；脚本本身不会上传 Kaggle。

无乘客特征的参照点，用训练折的多数类别预测；未提交 Kaggle。

同一训练集被多次用于方案比较，较高的本地分数可能受到选择偏差影响；不要把交叉验证分数当作公榜分数。
