# TIT-LR-001｜逻辑回归（Logistic Regression）第 1 次：四项基础特征

[返回全部实验](../EXPERIMENTS.md) · 旧编号 `T01` · 任务：Titanic 生还二分类（binary classification）

## 配置（Configuration）

| 项目 | 内容 |
| --- | --- |
| 算法 | `logistic_regression` |
| 算法内尝试序号 | 1 |
| 输入原始列 | `Pclass`, `Sex`, `Age`, `Fare` |
| 进入模型的特征 | `Pclass`, `Sex`, `Age`, `Fare` |
| 超参数（hyperparameters） | `{"max_iter": 1000, "random_state": 42}` |
| 数据 SHA256 | `bb1bda464cc6819d412b41d34be69fd89d26b372dc24c09421c3dbca1b0dbe9f` |

预处理（preprocessing）：数值缺失值用训练折中位数填充并缩放；Sex 缺失值用训练折众数填充，再做独热编码（one-hot encoding）。

## 本地验证（Local CV）

- 分层五折，打乱，随机种子 42；准确率（accuracy）均分 **78.56%**，标准差 `0.012741`。
- 五折分数：`0.798883`、`0.797753`、`0.780899`、`0.764045`、`0.786517`。
- 种子 42–46 的五次五折均分平均为 **78.86%**；各次均分：`42: 0.785619`、`43: 0.792386`、`44: 0.792392`、`45: 0.787879`、`46: 0.784527`。

## Kaggle 云端结果（Public leaderboard）

- 提交编号 `56628386`；提交时间 `2026-09-28T04:06:20.497Z`；公开分数 **0.75837**。
- 排名快照 `2026-09-28T12:59:17Z`：**9028 / 10267** 队。名次会随滚动榜单变化。
- 已提交预测 CSV 的 SHA256：`2f4002ed193581244713c4089f5fffa999fa0dd0230a2c8b3d05516392df5b17`。
- 该排名快照在提交数小时后抓取，不能称为提交瞬间排名。

## 复现与判断

本地运行：`.local/venv/bin/python titanic_poc/run.py --experiment-id TIT-LR-001`。结果与预测文件分别保存在 `.local/titanic/experiments/TIT-LR-001/results.json` 和 `submission.csv`；脚本本身不会上传 Kaggle。

第一版四特征模型；已经提交，公开分数低于本地交叉验证。

同一训练集被多次用于方案比较，较高的本地分数可能受到选择偏差影响；不要把交叉验证分数当作公榜分数。
