# TIT-RF-001｜随机森林（Random Forest）第 1 次：四项基础特征

[返回全部实验](../EXPERIMENTS.md) · 旧编号 `T07` · 任务：Titanic 生还二分类（binary classification）

## 配置（Configuration）

| 项目 | 内容 |
| --- | --- |
| 算法 | `random_forest` |
| 算法内尝试序号 | 1 |
| 输入原始列 | `Pclass`, `Sex`, `Age`, `Fare` |
| 进入模型的特征 | `Pclass`, `Sex`, `Age`, `Fare` |
| 超参数（hyperparameters） | `{"max_depth": 5, "min_samples_leaf": 5, "n_estimators": 300, "n_jobs": -1, "random_state": 42}` |
| 数据 SHA256 | `bb1bda464cc6819d412b41d34be69fd89d26b372dc24c09421c3dbca1b0dbe9f` |

预处理（preprocessing）：数值缺失值用训练折中位数填充并缩放；Sex 缺失值用训练折众数填充，再做独热编码（one-hot encoding）。

## 本地验证（Local CV）

- 分层五折，打乱，随机种子 42；准确率（accuracy）均分 **82.49%**，标准差 `0.019797`。
- 五折分数：`0.849162`、`0.831461`、`0.792135`、`0.814607`、`0.837079`。
- 种子 42–46 的五次五折均分平均为 **81.93%**；各次均分：`42: 0.824889`、`43: 0.815950`、`44: 0.823771`、`45: 0.811412`、`46: 0.820432`。

## Kaggle 云端结果（Public leaderboard）

- 提交编号 `56710547`；提交时间 `2026-09-30T14:45:03.503Z`；公开分数 **0.77751**。
- 排名快照 `2026-09-30T14:45:34Z`：**3998 / 10428** 队。名次会随滚动榜单变化。
- 已提交预测 CSV 的 SHA256：`82919c61fb81fa85c4798d5b19c929eec19b705fa7236dbfcf2c5e31b8730f39`。

## 复现与判断

本地运行：`.local/venv/bin/python titanic_poc/run.py --experiment-id TIT-RF-001`。结果与预测文件分别保存在 `.local/titanic/experiments/TIT-RF-001/results.json` 和 `submission.csv`；脚本本身不会上传 Kaggle。

不增加额外特征，只将四特征模型换成限制深度的随机森林；本地和公榜均高于逻辑回归第一版。

同一训练集被多次用于方案比较，较高的本地分数可能受到选择偏差影响；不要把交叉验证分数当作公榜分数。
