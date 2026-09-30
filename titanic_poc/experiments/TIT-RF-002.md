# TIT-RF-002｜随机森林（Random Forest）第 2 次：四项基础特征 + FamilySize

[返回全部实验](../EXPERIMENTS.md) · 旧编号 `T08` · 任务：Titanic 生还二分类（binary classification）

## 配置（Configuration）

| 项目 | 内容 |
| --- | --- |
| 算法 | `random_forest` |
| 算法内尝试序号 | 2 |
| 输入原始列 | `Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, `Parch` |
| 进入模型的特征 | `Pclass`, `Sex`, `Age`, `Fare`, `FamilySize` |
| 超参数（hyperparameters） | `{"max_depth": 5, "min_samples_leaf": 5, "n_estimators": 300, "n_jobs": -1, "random_state": 42}` |
| 数据 SHA256 | `bb1bda464cc6819d412b41d34be69fd89d26b372dc24c09421c3dbca1b0dbe9f` |

预处理（preprocessing）：数值缺失值用训练折中位数填充并缩放；Sex 缺失值用训练折众数填充，再做独热编码（one-hot encoding）。 FamilySize 在每折流程内由 SibSp + Parch + 1 构造。

## 本地验证（Local CV）

- 分层五折，打乱，随机种子 42；准确率（accuracy）均分 **82.71%**，标准差 `0.016552`。
- 五折分数：`0.837989`、`0.820225`、`0.808989`、`0.814607`、`0.853933`。
- 种子 42–46 的五次五折均分平均为 **82.15%**；各次均分：`42: 0.827148`、`43: 0.823796`、`44: 0.820400`、`45: 0.818147`、`46: 0.818178`。

## Kaggle 云端结果（Public leaderboard）

- 提交编号 `56710581`；提交时间 `2026-09-30T14:46:09.843Z`；公开分数 **0.77990**。
- 排名快照 `2026-09-30T14:46:36Z`：**3271 / 10428** 队。名次会随滚动榜单变化。
- 已提交预测 CSV 的 SHA256：`52232f5ccb88bec09e0f2a0b1d1591a32d8715a2dccd8db1aab80fabb85a38ad`。

## 复现与判断

本地运行：`.local/venv/bin/python titanic_poc/run.py --experiment-id TIT-RF-002`。结果与预测文件分别保存在 `.local/titanic/experiments/TIT-RF-002/results.json` 和 `submission.csv`；脚本本身不会上传 Kaggle。

在同样森林参数下增加 FamilySize；公开分数比 TIT-RF-001 高 0.00239，但两版只有 9 个测试预测不同。

同一训练集被多次用于方案比较，较高的本地分数可能受到选择偏差影响；不要把交叉验证分数当作公榜分数。
