# TIT-RF-004｜随机森林（Random Forest）第 4 次：四项基础特征 + Cabin + SibSp/Parch

[返回全部实验](../EXPERIMENTS.md) · 旧编号 `T10` · 任务：Titanic 生还二分类（binary classification）

## 配置（Configuration）

| 项目 | 内容 |
| --- | --- |
| 算法 | `random_forest` |
| 算法内尝试序号 | 4 |
| 输入原始列 | `Pclass`, `Sex`, `Age`, `Fare`, `Cabin`, `SibSp`, `Parch` |
| 进入模型的特征 | `Pclass`, `Sex`, `Age`, `Fare`, `Cabin`, `SibSp`, `Parch` |
| 超参数（hyperparameters） | `{"max_depth": 5, "min_samples_leaf": 5, "n_estimators": 300, "n_jobs": -1, "random_state": 42}` |
| 数据 SHA256 | `bb1bda464cc6819d412b41d34be69fd89d26b372dc24c09421c3dbca1b0dbe9f` |

预处理（preprocessing）：数值缺失值用训练折中位数填充并缩放；Sex 缺失值用训练折众数填充，再做独热编码（one-hot encoding）。 Cabin 在每折流程内提取甲板首字母；缺失值为 Unknown。

## 本地验证（Local CV）

- 分层五折，打乱，随机种子 42；准确率（accuracy）均分 **81.48%**，标准差 `0.016975`。
- 五折分数：`0.810056`、`0.803371`、`0.808989`、`0.803371`、`0.848315`。
- 种子 42–46 的五次五折均分平均为 **81.64%**；各次均分：`42: 0.814820`、`43: 0.822698`、`44: 0.817080`、`45: 0.811418`、`46: 0.815925`。

## Kaggle 云端结果（Public leaderboard）

- 未提交；云端分数与名次均未知。

## 复现与判断

本地运行：`.local/venv/bin/python titanic_poc/run.py --experiment-id TIT-RF-004`。结果与预测文件分别保存在 `.local/titanic/experiments/TIT-RF-004/results.json` 和 `submission.csv`；脚本本身不会上传 Kaggle。

森林加入 Cabin 甲板并保留 SibSp、Parch 两列；重复交叉验证与 TIT-RF-003 接近，未提交。

同一训练集被多次用于方案比较，较高的本地分数可能受到选择偏差影响；不要把交叉验证分数当作公榜分数。
