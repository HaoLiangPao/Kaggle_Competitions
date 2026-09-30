# TIT-LR-006｜逻辑回归（Logistic Regression）第 6 次：四项基础特征 + Cabin + FamilySize

[返回全部实验](../EXPERIMENTS.md) · 旧编号 `T06` · 任务：Titanic 生还二分类（binary classification）

## 配置（Configuration）

| 项目 | 内容 |
| --- | --- |
| 算法 | `logistic_regression` |
| 算法内尝试序号 | 6 |
| 输入原始列 | `Pclass`, `Sex`, `Age`, `Fare`, `Cabin`, `SibSp`, `Parch` |
| 进入模型的特征 | `Pclass`, `Sex`, `Age`, `Fare`, `Cabin`, `FamilySize` |
| 超参数（hyperparameters） | `{"max_iter": 1000, "random_state": 42}` |
| 数据 SHA256 | `bb1bda464cc6819d412b41d34be69fd89d26b372dc24c09421c3dbca1b0dbe9f` |

预处理（preprocessing）：数值缺失值用训练折中位数填充并缩放；Sex 缺失值用训练折众数填充，再做独热编码（one-hot encoding）。 Cabin 在每折流程内提取甲板首字母；缺失值为 Unknown。 FamilySize 在每折流程内由 SibSp + Parch + 1 构造。

## 本地验证（Local CV）

- 分层五折，打乱，随机种子 42；准确率（accuracy）均分 **79.68%**，标准差 `0.025826`。
- 五折分数：`0.810056`、`0.808989`、`0.764045`、`0.769663`、`0.831461`。
- 种子 42–46 的五次五折均分平均为 **79.87%**；各次均分：`42: 0.796843`、`43: 0.795738`、`44: 0.796855`、`45: 0.799096`、`46: 0.804720`。

## Kaggle 云端结果（Public leaderboard）

- 提交编号 `56652901`；提交时间 `2026-09-28T21:43:11.720Z`；公开分数 **0.76076**。
- 排名快照 `2026-09-28T21:43:26Z`：**8866 / 10248** 队。名次会随滚动榜单变化。
- 已提交预测 CSV 的 SHA256：`b1a56c9145c2143fcb0e4cd922bd614f0d33ccdf71b82ed899ea6ff58cb80b70`。

## 复现与判断

本地运行：`.local/venv/bin/python titanic_poc/run.py --experiment-id TIT-LR-006`。结果与预测文件分别保存在 `.local/titanic/experiments/TIT-LR-006/results.json` 和 `submission.csv`；脚本本身不会上传 Kaggle。

FamilySize 与 Cabin 甲板一起输入；公开分数较第一版略高，不能单凭一次提交断定特征稳定有效。

同一训练集被多次用于方案比较，较高的本地分数可能受到选择偏差影响；不要把交叉验证分数当作公榜分数。
