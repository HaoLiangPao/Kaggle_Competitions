# Titanic 实验记录（Experiment Log）

每一行对应一份独立实验文档。ID 格式是 `TIT-算法缩写-尝试序号`，例如 `TIT-RF-002` 表示 Titanic 的随机森林第 2 次尝试；旧的 T00–T10 编号保留用于追溯先前讨论。点击 ID 可查看参数、逐折结果、Kaggle 提交及复现命令。

**基础特征 B**：`Pclass`（舱位等级）、`Sex`（性别）、`Age`（年龄）、`Fare`（票价）。除多数类基线外，每行都包含 B。评分指标（metric）统一为准确率（accuracy）。本地交叉验证（cross-validation）与 Kaggle 公开榜（public leaderboard）使用不同数据，不要把两个分数当作同一测试结果。

| 实验 ID / 文档 | 旧 ID | 名称 | 算法与尝试 | 额外特征 | 本地 5 折 | 本地重复 5×5 折 | Kaggle 公榜 | 排名快照 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [TIT-BL-001](experiments/TIT-BL-001.md) | T00 | 永远预测训练折的多数类 | 多数类基线（Majority Baseline） · 第 1 次 | 无 | 61.62% | — | — | — |
| [TIT-LR-001](experiments/TIT-LR-001.md) | T01 | 四项基础特征 | 逻辑回归（Logistic Regression） · 第 1 次 | 无 | 78.56% | 78.86% | 0.75837 · #56628386 | 9028 / 10267 |
| [TIT-LR-002](experiments/TIT-LR-002.md) | T02 | 四项基础特征 + Cabin 甲板 | 逻辑回归（Logistic Regression） · 第 2 次 | Cabin 甲板 | 79.68% | — | — | — |
| [TIT-LR-003](experiments/TIT-LR-003.md) | T03 | 四项基础特征 + SibSp/Parch | 逻辑回归（Logistic Regression） · 第 3 次 | SibSp + Parch（分开） | 79.01% | — | — | — |
| [TIT-LR-004](experiments/TIT-LR-004.md) | T04 | 四项基础特征 + Cabin + SibSp/Parch | 逻辑回归（Logistic Regression） · 第 4 次 | Cabin 甲板、SibSp + Parch（分开） | 79.80% | — | — | — |
| [TIT-LR-005](experiments/TIT-LR-005.md) | T05 | 四项基础特征 + FamilySize | 逻辑回归（Logistic Regression） · 第 5 次 | FamilySize | 79.23% | 79.37% | — | — |
| [TIT-LR-006](experiments/TIT-LR-006.md) | T06 | 四项基础特征 + Cabin + FamilySize | 逻辑回归（Logistic Regression） · 第 6 次 | Cabin 甲板、FamilySize | 79.68% | 79.87% | 0.76076 · #56652901 | 8866 / 10248 |
| [TIT-RF-001](experiments/TIT-RF-001.md) | T07 | 四项基础特征 | 随机森林（Random Forest） · 第 1 次 | 无 | 82.49% | 81.93% | 0.77751 · #56710547 | 3998 / 10428 |
| [TIT-RF-002](experiments/TIT-RF-002.md) | T08 | 四项基础特征 + FamilySize | 随机森林（Random Forest） · 第 2 次 | FamilySize | 82.71% | 82.15% | 0.77990 · #56710581 | 3271 / 10428 |
| [TIT-RF-003](experiments/TIT-RF-003.md) | T09 | 四项基础特征 + Cabin + FamilySize | 随机森林（Random Forest） · 第 3 次 | Cabin 甲板、FamilySize | 82.16% | 81.62% | — | — |
| [TIT-RF-004](experiments/TIT-RF-004.md) | T10 | 四项基础特征 + Cabin + SibSp/Parch | 随机森林（Random Forest） · 第 4 次 | Cabin 甲板、SibSp + Parch（分开） | 81.48% | 81.64% | — | — |

## 如何比较

- `—` 表示尚未做该项验证或尚未提交，绝不表示得分为零。只有实际提交并确认完成的版本才填写 Kaggle 分数。
- 单次本地结果：随机种子 42 的分层五折（Stratified 5-fold CV）。重复结果：种子 42–46 各做一次五折，再平均五个均分。优先在同一列比较模型；重复划分共享训练样本，不是五份独立测试集。
- 排名是团队在提交后抓取的榜单快照，不是永久名次。每个已提交版本的文档都写明提交时间、快照时间及当时队伍总数；旧版 T01 的快照在提交数小时后抓取。
- FamilySize = SibSp + Parch + 1；Cabin 仅取首字母作为甲板，缺失时记为 `Unknown`。这些是输入特征（features），准确率才是评分指标（metric）。
- 官方数据压缩包 SHA256：`bb1bda464cc6819d412b41d34be69fd89d26b372dc24c09421c3dbca1b0dbe9f`。原始比赛数据、预测 CSV 和本机状态保存在被 Git 忽略的 `.local/titanic/`；本仓库只追踪配置、聚合结果和提交元数据。

## 实验工作流

```text
官方数据 → 固定实验 ID → 特征与预处理 Pipeline → 算法 → 分层交叉验证
         → 本地结果 → 全量训练与预测文件 → 经授权提交 → 公榜快照
```

运行 `.local/venv/bin/python titanic_poc/run.py --list-experiments` 查看全部 ID；使用 `--experiment-id TIT-RF-002` 复现某一版。每个 ID 的输出单独写入 `.local/titanic/experiments/<ID>/`；内容相同的重跑会复用文件，内容不同时会报错，避免覆盖历史结果。重复交叉验证使用 `.local/venv/bin/python -m titanic_poc.compare_models`。

实验配置在 `experiment_registry.py`，可审查的历史结果在 `experiment_records.json`。更新记录后运行 `.local/venv/bin/python -m titanic_poc.render_experiments` 重新生成本索引与各实验文档；`--check` 仅检查文件是否同步。新算法或新特征应使用新 ID，保留旧记录。
