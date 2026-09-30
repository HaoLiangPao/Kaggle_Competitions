# Titanic 实验记录（Experiment Log）

预测目标（target）：`Survived`。评分指标（metric）统一为准确率（accuracy），即预测正确的人数占比。表中“本地”来自训练集的交叉验证（cross-validation, CV）；“Kaggle 公榜”来自提交后的测试集公开分数（public score）。两者不是同一批评估数据。

**基础特征 B**：`Pclass`（舱位等级）、`Sex`（性别）、`Age`（年龄）、`Fare`（票价）。下表除多数类基线外，**每一行都包含 B**；“额外特征”仅列另外添加的字段。

| ID | 方法（Method） | 额外特征（Features） | 本地 5 折准确率 | 本地重复 5×5 折准确率 | Kaggle 公榜分数 | 提交后排名快照 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| T00 | 多数类基线（Majority baseline） | 无，也不使用 B | 61.62% | — | — | — |
| T01 | 逻辑回归（Logistic Regression） | 无 | 78.56% | 78.86% | **0.75837** · #56628386 | **9028 / 10267**¹ |
| T02 | 逻辑回归（Logistic Regression） | Cabin 甲板（Deck） | 79.68% | — | — | — |
| T03 | 逻辑回归（Logistic Regression） | `SibSp`、`Parch` 分开输入 | 79.01% | — | — | — |
| T04 | 逻辑回归（Logistic Regression） | Cabin 甲板、`SibSp`、`Parch` 分开输入 | 79.80% | — | — | — |
| T05 | 逻辑回归（Logistic Regression） | FamilySize | 79.23% | 79.37% | — | — |
| T06 | 逻辑回归（Logistic Regression） | Cabin 甲板、FamilySize | 79.68% | 79.87% | **0.76076** · #56652901 | **8866 / 10248**² |
| T07 | 随机森林（Random Forest） | 无 | 82.49% | 81.93% | — | — |
| T08 | 随机森林（Random Forest） | FamilySize | **82.71%** | **82.15%** | — | — |
| T09 | 随机森林（Random Forest） | Cabin 甲板、FamilySize | 82.16% | 81.62% | — | — |
| T10 | 随机森林（Random Forest） | Cabin 甲板、`SibSp`、`Parch` 分开输入 | 81.48% | 81.64% | — | — |

¹ 2026-09-28 **12:59:17 UTC** 榜单快照，首版提交时间是 04:06:20 UTC；这是同日稍后记录的名次，不是提交瞬间的名次。  
² 2026-09-28 **21:43:26 UTC** 榜单快照，FamilySize 版本提交时间是 21:43:11 UTC。Kaggle 是滚动榜单，人数和名次会变。

## 如何读这张表

- `—` 表示**尚未验证或尚未提交**，绝不表示得分为 0。只有 T01、T06 有 Kaggle 公开分数；随机森林目前只有本地结果。
- 单次本地结果是分层五折（Stratified 5-fold CV），打乱后固定随机种子 42。重复结果用种子 42–46 各做一次五折，并取五个均分的平均。比较模型时优先在**同一列**比较；重复折之间共享样本，不能当作独立测试集。
- `FamilySize = SibSp + Parch + 1`（同行家庭规模）。Cabin 只提取首字母作为甲板，缺失值单独记作 `Unknown`。这些是输入特征（features），不是评分指标（metrics）。
- 逻辑回归用每折训练数据填充数值缺失、缩放数值并做类别独热编码（one-hot encoding）。随机森林沿用同一预处理以保证对照可比，固定 300 棵树、最大深度 5、叶节点至少 5 条训练记录。两种方法都没有写入“某类乘客必定生还”的人工预测规则。
- 所有行使用同一份官方 `titanic.zip`，SHA256 为 `bb1bda464cc6819d412b41d34be69fd89d26b372dc24c09421c3dbca1b0dbe9f`；本地 Python 环境为 scikit-learn 1.9.1。预处理在每折训练流程中拟合。多个方案复用同一训练数据来挑选较高分时会有选择偏差；本地 82.15% **不能**当作 Kaggle 公榜成绩或保证未来提交会提高。

## 复现与更新

原始五折实验：运行 `python titanic_poc/run.py`，按需要加 `--include-cabin`、`--include-family` 或 `--include-family-size`。随机森林对照：运行 `.local/venv/bin/python -m titanic_poc.compare_models`。逐折结果保存在 Git 忽略的 `.local/titanic/*_results.json` 和 `.local/titanic/model_comparison.json`；原始数据及预测文件也留在 `.local/titanic/`，不加入 Git。

以后每做一个版本，先新增一行并记录方法、全部基础特征之外的输入、验证方案及本地分数。只有实际提交并确认 `COMPLETE` 后才填 Kaggle 分数；排名必须带抓取时间和当时队伍总数。若更换数据版本或评分指标，应另起一组记录，避免把不能直接比较的数字放在一起。
