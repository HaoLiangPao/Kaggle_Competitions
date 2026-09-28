# Titanic：第一个本地实验

目标：根据乘客资料预测 `Survived`（0 = 未生还，1 = 生还）。Kaggle 使用准确率评分；提交文件需要 `PassengerId,Survived` 两列。比赛介绍与提交说明见 [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/overview/evaluation) 和 [Kaggle CLI 教程](https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md#tutorial-how-to-submit-to-a-competition)。

## 已完成的实验

数据来源：2026-09-27 使用 Kaggle CLI 下载的官方 `titanic.zip`，本地保存在 `.local/titanic/`，不会加入 Git。压缩包包含 891 行训练数据、418 行待预测数据和一份提交样例。训练集有 549 位未生还、342 位生还；`Age` 缺失 177 行。训练集女性 314 人，生还比例 74.2%；男性 577 人，生还比例 18.9%。这些是训练数据的描述，不能当作因果解释或待预测数据的真实结果。

| 方法 | 使用的资料 | 5 折平均准确率 |
| --- | --- | ---: |
| 永远猜训练折的多数类 | 无 | 61.6% |
| 逻辑回归 | 舱位、性别、年龄、票价 | 78.6% |

每折都重新学习缺失值填充和数值缩放，再训练模型，避免把验证折的信息带入训练。5 折分层、打乱，随机种子为 42。具体每折得分、环境版本和数据 SHA256 见本地 `.local/titanic/poc_results.json`。逻辑回归各折准确率范围约 76.4%–79.9%；这是本地验证结果，不是 Kaggle 榜单分数。

## 复现

在仓库根目录执行（`api_token` 为已忽略的本地 Kaggle API token 文件；先按照仓库根目录 README 安装固定版本依赖）：

```bash
mkdir -p .local/titanic
KAGGLE_API_TOKEN="$(cat api_token)" kaggle competitions download titanic -p .local/titanic
python titanic_poc/run.py
```

脚本直接读取压缩包，用全部训练数据拟合后，生成 `.local/titanic/submission_poc.csv`。它会核对提交列名、行数、乘客 ID 顺序和预测值范围。脚本本身不会加入比赛或上传文件。

## 首次 Kaggle 提交（2026-09-28）

将本 POC 生成的 CSV 提交至 Titanic；Kaggle 返回成功。随后查询到提交编号 `56628386`、状态 `COMPLETE`、公开榜准确率 **0.75837**。2026-09-28T12:59:17 的 Kaggle 公榜快照显示 **第 9028 名 / 10267 队**，排名会随榜单变化。本地 5 折平均准确率为 0.785619；两者评估的数据不同。提交文件及下载数据继续只保存在被 Git 忽略的 `.local/titanic/`。

## 下一步与权限

本机 Kaggle token 能列文件、下载数据并提交。Titanic [规则页](https://www.kaggle.com/c/titanic/rules) 当前注明无需接受规则；第一次提交后，账号的“已参加比赛”列表出现 Titanic。本次提交已由用户明确授权。旧版 Kaggle CLI 1.8.3 中 `competitions submissions` 命令出现 `page_number` 参数兼容错误。项目现固定使用已实测的 2.2.4，可直接运行 `kaggle competitions submissions titanic --format json` 读取提交记录。

下一轮可以先加入 `SibSp`、`Parch` 和 `Embarked`，与当前模型用相同的 5 折划分比较。先看验证结果，再决定是否提交。
