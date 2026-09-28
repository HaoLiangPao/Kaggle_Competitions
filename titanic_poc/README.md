# Titanic：第一个本地实验

目标：根据乘客资料预测 `Survived`（0 = 未生还，1 = 生还）。Kaggle 使用准确率评分；提交文件需要 `PassengerId,Survived` 两列。比赛介绍与提交说明见 [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/overview/evaluation) 和 [Kaggle CLI 教程](https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md#tutorial-how-to-submit-to-a-competition)。

## 已完成的实验

数据来源：2026-09-27 使用 Kaggle CLI 下载的官方 `titanic.zip`，本地保存在 `.local/titanic/`，不会加入 Git。压缩包包含 891 行训练数据、418 行待预测数据和一份提交样例。训练集有 549 位未生还、342 位生还；`Age` 缺失 177 行。训练集女性 314 人，生还比例 74.2%；男性 577 人，生还比例 18.9%。这些是训练数据的描述，不能当作因果解释或待预测数据的真实结果。

| 方法 | 使用的资料 | 5 折平均准确率 |
| --- | --- | ---: |
| 永远猜训练折的多数类 | 无 | 61.6% |
| 逻辑回归（已提交） | 舱位、性别、年龄、票价 | 78.6% |
| 逻辑回归 + Cabin 甲板（本地实验） | 上述四项 + 舱房首字母／Unknown | 79.7% |
| 逻辑回归 + 家庭人数（本地实验） | 四字段 + `SibSp`、`Parch` | 79.0% |
| 逻辑回归 + Cabin + 家庭人数（本地实验） | 四字段 + Cabin 甲板 + `SibSp`、`Parch` | 79.8% |

每折都重新学习缺失值填充和数值缩放，再训练模型，避免把验证折的信息带入训练。5 折分层、打乱，随机种子为 42。原模型的各折得分、环境版本和数据 SHA256 见本地 `.local/titanic/poc_results.json`。`SibSp` 和 `Parch` 在训练、测试数据中均存在且没有缺失值；原四字段模型与 Cabin 实验未使用它们。

Cabin 训练集缺失 687/891、测试集缺失 327/418，因此新实验不直接记忆舱房号，而是在每折预处理内部提取首字母作为甲板，缺失的单独编码为 `Unknown`。新实验的 5 折均分为 0.796843（原模型 0.785619），但单折分数从 0.752809 到 0.837079，波动较大。用另外四组随机划分复核时，四组平均分提升、一组基本持平。结果仅代表本地验证，没有提交 Kaggle；详见 `.local/titanic/cabin_results.json`。

家庭人数实验将 `SibSp` 与 `Parch` 分别作为数值输入，并在每折内同其他数值字段一起缩放。相同 5 折下，四字段 + 家庭人数为 **0.790120**，四字段 + Cabin + 家庭人数为 **0.797979**；后者只比 Cabin 模型高约 0.11 个百分点。换五组随机划分比较 Cabin 与 Cabin + 家庭人数，三组提升、一组持平、一组略低，因此暂不能断言家庭字段带来稳定收益。输出分别保存在 `.local/titanic/family_results.json` 和 `.local/titanic/cabin_family_results.json`，均未提交。

姓名中的姓氏可作为下一轮备选，但不是本轮输入。训练集有 667 个不同姓氏；测试集 418 人中有 188 人的姓氏在训练集出现。单独按姓氏记忆容易过拟合，也不能把同姓直接当作同一家庭。

## 复现

在仓库根目录执行（`api_token` 为已忽略的本地 Kaggle API token 文件；先按照仓库根目录 README 安装固定版本依赖）：

```bash
mkdir -p .local/titanic
KAGGLE_API_TOKEN="$(cat api_token)" kaggle competitions download titanic -p .local/titanic
python titanic_poc/run.py                  # 复现已提交的四字段模型
python titanic_poc/run.py --include-cabin  # 独立运行 Cabin 甲板实验
python titanic_poc/run.py --include-family  # 独立运行家庭人数实验
python titanic_poc/run.py --include-cabin --include-family  # 两组特征一起使用
```

脚本直接读取压缩包，用全部训练数据拟合后，为四种配置分别生成 `.local/titanic/submission_<配置>.csv`（原版为 `submission_poc.csv`）。每份文件都会核对提交列名、行数、乘客 ID 顺序和预测值范围。脚本本身不会加入比赛或上传文件；后三版尚未提交。

## 首次 Kaggle 提交（2026-09-28）

将本 POC 生成的 CSV 提交至 Titanic；Kaggle 返回成功。随后查询到提交编号 `56628386`、状态 `COMPLETE`、公开榜准确率 **0.75837**。2026-09-28T12:59:17 的 Kaggle 公榜快照显示 **第 9028 名 / 10267 队**，排名会随榜单变化。本地 5 折平均准确率为 0.785619；两者评估的数据不同。提交文件及下载数据继续只保存在被 Git 忽略的 `.local/titanic/`。

## 下一步与权限

本机 Kaggle token 能列文件、下载数据并提交。Titanic [规则页](https://www.kaggle.com/c/titanic/rules) 当前注明无需接受规则；第一次提交后，账号的“已参加比赛”列表出现 Titanic。本次提交已由用户明确授权。旧版 Kaggle CLI 1.8.3 中 `competitions submissions` 命令出现 `page_number` 参数兼容错误。项目现固定使用已实测的 2.2.4，可直接运行 `kaggle competitions submissions titanic --format json` 读取提交记录。

下一轮可把 `SibSp + Parch + 1` 变成家庭规模、比较“独自旅行”与同行旅客，并把姓名中的称谓或姓氏作为备选特征。新特征都应放在交叉验证流程内，并使用相同划分比较；不要直接记忆测试集乘客的已知答案。先看稳定性，再决定是否提交。
