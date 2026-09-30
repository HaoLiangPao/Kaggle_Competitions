# 常见机器学习算法地图（ML Algorithm Guide）

这是一份面向本项目的常见算法目录，覆盖分类、回归、聚类、异常检测、降维和部分专门任务；它不是声称列尽所有机器学习算法。先选**任务和评分指标（metric）**，再选模型。Titanic 是二分类（binary classification），官方按准确率（accuracy）评分。[Titanic 实验索引](../titanic_poc/EXPERIMENTS.md)记录了实际跑过的版本。

## 我们现在有没有 ML Pipeline？

**有，但要分清三样东西。** Titanic 当前使用的是 `scikit-learn Pipeline` / `ColumnTransformer`：特征构造、缺失值填充、数值缩放、类别编码和最后的估计器（estimator）串在一起。`TIT-LR-*` 的末端是逻辑回归（Logistic Regression）；`TIT-RF-*` 只把末端换成随机森林（Random Forest）。交叉验证（CV）每折分别拟合预处理，避免使用验证折的统计量。[scikit-learn Pipeline 文档](https://scikit-learn.org/stable/modules/compose.html)说明了这种组合方式。

仓库里**另有** `ml_pipeline/` 包，但 Titanic **没有导入或运行它**。该包的 `ModelWrapper` 目前注册的是回归模型，`Evaluator` 提供 RMSE、MAE、R²、MAPE。`kaggle_s6_e1/train.py` 还在交叉验证之前对全部训练数据调用 `preprocessor.fit_transform`，因此它当前的 CV 流程不能直接用作无泄漏的 Titanic 分类验证。要统一两个方向，应先让任务类型、评分和每折预处理成为通用接口，不能只把 Titanic 的调用改指向这个包。

现在还建立了**Titanic 项目级实验流程**：稳定 ID → 固定配置 → 本地验证 → 独立预测/结果目录 → 经用户授权的 Kaggle 提交 → 可追溯的实验记录。当前实现范围是多数类基线、逻辑回归和随机森林；评分指标仅为 accuracy。其他任务可以复用 ID 与记录规范，但需要适合该任务的数据划分、指标和训练器。训练脚本不会自动提交 Kaggle。

```text
实验 ID + 数据版本
       ↓
特征工程 → 缺失值/编码/缩放 → 模型（可替换）
       ↓
交叉验证 + 指标 → 全量拟合 → 预测文件
       ↓
实验记录 ← 提交后公榜分数/排名快照
```

**表格读法**：`可系统抽象 ✅` 表示能放入某种可复现的训练/验证/记录流程，括号中是所需流程类型；这**不保证**可以原封不动放进当前 Titanic 二分类 Pipeline。`本仓库已建 ✅` 只表示已有 ID、可运行代码、已记录结果和文档；`⬜` 表示尚未接入或验证。每个调参建议都应在固定数据划分下比较，并记录搜索范围与代价。模型越多、调参越频繁，越容易对本地验证集产生选择偏差；公榜也不应被当作无限次调参的训练集。[scikit-learn 交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)、[超参数搜索](https://scikit-learn.org/stable/modules/grid_search.html)。

## A. 有标签的表格分类（Supervised tabular classification）

| 算法（中 / EN） | 适合什么 | 先调哪些参数或步骤来改善表现 | 可系统抽象 | 本仓库已建 |
| --- | --- | --- | --- | --- |
| 多数类基线 / Dummy Classifier | 核对数据、评分和提交流程的最低参照 | 不追求调高；检查类别比例及分层划分 | ✅ 分类 | ✅ `TIT-BL-001` |
| 逻辑回归 / Logistic Regression | 小到中等表格、可解释的线性概率基线；名字含“回归”但此处做分类 | `C` 控制正则化强度（越小约束越强）；再比较 `penalty`、`class_weight`、交互特征与概率阈值 | ✅ 分类 | ✅ `TIT-LR-001`–`006` |
| 朴素贝叶斯 / Naive Bayes | 稀疏文本计数或快速小数据基线；假设特征条件独立 | 文本先选 Multinomial/Complement 与 `alpha`；连续特征可试 Gaussian 的 `var_smoothing`，并核对数据表示 | ✅ 分类 | ⬜ |
| k 近邻 / k-Nearest Neighbors | 距离有意义、样本量适中的任务 | 先缩放；试 `n_neighbors`、`weights`、距离 `metric`；小 k 易受噪声影响 | ✅ 分类 | ⬜ |
| 决策树 / Decision Tree | 想看直观分支规则、非线性交互 | 限制 `max_depth`、提高 `min_samples_leaf`、比较 `ccp_alpha` 剪枝 | ✅ 分类 | ⬜ |
| 随机森林 / Random Forest | 中小型表格、非线性与特征交互；可作稳定树模型基线 | 先调 `max_depth`、`min_samples_leaf`、`max_features` 抑制过拟合；再增 `n_estimators` 看是否稳定 | ✅ 分类 | ✅ `TIT-RF-001`–`004` |
| 极端随机树 / Extra Trees | 表格分类，想比较更强随机性的树集成 | 比较 `max_features`、`min_samples_leaf`、`max_depth`、`n_estimators` | ✅ 分类 | ⬜ |
| 支持向量机 / SVM | 中小数据、特征缩放后可能有清晰边界 | 先标准化；比较 `kernel`，再调 `C`；RBF 核另调 `gamma` | ✅ 分类 | ⬜ |
| 线性判别分析 / LDA | 小数据、类别分布较规整的线性基线 | 比较 `solver`、`shrinkage`；检查协方差估计是否稳定 | ✅ 分类 | ⬜ |
| 梯度提升树 / Gradient Boosting、HistGradientBoosting | 表格非线性；常作为树集成进阶对照 | 联动 `learning_rate` 与迭代次数；调 `max_leaf_nodes`/`max_depth`、`min_samples_leaf`，用早停 | ✅ 分类 | ⬜ |
| XGBoost / Extreme Gradient Boosting | 表格竞赛、需要精细控制正则化和采样 | 联动 `learning_rate`/`n_estimators`；再看 `max_depth`、`min_child_weight`、`subsample`、`colsample_bytree`、`reg_lambda` 与早停 | ✅ 分类适配 | ⬜ |
| LightGBM / Light Gradient Boosting Machine | 较大表格、训练速度重要 | 控制 `num_leaves` 与 `min_data_in_leaf` 防过拟合；配合 `learning_rate`/迭代次数、`feature_fraction`、早停 | ✅ 分类适配 | ⬜ |
| CatBoost / Categorical Boosting | 类别字段较多的表格；可处理原生类别特征 | 比较 `depth`、`iterations`、`learning_rate`、`l2_leaf_reg` 与早停；原生类别流程需单独适配 | ✅ 分类适配 | ⬜ |
| 多层感知机 / MLP Neural Network | 有足够数据、非线性模式明显；小表格先谨慎对照 | 缩放输入；从小网络开始调 `hidden_layer_sizes`、`alpha`、`learning_rate_init`、早停 | ✅ 分类 | ⬜ |

这些方法的类别和接口以 [scikit-learn 监督学习目录](https://scikit-learn.org/stable/supervised_learning.html)为主；外部提升树参数见 [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html)、[LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters.html)和 [CatBoost](https://catboost.ai/docs/en/concepts/parameter-tuning) 官方文档。三种外部库目前**不在本项目依赖中**，表中仅说明可行的接入方向，未测试其本机 API 或分数。

## B. 连续数值预测（Regression）

例如预测房价或销量；指标通常换成 MAE、RMSE 等，不能直接拿 Titanic 的 accuracy 比较。[scikit-learn 指标文档](https://scikit-learn.org/stable/modules/model_evaluation.html)。

| 算法（中 / EN） | 适合什么 | 先调哪些参数或步骤 | 可系统抽象 | 本仓库已建 |
| --- | --- | --- | --- | --- |
| 线性回归 / Linear Regression | 数值目标的最简可解释基线 | 重点检查特征、异常值与残差；模型本身几乎无正则化旋钮 | ✅ 回归流程 | ⬜ |
| 岭回归 / Ridge | 数值特征相关性强时的稳定线性模型 | 在缩放后的数据上调 `alpha`（越大约束越强） | ✅ 回归流程 | ⬜ |
| Lasso / Elastic Net | 希望部分系数收缩为零；特征很多 | 调 `alpha`；Elastic Net 另调 `l1_ratio`，放在交叉验证内选择 | ✅ 回归流程 | ⬜ |
| 随机森林回归 / Random Forest Regressor | 非线性数值预测 | 调 `max_depth`、`min_samples_leaf`、`max_features`、`n_estimators` | ✅ 回归流程 | ⬜ |
| 支持向量回归 / SVR | 中小数据的非线性数值预测 | 缩放输入和目标后比较 `C`、`epsilon`、`gamma`、核函数 | ✅ 回归流程 | ⬜ |

## C. 没有生存标签的探索（Unsupervised learning）

这些方法可帮助发现群组、异常或压缩特征，但**不能直接替代 Titanic 的生存预测器**。应选择相应的无监督评估或在下游预测模型内验证是否真的有帮助。[scikit-learn 无监督学习目录](https://scikit-learn.org/stable/unsupervised_learning.html)。

| 算法（中 / EN） | 适合什么 | 先调哪些参数或步骤 | 可系统抽象 | 本仓库已建 |
| --- | --- | --- | --- | --- |
| K 均值 / K-Means | 寻找大致球状、数量可预估的群组 | 缩放数值；比较 `n_clusters`、`n_init`，再检查簇是否有业务意义 | ✅ 聚类流程 | ⬜ |
| 高斯混合 / Gaussian Mixture Model | 需要柔性概率分群 | 调 `n_components`、`covariance_type`、`reg_covar`，注意小样本不稳定 | ✅ 聚类流程 | ⬜ |
| 密度聚类 / DBSCAN、HDBSCAN | 形状不规则的群组和噪声点 | DBSCAN 调 `eps`、`min_samples`、距离度量；HDBSCAN 调最小簇规模，先缩放 | ✅ 聚类流程 | ⬜ |
| 主成分分析 / PCA | 高维数值特征压缩或可视化 | 缩放后调 `n_components`；若做监督预测，只能在训练折内拟合 PCA | ✅ 降维流程 | ⬜ |
| 隔离森林 / Isolation Forest | 异常点检测 | 调 `contamination`（决定异常阈值）、`max_samples`、`n_estimators`，用有标签样本核对误报 | ✅ 异常检测流程 | ⬜ |

## D. 专门数据和目标（Specialized tasks）

| 算法（中 / EN） | 适合什么 | 先调哪些参数或步骤 | 可系统抽象 | 本仓库已建 |
| --- | --- | --- | --- | --- |
| 卷积神经网络 / CNN | 图像、局部空间结构 | 调网络容量、数据增强、学习率、批量大小和权重衰减；按主体分组验证防重复图像泄漏 | ✅ 深度学习训练器 | ⬜ |
| Transformer | 文本、序列、图像 token；通常需要更多数据或预训练模型 | 调模型容量、上下文长度、学习率、批量大小、正则化；优先验证数据与算力成本 | ✅ 深度学习训练器 | ⬜ |
| ARIMA / SARIMA | 有时间顺序的单变量或少量外生变量预测 | 调 `(p,d,q)` 和季节项；必须用时间顺序切分而非随机折 | ✅ 时间序列流程 | ⬜ |
| LambdaMART / Learning to Rank | 搜索结果、推荐候选的排序 | 按查询组（group）切分；调树复杂度、学习率、迭代次数，并使用 NDCG 等排序指标 | ✅ 排序专用流程 | ⬜ |

深度学习需自己的训练循环或适配器，[PyTorch 优化教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)解释了学习率与批量大小；[statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)和 [LightGBM Ranker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)提供任务专用接口。对时间序列、排序、聚类等任务，可沿用实验 ID 和记录规范，但不能直接复用 Titanic 的随机五折和 accuracy。

## 对 Titanic 的下一步学习顺序

1. 保留 `TIT-LR-*`、`TIT-RF-*` 作为可复现参照。先在**相同特征与折划分**上比较方法，再讨论特征作用。
2. 下一种模型可从 HistGradientBoosting 或 Extra Trees 中选一个；先固定少数超参数做本地对照，胜出后才考虑 Kaggle 提交。
3. 加入标题（Title）等来自现有字段的特征时，分配新实验 ID，先记录是否提升多次本地验证，再看公榜；不要根据公榜反复手调测试样本答案。
4. 任何新增指标（如 F1、ROC AUC）只用于补充诊断；Titanic 排名仍以 Kaggle 指定的 accuracy 为准。多指标结果应分别标注，不混成一个“总分”。
