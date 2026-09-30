# Kaggle Competitions

This repository contains code and pipelines for Kaggle competitions.

## ML Analysis Pipeline (High Level)

This pipeline is designed to be **clean, reusable, and extensible** - suitable for production-grade ML workflows. The architecture clearly separates automated pipeline steps from human decision points.

---

### Pipeline Architecture Overview

```
Raw Data (CSV/Parquet)
   │
   ▼
🤖 Data Loading
   │
   ▼
🤖 Data Validation & Schema Check
   │
   ▼
👤 EDA / Data Profiling (offline)
   │
   ▼
🤖 Feature Processing
   ├── Numerical features (scaling, missing values)
   ├── Categorical features (encoding)
   └── Feature engineering
   │
   ▼
🤖 Train / Validation Split (or CV)
   │
   ▼
🤖 Model Training
   ├── Linear Regression (baseline)
   └── GBDT (LightGBM / XGBoost)
   │
   ▼
🤖 Model Evaluation
   │
   ▼
🤖 Model Artifact Saving
   │
   ▼
🤖 Inference Interface (batch / API)
```

**Legend:**
- 🤖 **Automated Pipeline Steps** - Handled by code
- 👤 **Human Decision Steps** - Require manual analysis/judgment

---

## Pipeline Components

### 1. Data Layer

#### 🤖 Data Loading
- Load data from CSV/Parquet/Database
- Define feature columns and target column
- Keep data source abstraction (easy to swap sources)

#### 🤖 Data Validation & Schema Check
- Verify required columns exist
- Check data types match expectations
- Detect empty columns or data quality issues
- **Why**: Prevents silent failures in production

#### 👤 EDA (Exploratory Data Analysis)
- **NOT part of automated pipeline** - done in notebooks
- Analyze target distribution
- Feature distributions and correlations
- Missing value patterns
- Feature vs target relationships
- **Output**: Insights to guide feature engineering decisions

---

### 2. Feature Processing Layer

#### 🤖 Feature Preprocessing

**For Numerical Features:**
- Missing value imputation (mean/median)
- Standardization (StandardScaler for LR/SVM)
- Note: GBDT models don't need scaling

**For Categorical Features:**
- One-hot encoding (simple and stable)
- Target encoding (for high-cardinality features)

**Why separate preprocessing?**
- Training and inference must use **identical** preprocessing
- Preprocessor is saved as an artifact
- Foundation for MLOps deployment

**Key Principle:**
> Linear Regression → **needs scaling**  
> GBDT (tree-based) → **no scaling needed**

---

### 3. Data Splitting

#### 🤖 Train/Validation Split

**Strategy: K-Fold Cross Validation**
- Evaluates model stability
- Industry standard for reliable performance estimates
- Better than simple train/test split for limited data

---

### 4. Model Layer

#### 🤖 Model Training

**Current Approach:**
1. Start with **Linear Regression** (baseline)
   - Validates feature engineering and pipeline correctness
   - Simple, interpretable, fast
2. Upgrade to **GBDT** (LightGBM/XGBoost)
   - Production-grade performance
   - Handles non-linear relationships

**Model Interface Design:**
```python
Model:
 ├── fit(X, y)
 ├── predict(X)
 └── save() / load()
```

This interface allows **swapping models without changing the pipeline**.

---

### 5. Evaluation Layer

#### 🤖 Model Evaluation

**Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

**Why separate evaluation module?**
- Metrics logic ≠ training logic
- Easy to add new metrics
- Consistent evaluation across models

---

### 6. Artifact Management

#### 🤖 Model Artifact Saving

**What gets saved:**
- Trained model (joblib/pickle)
- Feature preprocessor (with fitted parameters)
- Metadata (metrics, timestamp, hyperparameters)

**Why**: Ensures reproducibility and enables deployment

---

### 7. Inference Layer

#### 🤖 Inference Interface

**Current: Batch Inference**
- Input: CSV → Output: Predictions

**Future Extensions:**
- REST API
- Streaming inference
- Real-time scoring

---

## Implementation

### Reusable Pipeline Package

See `ml_pipeline/` for the modular pipeline implementation:
- `data_loader.py` - Data loading and validation
- `preprocessor.py` - Feature preprocessing
- `model_wrapper.py` - Unified model interface
- `evaluator.py` - Metrics and evaluation
- `pipeline.py` - End-to-end orchestrator

### Competition Applications

Each competition folder contains:
- `config.yaml` - Pipeline configuration
- `train.py` - Training script
- `predict.py` - Inference script
- `eda.ipynb` - Exploratory analysis (👤 human step)

---

## Design Philosophy

**"First make it work, then make it better"**

1. **Start simple**: Linear Regression baseline
2. **Validate pipeline**: Ensure end-to-end flow works
3. **Iterate**: Upgrade to complex models (GBDT)
4. **Production-ready**: Modular, testable, deployable

This approach is **professional and production-oriented**, not just Kaggle optimization.

## Kaggle 比赛发现实验

第一个真实数据 POC：见 [Titanic 实验记录表](titanic_poc/EXPERIMENTS.md)和 [Titanic 本地实验说明](titanic_poc/README.md)，包含模型、输入特征、本地验证、Kaggle 公开分数与历史排名快照。

第一版只读取 Kaggle 比赛列表，不下载数据、不加入比赛、不提交结果。它会过滤已截止比赛，并用本地 SQLite 文件记录已经见过的比赛。首次 scan 只建立基线；以后出现的新比赛才会处于待通知状态。

需要 Python 3.11+、`requirements.txt` 中固定的 Kaggle CLI 2.2.4 和 Kaggle 登录凭据。旧版 1.8.3 的 `competitions submissions` 存在参数兼容错误；2.2.4 已实测可正常查询提交记录。本机可将已忽略的 api_token 文件显式传给脚本，或使用 KAGGLE_API_TOKEN 环境变量。

    python -m venv .local/venv
    source .local/venv/bin/activate
    python -m pip install -r requirements.txt

    # 先看当前列表，不修改状态
    python -m kaggle_watch preview --token-file api_token

    # 首次扫描：建立基线，不发送 Slack
    python -m kaggle_watch scan --token-file api_token

    # 后续扫描：显示新比赛的通知预览，待通知状态保留
    python -m kaggle_watch scan --token-file api_token

要发送到 Slack，需要先在运行环境中设置 SLACK_WEBHOOK_URL，然后运行 python -m kaggle_watch scan --token-file api_token --send-slack。只有 webhook 确认成功的比赛会被标为已发送。若由 ChatGPT 桌面版的 Slack 连接器发送，确认消息成功后再运行 python -m kaggle_watch ack --slug 比赛slug；它只接受处于待通知状态的比赛。不要把 webhook 写到仓库、命令参数或日志中。状态存于已忽略的 .local/kaggle_watch.sqlite3。默认检查最近 3 页，每次最多显示或发送 5 场；可以使用 --pages 和 --limit 调整。

通知里的“适合先看”仅根据 Kaggle 列表类别给出提示。比赛任务、数据规模和规则，需要选中比赛后再逐项核对。

定时任务必须使用同一个本地 checkout 和状态路径；在新的临时 worktree 中运行会产生另一份基线。第一版建议每天运行一次 scan，有待通知比赛时由连接器发送，发送成功后逐个 ack。


### 完整比赛雷达与评分

使用 python -m kaggle_watch report --token-file api_token 查看 Kaggle CLI general 列表中的全部未截止比赛。它输出一个适合 Slack 主消息的推荐摘要，以及完整比赛清单；添加 --json 可分别取得 summary 和 catalog，发送主消息后把 catalog 放在同一条 Slack 线程中。report 不改变去重状态。

难度和适合度均为 1–5 的学习路线估计。只对已核对比赛介绍与评分规则的比赛评分；适合度会因临近截止而降低。候选比赛的任务、评分规则、理由与来源保存在 kaggle_watch/profiles.py。文件体量取自 2026-09-27 的 Kaggle files 列表，未下载数据；比赛规则与数据可能更新，应在参与前重新核对。

当前 Slack 专用频道为私有 #kaggle-learning（频道 ID：C0C4Q6MPYPP）。通过 ChatGPT 的 Slack 连接器发送时无需 webhook；代码直发 webhook 是另一种可选路径。状态确认仍只用于新比赛通知，日常完整雷达报告不改变已通知记录。
