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
