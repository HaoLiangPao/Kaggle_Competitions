# Kaggle S6E1 - Student Exam Score Prediction

## Competition Overview

This is a Playground Series competition focused on predicting student exam scores based on various features including study habits, class attendance, sleep quality, and other factors.

## Dataset

- **train.csv**: Training data with exam scores
- **test.csv**: Test data for predictions
- **sample_submission.csv**: Example submission format

### Features
- `age`: Student age
- `gender`: Student gender
- `course`: Course type (b.sc, diploma, bca, b.com, ba, bba, b.tech)
- `study_hours`: Hours spent studying
- `class_attendance`: Class attendance percentage
- `internet_access`: Whether student has internet access (yes/no)
- `sleep_hours`: Average sleep hours
- `sleep_quality`: Quality of sleep (poor/average/good)
- `study_method`: Study method used (self-study, online videos, coaching, group study, mixed)
- `facility_rating`: Rating of study facilities (low/medium/high)
- `exam_difficulty`: Difficulty of exam (easy/moderate/hard)

### Target
- `exam_score`: Student exam score (0-100)

## Pipeline Architecture

This project uses a reusable ML pipeline (`ml_pipeline/`) with the following components:

```
🤖 Data Loading → 🤖 Preprocessing → 🤖 Model Training → 🤖 Evaluation → 🤖 Inference
```

### Pipeline Steps

1. **Data Loading & Validation**: Load CSV, validate schema, check data quality
2. **Feature Preprocessing**: 
   - Numerical features: imputation + standardization
   - Categorical features: one-hot encoding
3. **Model Training**: Linear Regression baseline with 5-fold CV
4. **Evaluation**: RMSE, MAE, R² metrics
5. **Artifact Saving**: Model, preprocessor, and metrics saved
6. **Inference**: Generate predictions for test set

## How to Run

### Install Dependencies

```bash
pip install pandas numpy scikit-learn pyyaml joblib
```

### Training

Train the Linear Regression model with cross-validation:

```bash
cd kaggle_s6_e1
python train.py
```

This will:
- Load and preprocess the training data
- Perform 5-fold cross-validation
- Train final model on full training set
- Save artifacts to `artifacts/` directory

### Inference

Generate predictions for the test set:

```bash
cd kaggle_s6_e1
python predict.py
```

This will:
- Load trained model and preprocessor
- Process test data
- Generate predictions
- Create `submission.csv`

## Results

Training results and metrics will be saved to:
- `artifacts/model.pkl` - Trained model
- `artifacts/preprocessor.pkl` - Fitted preprocessor
- `artifacts/cv_results.csv` - Cross-validation results
- `artifacts/cv_summary.json` - CV metrics summary
- `training.log` - Training logs

## Configuration

Edit `config.yaml` to customize:
- Model type (`lr`, `ridge`, `lasso`, `gbdt`)
- Model parameters
- CV folds
- Preprocessing options
- Output paths

### Example: Switch to Ridge Regression

```yaml
model:
  type: "ridge"
  params:
    alpha: 1.0
```

## Next Steps

1. **Exploratory Data Analysis**: Create `eda.ipynb` for feature analysis
2. **Feature Engineering**: Add domain-specific features
3. **Model Upgrade**: Switch to GBDT (LightGBM/XGBoost) for better performance
4. **Hyperparameter Tuning**: Optimize model parameters

## Pipeline Benefits

✅ **Modular**: Easy to swap models or preprocessing steps  
✅ **Reusable**: Apply same pipeline to other competitions  
✅ **Production-ready**: Proper train/test separation, artifact management  
✅ **Extensible**: Add new features without changing core pipeline
