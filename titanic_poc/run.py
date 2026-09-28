"""Small, reproducible Titanic classification experiment.

Run after downloading the official competition zip into .local/titanic/.
Nothing is submitted to Kaggle by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = ["Pclass", "Sex", "Age", "Fare"]
TARGET = "Survived"
SEED = 42


def read_competition_data(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with ZipFile(zip_path) as archive:
        train = pd.read_csv(archive.open("train.csv"))
        test = pd.read_csv(archive.open("test.csv"))
        sample = pd.read_csv(archive.open("gender_submission.csv"))

    for name, frame, columns in (
        ("train.csv", train, FEATURES + [TARGET]),
        ("test.csv", test, FEATURES + ["PassengerId"]),
        ("gender_submission.csv", sample, ["PassengerId", TARGET]),
    ):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if test["PassengerId"].tolist() != sample["PassengerId"].tolist():
        raise ValueError("Test passenger IDs do not match the sample submission")
    if train[TARGET].isna().any() or not set(train[TARGET].unique()) <= {0, 1}:
        raise ValueError("Train labels must be binary and nonmissing")
    return train, test, sample


def make_model():
    preprocessing = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["Pclass", "Age", "Fare"]),
            ("sex", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), ["Sex"]),
        ]
    )
    return make_pipeline(preprocessing, LogisticRegression(max_iter=1000, random_state=SEED))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Titanic baseline and produce a submission-shaped CSV")
    parser.add_argument("--zip", type=Path, default=Path(".local/titanic/titanic.zip"))
    parser.add_argument("--output", type=Path, default=Path(".local/titanic/submission_poc.csv"))
    parser.add_argument("--results", type=Path, default=Path(".local/titanic/poc_results.json"))
    args = parser.parse_args()

    train, test, sample = read_competition_data(args.zip)
    x, y = train[FEATURES], train[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for name, model in (("majority", DummyClassifier(strategy="most_frequent")), ("logistic_regression", make_model())):
        values = cross_val_score(model, x, y, scoring="accuracy", cv=cv)
        scores[name] = {"folds": [round(float(v), 6) for v in values], "mean": round(float(values.mean()), 6), "std": round(float(values.std()), 6)}

    model = make_model().fit(x, y)
    predictions = model.predict(test[FEATURES])
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], TARGET: predictions.astype(int)})
    if submission.columns.tolist() != sample.columns.tolist() or len(submission) != len(sample):
        raise ValueError("Generated submission does not match the sample submission shape")
    if not set(submission[TARGET].unique()) <= {0, 1}:
        raise ValueError("Predictions must be 0 or 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    results = {
        "competition": "titanic",
        "data_sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        "train_rows": len(train),
        "test_rows": len(test),
        "features": FEATURES,
        "target": TARGET,
        "metric": "accuracy",
        "validation": {"method": "StratifiedKFold", "folds": 5, "shuffle": True, "random_state": SEED},
        "scores": scores,
        "submission_rows": len(submission),
        "versions": {"pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
