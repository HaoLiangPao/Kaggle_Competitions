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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

if __package__:
    from .experiment_registry import EXPERIMENTS
else:
    from experiment_registry import EXPERIMENTS


FEATURES = ["Pclass", "Sex", "Age", "Fare"]
TARGET = "Survived"
SEED = 42
FOREST_PARAMS = {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 5, "random_state": SEED, "n_jobs": -1}


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


def cabin_to_deck(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract a broad deck category; retain missing cabins as Unknown."""
    deck = frame.iloc[:, 0].fillna("").astype(str).str.strip().str[:1].replace("", "Unknown")
    return deck.to_frame(name="Deck")


def add_family_size(frame: pd.DataFrame) -> pd.DataFrame:
    """Count the passenger and their relatives travelling aboard."""
    frame = frame.copy()
    frame["FamilySize"] = frame["SibSp"] + frame["Parch"] + 1
    return frame


def make_model(include_cabin: bool = False, include_family: bool = False, include_family_size: bool = False, estimator=None):
    numeric_features = ["Pclass", "Age", "Fare"]
    if include_family:
        numeric_features += ["SibSp", "Parch"]
    if include_family_size:
        numeric_features += ["FamilySize"]
    transformers = [
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_features),
        ("sex", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), ["Sex"]),
    ]
    if include_cabin:
        transformers.append((
            "cabin_deck",
            make_pipeline(FunctionTransformer(cabin_to_deck, validate=False), OneHotEncoder(handle_unknown="ignore")),
            ["Cabin"],
        ))
    preprocessing = ColumnTransformer(transformers)
    steps = []
    if include_family_size:
        steps.append(FunctionTransformer(add_family_size, validate=False))
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000, random_state=SEED)
    return make_pipeline(*steps, preprocessing, estimator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Titanic baseline and produce a submission-shaped CSV")
    parser.add_argument("--zip", type=Path, default=Path(".local/titanic/titanic.zip"))
    parser.add_argument("--include-cabin", action="store_true", help="Add Cabin deck and an Unknown category")
    parser.add_argument("--include-family", action="store_true", help="Add SibSp and Parch as numeric features")
    parser.add_argument("--include-family-size", action="store_true", help="Add SibSp + Parch + 1 as one numeric feature")
    parser.add_argument("--model", choices=("majority", "logistic_regression", "random_forest"), default="logistic_regression")
    parser.add_argument("--experiment-id", choices=tuple(EXPERIMENTS), help="Use a registered configuration and ID-specific local output directory")
    parser.add_argument("--list-experiments", action="store_true", help="List stable experiment IDs and exit")
    parser.add_argument("--output", type=Path, help="Prediction CSV path; defaults depend on experiment")
    parser.add_argument("--results", type=Path, help="Results JSON path; defaults depend on experiment")
    args = parser.parse_args()
    if args.list_experiments:
        for registered in EXPERIMENTS.values():
            print(f"{registered.experiment_id}  {registered.algorithm}  {registered.name}")
        return
    spec = EXPERIMENTS.get(args.experiment_id)
    if spec:
        if args.include_cabin or args.include_family or args.include_family_size or args.model != "logistic_regression":
            parser.error("--experiment-id supplies the model and feature flags; do not combine them")
        args.include_cabin = spec.include_cabin
        args.include_family = spec.include_family
        args.include_family_size = spec.include_family_size
        args.model = spec.algorithm
    if args.include_family and args.include_family_size:
        parser.error("Choose either --include-family or --include-family-size")

    train, test, sample = read_competition_data(args.zip)
    features = FEATURES + (["Cabin"] if args.include_cabin else []) + (["SibSp", "Parch"] if args.include_family or args.include_family_size else [])
    for name in features:
        if name not in train or name not in test:
            raise ValueError(f"{name} is required for this experiment")
    x, y = train[features], train[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    estimator = RandomForestClassifier(**FOREST_PARAMS) if args.model == "random_forest" else None
    candidate = DummyClassifier(strategy="most_frequent") if args.model == "majority" else make_model(args.include_cabin, args.include_family, args.include_family_size, estimator=estimator)
    score_models = (("majority", candidate),) if args.model == "majority" else (("majority", DummyClassifier(strategy="most_frequent")), (args.model, candidate))
    scores = {}
    for name, model in score_models:
        values = cross_val_score(model, x, y, scoring="accuracy", cv=cv)
        scores[name] = {"folds": [round(float(v), 6) for v in values], "mean": round(float(values.mean()), 6), "std": round(float(values.std()), 6)}

    model = candidate.fit(x, y)
    predictions = model.predict(test[features])
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], TARGET: predictions.astype(int)})
    if submission.columns.tolist() != sample.columns.tolist() or len(submission) != len(sample):
        raise ValueError("Generated submission does not match the sample submission shape")
    if not set(submission[TARGET].unique()) <= {0, 1}:
        raise ValueError("Predictions must be 0 or 1")

    variant = "_".join(name for name, included in (("cabin", args.include_cabin), ("family", args.include_family), ("family_size", args.include_family_size)) if included) or "poc"
    if args.model != "logistic_regression":
        variant += f"_{args.model}"
    output = args.output or (Path(f".local/titanic/experiments/{spec.experiment_id}/submission.csv") if spec else Path(f".local/titanic/submission_{variant}.csv"))
    results_path = args.results or (Path(f".local/titanic/experiments/{spec.experiment_id}/results.json") if spec else Path(f".local/titanic/{variant}_results.json"))
    model_features = FEATURES + (["Cabin"] if args.include_cabin else [])
    if args.include_family:
        model_features += ["SibSp", "Parch"]
    if args.include_family_size:
        model_features += ["FamilySize"]
    if args.model == "majority":
        model_features = []
    family_encoding = None
    if args.include_family:
        family_encoding = "SibSp and Parch as scaled numbers"
    elif args.include_family_size:
        family_encoding = "FamilySize = SibSp + Parch + 1, scaled"
    results = {
        "experiment_id": spec.experiment_id if spec else None,
        "experiment_name": spec.name if spec else None,
        "algorithm_attempt": spec.attempt if spec else None,
        "experiment_document": spec.document if spec else None,
        "competition": "titanic",
        "data_sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        "train_rows": len(train),
        "test_rows": len(test),
        "features": model_features,
        "source_columns": features,
        "cabin_encoding": "first character; missing=Unknown" if args.include_cabin else None,
        "family_encoding": family_encoding,
        "algorithm": args.model,
        "model_params": FOREST_PARAMS if args.model == "random_forest" else {"strategy": "most_frequent"} if args.model == "majority" else {"max_iter": 1000, "random_state": SEED},
        "target": TARGET,
        "metric": "accuracy",
        "validation": {"method": "StratifiedKFold", "folds": 5, "shuffle": True, "random_state": SEED},
        "scores": scores,
        "submission_rows": len(submission),
        "versions": {"pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    if output.resolve() == results_path.resolve():
        raise ValueError("Prediction and results paths must be different")
    csv_content = submission.to_csv(index=False)
    results_content = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    for path, content in ((output, csv_content), (results_path, results_content)):
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f"Existing experiment artifact differs; choose a new path or experiment ID: {path}")
    for path, content in ((output, csv_content), (results_path, results_content)):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
