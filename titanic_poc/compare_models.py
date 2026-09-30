"""Compare Titanic feature sets and classifiers locally; never submit to Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from titanic_poc.run import FEATURES, FOREST_PARAMS, TARGET, make_model, read_competition_data


SEEDS = [42, 43, 44, 45, 46]
CASES = [
    ("logistic_basic", "logistic_regression", False, False, False),
    ("logistic_family_size", "logistic_regression", False, False, True),
    ("logistic_cabin_family_size", "logistic_regression", True, False, True),
    ("forest_basic", "random_forest", False, False, False),
    ("forest_family_size", "random_forest", False, False, True),
    ("forest_cabin_family_size", "random_forest", True, False, True),
    ("forest_cabin_separate_family", "random_forest", True, True, False),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed Titanic models using the same validation splits")
    parser.add_argument("--zip", type=Path, default=Path(".local/titanic/titanic.zip"))
    parser.add_argument("--results", type=Path, default=Path(".local/titanic/model_comparison.json"))
    args = parser.parse_args()

    train, _, _ = read_competition_data(args.zip)
    comparisons = []
    for name, algorithm, cabin, separate_family, family_size in CASES:
        source_columns = FEATURES + (["Cabin"] if cabin else [])
        if separate_family or family_size:
            source_columns += ["SibSp", "Parch"]
        x, y = train[source_columns], train[TARGET]
        estimator = RandomForestClassifier(**FOREST_PARAMS) if algorithm == "random_forest" else None
        model = make_model(cabin, separate_family, family_size, estimator=estimator)
        seeds = []
        for seed in SEEDS:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            scores = cross_val_score(model, x, y, scoring="accuracy", cv=cv)
            seeds.append({"seed": seed, "folds": [round(float(v), 6) for v in scores], "mean": round(float(scores.mean()), 6)})
        comparisons.append({
            "name": name,
            "algorithm": algorithm,
            "features": FEATURES + (["Cabin deck"] if cabin else []) + (["SibSp", "Parch"] if separate_family else []) + (["FamilySize"] if family_size else []),
            "source_columns": source_columns,
            "seed_results": seeds,
            "mean_of_seed_means": round(sum(item["mean"] for item in seeds) / len(seeds), 6),
        })

    results = {
        "competition": "titanic",
        "data_sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        "metric": "accuracy",
        "validation": "Five stratified folds, shuffled, for each listed seed; preprocessing fitted within each fold",
        "seeds": SEEDS,
        "forest_params": FOREST_PARAMS,
        "note": "Exploratory local comparison; models were selected on these same validation splits. This script does not submit or report Kaggle scores; see EXPERIMENTS.md for separate submission records.",
        "sklearn_version": sklearn.__version__,
        "comparisons": comparisons,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({item["name"]: item["mean_of_seed_means"] for item in comparisons}, indent=2))


if __name__ == "__main__":
    main()
