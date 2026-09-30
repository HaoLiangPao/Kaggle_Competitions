"""Stable Titanic experiment identities and their model/feature configurations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    legacy_id: str
    name: str
    algorithm: str
    include_cabin: bool = False
    include_family: bool = False
    include_family_size: bool = False

    @property
    def attempt(self) -> int:
        return int(self.experiment_id.rsplit("-", 1)[1])

    @property
    def document(self) -> str:
        return f"titanic_poc/experiments/{self.experiment_id}.md"


SPECS = (
    ExperimentSpec("TIT-BL-001", "T00", "永远预测训练折的多数类", "majority"),
    ExperimentSpec("TIT-LR-001", "T01", "四项基础特征", "logistic_regression"),
    ExperimentSpec("TIT-LR-002", "T02", "四项基础特征 + Cabin 甲板", "logistic_regression", include_cabin=True),
    ExperimentSpec("TIT-LR-003", "T03", "四项基础特征 + SibSp/Parch", "logistic_regression", include_family=True),
    ExperimentSpec("TIT-LR-004", "T04", "四项基础特征 + Cabin + SibSp/Parch", "logistic_regression", include_cabin=True, include_family=True),
    ExperimentSpec("TIT-LR-005", "T05", "四项基础特征 + FamilySize", "logistic_regression", include_family_size=True),
    ExperimentSpec("TIT-LR-006", "T06", "四项基础特征 + Cabin + FamilySize", "logistic_regression", include_cabin=True, include_family_size=True),
    ExperimentSpec("TIT-RF-001", "T07", "四项基础特征", "random_forest"),
    ExperimentSpec("TIT-RF-002", "T08", "四项基础特征 + FamilySize", "random_forest", include_family_size=True),
    ExperimentSpec("TIT-RF-003", "T09", "四项基础特征 + Cabin + FamilySize", "random_forest", include_cabin=True, include_family_size=True),
    ExperimentSpec("TIT-RF-004", "T10", "四项基础特征 + Cabin + SibSp/Parch", "random_forest", include_cabin=True, include_family=True),
)

EXPERIMENTS = {spec.experiment_id: spec for spec in SPECS}
assert len(EXPERIMENTS) == len(SPECS)
