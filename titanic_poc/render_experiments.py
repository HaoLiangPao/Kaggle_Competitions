"""Render the Titanic experiment index and one document per stable experiment ID."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from titanic_poc.experiment_registry import EXPERIMENTS, SPECS, ExperimentSpec
from titanic_poc.run import FEATURES

ROOT = Path(__file__).resolve().parents[1]
RECORDS_PATH = ROOT / "titanic_poc/experiment_records.json"
INDEX_PATH = ROOT / "titanic_poc/EXPERIMENTS.md"
ALGORITHM_NAMES = {
    "majority": "多数类基线（Majority Baseline）",
    "logistic_regression": "逻辑回归（Logistic Regression）",
    "random_forest": "随机森林（Random Forest）",
}


def percent(value: float | None) -> str:
    if value is None:
        return "—"
    rounded = (Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded}%"


def validate_records(data: dict) -> None:
    if data["schema_version"] != 1 or set(data["records"]) != set(EXPERIMENTS):
        raise ValueError("Record schema or experiment IDs do not match the registry")
    for spec in SPECS:
        record = data["records"][spec.experiment_id]
        if record["algorithm"] != spec.algorithm or record["legacy_id"] != spec.legacy_id:
            raise ValueError(f"Algorithm or legacy ID mismatch: {spec.experiment_id}")
        expected_source = FEATURES + (["Cabin"] if spec.include_cabin else [])
        if spec.include_family or spec.include_family_size:
            expected_source += ["SibSp", "Parch"]
        expected_features = [] if spec.algorithm == "majority" else FEATURES + (["Cabin"] if spec.include_cabin else [])
        if spec.include_family:
            expected_features += ["SibSp", "Parch"]
        if spec.include_family_size:
            expected_features += ["FamilySize"]
        if record["source_columns"] != expected_source or record["features"] != expected_features:
            raise ValueError(f"Feature mismatch: {spec.experiment_id}")
        folds = record["single_cv"]["folds"]
        if len(folds) != 5 or abs(sum(folds) / 5 - record["single_cv"]["mean"]) > 0.000002:
            raise ValueError(f"Single CV score mismatch: {spec.experiment_id}")
        repeated = record["repeated_cv"]
        if repeated:
            if [item["seed"] for item in repeated["seeds"]] != data["validation_repeated_seeds"]:
                raise ValueError(f"Repeated CV seed mismatch: {spec.experiment_id}")
            means = [item["mean"] for item in repeated["seeds"]]
            if abs(sum(means) / len(means) - repeated["mean"]) > 0.000002:
                raise ValueError(f"Repeated CV score mismatch: {spec.experiment_id}")
        submission = record["submission"]
        if submission:
            snapshot = submission["rank_snapshot"]
            if not (0 <= submission["public_score"] <= 1 and 1 <= snapshot["rank"] <= snapshot["teams"]):
                raise ValueError(f"Invalid Kaggle submission: {spec.experiment_id}")


def extra_features(spec: ExperimentSpec) -> str:
    extra = []
    if spec.include_cabin:
        extra.append("Cabin 甲板")
    if spec.include_family:
        extra.append("SibSp + Parch（分开）")
    if spec.include_family_size:
        extra.append("FamilySize")
    return "、".join(extra) if extra else "无"


def index_markdown(data: dict) -> str:
    lines = [
        "# Titanic 实验记录（Experiment Log）",
        "",
        "每一行对应一份独立实验文档。ID 格式是 `TIT-算法缩写-尝试序号`，例如 `TIT-RF-002` 表示 Titanic 的随机森林第 2 次尝试；旧的 T00–T10 编号保留用于追溯先前讨论。点击 ID 可查看参数、逐折结果、Kaggle 提交及复现命令。",
        "",
        "**基础特征 B**：`Pclass`（舱位等级）、`Sex`（性别）、`Age`（年龄）、`Fare`（票价）。除多数类基线外，每行都包含 B。评分指标（metric）统一为准确率（accuracy）。本地交叉验证（cross-validation）与 Kaggle 公开榜（public leaderboard）使用不同数据，不要把两个分数当作同一测试结果。",
        "",
        "| 实验 ID / 文档 | 旧 ID | 名称 | 算法与尝试 | 额外特征 | 本地 5 折 | 本地重复 5×5 折 | Kaggle 公榜 | 排名快照 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for spec in SPECS:
        record = data["records"][spec.experiment_id]
        submission = record["submission"]
        cloud = f"{submission['public_score']:.5f} · #{submission['submission_id']}" if submission else "—"
        snapshot = submission["rank_snapshot"] if submission else None
        rank = f"{snapshot['rank']} / {snapshot['teams']}" if snapshot else "—"
        repeated = record["repeated_cv"]
        lines.append(
            f"| [{spec.experiment_id}](experiments/{spec.experiment_id}.md) | {spec.legacy_id} | {spec.name} | "
            f"{ALGORITHM_NAMES[spec.algorithm]} · 第 {spec.attempt} 次 | {extra_features(spec)} | "
            f"{percent(record['single_cv']['mean'])} | {percent(repeated['mean'] if repeated else None)} | {cloud} | {rank} |"
        )
    lines += [
        "",
        "## 如何比较",
        "",
        "- `—` 表示尚未做该项验证或尚未提交，绝不表示得分为零。只有实际提交并确认完成的版本才填写 Kaggle 分数。",
        "- 单次本地结果：随机种子 42 的分层五折（Stratified 5-fold CV）。重复结果：种子 42–46 各做一次五折，再平均五个均分。优先在同一列比较模型；重复划分共享训练样本，不是五份独立测试集。",
        "- 排名是团队在提交后抓取的榜单快照，不是永久名次。每个已提交版本的文档都写明提交时间、快照时间及当时队伍总数；旧版 T01 的快照在提交数小时后抓取。",
        "- FamilySize = SibSp + Parch + 1；Cabin 仅取首字母作为甲板，缺失时记为 `Unknown`。这些是输入特征（features），准确率才是评分指标（metric）。",
        f"- 官方数据压缩包 SHA256：`{data['data_sha256']}`。原始比赛数据、预测 CSV 和本机状态保存在被 Git 忽略的 `.local/titanic/`；本仓库只追踪配置、聚合结果和提交元数据。",
        "",
        "## 实验工作流",
        "",
        "```text",
        "官方数据 → 固定实验 ID → 特征与预处理 Pipeline → 算法 → 分层交叉验证",
        "         → 本地结果 → 全量训练与预测文件 → 经授权提交 → 公榜快照",
        "```",
        "",
        "运行 `.local/venv/bin/python titanic_poc/run.py --list-experiments` 查看全部 ID；使用 `--experiment-id TIT-RF-002` 复现某一版。每个 ID 的输出单独写入 `.local/titanic/experiments/<ID>/`；内容相同的重跑会复用文件，内容不同时会报错，避免覆盖历史结果。重复交叉验证使用 `.local/venv/bin/python -m titanic_poc.compare_models`。",
        "",
        "实验配置在 `experiment_registry.py`，可审查的历史结果在 `experiment_records.json`。更新记录后运行 `.local/venv/bin/python -m titanic_poc.render_experiments` 重新生成本索引与各实验文档；`--check` 仅检查文件是否同步。新算法或新特征应使用新 ID，保留旧记录。",
        "",
    ]
    return "\n".join(lines)


def detail_markdown(spec: ExperimentSpec, data: dict) -> str:
    record = data["records"][spec.experiment_id]
    local = record["single_cv"]
    repeated = record["repeated_cv"]
    submission = record["submission"]
    lines = [
        f"# {spec.experiment_id}｜{ALGORITHM_NAMES[spec.algorithm]}第 {spec.attempt} 次：{spec.name}",
        "",
        f"[返回全部实验](../EXPERIMENTS.md) · 旧编号 `{spec.legacy_id}` · 任务：Titanic 生还二分类（binary classification）",
        "",
        "## 配置（Configuration）",
        "",
        "| 项目 | 内容 |",
        "| --- | --- |",
        f"| 算法 | `{spec.algorithm}` |",
        f"| 算法内尝试序号 | {spec.attempt} |",
        f"| 输入原始列 | {', '.join('`'+x+'`' for x in record['source_columns'])} |",
        f"| 进入模型的特征 | {', '.join('`'+x+'`' for x in record['features']) if record['features'] else '无；多数类模型忽略乘客特征'} |",
        f"| 超参数（hyperparameters） | `{json.dumps(record['model_params'], ensure_ascii=False, sort_keys=True)}` |",
        f"| 数据 SHA256 | `{data['data_sha256']}` |",
        "",
    ]
    if spec.algorithm != "majority":
        steps = "数值缺失值用训练折中位数填充并缩放；Sex 缺失值用训练折众数填充，再做独热编码（one-hot encoding）。"
        if spec.include_cabin:
            steps += " Cabin 在每折流程内提取甲板首字母；缺失值为 Unknown。"
        if spec.include_family_size:
            steps += " FamilySize 在每折流程内由 SibSp + Parch + 1 构造。"
        lines += ["预处理（preprocessing）：" + steps, ""]
    else:
        lines += ["多数类基线忽略所有输入特征，只在每个训练折统计标签中的多数类。", ""]
    lines += [
        "## 本地验证（Local CV）",
        "",
        f"- 分层五折，打乱，随机种子 42；准确率（accuracy）均分 **{percent(local['mean'])}**，标准差 `{local['std']:.6f}`。",
        "- 五折分数：" + "、".join(f"`{score:.6f}`" for score in local["folds"]) + "。",
    ]
    if repeated:
        lines += [
            f"- 种子 42–46 的五次五折均分平均为 **{percent(repeated['mean'])}**；各次均分："
            + "、".join(f"`{item['seed']}: {item['mean']:.6f}`" for item in repeated["seeds"]) + "。",
        ]
    else:
        lines += ["- 尚未完成五组随机种子的重复交叉验证。"]
    lines += ["", "## Kaggle 云端结果（Public leaderboard）", ""]
    if submission:
        snapshot = submission["rank_snapshot"]
        lines += [
            f"- 提交编号 `{submission['submission_id']}`；提交时间 `{submission['submitted_at_utc']}`；公开分数 **{submission['public_score']:.5f}**。",
            f"- 排名快照 `{snapshot['captured_at_utc']}`：**{snapshot['rank']} / {snapshot['teams']}** 队。名次会随滚动榜单变化。",
            f"- 已提交预测 CSV 的 SHA256：`{submission['submission_sha256']}`。",
        ]
        if spec.experiment_id == "TIT-LR-001":
            lines += ["- 该排名快照在提交数小时后抓取，不能称为提交瞬间排名。"]
    else:
        lines += ["- 未提交；云端分数与名次均未知。"]
    lines += [
        "",
        "## 复现与判断",
        "",
        f"本地运行：`.local/venv/bin/python titanic_poc/run.py --experiment-id {spec.experiment_id}`。结果与预测文件分别保存在 `.local/titanic/experiments/{spec.experiment_id}/results.json` 和 `submission.csv`；脚本本身不会上传 Kaggle。",
        "",
        record["note"],
        "",
        "同一训练集被多次用于方案比较，较高的本地分数可能受到选择偏差影响；不要把交叉验证分数当作公榜分数。",
        "",
    ]
    return "\n".join(lines)


def render(check: bool = False) -> None:
    data = json.loads(RECORDS_PATH.read_text())
    validate_records(data)
    files = {INDEX_PATH: index_markdown(data)}
    for spec in SPECS:
        files[ROOT / spec.document] = detail_markdown(spec, data)
    for path, content in files.items():
        if check:
            if not path.exists() or path.read_text() != content:
                raise ValueError(f"Experiment document is out of sync: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(f"{'Checked' if check else 'Rendered'} {len(files)} experiment documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify generated documents without changing files")
    render(check=parser.parse_args().check)
