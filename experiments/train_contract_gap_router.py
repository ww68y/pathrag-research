import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_answer_contract_agent import clean_text, load_jsonl
from build_claim_evidence_agent import as_float, content_tokens


ROOT_DIR = Path(__file__).resolve().parents[1]
NUMERIC_FEATURES = [
    "question_chars", "answer_chars", "answer_claims", "mean_support",
    "unsupported_ratio", "supported_ratio", "partial_ratio", "minimum_points",
    "required_facets", "top_text_score", "top_graph_score", "question_answer_overlap",
    "answer_question_length_ratio", "uncertainty_flag", "low_support_flag",
    "overlong_flag", "shape_flag",
]
CATEGORICAL_FEATURES = ["source_dataset", "task_dataset", "contract_type"]


def parse_spec(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split("|", 2)
    if len(parts) != 3:
        raise ValueError("Dataset specs must be NAME|RESULT_JSONL|SOURCE_JSONL")
    return parts[0], ROOT_DIR / parts[1], ROOT_DIR / parts[2]


def context_group(name: str, source: dict) -> str:
    identity = (
        clean_text(source.get("source_context_hash"))
        or clean_text(source.get("context_id"))
    )
    if not identity:
        context = clean_text(source.get("context", ""))
        identity = hashlib.sha1(context.encode("utf-8")).hexdigest()[:16]
    return f"{name}:{identity}"


def overlap(left: str, right: str) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    return len(left_tokens & right_tokens) / max(1, len(left_tokens))


def build_records(specs: list[str]) -> list[dict]:
    records = []
    for spec in specs:
        name, result_path, source_path = parse_spec(spec)
        results = load_jsonl(result_path)
        sources = load_jsonl(source_path)
        source_by_line = {
            int(source.get("_subset_line_no", index)): source for index, source in enumerate(sources)
        }
        for row in results:
            agent = row.get("contract_completion_agent") or {}
            if not agent.get("verification_attempted") or not agent.get("verification"):
                continue
            line_no = int(row.get("line_no", 0))
            source = source_by_line.get(line_no, {})
            metrics = agent.get("pre_claim_metrics") or {}
            contract = agent.get("contract") or {}
            reasons = set(agent.get("risk_reasons") or [])
            question = clean_text(row.get("question", ""))
            answer = clean_text(
                row.get("prediction_before_contract_completion") or row.get("prediction", "")
            )
            claims = int(metrics.get("claims", 0))
            supported = int(metrics.get("supported_claims", 0))
            partial = int(metrics.get("partial_claims", 0))
            text_hits = agent.get("question_text_evidence") or []
            graph_hits = agent.get("question_graph_evidence") or []
            features = {
                "question_chars": len(question),
                "answer_chars": len(answer),
                "answer_claims": claims,
                "mean_support": as_float(metrics.get("mean_support")),
                "unsupported_ratio": as_float(metrics.get("unsupported_ratio")),
                "supported_ratio": supported / max(1, claims),
                "partial_ratio": partial / max(1, claims),
                "minimum_points": int(contract.get("minimum_distinct_points", 1)),
                "required_facets": len(contract.get("required_facets") or []),
                "top_text_score": as_float(text_hits[0].get("score")) if text_hits else 0.0,
                "top_graph_score": as_float(graph_hits[0].get("score")) if graph_hits else 0.0,
                "question_answer_overlap": overlap(question, answer),
                "answer_question_length_ratio": len(answer) / max(1, len(question)),
                "uncertainty_flag": int("uncertainty_or_hedging" in reasons),
                "low_support_flag": int("low_claim_support" in reasons),
                "overlong_flag": int("overlong_answer_may_drift" in reasons),
                "shape_flag": int("answer_shape_below_contract" in reasons),
                "source_dataset": name,
                "task_dataset": clean_text(row.get("dataset", "unknown")),
                "contract_type": clean_text(contract.get("type", "general")),
            }
            records.append({
                "source_dataset": name,
                "line_no": line_no,
                "group": context_group(name, source),
                "label": int(bool(agent.get("repair_needed"))),
                "features": features,
            })
    return records


def make_model(
    numeric_features: list[str],
    categorical_features: list[str],
    classifier_name: str,
) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transform = ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("categorical", categorical, categorical_features),
    ])
    if classifier_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    else:
        classifier = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.7)
    return Pipeline([
        ("features", transform),
        ("classifier", classifier),
    ])


def feature_rows(records: list[dict]) -> list[dict]:
    return [record["features"] for record in records]


def tune_threshold(labels: np.ndarray, probabilities: np.ndarray, min_recall: float) -> tuple[float, dict]:
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        recall = recall_score(labels, predictions, zero_division=0)
        precision = precision_score(labels, predictions, zero_division=0)
        selected = int(predictions.sum())
        metrics = {
            "threshold": round(float(threshold), 6),
            "recall": round(float(recall), 6),
            "precision": round(float(precision), 6),
            "f1": round(float(f1_score(labels, predictions, zero_division=0)), 6),
            "selected": selected,
            "verification_reduction": round(1 - selected / len(labels), 6),
        }
        if recall >= min_recall:
            candidates.append(metrics)
    if not candidates:
        threshold = 0.0
        predictions = np.ones_like(labels)
        return threshold, {
            "threshold": threshold,
            "recall": 1.0,
            "precision": round(float(labels.mean()), 6),
            "f1": round(float(f1_score(labels, predictions)), 6),
            "selected": len(labels),
            "verification_reduction": 0.0,
        }
    best = max(candidates, key=lambda item: (item["verification_reduction"], item["precision"], item["f1"]))
    return float(best["threshold"]), best


def feature_weights(model: Pipeline, limit: int = 12) -> list[tuple[str, float]]:
    names = model.named_steps["features"].get_feature_names_out()
    classifier = model.named_steps["classifier"]
    weights = (
        classifier.feature_importances_
        if hasattr(classifier, "feature_importances_")
        else classifier.coef_[0]
    )
    ranked = sorted(zip(names, weights), key=lambda item: abs(item[1]), reverse=True)
    return [(str(name), round(float(weight), 6)) for name, weight in ranked[:limit]]


def write_report(
    path: Path,
    records: list[dict],
    probabilities: np.ndarray,
    threshold_metrics: dict,
    weights: list[tuple[str, float]],
    folds: int,
    version: str,
    feature_mode: str,
    classifier_name: str,
) -> None:
    labels = np.array([record["label"] for record in records])
    predictions = (probabilities >= threshold_metrics["threshold"]).astype(int)
    dataset_counts = {}
    for record in records:
        counts = dataset_counts.setdefault(record["source_dataset"], [0, 0])
        counts[0] += 1
        counts[1] += record["label"]
    lines = [
        f"# SAGE {version} Learned Contract-Gap Router\n\n",
        f"- Samples: {len(records)}\n",
        f"- Positive severe-gap labels: {int(labels.sum())}\n",
        f"- Context groups: {len(set(record['group'] for record in records))}\n",
        f"- Stratified group folds: {folds}\n",
        f"- Feature mode: `{feature_mode}`\n",
        f"- Classifier: `{classifier_name}`\n",
        "- Reference answers and evaluation scores are not model features or labels.\n\n",
        "## Out-of-Fold Metrics\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| ROC-AUC | {roc_auc_score(labels, probabilities):.4f} |\n",
        f"| PR-AUC | {average_precision_score(labels, probabilities):.4f} |\n",
        f"| Threshold | {threshold_metrics['threshold']:.3f} |\n",
        f"| Severe-gap recall | {threshold_metrics['recall'] * 100:.2f}% |\n",
        f"| Precision | {threshold_metrics['precision'] * 100:.2f}% |\n",
        f"| F1 | {threshold_metrics['f1']:.4f} |\n",
        f"| Predicted verifier calls | {int(predictions.sum())} / {len(records)} |\n",
        f"| Verification reduction | {threshold_metrics['verification_reduction'] * 100:.2f}% |\n\n",
        "## Dataset Coverage\n\n",
        "| Dataset | Verified labels | Severe gaps |\n",
        "|---|---:|---:|\n",
    ]
    for name, (samples, positives) in sorted(dataset_counts.items()):
        lines.append(f"| {name} | {samples} | {positives} |\n")
    lines.extend(["\n## Largest Model Weights\n\n", "| Feature | Weight |\n", "|---|---:|\n"])
    for name, weight in weights:
        lines.append(f"| `{name}` | {weight} |\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--version", default="v30")
    parser.add_argument("--feature-mode", choices=["full", "invariant"], default="full")
    parser.add_argument("--classifier", choices=["logistic", "random_forest"], default="logistic")
    args = parser.parse_args()

    records = build_records(args.dataset)
    labels = np.array([record["label"] for record in records])
    groups = np.array([record["group"] for record in records])
    rows = pd.DataFrame(feature_rows(records))
    numeric_features = NUMERIC_FEATURES
    categorical_features = (
        CATEGORICAL_FEATURES if args.feature_mode == "full" else ["contract_type"]
    )
    folds = min(args.folds, len(set(groups)))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    probabilities = np.zeros(len(records), dtype=float)
    fold_ids = np.zeros(len(records), dtype=int)
    for fold, (train_index, test_index) in enumerate(splitter.split(rows, labels, groups), start=1):
        model = make_model(numeric_features, categorical_features, args.classifier)
        train_rows = rows.iloc[train_index]
        test_rows = rows.iloc[test_index]
        model.fit(train_rows, labels[train_index])
        probabilities[test_index] = model.predict_proba(test_rows)[:, 1]
        fold_ids[test_index] = fold

    threshold, threshold_metrics = tune_threshold(labels, probabilities, args.min_recall)
    final_model = make_model(numeric_features, categorical_features, args.classifier)
    final_model.fit(rows, labels)
    artifact = {
        "model": final_model,
        "threshold": threshold,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_mode": args.feature_mode,
        "classifier_name": args.classifier,
    }
    model_path = ROOT_DIR / args.model_output
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    output_rows = []
    for index, record in enumerate(records):
        output_rows.append({
            **record,
            "fold": int(fold_ids[index]),
            "oof_probability": round(float(probabilities[index]), 6),
            "threshold": threshold,
            "predicted_severe_gap": bool(probabilities[index] >= threshold),
        })
    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in output_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    weights = feature_weights(final_model)
    write_report(
        ROOT_DIR / args.report,
        records,
        probabilities,
        threshold_metrics,
        weights,
        folds,
        args.version,
        args.feature_mode,
        args.classifier,
    )
    print(f"Saved OOF predictions to {output_path}")
    print(f"Saved learned router to {model_path}")
    print(f"Threshold={threshold:.3f} recall={threshold_metrics['recall']:.3f} reduction={threshold_metrics['verification_reduction']:.3f}")


if __name__ == "__main__":
    main()
