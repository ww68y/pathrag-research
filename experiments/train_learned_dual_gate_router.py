import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_answer_contract_agent import (
    answer_shape_signals,
    clean_text,
    derive_answer_contract,
    load_jsonl,
    row_map,
)
from build_text_first_contract_agent import (
    accept_graph_rescue,
    has_explicit_uncertainty,
    route_text_answer,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS = ["correctness", "coverage", "relevance", "conciseness", "overall"]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "their", "this", "to", "under",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}
BUDGETS = {
    "economy": {"max_attempt_rate": 0.15, "cost_penalty": 0.20},
    "balanced": {"max_attempt_rate": 0.40, "cost_penalty": 0.08},
    "quality": {"max_attempt_rate": 1.00, "cost_penalty": 0.00},
}


def as_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-zA-Z][a-zA-Z-]+", clean_text(text).lower())
        if token not in STOPWORDS and len(token) > 2
    }


def overlap_ratio(source: str, target: str) -> float:
    source_tokens = tokens(source)
    if not source_tokens:
        return 0.0
    return len(source_tokens & tokens(target)) / len(source_tokens)


def jaccard(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def eval_map(path: Path) -> dict[int, dict]:
    return {row.get("line_no"): row.get("scores", {}) for row in load_jsonl(path)}


def parse_dataset_spec(spec: str) -> tuple[str, Path, Path, Path, Path]:
    parts = spec.split(":", 4)
    if len(parts) != 5:
        raise ValueError(
            "--dataset must use name:text_jsonl:graph_jsonl:text_eval_jsonl:graph_eval_jsonl"
        )
    name, text_path, graph_path, text_eval_path, graph_eval_path = parts
    return (
        name,
        ROOT_DIR / text_path,
        ROOT_DIR / graph_path,
        ROOT_DIR / text_eval_path,
        ROOT_DIR / graph_eval_path,
    )


def text_features(dataset: str, row: dict) -> dict:
    question = clean_text(row.get("question"))
    answer = clean_text(row.get("prediction"))
    contract = derive_answer_contract(question)
    shape = answer_shape_signals(row, contract)
    chunks = row.get("retrieved_chunks") or []
    scores = [as_float(chunk.get("score")) for chunk in chunks]
    chunk_chars = [as_float(chunk.get("chars")) for chunk in chunks]
    hyde_query = clean_text(row.get("hyde_query"))
    question_tokens = tokens(question)
    capitalized = re.findall(r"\b[A-Z][a-zA-Z-]+\b", question)
    rule_action, _, _, rule_reasons = route_text_answer(row)

    features = {
        f"dataset={dataset}": 1.0,
        f"contract={contract.get('type', 'general')}": 1.0,
        f"question_start={question.split(' ', 1)[0].lower() if question else 'empty'}": 1.0,
        "question_chars": len(question),
        "question_token_count": len(question_tokens),
        "question_capitalized_count": len(capitalized),
        "question_compound": float(contract.get("compound_question", False)),
        "question_numeric_expected": float(contract.get("numeric_answer_expected", False)),
        "answer_chars": shape.get("answer_chars", 0),
        "answer_sentence_count": shape.get("sentence_count", 0),
        "answer_distinct_points": shape.get("estimated_distinct_points", 0),
        "answer_contrast_markers": shape.get("contrast_markers", 0),
        "answer_causal_markers": shape.get("causal_markers", 0),
        "answer_numeric_mentions": shape.get("numeric_mentions", 0),
        "answer_uncertainty_count": len(has_explicit_uncertainty(answer)),
        "question_answer_overlap": overlap_ratio(question, answer),
        "answer_question_length_ratio": len(answer) / max(1, len(question)),
        "retrieved_chunk_count": len(chunks),
        "retrieval_score_max": max(scores) if scores else 0.0,
        "retrieval_score_mean": sum(scores) / len(scores) if scores else 0.0,
        "retrieval_score_min": min(scores) if scores else 0.0,
        "retrieval_score_spread": (max(scores) - min(scores)) if scores else 0.0,
        "retrieved_chars_total": sum(chunk_chars),
        "hyde_query_chars": len(hyde_query),
        "question_hyde_overlap": overlap_ratio(question, hyde_query),
        "hyde_answer_overlap": jaccard(hyde_query, answer),
        "context_chars": as_float(row.get("context_chars")),
        "num_chunks": as_float(row.get("num_chunks")),
        "rule_pre_attempt": float(rule_action == "graph_rescue"),
        "rule_reason_count": len(rule_reasons) if rule_action == "graph_rescue" else 0,
    }
    for reason in rule_reasons if rule_action == "graph_rescue" else []:
        features[f"rule_reason={reason.split(':', 1)[0]}"] = 1.0
    return features


def post_features(dataset: str, text_row: dict, graph_row: dict) -> dict:
    features = text_features(dataset, text_row)
    question = clean_text(text_row.get("question"))
    text_answer = clean_text(text_row.get("prediction"))
    graph_answer = clean_text(graph_row.get("prediction"))
    contract = derive_answer_contract(question)
    graph_shape = answer_shape_signals(graph_row, contract)
    text_shape = answer_shape_signals(text_row, contract)
    rule_action, _, _, rule_reasons = route_text_answer(text_row)
    rule_accept, guard = accept_graph_rescue(
        text_row,
        graph_row,
        contract,
        text_shape,
        rule_reasons,
    )

    features.update({
        "graph_answer_chars": graph_shape.get("answer_chars", 0),
        "graph_sentence_count": graph_shape.get("sentence_count", 0),
        "graph_distinct_points": graph_shape.get("estimated_distinct_points", 0),
        "graph_contrast_markers": graph_shape.get("contrast_markers", 0),
        "graph_causal_markers": graph_shape.get("causal_markers", 0),
        "graph_numeric_mentions": graph_shape.get("numeric_mentions", 0),
        "graph_uncertainty_count": len(has_explicit_uncertainty(graph_answer)),
        "question_graph_overlap": overlap_ratio(question, graph_answer),
        "text_graph_jaccard": jaccard(text_answer, graph_answer),
        "graph_text_length_ratio": len(graph_answer) / max(1, len(text_answer)),
        "graph_sentence_gain": graph_shape.get("sentence_count", 0) - text_shape.get("sentence_count", 0),
        "graph_point_gain": graph_shape.get("estimated_distinct_points", 0) - text_shape.get("estimated_distinct_points", 0),
        "graph_query_seconds": as_float(graph_row.get("query_seconds")),
        "graph_nodes": as_float(graph_row.get("graph_nodes")),
        "graph_edges": as_float(graph_row.get("graph_edges")),
        "rule_graph_accept": float(rule_accept),
        "rule_utility_gain": as_float(guard.get("utility_gain")),
        "rule_length_ratio": as_float(guard.get("length_ratio")),
    })
    return features


def build_records(specs: list[str]) -> list[dict]:
    records = []
    for spec in specs:
        dataset, text_path, graph_path, text_eval_path, graph_eval_path = parse_dataset_spec(spec)
        text_rows = row_map(text_path)
        graph_rows = row_map(graph_path)
        text_evals = eval_map(text_eval_path)
        graph_evals = eval_map(graph_eval_path)
        common_lines = sorted(set(text_rows) & set(graph_rows) & set(text_evals) & set(graph_evals))
        for line_no in common_lines:
            text_row = text_rows[line_no]
            graph_row = graph_rows[line_no]
            text_scores = text_evals[line_no]
            graph_scores = graph_evals[line_no]
            text_overall = as_float(text_scores.get("overall"))
            graph_overall = as_float(graph_scores.get("overall"))
            context_hash = clean_text(graph_row.get("context_hash")) or f"line-{line_no}"
            contract = derive_answer_contract(clean_text(text_row.get("question")))
            rule_action, _, rule_signals, rule_reasons = route_text_answer(text_row)
            rule_accept, rule_guard = accept_graph_rescue(
                text_row,
                graph_row,
                contract,
                rule_signals,
                rule_reasons,
            )
            records.append({
                "dataset": dataset,
                "line_no": line_no,
                "group": f"{dataset}:{context_hash}",
                "text_row": text_row,
                "graph_row": graph_row,
                "text_scores": text_scores,
                "graph_scores": graph_scores,
                "text_features": text_features(dataset, text_row),
                "post_features": post_features(dataset, text_row, graph_row),
                "label": int(graph_overall > text_overall),
                "quality_gain": graph_overall - text_overall,
                "rule_attempt": rule_action == "graph_rescue",
                "rule_accept": bool(rule_accept),
                "rule_guard": rule_guard,
            })
    return records


def split_by_dataset_groups(records: list[dict], test_size: float, random_state: int) -> tuple[list[dict], list[dict]]:
    train_records = []
    test_records = []
    by_dataset = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)

    for dataset, dataset_records in sorted(by_dataset.items()):
        groups = np.array([record["group"] for record in dataset_records])
        indices = np.arange(len(dataset_records))
        unique_groups = sorted(set(groups))
        if len(unique_groups) < 2:
            train_records.extend(dataset_records)
            continue
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(indices, groups=groups))
        train_records.extend(dataset_records[index] for index in train_idx)
        test_records.extend(dataset_records[index] for index in test_idx)
    return train_records, test_records


def make_model(random_state: int) -> Pipeline:
    return Pipeline([
        ("vectorizer", DictVectorizer(sparse=False)),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                C=0.35,
                max_iter=2000,
                random_state=random_state,
            ),
        ),
    ])


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    auc = roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else 0.5
    return {
        "auc": round(float(auc), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def feature_weights(model: Pipeline, limit: int = 10) -> dict[str, list[tuple[str, float]]]:
    names = model.named_steps["vectorizer"].get_feature_names_out()
    weights = model.named_steps["classifier"].coef_[0]
    ranked = sorted(zip(names, weights), key=lambda item: item[1])
    return {
        "text_preferred": [(name, round(float(weight), 4)) for name, weight in ranked[:limit]],
        "graph_preferred": [(name, round(float(weight), 4)) for name, weight in ranked[-limit:][::-1]],
    }


def tune_thresholds(
    records: list[dict],
    pre_probs: np.ndarray,
    post_probs: np.ndarray,
    max_attempt_rate: float,
    cost_penalty: float,
    budget: str,
) -> dict:
    thresholds = np.arange(0.20, 0.86, 0.05)
    best = None
    rule_attempts = np.array([record["rule_attempt"] for record in records], dtype=bool)
    rule_accepts = np.array([record["rule_accept"] for record in records], dtype=bool)
    utility_gains = np.array([as_float(record["rule_guard"].get("utility_gain")) for record in records])
    length_ratios = np.array([as_float(record["rule_guard"].get("length_ratio")) for record in records])
    for pre_threshold in thresholds:
        learned_attempts = pre_probs >= pre_threshold
        attempts = rule_attempts if budget == "economy" else (rule_attempts | learned_attempts)
        attempt_rate = float(np.mean(attempts))
        if attempt_rate > max_attempt_rate + 1e-9:
            continue
        for post_threshold in thresholds:
            learned_safe = (
                (post_probs >= post_threshold)
                & (utility_gains >= 0.5)
                & ((length_ratios <= 3.0) | (post_probs >= max(0.80, post_threshold)))
            )
            accepts = attempts & (
                rule_accepts if budget == "economy" else (rule_accepts | learned_safe)
            )
            selected = [
                record["graph_scores"] if accepts[index] else record["text_scores"]
                for index, record in enumerate(records)
            ]
            overall = float(np.mean([as_float(scores.get("overall")) for scores in selected]))
            objective = overall - cost_penalty * attempt_rate
            candidate = {
                "pre_threshold": round(float(pre_threshold), 3),
                "post_threshold": round(float(post_threshold), 3),
                "train_overall": round(overall, 4),
                "train_attempt_rate": round(attempt_rate, 4),
                "train_accept_rate": round(float(np.mean(accepts)), 4),
                "objective": round(objective, 6),
            }
            if best is None or (candidate["objective"], candidate["train_overall"]) > (
                best["objective"], best["train_overall"]
            ):
                best = candidate

    if best is None:
        return {
            "pre_threshold": 0.85,
            "post_threshold": 0.85,
            "train_overall": 0.0,
            "train_attempt_rate": 0.0,
            "train_accept_rate": 0.0,
            "objective": 0.0,
        }
    return best


def evaluate_budget(
    records: list[dict],
    pre_probs: np.ndarray,
    post_probs: np.ndarray,
    thresholds: dict,
    budget: str,
) -> tuple[list[dict], dict]:
    pre_threshold = thresholds["pre_threshold"]
    post_threshold = thresholds["post_threshold"]
    rule_attempts = np.array([record["rule_attempt"] for record in records], dtype=bool)
    rule_accepts = np.array([record["rule_accept"] for record in records], dtype=bool)
    utility_gains = np.array([as_float(record["rule_guard"].get("utility_gain")) for record in records])
    length_ratios = np.array([as_float(record["rule_guard"].get("length_ratio")) for record in records])
    learned_attempts = pre_probs >= pre_threshold
    attempts = rule_attempts if budget == "economy" else (rule_attempts | learned_attempts)
    learned_safe = (
        (post_probs >= post_threshold)
        & (utility_gains >= 0.5)
        & ((length_ratios <= 3.0) | (post_probs >= max(0.80, post_threshold)))
    )
    accepts = attempts & (rule_accepts if budget == "economy" else (rule_accepts | learned_safe))
    outputs = []
    metric_sums = defaultdict(float)
    text_metric_sums = defaultdict(float)
    graph_metric_sums = defaultdict(float)
    selected_contexts = {}
    all_contexts = {}
    text_query_seconds = 0.0
    graph_query_seconds = 0.0
    full_graph_query_seconds = 0.0

    for index, record in enumerate(records):
        text_row = record["text_row"]
        graph_row = record["graph_row"]
        chosen_graph = bool(accepts[index])
        attempted = bool(attempts[index])
        chosen_row = graph_row if chosen_graph else text_row
        scores = record["graph_scores"] if chosen_graph else record["text_scores"]
        new_row = dict(chosen_row)
        if chosen_graph:
            new_row["prediction_before_learned_graph_accept"] = text_row.get("prediction", "")
        elif attempted:
            new_row["rejected_learned_graph_prediction"] = graph_row.get("prediction", "")

        new_row["method"] = "sage_v23_learned_dual_gate"
        new_row["learned_dual_gate"] = {
            "enabled": True,
            "pre_graph_probability": round(float(pre_probs[index]), 6),
            "post_accept_probability": round(float(post_probs[index]), 6),
            "pre_threshold": pre_threshold,
            "post_threshold": post_threshold,
            "graph_attempted": attempted,
            "graph_accepted": chosen_graph,
            "chosen_policy": "graph" if chosen_graph else "text",
            "rule_graph_attempt": bool(rule_attempts[index]),
            "rule_graph_accept": bool(rule_accepts[index]),
            "learned_graph_attempt": bool(learned_attempts[index]),
            "learned_safety_passed": bool(learned_safe[index]),
            "rescue_utility_gain": round(float(utility_gains[index]), 6),
            "graph_text_length_ratio": round(float(length_ratios[index]), 6),
            "contract": derive_answer_contract(clean_text(text_row.get("question"))),
            "text_query_seconds": round(as_float(text_row.get("query_seconds")), 3),
            "graph_query_seconds": round(as_float(graph_row.get("query_seconds")) if attempted else 0.0, 3),
        }
        outputs.append(new_row)

        for metric in METRICS:
            metric_sums[metric] += as_float(scores.get(metric))
            text_metric_sums[metric] += as_float(record["text_scores"].get(metric))
            graph_metric_sums[metric] += as_float(record["graph_scores"].get(metric))

        text_query_seconds += as_float(text_row.get("query_seconds"))
        full_graph_query_seconds += as_float(graph_row.get("query_seconds"))
        if attempted:
            graph_query_seconds += as_float(graph_row.get("query_seconds"))
            context_hash = clean_text(graph_row.get("context_hash"))
            if context_hash:
                selected_contexts.setdefault(context_hash, as_float(graph_row.get("insert_seconds")))
        context_hash = clean_text(graph_row.get("context_hash"))
        if context_hash:
            all_contexts.setdefault(context_hash, as_float(graph_row.get("insert_seconds")))

    n = max(1, len(records))
    summary = {
        "samples": len(records),
        "attempts": int(np.sum(attempts)),
        "accepts": int(np.sum(accepts)),
        "attempt_rate": round(float(np.mean(attempts)), 4),
        "accept_rate": round(float(np.mean(accepts)), 4),
        "scores": {metric: round(metric_sums[metric] / n, 4) for metric in METRICS},
        "text_scores": {metric: round(text_metric_sums[metric] / n, 4) for metric in METRICS},
        "graph_scores": {metric: round(graph_metric_sums[metric] / n, 4) for metric in METRICS},
        "text_query_seconds": round(text_query_seconds, 3),
        "added_graph_query_seconds": round(graph_query_seconds, 3),
        "full_graph_query_seconds": round(full_graph_query_seconds, 3),
        "selected_graph_contexts": len(selected_contexts),
        "all_graph_contexts": len(all_contexts),
        "selected_index_seconds": round(sum(selected_contexts.values()), 3),
        "full_index_seconds": round(sum(all_contexts.values()), 3),
    }
    return outputs, summary


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    train_records: list[dict],
    test_records: list[dict],
    pre_metrics: dict,
    post_metrics: dict,
    thresholds: dict[str, dict],
    budget_results: dict[str, dict[str, dict]],
    split_manifest: dict,
    pre_weights: dict[str, list[tuple[str, float]]],
    post_weights: dict[str, list[tuple[str, float]]],
) -> None:
    train_labels = Counter(record["label"] for record in train_records)
    test_labels = Counter(record["label"] for record in test_records)
    lines = [
        "# SAGE v23 Learned Dual-Gate Router Report\n\n",
        "## Split\n\n",
        f"- Train samples: {len(train_records)}\n",
        f"- Test samples: {len(test_records)}\n",
        f"- Train graph-beneficial labels: {train_labels.get(1, 0)} / {len(train_records)}\n",
        f"- Test graph-beneficial labels: {test_labels.get(1, 0)} / {len(test_records)}\n",
        "- Split unit: knowledge-base context hash, independently within each dataset\n",
        "- Runtime features do not include references or evaluation scores\n\n",
        "## Gate Classification\n\n",
        "| Gate | AUC | Balanced accuracy | Precision | Recall | F1 |\n",
        "|---|---:|---:|---:|---:|---:|\n",
        f"| Pre-route | {pre_metrics['auc']} | {pre_metrics['balanced_accuracy']} | {pre_metrics['precision']} | {pre_metrics['recall']} | {pre_metrics['f1']} |\n",
        f"| Post-rescue | {post_metrics['auc']} | {post_metrics['balanced_accuracy']} | {post_metrics['precision']} | {post_metrics['recall']} | {post_metrics['f1']} |\n",
        "\n## Learned Budget Thresholds\n\n",
        "| Budget | Pre threshold | Post threshold | Train attempt rate | Train overall |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for budget in BUDGETS:
        item = thresholds[budget]
        lines.append(
            f"| {budget} | {item['pre_threshold']} | {item['post_threshold']} | "
            f"{item['train_attempt_rate']} | {item['train_overall']} |\n"
        )

    lines.extend([
        "\n## Interpretable Feature Weights\n\n",
        "Positive weights favor graph retrieval/acceptance; negative weights favor retaining text.\n\n",
        "| Gate | Direction | Feature | Weight |\n",
        "|---|---|---|---:|\n",
    ])
    for gate, weights in [("pre", pre_weights), ("post", post_weights)]:
        for direction in ["graph_preferred", "text_preferred"]:
            for name, weight in weights[direction]:
                lines.append(f"| {gate} | {direction} | {name} | {weight} |\n")

    lines.extend([
        "\n## Test Quality and Routing\n\n",
        "| Dataset | Budget | Samples | Attempts | Accepts | Correctness | Coverage | Relevance | Conciseness | Overall | Text baseline | Graph baseline |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ])
    for dataset, dataset_results in sorted(budget_results.items()):
        for budget, result in dataset_results.items():
            scores = result["scores"]
            lines.append(
                f"| {dataset} | {budget} | {result['samples']} | {result['attempts']} | {result['accepts']} | "
                f"{scores['correctness']} | {scores['coverage']} | {scores['relevance']} | "
                f"{scores['conciseness']} | {scores['overall']} | "
                f"{result['text_scores']['overall']} | {result['graph_scores']['overall']} |\n"
            )

    lines.extend([
        "\n## Test Replay Cost\n\n",
        "| Dataset | Budget | Text query | Added graph query | Graph contexts | Selected indexing | Full indexing |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ])
    for dataset, dataset_results in sorted(budget_results.items()):
        for budget, result in dataset_results.items():
            lines.append(
                f"| {dataset} | {budget} | {result['text_query_seconds']} | "
                f"{result['added_graph_query_seconds']} | "
                f"{result['selected_graph_contexts']}/{result['all_graph_contexts']} | "
                f"{result['selected_index_seconds']} | {result['full_index_seconds']} |\n"
            )

    lines.extend([
        "\n## Split Manifest\n\n",
        "```json\n",
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        "\n```\n",
    ])
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-dir", default="results/v23")
    parser.add_argument("--model-output", default="results/v23/sage_v23_dual_gate.joblib")
    parser.add_argument("--report", default="results/historical_sage/mse_pathrag_sage_v23_learned_dual_gate_report.md")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--holdout-dataset",
        action="append",
        default=[],
        help="Dataset name(s) reserved entirely for external testing",
    )
    args = parser.parse_args()

    records = build_records(args.dataset)
    holdouts = set(args.holdout_dataset)
    if holdouts:
        train_records = [record for record in records if record["dataset"] not in holdouts]
        test_records = [record for record in records if record["dataset"] in holdouts]
    else:
        train_records, test_records = split_by_dataset_groups(records, args.test_size, args.random_state)
    if len({record["label"] for record in train_records}) < 2:
        raise RuntimeError("Training split must contain both graph-beneficial and text-preferred labels")

    pre_model = make_model(args.random_state)
    post_model = make_model(args.random_state)
    train_labels = np.array([record["label"] for record in train_records])
    test_labels = np.array([record["label"] for record in test_records])
    pre_model.fit([record["text_features"] for record in train_records], train_labels)
    post_model.fit([record["post_features"] for record in train_records], train_labels)

    train_pre_probs = pre_model.predict_proba([record["text_features"] for record in train_records])[:, 1]
    train_post_probs = post_model.predict_proba([record["post_features"] for record in train_records])[:, 1]
    test_pre_probs = pre_model.predict_proba([record["text_features"] for record in test_records])[:, 1]
    test_post_probs = post_model.predict_proba([record["post_features"] for record in test_records])[:, 1]

    pre_metrics = classification_metrics(test_labels, test_pre_probs)
    post_metrics = classification_metrics(test_labels, test_post_probs)
    thresholds = {
        budget: tune_thresholds(
            train_records,
            train_pre_probs,
            train_post_probs,
            config["max_attempt_rate"],
            config["cost_penalty"],
            budget,
        )
        for budget, config in BUDGETS.items()
    }

    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_results: dict[str, dict[str, dict]] = defaultdict(dict)
    for dataset in sorted({record["dataset"] for record in test_records}):
        indices = [index for index, record in enumerate(test_records) if record["dataset"] == dataset]
        dataset_records = [test_records[index] for index in indices]
        dataset_pre_probs = test_pre_probs[indices]
        dataset_post_probs = test_post_probs[indices]
        for budget in BUDGETS:
            outputs, summary = evaluate_budget(
                dataset_records,
                dataset_pre_probs,
                dataset_post_probs,
                thresholds[budget],
                budget,
            )
            for row in outputs:
                row["learned_dual_gate"]["budget"] = budget
            result_path = output_dir / f"sage_v23_{budget}_{dataset}_test.jsonl"
            write_jsonl(outputs, result_path)
            budget_results[dataset][budget] = summary

    split_manifest = {
        "random_state": args.random_state,
        "test_size": args.test_size,
        "holdout_datasets": sorted(holdouts),
        "train_groups": sorted({record["group"] for record in train_records}),
        "test_groups": sorted({record["group"] for record in test_records}),
        "train_samples_by_dataset": dict(Counter(record["dataset"] for record in train_records)),
        "test_samples_by_dataset": dict(Counter(record["dataset"] for record in test_records)),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_path = ROOT_DIR / args.model_output
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pre_model": pre_model,
            "post_model": post_model,
            "thresholds": thresholds,
            "split_manifest": split_manifest,
        },
        model_path,
    )
    write_report(
        ROOT_DIR / args.report,
        train_records,
        test_records,
        pre_metrics,
        post_metrics,
        thresholds,
        budget_results,
        split_manifest,
        feature_weights(pre_model),
        feature_weights(post_model),
    )
    print(f"Saved v23 model to {model_path}")
    print(f"Saved v23 report to {ROOT_DIR / args.report}")


if __name__ == "__main__":
    main()
