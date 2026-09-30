import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.vsr_pathrag import optimize_evidence_portfolio


OUT_JSON = ROOT_DIR / "results" / "vsr_pathrag_v2_portfolio_guard_case.json"
OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v2_portfolio_guard_case_summary.md"


def main() -> None:
    question = "How did Aurora improve safety and charging?"
    query_graph = {
        "question": question,
        "relation_intents": ["mechanism"],
        "required_facets": ["safety", "charging"],
    }
    relationships = [
        {
            "src": "AURORA",
            "tgt": "SAFETY MONITORING",
            "keywords": "improve safety through monitoring",
            "description": "Aurora improved safety through real-time temperature monitoring.",
            "source_ids": ["chunk-safety"],
        },
        {
            "src": "AURORA",
            "tgt": "CHARGING CONTROL",
            "keywords": "improve charging through control",
            "description": "Aurora improved charging through adaptive charging control.",
            "source_ids": ["chunk-charging"],
        },
        {
            "src": "AURORA",
            "tgt": "SAFETY SENSOR",
            "keywords": "improve safety through monitoring",
            "description": "Aurora improved safety through temperature monitoring sensors.",
            "source_ids": ["chunk-safety"],
        },
    ]
    diversity_case = optimize_evidence_portfolio(
        question,
        query_graph,
        relationships,
        max_paths=2,
        token_budget=120,
    )

    risky_repair_case = optimize_evidence_portfolio(
        "How did Aurora improve safety?",
        {
            "question": "How did Aurora improve safety?",
            "relation_intents": ["mechanism"],
            "required_facets": ["safety"],
        },
        [
            {
                "src": "AURORA",
                "tgt": "PLASMA SHIELD",
                "keywords": "improve safety through",
                "description": "Aurora improved safety through a plasma shield.",
                "source_ids": ["chunk-risk"],
                "repair_id": "repair-risk",
                "repair_confidence": 0.62,
                "temporary": True,
            }
        ],
        max_paths=1,
        token_budget=80,
    )

    selected_sources = sorted(
        {
            source
            for path in diversity_case["selected_paths"]
            for source in path["source_ids"]
        }
    )
    checks = {
        "complementary_sources_selected": selected_sources == ["chunk-charging", "chunk-safety"],
        "path_budget_respected": diversity_case["selected_count"] <= 2,
        "token_budget_respected": diversity_case["estimated_tokens"] <= 120,
        "repair_only_goal_requires_verification": risky_repair_case["verification_required"],
        "repair_only_goal_is_high_risk": risky_repair_case["risk_level"] == "high",
    }
    payload = {
        "case_type": "risk_aware_path_portfolio",
        "diversity_case": diversity_case,
        "risky_repair_case": risky_repair_case,
        "checks": checks,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# VSR-PathRAG v2 Risk-Aware Portfolio Guard Case",
        "",
        "## Complementary Evidence Case",
        "",
        f"- Candidate paths: `{diversity_case['candidate_count']}`",
        f"- Selected paths: `{diversity_case['selected_count']}`",
        f"- Selected sources: `{selected_sources}`",
        f"- Goal coverage: `{diversity_case['goal_coverage_ratio']}`",
        f"- Estimated tokens: `{diversity_case['estimated_tokens']}/120`",
        f"- Mean redundancy: `{diversity_case['mean_pairwise_redundancy']}`",
        "",
        "## Repair Dependency Case",
        "",
        f"- Repair dependency ratio: `{risky_repair_case['repair_dependency_ratio']}`",
        f"- Verification required: `{risky_repair_case['verification_required']}`",
        f"- Risk level: `{risky_repair_case['risk_level']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in checks.items())
    lines.append("")
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_SUMMARY}")
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
