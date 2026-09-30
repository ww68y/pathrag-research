import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.vsr_pathrag import (
    build_signed_claim_evidence_graph,
    rollback_overlay_evidence,
)


OUT_JSON = ROOT_DIR / "results" / "vsr_pathrag_v1_counterfactual_guard_case.json"
OUT_SUMMARY = ROOT_DIR / "results" / "vsr_pathrag_v1_counterfactual_guard_case_summary.md"


def main() -> None:
    claim = "Aurora improved battery safety through an experimental plasma shield."
    repair_id = "repair-controlled-weak-edge"
    evidence = [
        {
            "kind": "relationship",
            "label": "AURORA -> EXPERIMENTAL PLASMA SHIELD",
            "text": claim,
            "repair_id": repair_id,
            "repair_confidence": 0.61,
            "source_ids": ["controlled-adversarial-source"],
        }
    ]
    overlay = {
        "relationships": [
            {
                "src": "AURORA",
                "tgt": "EXPERIMENTAL PLASMA SHIELD",
                "description": claim,
                "repair_id": repair_id,
                "repair_confidence": 0.61,
                "temporary": True,
            }
        ],
        "paths": [
            {
                "path_id": "controlled-risk-path",
                "nodes": ["AURORA", "EXPERIMENTAL PLASMA SHIELD"],
                "edge_ids": [repair_id],
            }
        ],
    }

    claim_graph = build_signed_claim_evidence_graph([claim], evidence)
    rollback_ids = claim_graph["rollback_repair_ids"]
    rolled_overlay = rollback_overlay_evidence(overlay, rollback_ids)
    payload = {
        "case_type": "controlled_weak_sole_repair",
        "claim": claim,
        "claim_evidence_graph": claim_graph,
        "overlay_before_rollback": overlay,
        "overlay_after_rollback": rolled_overlay,
        "checks": {
            "weak_repair_identified": repair_id in rollback_ids,
            "temporary_relation_removed": not rolled_overlay["relationships"],
            "dependent_path_removed": not rolled_overlay["paths"],
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    contribution = claim_graph["claims"][0]["counterfactual_contributions"][0]
    checks = payload["checks"]
    lines = [
        "# VSR-PathRAG v1 Counterfactual Guard Case",
        "",
        "This controlled case injects a low-confidence temporary relation that is the sole support for a claim.",
        "It verifies the second-line guard after provenance validation, without mutating the base graph.",
        "",
        "## Result",
        "",
        f"- Claim status before rollback: `{claim_graph['claims'][0]['status']}`",
        f"- Repair confidence: `0.61`",
        f"- Leave-one-out contribution: `{contribution['delta']}`",
        f"- Rollback repair IDs: `{rollback_ids}`",
        f"- Weak repair identified: `{checks['weak_repair_identified']}`",
        f"- Temporary relation removed: `{checks['temporary_relation_removed']}`",
        f"- Dependent path removed: `{checks['dependent_path_removed']}`",
        "",
        "## Interpretation",
        "",
        "The claim initially appears supported only because of the injected relation. Removing that relation causes a large support drop, so the counterfactual guard marks it as unsafe and removes both the relation and every dependent path from the query-local overlay.",
        "",
    ]
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_SUMMARY}")
    print(json.dumps(payload["checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
