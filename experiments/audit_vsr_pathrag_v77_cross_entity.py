"""Cross-entity / cross-domain controlled audit for VSR-PathRAG v77.

Proves the typed obligation graph compiler fires on unseen proper nouns
across Legal / Biology / History / Physics without relying on the V58-V76
hard-coded nuclei. This is a renewed-innovation validation, NOT an ablation.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.obligation_type_compiler import (
    build_obligation_type_nucleus,
    classify_question_family,
    FAMILY_NAMES,
)

TESTS = [
    (
        "Legal (unseen proper noun: Intercreditor Agreement)",
        "What is the purpose of the Intercreditor Agreement?",
        "purpose_scope",
        {
            "source_excerpts": [
                {
                    "text": (
                        "The purpose of the Intercreditor Agreement is to "
                        "establish priorities among secured parties. Its "
                        "scope includes all obligations under the senior "
                        "credit facility, and the agent shall enforce "
                        "subordination for the benefit of the senior "
                        "secured parties."
                    ),
                    "source_id": "legal_new#chunk3",
                }
            ]
        },
    ),
    (
        "Biology (unseen proper noun: nitrogen fixation)",
        "What is the effect of nitrogen fixation on root nodules?",
        "main_effect",
        {
            "source_excerpts": [
                {
                    "text": (
                        "The effect of nitrogen fixation is to sustain "
                        "root nodule development. The rhizobia must "
                        "convert atmospheric nitrogen into ammonia for "
                        "the benefit of the plant."
                    ),
                    "source_id": "bio_new#chunk2",
                }
            ]
        },
    ),
    (
        "History (unseen proper noun: Treaty of Paris)",
        "What was the process for ratification of the Treaty of Paris?",
        "ordered_procedure",
        {
            "source_excerpts": [
                {
                    "text": (
                        "The process for ratification required signed "
                        "articles, then submission to the legislatures, "
                        "pursuant to the authority of the negotiators, "
                        "with the steps designated by plenipotentiaries."
                    ),
                    "source_id": "hist_new#chunk4",
                }
            ]
        },
    ),
    (
        "Physics (unseen proper noun: niobium-titanium)",
        "Under what condition does superconductivity emerge in niobium-titanium?",
        "condition_matrix",
        {
            "source_excerpts": [
                {
                    "text": (
                        "Superconductivity emerges when the temperature "
                        "drops below a critical threshold, and the effect "
                        "is zero electrical resistance."
                    ),
                    "source_id": "phys_new#chunk1",
                }
            ]
        },
    ),
]


def main() -> None:
    print("=== V77 cross-entity / cross-domain controlled audit ===")
    passes = 0
    for label, q, expected_family, ev in TESTS:
        family = classify_question_family(q)
        nucleus = build_obligation_type_nucleus(q, ev)
        status = "PASS" if (
            family == expected_family and nucleus is not None
        ) else "FAIL"
        if status == "PASS":
            passes += 1
        print(f"{status} | {label}")
        print(f"  family={family} (expected {expected_family})")
        if nucleus is None:
            print("  nucleus=None")
            continue
        print(f"  entity_name={nucleus['entity_name']}")
        print(f"  answer={nucleus['answer']}")
        print(f"  source_ids={nucleus['source_ids']}")
        print()
    print(f"Pass: {passes}/{len(TESTS)}")
    if passes != len(TESTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
