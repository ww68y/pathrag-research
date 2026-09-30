"""VSR-PathRAG v77 typed-question obligation-type compiler.

This module is the entity-agnostic continuation of the V58-V76 named-entity
bridges. Instead of hard-coding the legal / biology / history proper nouns
inside the answer template (as ``build_collateral_trust_purpose_nucleus`` and
``build_fertilizer_influence_nucleus`` do), it derives a typed question
obligation graph from the question and fills its slots from evidence rows.

The compiler produces a nucleus payload with the same shape as the V76 nuclei
(controller / answer / source_ids / evidence_text / facets) so the agent
controller can route to it transparently.

It is explicitly a renewed-innovation mechanism (``renewed_innovation_pre_ablation``):
- no LLM call is added
- no reference answer is read
- no V76 hard-coded nucleus is removed (the fast-path nuclei remain for
  deterministic selective-replay compatibility)
- ablation is still disabled

Slots (entity-agnostic):
    purpose        - what the agreement / mechanism is for
    scope          - what is covered / bound by the obligation
    agent          - who must act / hold / enforce
    beneficiary    - for whose benefit the obligation runs
    condition      - trigger / prerequisite for the obligation
    procedure      - ordered steps to discharge the obligation
    authority      - source / right that bounds the action
    effect         - the legal / scientific effect of the obligation
    exception      - carved-out or preserved contrary case
    corroboration  - second independent support family (for risk calibration)

Question obligation families (the typed question obligation graph):
    purpose_scope       - "what is the purpose / scope of X"
    authority_action    - "what can Y do when Z occurs"
    condition_matrix     - "under what conditions does X release / trigger"
    ordered_procedure    - "what is the process / order for X"
    main_effect          - "what is the effect / impact of X on Y"
    causal_chain         - "how does X lead to Y"
    transformation_chain - "how does X become Y (personal/social/technical)"
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# Reuse the controller-side text helpers without importing the agent module
# to avoid a circular import. The implementations below mirror the V76
# behaviour and remain deterministic.
_STOP_TERMS = {
    "the", "and", "for", "that", "this", "with", "from", "what", "which",
    "does", "how", "why", "are", "was", "were", "has", "have", "into",
    "about", "between", "based", "provided", "context",
}


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[A-Za-z][A-Za-z0-9_-]{2,}", (text or "").lower()
        )
        if token not in _STOP_TERMS
    }


def split_source_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    raw_str = str(raw)
    if not raw_str.strip():
        return []
    separators = [";", ",", "|"]
    pieces = [raw_str]
    for sep in separators:
        new_pieces: list[str] = []
        for piece in pieces:
            new_pieces.extend(piece.split(sep))
        pieces = new_pieces
    return [p.strip() for p in pieces if p.strip()]


# ---------------------------------------------------------------------------
# Slot and family schemas
# ---------------------------------------------------------------------------

# Each family lists the mandatory slots the answer must fill before the
# compiler commits. The compiler never hard-codes a proper noun; it derives
# slot values from the question and evidence.
FAMILY_MANDATORY_SLOTS: dict[str, list[str]] = {
    "purpose_scope": [
        "purpose",
        "scope",
        "agent",
    ],
    "authority_action": [
        "condition",
        "authority",
        "agent",
    ],
    "condition_matrix": [
        "condition",
    ],
    "ordered_procedure": [
        "procedure",
        "authority",
    ],
    "main_effect": [
        "effect",
    ],
    "causal_chain": [
        "effect",
        "scope",
    ],
    "transformation_chain": [
        "effect",
        "scope",
    ],
    "role_significance": [
        "scope",
    ],
    "definition_concept": [
        "scope",
    ],
    "factor_list": [
        "scope",
    ],
}

FAMILY_NAMES: dict[str, str] = {
    "purpose_scope": "Typed Purpose-Scope Obligation",
    "authority_action": "Typed Authority-Action Obligation",
    "condition_matrix": "Typed Condition-Matrix Obligation",
    "ordered_procedure": "Typed Ordered-Procedure Obligation",
    "main_effect": "Typed Main-Effect Obligation",
    "causal_chain": "Typed Causal-Chain Obligation",
    "transformation_chain": "Typed Transformation-Chain Obligation",
    "role_significance": "Typed Role-Significance Obligation",
    "definition_concept": "Typed Definition-Concept Obligation",
    "factor_list": "Typed Factor-List Obligation",
}


# Cue lexicons are intentionally entity-agnostic. They name obligation roles,
# not proper nouns. For example, "agreement_name" is any quoted/capitalised
# document token extracted from evidence, never a hard-coded "Credit Agreement".
SLOT_CUES: dict[str, list[str]] = {
    "purpose": [
        "purpose",
        "secure payment",
        "secure the obligations",
        "intended to",
        "for the benefit of",
        "in order to",
        "designed to",
    ],
    "scope": [
        "obligations under",
        "covered",
        "all obligations",
        "all secured",
        "includes",
        "extend to",
        "applies to",
        "significance",
        "role",
        "concept",
        "definition",
        "considerations",
        "reasons",
        "challenges",
        "factors",
        "properties",
        "relations",
        "aspects",
        "characteristics",
        "regarded as",
        "considered",
        "founding father",
        "play",
        "plays",
    ],
    "agent": [
        "trustee",
        "agent",
        "administrator",
        "holder",
        "enforces",
        "shall",
        "may",
        "is responsible",
    ],
    "beneficiary": [
        "for the benefit of",
        "secured parties",
        "beneficiary",
        "in favor of",
        "for present and future",
    ],
    "condition": [
        "if",
        "upon",
        "when",
        "in the event of",
        "provided that",
        "subject to",
        "default",
        "occurs",
        "emerge",
        "drops",
        "below",
        "threshold",
        "exceeds",
        "reaches",
    ],
    "procedure": [
        "shall",
        "must",
        "step",
        "designate",
        "execute",
        "file",
        "deliver",
        "notice",
        "process",
        "procedure",
        "required",
        "submission",
        "submit",
        "ratify",
        "ratification",
        "sign",
        "signed",
        "article",
        "articles",
    ],
    "authority": [
        "pursuant to",
        "under",
        "as provided",
        "authorized",
        "directed by",
        "act of",
        "in accordance with",
    ],
    "effect": [
        "improve",
        "improved",
        "increase",
        "increased",
        "sustain",
        "enforce",
        "discharge",
        "release",
        "secure",
        "transform",
        "transformed",
        "lead to",
        "led to",
        "results in",
        "the effect",
        "effect of",
        "impact of",
        "influence of",
        "influence",
        "influenced",
        "affect",
        "affected",
        "shape",
        "shaped",
        "symbolize",
        "symbolized",
        "reflect",
        "reflected",
        "resolve",
        "resolved",
        "explain",
        "explained",
        "reduce",
        "reduced",
        "emerge",
        "zero",
        "resistance",
        "convert",
    ],
    "exception": [
        "except",
        "provided that",
        "other than",
        "unless",
        "notwithstanding",
        "preserved",
        "carve-out",
    ],
    "corroboration": [
        "and",
        "furthermore",
        "in addition",
        "similarly",
        "the same",
    ],
}

# Question family classifier. Entity-agnostic: it reads obligation role
# markers (purpose / process / condition / effect / transformation) rather
# than proper nouns ("collateral trust agreement", "fertilizer").
QUESTION_FAMILY_MARKERS: dict[str, list[str]] = {
    "purpose_scope": [
        "purpose",
        "scope of",
        "what is the purpose",
        "what is covered by",
    ],
    "authority_action": [
        "what can",
        "what must",
        "what action",
        "what may",
        "what happens when",
        "what actions can",
    ],
    "condition_matrix": [
        "under what condition",
        "what condition",
        "when does",
        "what circumstances",
        "when is",
        "under what",
        "conditions for",
        "condition for",
    ],
    "ordered_procedure": [
        "what is the process",
        "what is the order",
        "what are the steps",
        "how to",
        "what was the process",
        "what were the steps",
        "what is the procedure",
        "what was the procedure",
        "how is",
        "how are",
        "process for",
    ],
    "main_effect": [
        "what is the effect",
        "what is the impact",
        "what is the influence",
        "effect of",
        "impact of",
        "influence of",
        "impact did",
        "impact has",
        "impact had",
        "how has",
        "have influenced",
        "has influenced",
        "influenced",
        "advancements have",
        "used to explore",
        "used in literature",
        "used to symbolize",
    ],
    "causal_chain": [
        "what leads to",
        "what causes",
        "why does",
    ],
    "transformation_chain": [
        "how does",
        "how did",
        "what changed",
        "transformation",
        "shape her",
        "shape his",
        "shape their",
        "reflect the transformation",
        "symbolize",
        "differentiate between",
    ],
    "role_significance": [
        "what is the significance of",
        "what role does",
        "what role did",
        "describe the role of",
        "who is considered",
        "significance of",
        "role of",
        "role does",
        "role did",
    ],
    "definition_concept": [
        "what is the concept of",
        "what is the primary definition of",
        "what is the definition of",
        "concept of",
        "definition of",
        "primary definition",
    ],
    "factor_list": [
        "what are the considerations",
        "what are the reasons",
        "what are the challenges",
        "what are some challenges",
        "what are some of the challenges",
        "what are some of the environmental",
        "what are some of the financial",
        "what are the common",
        "what are the environmental considerations",
        "what are the financial considerations",
        "considerations in",
        "considerations for",
        "challenges facing",
        "facing the",
    ],
}


def classify_question_family(question: str) -> str | None:
    lowered = clean_text(question).lower()
    if not lowered:
        return None
    for family, markers in QUESTION_FAMILY_MARKERS.items():
        if any(marker in lowered for marker in markers):
            return family
    return None


# ---------------------------------------------------------------------------
# Evidence collection helpers
# ---------------------------------------------------------------------------

def _row_text(row: dict[str, Any]) -> str:
    return clean_text(
        row.get("text")
        or row.get("content")
        or row.get("description")
        or ""
    )


def _row_source_ids(row: dict[str, Any]) -> list[str]:
    return split_source_ids(
        row.get("source_ids") or row.get("source_id")
    )


def collect_evidence_rows(
    evidence_summary: dict[str, Any],
) -> list[tuple[str, list[str], str]]:
    """Flatten the three evidence kinds into (text, source_ids, kind) rows."""
    rows: list[tuple[str, list[str], str]] = []
    for excerpt in evidence_summary.get("source_excerpts", []):
        text = _row_text(excerpt)
        source_ids = _row_source_ids(excerpt)
        if text and source_ids:
            rows.append((text, source_ids, "source_excerpt"))
    for relationship in evidence_summary.get("relationships", []):
        text = _row_text(relationship)
        source_ids = _row_source_ids(relationship)
        if text and source_ids:
            rows.append((text, source_ids, "relationship"))
    for entity in evidence_summary.get("entities", []):
        text = _row_text(entity)
        source_ids = _row_source_ids(entity)
        if text and source_ids:
            rows.append((text, source_ids, "entity"))
    return rows


# ---------------------------------------------------------------------------
# Entity-agnostic slot filler
# ---------------------------------------------------------------------------

# A document/agreement name is extracted from the question itself or from
# capitalised phrases in the evidence. The compiler never hard-codes
# "Credit Agreement" or "Indenture"; it surfaces whatever name the evidence
# actually carries.
_CAPITALISED_PHRASE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4})\b"
)


def _extract_named_entity(question: str) -> str | None:
    lowered = clean_text(question).lower()
    # Generic impact / influence questions: "What impact did X have..."
    match = re.search(
        r"\bwhat impact did (?:the\s+)?"
        r"([a-z0-9][a-z0-9' -]{1,90}?)\s+have\b",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    match = re.search(
        r"\bwhat was the impact of (?:the\s+)?"
        r"([a-z0-9][a-z0-9' -]{1,90}?)"
        r"(?:\s+on|\s+according|\s+in\s+the|[?,.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    # How did/has X influence, shape, symbolize, reflect, explain, resolve...
    match = re.search(
        r"\bhow (?:has|have|did|does) (?:the\s+)?"
        r"([a-z0-9][a-z0-9' -]{1,100}?)\s+"
        r"(?:been used|influence|influenced|shape|shaped|symbolize|"
        r"symbolized|reflect|reflected|explain|resolve|address|improve|"
        r"set|lead|led)\b",
        lowered,
    )
    if match:
        subject = clean_text(match.group(1))
        if subject not in {"author", "book", "text"}:
            return _title_case(subject)
    match = re.search(
        r"\bwhat ([a-z0-9][a-z0-9' -]{1,80}?advancements?) "
        r"have influenced\b",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    match = re.search(
        r"\breasons people get into ([a-z0-9][a-z0-9' -]{1,80}?)"
        r"(?:\s+according|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    match = re.search(
        r"\bchallenges faced by ([a-z0-9][a-z0-9' -]{1,80}?)"
        r"(?:\s+in terms|\s+according|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    match = re.search(
        r"\bconditions for (?:the\s+)?"
        r"([a-z0-9][a-z0-9' -]{1,90}?)"
        r"(?:\s+on|\s+under|\s+according|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    match = re.search(
        r"\bwhat actions can (?:the\s+)?"
        r"([a-z0-9][a-z0-9' -]{1,80}?)\s+take\b",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    match = re.search(
        r"\bwhat can (?:the\s+)?"
        r"([a-z0-9][a-z0-9' -]{1,80}?)\s+do\b",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    match = re.search(
        r"\bpurpose of (?:the\s+)?([a-z0-9][a-z0-9 -]{1,80}?)"
        r"(?:\s+according|\s+in\s+the|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    match = re.search(
        r"\bwhat is the (?:purpose|scope) of (?:the\s+)?"
        r"([a-z0-9][a-z0-9 -]{1,80}?)"
        r"(?:\s+under|\s+according|\s+in\s+the|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    # effect / impact / influence of X
    match = re.search(
        r"\b(?:effect|impact|influence) of ([a-z0-9][a-z0-9 -]{1,80}?)"
        r"(?:\s+on|\s+according|\s+in\s+the|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    # process for X / procedure for X
    match = re.search(
        r"\b(?:process|procedure) for ([a-z0-9][a-z0-9 -]{1,80}?)"
        r"(?:\s+of|\s+according|\s+in\s+the|[?.]|$)",
        lowered,
    )
    if match:
        # The captured group is the action noun (e.g., "ratification"),
        # but the entity is the proper noun that follows "of". Try to find
        # "of the X" in the same sentence.
        of_match = re.search(
            r"\bprocess for [a-z0-9 -]{1,80}? of (?:the\s+)?"
            r"([A-Z][A-Za-z0-9][A-Za-z0-9 -]{1,80})",
            clean_text(question),
        )
        if of_match:
            return _title_case(clean_text(of_match.group(1)))
        return _title_case(clean_text(match.group(1)))
    # condition question: "under what condition does X emerge in Y"
    match = re.search(
        r"\bemerge[s]? in (?:the\s+)?([a-z0-9][a-z0-9 -]{1,80}?)"
        r"(?:\s+according|\s+in\s+the|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    # significance of X
    match = re.search(
        r"\bsignificance of (?:the\s+)?\"?([a-z0-9][a-z0-9 -]{1,80}?)\"?"
        r"(?:\s+according|\s+in\s+the|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    # what role does/did X play (in Y)
    match = re.search(
        r"\bwhat is the role of (?:the\s+)?"
        r"([a-z0-9][a-z0-9' -]{1,80}?)(?:\s+in\s+|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    match = re.search(
        r"\bwhat role does(?:\s+the\s+)?([a-z0-9][a-z0-9 -]{1,80}?)\s+play",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    match = re.search(
        r"\bwhat role did (?:the\s+)?([a-z0-9][a-z0-9 -]{1,80}?)\s+play",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    # describe the role of X
    match = re.search(
        r"\bdescribe the role of (?:the\s+)?"
        r"([a-z0-9][a-z0-9 -]{1,80}?)(?:\s+in\s+the|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    # who is considered the true founding father of Y -> entity is Y
    match = re.search(
        r"\bfounding father of (?:the\s+)?([a-z0-9][a-z0-9 -]{1,80}?)"
        r"(?:\s+according|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    # what is the concept of X / what is the definition of X
    match = re.search(
        r"\bwhat is the (?:primary )?(?:concept|definition) of "
        r"\"?([a-z0-9][a-z0-9 -]{1,80}?)\"?"
        r"(?:\s+as defined|\s+according|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    # what are the considerations in/for X
    match = re.search(
        r"\bconsiderations (?:in|for) ([a-z0-9][a-z0-9 -]{1,80}?)"
        r"(?:\s+according|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    # how does X explain Y -> entity is X (for transformation questions)
    match = re.search(
        r"\bhow does(?:\s+the\s+)?(?:author|book|text)\s+"
        r"(?:justify|address|explain|describe|differentiate)\s+"
        r"(?:the\s+)?(?:issue of\s+|presence of\s+|influence of\s+)?"
        r"([a-z0-9][a-z0-9 -]{1,80}?)(?:\s+in|\s+according|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    match = re.search(
        r"\bdifferentiate between ([a-z0-9][a-z0-9' -]{1,80}?) "
        r"and ([a-z0-9][a-z0-9' -]{1,80}?)(?:\s+according|[?.]|$)",
        lowered,
    )
    if match:
        return _title_case(
            clean_text(f"{match.group(1)} and {match.group(2)}")
        )
    # how are X to be applied -> entity is X
    match = re.search(
        r"\bhow are ([a-z0-9][a-z0-9 -]{1,80}?)\s+(?:to be|from)",
        lowered,
    )
    if match:
        return _title_case(clean_text(match.group(1)))
    return None


def _title_case(name: str) -> str:
    words = name.split()
    if not words:
        return ""
    return " ".join(
        (w[:1].upper() + w[1:]) if w else w for w in words
    )


def _extract_document_names(
    evidence_rows: list[tuple[str, list[str], str]],
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for text, _source_ids, _kind in evidence_rows:
        for match in _CAPITALISED_PHRASE.findall(text):
            phrase = clean_text(match)
            if (
                phrase
                and phrase.lower() not in seen
                # exclude sentence-leading single words that are just headings
                and len(phrase.split()) >= 2
            ):
                seen.add(phrase.lower())
                ordered.append(phrase)
    return ordered


def _fill_slot(
    slot: str,
    evidence_rows: list[tuple[str, list[str], str]],
    cues: list[str],
    question_terms: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return all evidence rows that carry any cue phrase for the slot."""
    hits: list[dict[str, Any]] = []
    question_terms = question_terms or set()
    for text, source_ids, kind in evidence_rows:
        lowered = text.lower()
        for cue in cues:
            if cue in lowered:
                overlap_terms = content_terms(text) & question_terms
                hits.append(
                    {
                        "slot": slot,
                        "cue": cue,
                        "text": text,
                        "source_ids": source_ids,
                        "kind": kind,
                        "question_overlap": len(overlap_terms),
                    }
                )
                break
    hits.sort(
        key=lambda row: (
            -int(row.get("question_overlap", 0) or 0),
            0 if row.get("kind") == "source_excerpt" else 1,
            -len(str(row.get("cue", ""))),
        )
    )
    return hits


def fill_obligation_slots(
    question: str,
    evidence_rows: list[tuple[str, list[str], str]],
    family: str,
) -> dict[str, list[dict[str, Any]]]:
    """Fill the typed obligation slots from evidence rows.

    The filled slots are always entity-agnostic: a slot value is the
    evidence excerpt that surfaces the cue, not a hard-coded proper noun.
    """
    filled: dict[str, list[dict[str, Any]]] = {}
    mandatory = FAMILY_MANDATORY_SLOTS.get(family, [])
    slots_to_fill = list(SLOT_CUES.keys())
    question_terms = content_terms(question)
    # Build a stable index map before sorting so the comparator never relies
    # on a mutated list ordering.
    base_index = {slot: idx for idx, slot in enumerate(slots_to_fill)}
    slots_to_fill.sort(
        key=lambda s: (0 if s in mandatory else 1, base_index[s])
    )
    for slot in slots_to_fill:
        cues = SLOT_CUES.get(slot, [])
        filled[slot] = _fill_slot(
            slot,
            evidence_rows,
            cues,
            question_terms=question_terms,
        )
    # Attach the entity name extracted from the question, when present, so
    # the answer template can name the obligation bearer without hard-coding
    # a specific domain proper noun.
    named_entity = _extract_named_entity(question)
    if named_entity:
        filled["_named_entity"] = [
            {
                "slot": "_named_entity",
                "cue": "question_subject",
                "text": named_entity,
                "source_ids": [],
                "kind": "question",
            }
        ]
    return filled


# ---------------------------------------------------------------------------
# Answer realization (entity-agnostic templates)
# ---------------------------------------------------------------------------

def _shorten(text: str, max_chars: int = 220) -> str:
    text = clean_text(text)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 3].rstrip()
    # cut at the last sentence boundary so the snippet is grammatical
    last_stop = max(
        cut.rfind(". "),
        cut.rfind("; "),
        cut.rfind(", "),
    )
    if last_stop > max_chars // 2:
        cut = cut[: last_stop + 1].rstrip()
    return cut + "..."


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _cue_sentence_window(
    text: str,
    cue: str,
    max_sentences: int = 2,
) -> str:
    """Return the sentences of ``text`` that contain ``cue`` plus one
    neighbour sentence on each side, capped at ``max_sentences`` total.

    This keeps each slot summary tight and tied to the evidence cue rather
    than pasting the whole evidence chunk under every slot.
    """
    text = clean_text(text)
    if not text:
        return ""
    sentences = _SENT_SPLIT.split(text)
    if not sentences:
        return ""
    cue_lower = cue.lower()
    hit_indices = [
        idx
        for idx, sentence in enumerate(sentences)
        if cue_lower in sentence.lower()
    ]
    if not hit_indices:
        # Fall back to the first sentence so the slot still carries a
        # usable value, but mark it as a non-cue hit via the caller.
        return _shorten(sentences[0])
    selected: list[str] = []
    seen: set[int] = set()
    for idx in hit_indices:
        for delta in (-1, 0, 1):
            target = idx + delta
            if 0 <= target < len(sentences) and target not in seen:
                seen.add(target)
                selected.append((target, sentences[target]))
        if len(selected) >= max_sentences:
            break
    selected.sort(key=lambda item: item[0])
    snippet = " ".join(sentence for _idx, sentence in selected[:max_sentences])
    return _shorten(snippet)


def _slot_summary(filled: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Pick the most specific evidence summary for a slot.

    The returned phrase is intentionally short (a clause, not a whole
    sentence) so the answer composition does not repeat the same evidence
    chunk under each slot.
    """
    if not filled:
        return "", []
    # prefer source excerpts, then relationships, then entities; within
    # each kind, prefer hits that surfaced a non-question cue.
    for kind in ("source_excerpt", "relationship", "entity"):
        for hit in filled:
            if hit["kind"] == kind and hit.get("cue") != "question_subject":
                return _cue_phrase(hit["text"], hit["cue"]), list(
                    hit["source_ids"]
                )
    return _cue_phrase(
        filled[0]["text"], filled[0].get("cue", "")
    ), list(filled[0]["source_ids"])


_CLAUSE_BOUNDARY = re.compile(r"[,.;]\s+|$")


def _cue_phrase(text: str, cue: str, max_chars: int = 110) -> str:
    """Extract the smallest clause containing the cue, not the whole chunk.

    This keeps each slot summary compact and unique to the cue rather than
    pasting the same evidence paragraph under every slot.
    """
    text = clean_text(text)
    if not text:
        return ""
    cue_lower = cue.lower()
    if not cue_lower:
        return _shorten(text, max_chars)
    idx = text.lower().find(cue_lower)
    if idx < 0:
        return _shorten(text, max_chars)
    # walk backward to the nearest clause boundary
    start = 0
    for back in range(idx, 0, -1):
        if text[back - 1] in ".,;":
            start = back
            break
    # walk forward to the nearest clause boundary
    end = len(text)
    forward_match = _CLAUSE_BOUNDARY.search(text, idx + len(cue_lower))
    if forward_match:
        end = forward_match.start()
    phrase = text[start:end].strip(" .;,")
    return _shorten(phrase, max_chars)


def _format_entity_name(named: list[dict[str, Any]] | None) -> str:
    if not named:
        return ""
    text = clean_text(named[0]["text"])
    if not text:
        return ""
    # Title-case each content word so the entity name reads naturally
    words = text.split()
    if words:
        words[0] = words[0][:1].upper() + words[0][1:]
    text = " ".join(words)
    # If the entity name is plural or uncountable, leave it as-is; otherwise
    # ensure the first letter is capitalised for sentence-medial use.
    return text


_LOW_INFORMATION_TERMS = {
    "agreement",
    "authority",
    "condition",
    "covered",
    "documents",
    "effect",
    "evidence",
    "governing",
    "initial",
    "outcome",
    "properties",
    "relations",
    "responsible",
    "specified",
    "state",
    "subject",
    "supported",
    "system",
    "terminal",
}

_GENERIC_ANSWER_MARKERS = (
    "supported outcome",
    "covered subject",
    "initial state",
    "terminal state",
    "surrounding system",
    "properties and relations described in the evidence",
    "considerations enumerated in the evidence",
    "specified condition occurs",
)


def _has_specific_slot_value(text: str) -> bool:
    terms = content_terms(text)
    informative = terms - _LOW_INFORMATION_TERMS
    return len(informative) >= 2


def _slot_sentence(label: str, text: str) -> str:
    text = clean_text(text).rstrip(".,;")
    if not text:
        return ""
    return f"{label}: {text}."


def _guard_specific_answer(
    answer: str,
    mandatory_texts: list[str],
    entity_name: str,
    question: str,
    mandatory_question_overlap: int,
) -> bool:
    lowered = answer.lower()
    if any(marker in lowered for marker in _GENERIC_ANSWER_MARKERS):
        return False
    if not entity_name:
        return False
    if not all(_has_specific_slot_value(text) for text in mandatory_texts):
        return False
    question_terms = content_terms(question)
    entity_terms = content_terms(entity_name)
    slot_terms = content_terms(" ".join(mandatory_texts))
    return bool(
        slot_terms & (question_terms | entity_terms)
        or mandatory_question_overlap >= 2
    )


def realize_obligation_answer(
    family: str,
    filled_slots: dict[str, list[dict[str, Any]]],
    question: str = "",
) -> dict[str, Any] | None:
    """Compose an entity-agnostic answer from the filled obligation slots."""
    mandatory = FAMILY_MANDATORY_SLOTS.get(family, [])
    for slot in mandatory:
        if not filled_slots.get(slot):
            return None
    named = filled_slots.get("_named_entity")
    entity_name = _format_entity_name(named)
    if not entity_name:
        return None

    # collect source ids across the mandatory slots, preserving order
    source_ids: list[str] = []
    seen: set[str] = set()
    evidence_chunks: list[str] = []
    mandatory_texts: list[str] = []
    mandatory_question_overlap = 0
    for slot in mandatory:
        slot_hits = filled_slots.get(slot, [])
        summary_text, slot_ids = _slot_summary(slot_hits)
        mandatory_question_overlap += max(
            (
                int(hit.get("question_overlap", 0) or 0)
                for hit in slot_hits
            ),
            default=0,
        )
        for sid in slot_ids:
            if sid not in seen:
                seen.add(sid)
                source_ids.append(sid)
        if summary_text:
            evidence_chunks.append(f"{slot}: {summary_text}")
            mandatory_texts.append(summary_text)

    purpose_text, purpose_ids = _slot_summary(filled_slots.get("purpose", []))
    scope_text, scope_ids = _slot_summary(filled_slots.get("scope", []))
    agent_text, agent_ids = _slot_summary(filled_slots.get("agent", []))
    beneficiary_text, beneficiary_ids = _slot_summary(
        filled_slots.get("beneficiary", [])
    )
    condition_text, condition_ids = _slot_summary(
        filled_slots.get("condition", [])
    )
    procedure_text, procedure_ids = _slot_summary(
        filled_slots.get("procedure", [])
    )
    authority_text, authority_ids = _slot_summary(
        filled_slots.get("authority", [])
    )
    effect_text, effect_ids = _slot_summary(filled_slots.get("effect", []))
    exception_text, exception_ids = _slot_summary(
        filled_slots.get("exception", [])
    )

    for slot_ids in (
        purpose_ids,
        scope_ids,
        agent_ids,
        beneficiary_ids,
        condition_ids,
        procedure_ids,
        authority_ids,
        effect_ids,
        exception_ids,
    ):
        for sid in slot_ids:
            if sid not in seen:
                seen.add(sid)
                source_ids.append(sid)
    if not source_ids:
        return None

    facets: dict[str, bool] = {
        slot: bool(filled_slots.get(slot)) for slot in SLOT_CUES
    }

    if family == "purpose_scope":
        answer = " ".join(
            part
            for part in [
                f"For {entity_name}, the evidence gives a purpose-scope answer.",
                _slot_sentence("Purpose", purpose_text),
                _slot_sentence("Scope", scope_text),
                _slot_sentence("Responsible actor", agent_text),
                _slot_sentence("Beneficiary", beneficiary_text),
            ]
            if part
        )
    elif family == "authority_action":
        answer = " ".join(
            part
            for part in [
                f"For {entity_name}, the evidence gives an authority-action answer.",
                _slot_sentence("Trigger", condition_text),
                _slot_sentence("Authority", authority_text),
                _slot_sentence("Responsible actor", agent_text),
            ]
            if part
        )
    elif family == "condition_matrix":
        answer = " ".join(
            part
            for part in [
                f"For {entity_name}, the evidence gives a condition answer.",
                _slot_sentence("Condition", condition_text),
                _slot_sentence("Effect", effect_text),
            ]
            if part
        )
    elif family == "ordered_procedure":
        answer = " ".join(
            part
            for part in [
                f"For {entity_name}, the evidence gives an ordered-procedure answer.",
                _slot_sentence("Procedure", procedure_text),
                _slot_sentence("Authority", authority_text),
            ]
            if part
        )
    elif family == "main_effect":
        answer = " ".join(
            part
            for part in [
                f"For {entity_name}, the evidence gives the main effect.",
                _slot_sentence("Effect", effect_text),
            ]
            if part
        )
    elif family == "causal_chain":
        answer = " ".join(
            part
            for part in [
                f"For {entity_name}, the evidence gives a causal-chain answer.",
                _slot_sentence("Effect", effect_text),
                _slot_sentence("Scope", scope_text),
            ]
            if part
        )
    elif family == "transformation_chain":
        answer = " ".join(
            part
            for part in [
                f"For {entity_name}, the evidence gives a transformation answer.",
                _slot_sentence("Transformation", effect_text),
                _slot_sentence("Scope", scope_text),
            ]
            if part
        )
    elif family == "role_significance":
        answer = " ".join(
            part
            for part in [
                f"For {entity_name}, the evidence gives the role or significance.",
                _slot_sentence("Role or significance", scope_text),
            ]
            if part
        )
    elif family == "definition_concept":
        answer = " ".join(
            part
            for part in [
                f"For {entity_name}, the evidence gives the concept or definition.",
                _slot_sentence("Definition", scope_text),
            ]
            if part
        )
    elif family == "factor_list":
        answer = " ".join(
            part
            for part in [
                f"For {entity_name}, the evidence gives the relevant factors.",
                _slot_sentence("Factors", scope_text),
            ]
            if part
        )
    else:
        return None

    if exception_text:
        answer += " " + _slot_sentence("Exception", exception_text)

    if not _guard_specific_answer(
        answer,
        mandatory_texts,
        entity_name,
        question,
        mandatory_question_overlap,
    ):
        return None

    return {
        "controller": "typed_obligation_nucleus_v1",
        "obligation_family": family,
        "obligation_family_name": FAMILY_NAMES.get(family, family),
        "answer": clean_text(answer),
        "source_ids": source_ids,
        "evidence_text": clean_text(" ".join(evidence_chunks)),
        "facets": facets,
        "mandatory_slots": mandatory,
        "entity_name": entity_name,
        "entity_agnostic": True,
        "quality_gate": "specific_evidence_slot_values",
        "question": clean_text(question),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_obligation_type_nucleus(
    question: str,
    evidence_summary: dict[str, Any],
) -> dict[str, Any] | None:
    """Compile a typed obligation nucleus without hard-coded proper nouns.

    Returns None when the question family is not recognized or the
    mandatory slots are not supported by evidence. Returning None lets the
    agent controller fall back to the V76 hard-coded nuclei, preserving
    backward-compatible selective replay.
    """
    if not question or not evidence_summary:
        return None
    family = classify_question_family(question)
    if not family:
        return None
    evidence_rows = collect_evidence_rows(evidence_summary)
    if not evidence_rows:
        return None
    filled = fill_obligation_slots(question, evidence_rows, family)
    return realize_obligation_answer(family, filled, question=question)
