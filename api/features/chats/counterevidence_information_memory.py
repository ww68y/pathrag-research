from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INFORMATION_MEMORY_PATH = (
    ROOT_DIR
    / "data_repro"
    / "vsr_counterevidence_information_memory.jsonl"
)
_INFORMATION_MEMORY_LOCK = threading.RLock()


def _memory_path(path: Path | str | None) -> Path:
    return (
        Path(path)
        if path
        else Path(
            os.getenv(
                "VSR_COUNTEREVIDENCE_INFORMATION_MEMORY_PATH",
                str(DEFAULT_INFORMATION_MEMORY_PATH),
            )
        )
    )


def _memory_id(context: dict[str, Any]) -> str:
    payload = json.dumps(
        context,
        ensure_ascii=True,
        sort_keys=True,
    )
    return (
        "counterevidence-information-"
        + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    )


def load_counterevidence_information_memory(
    path: Path | str | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    memory_path = _memory_path(path)
    if not memory_path.exists():
        return []
    rows = []
    try:
        with memory_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    isinstance(row, dict)
                    and row.get("context")
                    and (
                        int(row.get("sample_count", 0) or 0) > 0
                        or int(
                            row.get(
                                "provisional_sample_count",
                                0,
                            )
                            or 0
                        )
                        > 0
                    )
                ):
                    rows.append(row)
    except (OSError, json.JSONDecodeError):
        return []
    return rows[-limit:]


def append_counterevidence_information_updates(
    updates: list[dict[str, Any]],
    path: Path | str | None = None,
    max_entries: int = 1000,
    current_time: float | None = None,
    provisional_ttl_days: float = 30.0,
    max_provisional_samples: int = 8,
) -> dict[str, Any]:
    enabled = os.getenv(
        "VSR_COUNTEREVIDENCE_INFORMATION_MEMORY_WRITE_ENABLED",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return {
            "written": False,
            "reason": "information_memory_write_disabled",
        }
    valid_updates = [
        row
        for row in updates
        if row.get("context")
        and int(row.get("sample_count", 0) or 0) > 0
    ]
    if not valid_updates:
        return {
            "written": False,
            "reason": "no_information_updates",
        }
    memory_path = _memory_path(path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    now = float(
        current_time if current_time is not None else time.time()
    )
    provisional_ttl_seconds = (
        max(0.0, float(provisional_ttl_days)) * 86400.0
    )
    max_provisional_samples = max(
        1,
        int(max_provisional_samples),
    )
    with _INFORMATION_MEMORY_LOCK:
        existing = load_counterevidence_information_memory(
            memory_path,
            limit=max_entries,
        )
        by_id = {
            row["memory_id"]: row
            for row in existing
            if row.get("memory_id")
        }
        provisional_update_count = 0
        confirmed_negative_update_count = 0
        immediate_positive_update_count = 0
        expired_provisional_count = 0
        capped_provisional_update_count = 0
        for update in valid_updates:
            context = update["context"]
            memory_id = _memory_id(context)
            previous = by_id.get(memory_id) or {}
            provisional_updated_at = float(
                previous.get("provisional_updated_at")
                or previous.get("updated_at")
                or now
            )
            provisional_is_expired = bool(
                int(
                    previous.get(
                        "provisional_sample_count",
                        0,
                    )
                    or 0
                )
                and now - provisional_updated_at
                > provisional_ttl_seconds
            )
            if provisional_is_expired:
                previous = {
                    **previous,
                    "provisional_sample_count": 0,
                    "provisional_mean_information_value": 0.0,
                    "provisional_information_m2": 0.0,
                    "provisional_query_family_hashes": [],
                    "provisional_retrieval_route_hashes": [],
                }
                expired_provisional_count += 1
            previous_count = int(
                previous.get("sample_count", 0) or 0
            )
            update_count = int(update["sample_count"])
            negative_only = bool(update.get("negative_only"))
            update_family_hashes = set(
                update.get("query_family_hashes") or []
            )
            update_route_hashes = set(
                update.get("retrieval_route_hashes") or []
            )
            require_route_diversity = bool(
                update.get("require_route_diversity")
            )
            if negative_only:
                provisional_count = int(
                    previous.get(
                        "provisional_sample_count",
                        0,
                    )
                    or 0
                )
                provisional_mean = float(
                    previous.get(
                        "provisional_mean_information_value",
                        0.0,
                    )
                )
                provisional_m2 = float(
                    previous.get(
                        "provisional_information_m2",
                        0.0,
                    )
                    or 0.0
                )
                provisional_families = set(
                    previous.get(
                        "provisional_query_family_hashes",
                        [],
                    )
                )
                combined_families = (
                    provisional_families | update_family_hashes
                )
                provisional_routes = set(
                    previous.get(
                        "provisional_retrieval_route_hashes",
                        [],
                    )
                )
                combined_routes = (
                    provisional_routes | update_route_hashes
                )
                update_mean_value = float(
                    update["mean_realized_information_value"]
                )
                update_m2_value = float(
                    update.get("information_m2", 0.0) or 0.0
                )
                combined_count = provisional_count + update_count
                provisional_delta = (
                    update_mean_value - provisional_mean
                )
                combined_mean = (
                    provisional_mean
                    + provisional_delta
                    * update_count
                    / combined_count
                    if provisional_count
                    else update_mean_value
                )
                combined_m2 = (
                    provisional_m2
                    + update_m2_value
                    + provisional_delta
                    * provisional_delta
                    * provisional_count
                    * update_count
                    / combined_count
                )
                if combined_count > max_provisional_samples:
                    combined_m2 *= (
                        max_provisional_samples / combined_count
                    )
                    combined_count = max_provisional_samples
                    capped_provisional_update_count += 1
                confirmation_ready = bool(
                    len(combined_families) >= 2
                    and (
                        not require_route_diversity
                        or len(combined_routes) >= 2
                    )
                )
                if not confirmation_ready:
                    by_id[memory_id] = {
                        **previous,
                        "memory_id": memory_id,
                        "updated_at": int(time.time()),
                        "context": context,
                        "sample_count": previous_count,
                        "provisional_sample_count": combined_count,
                        "provisional_mean_information_value": round(
                            combined_mean,
                            4,
                        ),
                        "provisional_information_m2": round(
                            combined_m2,
                            4,
                        ),
                        "provisional_query_family_hashes": sorted(
                            combined_families
                        ),
                        "provisional_retrieval_route_hashes": sorted(
                            combined_routes
                        ),
                        "require_route_diversity": (
                            require_route_diversity
                        ),
                        "provisional_updated_at": int(now),
                        "feedback_status": (
                            "provisional_negative"
                        ),
                    }
                    provisional_update_count += 1
                    continue
                update_count = combined_count
                update = {
                    **update,
                    "sample_count": combined_count,
                    "mean_realized_information_value": (
                        combined_mean
                    ),
                    "information_m2": combined_m2,
                    "low_value_streak": combined_count,
                }
                confirmed_negative_update_count += 1
            else:
                immediate_positive_update_count += 1
            sample_count = previous_count + update_count
            previous_mean = float(
                previous.get(
                    "mean_realized_information_value",
                    0.0,
                )
            )
            update_mean = float(
                update["mean_realized_information_value"]
            )
            previous_m2 = max(
                0.0,
                float(previous.get("information_m2", 0.0) or 0.0),
            )
            update_m2 = max(
                0.0,
                float(update.get("information_m2", 0.0) or 0.0),
            )
            delta = update_mean - previous_mean
            mean_value = (
                previous_mean
                + delta * update_count / sample_count
                if previous_count
                else update_mean
            )
            information_m2 = (
                previous_m2
                + update_m2
                + delta
                * delta
                * previous_count
                * update_count
                / sample_count
            )
            update_streak = int(
                update.get("low_value_streak", 0) or 0
            )
            low_value_streak = (
                int(previous.get("low_value_streak", 0) or 0)
                + update_streak
                if update_streak == update_count
                else update_streak
            )
            by_id[memory_id] = {
                "memory_id": memory_id,
                "updated_at": int(time.time()),
                "context": context,
                "sample_count": sample_count,
                "mean_realized_information_value": round(
                    mean_value,
                    4,
                ),
                "information_m2": round(information_m2, 4),
                "information_variance": round(
                    information_m2 / max(1, sample_count - 1),
                    4,
                ),
                "low_value_streak": low_value_streak,
                "provisional_sample_count": 0,
                "provisional_mean_information_value": 0.0,
                "provisional_information_m2": 0.0,
                "provisional_query_family_hashes": [],
                "provisional_retrieval_route_hashes": [],
                "provisional_updated_at": 0,
                "require_route_diversity": (
                    require_route_diversity
                ),
                "feedback_status": (
                    "confirmed_negative"
                    if negative_only
                    else "committed_positive"
                ),
            }
        retained = list(by_id.values())[-max_entries:]
        temporary_path = memory_path.with_suffix(
            memory_path.suffix + ".tmp"
        )
        with temporary_path.open("w", encoding="utf-8") as handle:
            for row in retained:
                handle.write(
                    json.dumps(row, ensure_ascii=True) + "\n"
                )
        os.replace(temporary_path, memory_path)
    return {
        "written": True,
        "path": str(memory_path),
        "update_count": len(valid_updates),
        "provisional_update_count": provisional_update_count,
        "confirmed_negative_update_count": (
            confirmed_negative_update_count
        ),
        "immediate_positive_update_count": (
            immediate_positive_update_count
        ),
        "expired_provisional_count": expired_provisional_count,
        "capped_provisional_update_count": (
            capped_provisional_update_count
        ),
        "provisional_ttl_days": provisional_ttl_days,
        "max_provisional_samples": max_provisional_samples,
        "memory_size": len(retained),
    }
