from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Mapping


def session_key(market: str, code: str) -> str:
    """Return the stable identifier used by the stock conversation store."""
    return f"{str(market).strip()}:{str(code).strip().upper()}"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def known_invested_principal(holding_snapshot: Mapping[str, Any] | None) -> float:
    """Only a known historical cost is treated as invested principal."""
    if not holding_snapshot:
        return 0.0
    try:
        value = float(holding_snapshot.get("cost_total_rmb") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(value, 0.0)


def build_analysis_messages(
    *,
    market: str,
    code: str,
    name: str,
    analysis: Mapping[str, Any],
    holding_state: str,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    """Create compact, serialisable chat messages from an existing analysis result."""
    timestamp = created_at or now_text()
    selected = analysis.get("selected_horizon") or {}
    suitability = analysis.get("suitability") or {}
    user_content = f"请分析{market} {code}（{name}），当前状态：{holding_state}。"
    assistant_content = (
        f"结论：{analysis.get('conclusion', '数据不足')}。"
        f"个人适配：{suitability.get('fit', '数据不足')}；"
        f"股票风险：{analysis.get('stock_risk_level', '数据不足')}；"
        f"建议复核／持有周期：{selected.get('name', '数据不足')}；"
        f"数据完整度：{float(analysis.get('data_confidence') or 0.0):.3f}%。"
    )
    return [
        {"role": "user", "content": user_content, "created_at": timestamp, "kind": "analysis_request"},
        {
            "role": "assistant",
            "content": assistant_content,
            "created_at": timestamp,
            "kind": "analysis_summary",
        },
    ]


def upsert_analysis_session(
    sessions: Mapping[str, Any] | None,
    *,
    event_id: str,
    market: str,
    code: str,
    name: str,
    analysis: Mapping[str, Any],
    holding_state: str,
    holding_snapshot: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    """Append one analysis to its stock conversation, once per analysis event."""
    updated = deepcopy(dict(sessions or {}))
    key = session_key(market, code)
    timestamp = now_text()
    record = updated.get(key)
    created = record is None
    if record is None:
        record = {
            "key": key,
            "market": market,
            "code": code,
            "name": name,
            "created_at": timestamp,
            "updated_at": timestamp,
            "principal_rmb": known_invested_principal(holding_snapshot),
            "principal_source": "已知持仓成本" if known_invested_principal(holding_snapshot) > 0 else "尚未记录实际投入",
            "messages": [],
            "recorded_event_ids": [],
        }
    event_ids = list(record.get("recorded_event_ids") or [])
    if event_id in event_ids:
        return updated, False

    record["market"] = market
    record["code"] = code
    record["name"] = name
    record["updated_at"] = timestamp
    record["messages"] = list(record.get("messages") or []) + build_analysis_messages(
        market=market,
        code=code,
        name=name,
        analysis=analysis,
        holding_state=holding_state,
        created_at=timestamp,
    )
    record["recorded_event_ids"] = event_ids + [event_id]
    record["latest_summary"] = {
        "conclusion": analysis.get("conclusion", "数据不足"),
        "conclusion_reason": analysis.get("conclusion_reason", ""),
        "investor_level": analysis.get("investor_level", "数据不足"),
        "stock_risk_level": analysis.get("stock_risk_level", "数据不足"),
        "suitability": (analysis.get("suitability") or {}).get("fit", "数据不足"),
        "selected_horizon": (analysis.get("selected_horizon") or {}).get("name", "数据不足"),
        "data_confidence": float(analysis.get("data_confidence") or 0.0),
        "holding_state": holding_state,
        "recorded_at": timestamp,
    }
    if created:
        known_principal = known_invested_principal(holding_snapshot)
        record["principal_rmb"] = known_principal
        record["principal_source"] = "已知持仓成本" if known_principal > 0 else "尚未记录实际投入"
    elif (
        float(record.get("principal_rmb") or 0.0) <= 0
        and record.get("principal_source") != "用户在会话中确认"
        and known_invested_principal(holding_snapshot) > 0
    ):
        record["principal_rmb"] = known_invested_principal(holding_snapshot)
        record["principal_source"] = "已知持仓成本"
    updated[key] = record
    return updated, True


def set_invested_principal(
    sessions: Mapping[str, Any] | None,
    key: str,
    amount: float,
) -> dict[str, Any]:
    """Update the actual principal recorded in one retained stock conversation."""
    parsed = float(amount)
    if parsed < 0:
        raise ValueError("投入本金不能小于0。")
    updated = deepcopy(dict(sessions or {}))
    if key not in updated:
        raise KeyError("股票会话不存在或已经删除。")
    updated[key]["principal_rmb"] = parsed
    updated[key]["principal_source"] = "用户在会话中确认"
    updated[key]["updated_at"] = now_text()
    return updated


def append_note(
    sessions: Mapping[str, Any] | None,
    key: str,
    content: str,
) -> dict[str, Any]:
    cleaned = str(content).strip()
    if not cleaned:
        return deepcopy(dict(sessions or {}))
    updated = deepcopy(dict(sessions or {}))
    if key not in updated:
        raise KeyError("股票会话不存在或已经删除。")
    timestamp = now_text()
    updated[key]["messages"] = list(updated[key].get("messages") or []) + [
        {"role": "user", "content": cleaned, "created_at": timestamp, "kind": "note"}
    ]
    updated[key]["updated_at"] = timestamp
    return updated


def delete_session(sessions: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    """Deleting a conversation is the business event 'fully sold'."""
    updated = deepcopy(dict(sessions or {}))
    updated.pop(key, None)
    return updated


def portfolio_from_sessions(sessions: Mapping[str, Any] | None) -> dict[str, Any]:
    """Calculate portfolio weights from retained conversation principals only."""
    rows: list[dict[str, Any]] = []
    for key, record in dict(sessions or {}).items():
        try:
            principal = max(float(record.get("principal_rmb") or 0.0), 0.0)
        except (TypeError, ValueError):
            principal = 0.0
        if principal <= 0:
            continue
        rows.append(
            {
                "key": key,
                "market": record.get("market", "—"),
                "code": record.get("code", "—"),
                "name": record.get("name", "—"),
                "principal_rmb": principal,
                "updated_at": record.get("updated_at", ""),
            }
        )
    rows.sort(key=lambda item: (-item["principal_rmb"], item["key"]))
    total = sum(item["principal_rmb"] for item in rows)
    for item in rows:
        item["weight"] = item["principal_rmb"] / total if total > 0 else 0.0
    return {
        "rows": rows,
        "total_principal_rmb": total,
        "position_count": len(rows),
        "conversation_count": len(dict(sessions or {})),
    }
