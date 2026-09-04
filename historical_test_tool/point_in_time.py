from __future__ import annotations

from datetime import date
from typing import Any


def build_point_in_time_evidence(core_module: Any, bundle: Any, as_of: date) -> tuple[Any, Any, dict[str, Any]]:
    """Build only evidence that can be proven available by the historical cutoff."""

    fundamental = core_module.EvidenceSnapshot(
        available=False,
        provider="历史时点财务快照未接入",
        score=None,
        notes=[
            "免费数据源未提供可统一核验的实际披露时间，因此没有把今天看到的财务数据回填到历史日期。",
            "本次基本面按现有Agent的缺失数据规则处理：不加分、不减分，并降低数据完整度。",
        ],
    )

    benchmark_available = bundle.benchmark is not None and not bundle.benchmark.empty
    if benchmark_available:
        regime = core_module.derive_market_regime(bundle.benchmark)
        regime["数据截止日"] = as_of.isoformat()
        macro = core_module.EvidenceSnapshot(
            available=True,
            provider="截至T日的对应市场基准",
            fields=regime,
            score=float(regime["市场分"]),
            notes=[
                "仅使用T日及以前的基准指数趋势、收益和波动。",
                "历史利率序列未完成逐发布日期核验，本次不使用LPR或美债利率修正。",
            ],
        )
    else:
        macro = core_module.EvidenceSnapshot(
            available=False,
            provider="历史市场基准未取得",
            score=None,
            notes=["基准与历史利率均未安全取得，宏观项不参与本次评分。"],
        )

    evidence_status = {
        "行情与成交量": "参与；已在T日截断",
        "市场基准": "参与；已在T日截断" if benchmark_available else "未取得，不参与",
        "财务数据": "未取得可核验披露日的历史快照，不参与",
        "历史利率": "未完成逐发布日期核验，不参与",
        "历史资讯": "未接入可核验历史资讯库，不参与",
        "处理原则": "缺失证据记为缺失，不用今天的数据倒填过去",
    }
    return fundamental, macro, evidence_status


def empty_historical_news_payload(market: str, code: str, name: str, as_of: date) -> dict[str, Any]:
    return {
        "version": "historical-safe-empty-v1",
        "market": market,
        "code": code,
        "name": name,
        "fetched_at": f"{as_of.isoformat()}T23:59:59",
        "window_days": 0,
        "providers_attempted": [],
        "providers_used": [],
        "items": [],
        "warnings": ["未接入带可核验历史发布时间的资讯库，资讯修正固定为0分。"],
    }


def build_point_in_time_news(market: str, code: str, name: str, as_of: date) -> tuple[dict[str, Any], str]:
    """Prefer the local news archive (publish time <= T); degrade honestly when absent."""

    try:
        from historical_test_tool import news_archive
    except Exception:
        import news_archive  # type: ignore[no-redef]
    try:
        payload = news_archive.load_historical_news_payload(market, code, name, as_of)
    except Exception:
        payload = None
    if payload:
        count = len(payload.get("items") or [])
        return payload, f"参与；仅使用档案中发布时间不晚于T的{count}条资讯"
    return empty_historical_news_payload(market, code, name, as_of), "本地档案未覆盖，不参与"

