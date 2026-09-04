"""A股历史资讯档案：抓一次、存档永久、回测只读档案。

设计原则（防封号 + 防未来数据）：
1. 抓取只在本地手动运行（build_archive），单线程、每页间隔≥1.5秒、遇限流自动退避；
   不在 Streamlit Cloud 上抓取。
2. 所有抓取结果写入 SQLite 档案（news_archive.db），同一条新闻只抓一次。
3. 回测读取档案时只取 发布时间 ≤ T 的条目，并把 T 作为“现在”交给
   news_analysis._enrich_items 做与正式Agent完全相同的情绪／相关度计算。
4. 档案不存在或没有覆盖数据时返回 None，调用方回退到“资讯不参与”的降级逻辑，
   绝不伪造资讯。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import time
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import news_analysis

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgentA-HistoricalNewsArchive/2.3 "
    "(research use; polite crawling; contact: yeti33-web@users.noreply.github.com)"
)

DEFAULT_DB = Path(__file__).resolve().parent.parent / "news_archive.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    market       TEXT NOT NULL,
    code         TEXT NOT NULL,
    title        TEXT NOT NULL,
    summary      TEXT,
    source       TEXT,
    published_at TEXT,
    url          TEXT NOT NULL,
    provider     TEXT,
    UNIQUE (market, code, url)
);
CREATE INDEX IF NOT EXISTS idx_news_point ON news (market, code, published_at);
"""


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    connection = sqlite3.connect(path)
    connection.executescript(_SCHEMA)
    return connection


def _strip_tags(text: Any) -> str:
    return news_analysis._strip_html(text)


def _eastmoney_page(code: str, page_index: int) -> list[dict[str, Any]]:
    """Fetch one page of public Eastmoney article search results for a stock code."""

    param = {
        "uid": "",
        "keyword": code,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "time",
                "pageIndex": page_index,
                "pageSize": 100,
                "preTag": "",
                "postTag": "",
            }
        },
    }
    url = (
        "https://search-api-web.eastmoney.com/search/jsonp"
        f"?cb=jQuery_cb&param={quote(json.dumps(param, ensure_ascii=False))}"
    )
    request = Request(url, headers={"User-Agent": USER_AGENT, "Referer": "https://so.eastmoney.com/"})
    with urlopen(request, timeout=15) as response:
        body = response.read().decode("utf-8", errors="replace")
    start = body.find("(")
    end = body.rfind(")")
    if start < 0 or end <= start:
        raise RuntimeError("东方财富资讯接口返回格式异常。")
    payload = json.loads(body[start + 1 : end])
    articles = ((payload.get("result") or {}).get("cmsArticleWebOld")) or []
    records: list[dict[str, Any]] = []
    for article in articles:
        title = _strip_tags(article.get("title"))
        link = str(article.get("url") or "").strip()
        published = str(article.get("date") or "").strip()
        if not title or not link or not published:
            continue
        records.append(
            {
                "title": title,
                "summary": news_analysis._truncate(_strip_tags(article.get("content"))),
                "source": _strip_tags(article.get("mediaName")) or "东方财富",
                "published_at": published,
                "url": link,
                "provider": "东方财富历史资讯档案",
            }
        )
    return records


def build_archive(
    market: str,
    code: str,
    start_date: date,
    max_pages: int = 20,
    sleep_seconds: float = 1.5,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Polite, paginated fetch of historical A-share news into the SQLite archive."""

    if market != "A股":
        raise ValueError("历史资讯档案当前仅支持A股（东方财富公开检索）。")
    connection = _connect(db_path)
    inserted = 0
    seen = 0
    stopped_reason = "页数用尽"
    try:
        for page in range(1, max_pages + 1):
            try:
                records = _eastmoney_page(code, page)
            except Exception as exc:
                text = str(exc)
                if "429" in text or "403" in text or "502" in text or "503" in text:
                    stopped_reason = f"第{page}页遇到限流，自动停止（已存档{inserted}条新资讯）"
                    break
                raise
            if not records:
                stopped_reason = "接口无更多结果"
                break
            oldest: datetime | None = None
            for record in records:
                seen += 1
                parsed = news_analysis._parse_datetime(record["published_at"])
                oldest = parsed if oldest is None else min(oldest, parsed or oldest)
                connection.execute(
                    "INSERT OR IGNORE INTO news (market, code, title, summary, source, published_at, url, provider)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        market,
                        code,
                        record["title"],
                        record["summary"],
                        record["source"],
                        record["published_at"],
                        record["url"],
                        record["provider"],
                    ),
                )
                inserted += 1
            connection.commit()
            if oldest is not None and oldest.date() < start_date:
                stopped_reason = f"已回溯到{oldest.date().isoformat()}，早于起点，停止"
                break
            time.sleep(max(0.0, sleep_seconds))
    finally:
        connection.close()
    return {"market": market, "code": code, "seen": seen, "inserted": inserted, "stop": stopped_reason}


def _raw_items_until(
    market: str,
    code: str,
    as_of: date,
    window_days: int,
    connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    upper = datetime(as_of.year, as_of.month, as_of.day, 23, 59, 59)
    lower = upper - timedelta(days=window_days + 1)
    rows = connection.execute(
        "SELECT title, summary, source, published_at, url, provider FROM news"
        " WHERE market = ? AND code = ? AND published_at <= ? AND published_at >= ?"
        " ORDER BY published_at DESC",
        (market, code, upper.isoformat(sep=" ", timespec="seconds"), lower.isoformat(sep=" ", timespec="seconds")),
    ).fetchall()
    return [
        {
            "title": row[0],
            "summary": row[1],
            "source": row[2],
            "published_at": row[3],
            "url": row[4],
            "provider": row[5],
        }
        for row in rows
    ]


def load_historical_news_payload(
    market: str,
    code: str,
    name: str,
    as_of: date,
    window_days: int = 14,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return an A-style news payload built only from archived items published by T.

    Returns None when the archive is missing or holds nothing for the window, so the
    caller can keep the honest “news not participating” fallback.
    """

    path = Path(db_path) if db_path else DEFAULT_DB
    if not path.exists():
        return None
    connection = _connect(path)
    try:
        raw = _raw_items_until(market, code, as_of, window_days, connection)
    finally:
        connection.close()
    if not raw:
        return None
    now_at_t = datetime(as_of.year, as_of.month, as_of.day, 23, 59, 59, tzinfo=timezone.utc)
    enriched = news_analysis._enrich_items(
        raw,
        market=market,
        code=code,
        name=name,
        now=now_at_t,
        max_days=window_days,
    )
    if not enriched:
        return None
    used = list(dict.fromkeys(str(item.get("provider") or "") for item in enriched if item.get("provider")))
    return {
        "version": "historical-archive-v1",
        "market": market,
        "code": code,
        "name": name,
        "fetched_at": now_at_t.isoformat(),
        "window_days": window_days,
        "providers_attempted": ["东方财富历史资讯档案（本地SQLite）"],
        "providers_used": used or ["东方财富历史资讯档案"],
        "items": enriched,
        "warnings": ["历史资讯仅来自本地档案中发布时间不晚于T的条目；未覆盖的日期不会伪造。"],
    }


def archive_stats(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    path = Path(db_path) if db_path else DEFAULT_DB
    if not path.exists():
        return []
    connection = _connect(path)
    try:
        rows = connection.execute(
            "SELECT market, code, COUNT(*), MIN(published_at), MAX(published_at) FROM news"
            " GROUP BY market, code ORDER BY market, code"
        ).fetchall()
    finally:
        connection.close()
    return [
        {"market": r[0], "code": r[1], "count": r[2], "earliest": r[3], "latest": r[4]}
        for r in rows
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the point-in-time news archive (local, polite).")
    parser.add_argument("--market", default="A股")
    parser.add_argument("--code", required=True)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=1.5)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    summary = build_archive(
        args.market,
        args.code,
        date.fromisoformat(args.start),
        max_pages=args.max_pages,
        sleep_seconds=args.sleep,
        db_path=args.db,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
