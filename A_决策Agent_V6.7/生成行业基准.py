# -*- coding: utf-8 -*-
"""生成 A 股行业基准（净利率／负债率／PE 中位数），供基本面行业标准化使用。

用法：
    python 生成行业基准.py [--samples 12] [--workers 6] [--industries 行业名1,行业名2,...]
不带 --industries 时扫描全部行业（BaoStock 证监会分类）。结果写入行业基准_V6.7.csv。
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import partial

import baostock as bs

OUT = "行业基准_V6.7.csv"
# profit_data 字段：code,pubDate,statDate,roeAvg,npMargin,gpMargin,netProfit,epsTTM,...
# balance_data 字段：code,pubDate,statDate,...,liabilityToAsset,...
_NP_IDX = 4
_EPS_IDX = 7
_DEBT_IDX = 7


def _recent_quarters(count: int = 4) -> list[tuple[int, int]]:
    today = date.today()
    year, quarter = today.year, max(1, (today.month - 1) // 3)
    out = []
    for _ in range(count):
        out.append((year, quarter))
        quarter -= 1
        if quarter == 0:
            year, quarter = year - 1, 4
    return out


def _retry(fn, tries: int = 3):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(0.4)


def _profit_row(code: str, year: int, quarter: int):
    rp = bs.query_profit_data(code=code, year=year, quarter=quarter)
    while (rp.error_code == "0") and rp.next():
        return rp.get_row_data()
    return None


def _balance_row(code: str, year: int, quarter: int):
    rb = bs.query_balance_data(code=code, year=year, quarter=quarter)
    while (rb.error_code == "0") and rb.next():
        return rb.get_row_data()
    return None


def _last_close(code: str):
    rk = bs.query_history_k_data_plus(
        code, "close", start_date="2024-06-01", end_date=date.today().isoformat(),
        frequency="d", adjustflag="2",
    )
    last = None
    while (rk.error_code == "0") and rk.next():
        row = rk.get_row_data()
        if row and row[0]:
            last = float(row[0])
    return last


def scan_industry(ind: str, samples: int) -> tuple:
    """单个行业扫描（独立进程、独立 baostock 连接）。"""
    bs.login()
    try:
        rs = bs.query_stock_industry()
        codes: list[str] = []
        while (rs.error_code == "0") and rs.next():
            r = rs.get_row_data()
            if len(r) > 3 and r[3] == ind:
                codes.append(r[1])
        codes = codes[:samples]
        quarters = _recent_quarters(4)
        # 探测：前3只样本中至少2只有数据的最近季度作为基准报告期
        base_q = None
        for (y, q) in quarters:
            got = 0
            for code in codes[:3]:
                try:
                    if _retry(lambda c=code, y=y, q=q: _profit_row(c, y, q)) is not None:
                        got += 1
                except Exception:
                    pass
            if got >= 2:
                base_q = (y, q)
                break
        if base_q is None:
            base_q = quarters[0]
        y, q = base_q

        margins, debts, eps_vals, prices, pes = [], [], [], [], []
        ok = 0
        for code in codes:
            try:
                prow = _retry(lambda c=code: _profit_row(c, y, q))
                if prow is None:
                    continue
                margins.append(float(prow[_NP_IDX]))
                eps = float(prow[_EPS_IDX])
                eps_vals.append(eps)
                brow = _retry(lambda c=code: _balance_row(c, y, q))
                if brow is not None:
                    debts.append(float(brow[_DEBT_IDX]))
                close = _retry(lambda c=code: _last_close(c))
                if close is not None and close > 0:
                    prices.append(close)
                    if eps > 0:
                        pes.append(close / eps)
                ok += 1
            except Exception:
                continue

        def med(v):
            return round(statistics.median(v), 4) if v else None

        return (ind, ok, med(margins), med(debts), med(pes), f"{y}Q{q}")
    finally:
        bs.logout()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--industries", type=str, default="")
    args = parser.parse_args()

    lg = bs.login()
    rs = bs.query_stock_industry()
    inds_set: dict[str, bool] = {}
    while (rs.error_code == "0") and rs.next():
        r = rs.get_row_data()
        if len(r) > 3 and r[3]:
            inds_set[r[3]] = True
    bs.logout()
    all_inds = sorted(inds_set)
    if args.industries:
        want = set(args.industries.split(","))
        inds = [x for x in all_inds if x in want]
    else:
        inds = all_inds
    print(f"待扫描行业: {len(inds)} 个（每行业最多 {args.samples} 只，进程池 {args.workers}）")

    t0 = time.time()
    # 断点续跑：跳过今天已成功写入的行业
    done_ok: set[str] = set()
    try:
        with open(OUT, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("样本数") and int(row["样本数"]) > 0 and row.get("生成日期") == time.strftime("%Y-%m-%d"):
                    done_ok.add(row["行业"])
    except FileNotFoundError:
        pass
    todo = [x for x in inds if x not in done_ok]
    print(f"待扫描行业: {len(inds)}（今日已完成 {len(done_ok)}，实际执行 {len(todo)}），每行业最多 {args.samples} 只，进程池 {args.workers}")

    results: dict[str, tuple] = {}
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for ind, res in zip(todo, ex.map(partial(scan_industry, samples=args.samples), todo)):
                results[ind] = res
        # 样本为0的行业：单进程重试一次（并发易被BaoStock限流，单进程稳定）
        retry = [ind for ind, r in results.items() if (r[1] or 0) == 0]
        if retry:
            print(f"并发失败 {len(retry)} 个行业，单进程重试: {retry[:5]}...")
            for ind in retry:
                results[ind] = scan_industry(ind, args.samples)
    # 合并今日已成功的旧结果
    if done_ok:
        with open(OUT, "r", encoding="utf-8-sig") as f:
            rows = [r for r in csv.reader(f)][1:]
        for r in rows:
            if r[0] in done_ok:
                results[r[0]] = (r[0], int(r[1]), float(r[2]) if r[2] else None,
                                 float(r[3]) if r[3] else None, float(r[4]) if r[4] else None, r[5])
    print(f"扫描完成，耗时 {time.time()-t0:.1f}s")

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["行业", "样本数", "净利率中位数", "负债率中位数", "PE中位数", "报告期", "生成日期"])
        for ind, r in sorted(results.items()):
            w.writerow(list(r[:5]) + [r[5], time.strftime("%Y-%m-%d")])
    print(f"已写入 {OUT}")
    for ind, r in sorted(results.items()):
        print(ind, r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
