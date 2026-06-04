from __future__ import annotations

import concurrent.futures
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .data_sources import import_akshare, infer_exchange
from .settings import CACHE_DIR
from .utils import safe_read_csv, to_number, write_csv


POSITIVE_EVENT_WORDS = (
    "订单",
    "中标",
    "签约",
    "增持",
    "回购",
    "突破",
    "创新",
    "涨价",
    "盈利",
    "预增",
    "超预期",
    "战略合作",
)
NEGATIVE_EVENT_WORDS = (
    "减持",
    "处罚",
    "调查",
    "问询",
    "亏损",
    "预减",
    "诉讼",
    "冻结",
    "退市",
    "终止",
    "违约",
    "风险",
)
EXPANSION_WORDS = ("设厂", "建厂", "扩产", "投产", "产能", "基地", "项目落地", "新工厂", "生产线", "厂房")
POLICY_WORDS = (
    "国务院",
    "发改委",
    "工信部",
    "财政部",
    "央行",
    "政策",
    "补贴",
    "专项债",
    "新质生产力",
    "国产替代",
    "人工智能",
    "半导体",
    "机器人",
    "低空经济",
    "新能源",
    "储能",
)


def score_words(text: str, positives: tuple[str, ...], negatives: tuple[str, ...] = ()) -> float:
    text = str(text)
    pos = sum(1 for word in positives if word in text)
    neg = sum(1 for word in negatives if word in text)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def with_market_prefix(code: str) -> str:
    exchange = infer_exchange(code)
    if exchange == "SH":
        return f"SH{str(code).zfill(6)}"
    if exchange == "SZ":
        return f"SZ{str(code).zfill(6)}"
    return str(code).zfill(6)


def normalize_code_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    code_col = next((col for col in out.columns if "代码" in str(col) or str(col).lower() in {"code", "symbol", "股票代码"}), None)
    if code_col is None:
        return pd.DataFrame()
    out = out.rename(columns={code_col: "code"})
    out["code"] = out["code"].astype(str).str.extract(r"(\d{6})")[0].str.zfill(6)
    return out[out["code"].notna()].copy()


def try_fetch(name: str, cache_key: str, refresh: bool, *args, **kwargs) -> pd.DataFrame:
    cache_path = CACHE_DIR / "alt" / f"{cache_key}.csv"
    if not refresh:
        cached = safe_read_csv(cache_path, dtype={"code": str})
        if not cached.empty:
            return cached
    ak = import_akshare()
    func = getattr(ak, name, None)
    if func is None:
        return pd.DataFrame()
    try:
        df = func(*args, **kwargs)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    write_csv(df, cache_path)
    return df


def fetch_hot_keywords(stocks: pd.DataFrame, refresh: bool = False, max_workers: int = 6) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    def job(code: str) -> pd.DataFrame:
        cache_path = CACHE_DIR / "alt" / "keywords" / f"{code}.csv"
        if not refresh:
            cached = safe_read_csv(cache_path, dtype={"code": str})
            if not cached.empty:
                return cached
        ak = import_akshare()
        try:
            df = ak.stock_hot_keyword_em(symbol=with_market_prefix(code))
        except Exception:
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["code"] = str(code).zfill(6)
        text_cols = [col for col in out.columns if "词" in str(col) or "keyword" in str(col).lower()]
        if text_cols:
            out["keyword_text"] = out[text_cols].astype(str).agg(" ".join, axis=1)
        else:
            out["keyword_text"] = out.astype(str).agg(" ".join, axis=1)
        out["keyword_policy_score"] = out["keyword_text"].map(lambda x: score_words(x, POLICY_WORDS))
        out["keyword_expansion_score"] = out["keyword_text"].map(lambda x: score_words(x, EXPANSION_WORDS))
        write_csv(out, cache_path)
        time.sleep(0.05)
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(job, code) for code in stocks["code"].astype(str).str.zfill(6)]
        for future in concurrent.futures.as_completed(futures):
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=["code", "keyword_text", "keyword_policy_score", "keyword_expansion_score"])
    return pd.concat(frames, ignore_index=True)


def fetch_fund_flow_rank(refresh: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for indicator in ("今日", "3日", "5日", "10日"):
        df = try_fetch("stock_individual_fund_flow_rank", f"fund_flow_rank_{indicator}", refresh, indicator=indicator)
        df = normalize_code_column(df)
        if df.empty:
            continue
        df["fund_indicator"] = indicator
        numeric_cols = [col for col in df.columns if any(key in str(col) for key in ("净流入", "净额", "流入", "涨跌幅"))]
        for col in numeric_cols:
            df[col] = to_number(df[col])
        df["fund_flow_score"] = 0.0
        for col in numeric_cols[:4]:
            rank = df[col].rank(pct=True)
            df["fund_flow_score"] += rank.fillna(0) / max(len(numeric_cols[:4]), 1)
        frames.append(df[["code", "fund_indicator", "fund_flow_score"]])
    if not frames:
        return pd.DataFrame(columns=["code", "fund_indicator", "fund_flow_score"])
    return pd.concat(frames, ignore_index=True)


def fetch_notices(refresh: bool = False, days: int = 45) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    today = datetime.now().date()
    for offset in range(days):
        day = today - timedelta(days=offset)
        date_str = day.strftime("%Y%m%d")
        df = try_fetch("stock_notice_report", f"notices_{date_str}", refresh, symbol="全部", date=date_str)
        df = normalize_code_column(df)
        if df.empty:
            continue
        title_col = next((col for col in df.columns if "标题" in str(col) or "公告" in str(col)), None)
        if title_col is None:
            df["notice_text"] = df.astype(str).agg(" ".join, axis=1)
        else:
            df["notice_text"] = df[title_col].astype(str)
        df["notice_date"] = date_str
        df["event_score"] = df["notice_text"].map(lambda x: score_words(x, POSITIVE_EVENT_WORDS, NEGATIVE_EVENT_WORDS))
        df["expansion_score"] = df["notice_text"].map(lambda x: score_words(x, EXPANSION_WORDS))
        df["risk_event_score"] = df["notice_text"].map(lambda x: score_words(x, (), NEGATIVE_EVENT_WORDS))
        frames.append(df[["code", "notice_date", "notice_text", "event_score", "expansion_score", "risk_event_score"]])
    if not frames:
        return pd.DataFrame(columns=["code", "notice_date", "notice_text", "event_score", "expansion_score", "risk_event_score"])
    return pd.concat(frames, ignore_index=True)


def fetch_research_reports(stocks: pd.DataFrame, refresh: bool = False, max_workers: int = 4) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    def job(code: str) -> pd.DataFrame:
        cache_path = CACHE_DIR / "alt" / "research" / f"{code}.csv"
        if not refresh:
            cached = safe_read_csv(cache_path, dtype={"code": str})
            if not cached.empty:
                return cached
        ak = import_akshare()
        try:
            df = ak.stock_research_report_em(symbol=str(code).zfill(6))
        except Exception:
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["code"] = str(code).zfill(6)
        out["research_text"] = out.astype(str).agg(" ".join, axis=1)
        out["research_score"] = out["research_text"].map(lambda x: score_words(x, POSITIVE_EVENT_WORDS, NEGATIVE_EVENT_WORDS))
        write_csv(out, cache_path)
        time.sleep(0.05)
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(job, code) for code in stocks["code"].astype(str).str.zfill(6)]
        for future in concurrent.futures.as_completed(futures):
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=["code", "research_text", "research_score"])
    return pd.concat(frames, ignore_index=True)


def fetch_policy_news(refresh: bool = False, days: int = 14) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    today = datetime.now().date()
    for offset in range(days):
        date_str = (today - timedelta(days=offset)).strftime("%Y%m%d")
        cctv = try_fetch("news_cctv", f"cctv_{date_str}", refresh, date=date_str)
        if not cctv.empty:
            cctv["policy_text"] = cctv.astype(str).agg(" ".join, axis=1)
            frames.append(cctv[["policy_text"]])
    caixin = try_fetch("stock_news_main_cx", "caixin_main", refresh)
    if not caixin.empty:
        caixin["policy_text"] = caixin.astype(str).agg(" ".join, axis=1)
        frames.append(caixin[["policy_text"]])
    if not frames:
        return pd.DataFrame(columns=["policy_text", "market_policy_score"])
    out = pd.concat(frames, ignore_index=True)
    out["market_policy_score"] = out["policy_text"].map(lambda x: score_words(x, POLICY_WORDS, NEGATIVE_EVENT_WORDS))
    return out


def summarize_alternative_data(
    stocks: pd.DataFrame,
    news: pd.DataFrame,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    keywords = fetch_hot_keywords(stocks, refresh=refresh)
    fund_flow = fetch_fund_flow_rank(refresh=refresh)
    notices = fetch_notices(refresh=refresh)
    research = fetch_research_reports(stocks, refresh=refresh)
    policy_news = fetch_policy_news(refresh=refresh)

    summaries: list[pd.DataFrame] = []
    if not keywords.empty:
        summaries.append(
            keywords.groupby("code").agg(
                keyword_policy_score=("keyword_policy_score", "mean"),
                keyword_expansion_score=("keyword_expansion_score", "mean"),
                keyword_count=("keyword_text", "count"),
            ).reset_index()
        )
    if not fund_flow.empty:
        summaries.append(fund_flow.groupby("code").agg(fund_flow_score=("fund_flow_score", "mean")).reset_index())
    if not notices.empty:
        summaries.append(
            notices.groupby("code").agg(
                notice_count=("notice_text", "count"),
                event_score=("event_score", "mean"),
                expansion_score=("expansion_score", "mean"),
                risk_event_score=("risk_event_score", "mean"),
            ).reset_index()
        )
    if not research.empty:
        summaries.append(
            research.groupby("code").agg(
                research_count=("research_text", "count"),
                research_score=("research_score", "mean"),
            ).reset_index()
        )
    if not news.empty:
        tmp = news.copy()
        tmp["news_text"] = tmp[["title", "content"]].fillna("").astype(str).agg(" ".join, axis=1)
        tmp["news_policy_score"] = tmp["news_text"].map(lambda x: score_words(x, POLICY_WORDS, NEGATIVE_EVENT_WORDS))
        tmp["news_expansion_score"] = tmp["news_text"].map(lambda x: score_words(x, EXPANSION_WORDS, NEGATIVE_EVENT_WORDS))
        tmp["news_event_score"] = tmp["news_text"].map(lambda x: score_words(x, POSITIVE_EVENT_WORDS, NEGATIVE_EVENT_WORDS))
        summaries.append(
            tmp.groupby("code").agg(
                news_policy_score=("news_policy_score", "mean"),
                news_expansion_score=("news_expansion_score", "mean"),
                news_event_score=("news_event_score", "mean"),
            ).reset_index()
        )

    if not summaries:
        summary = stocks[["code"]].copy()
    else:
        summary = summaries[0]
        for frame in summaries[1:]:
            summary = summary.merge(frame, on="code", how="outer")

    market_policy_score = float(policy_news["market_policy_score"].mean()) if not policy_news.empty else 0.0
    summary["market_policy_score"] = market_policy_score
    for col in [
        "keyword_policy_score",
        "keyword_expansion_score",
        "keyword_count",
        "fund_flow_score",
        "notice_count",
        "event_score",
        "expansion_score",
        "risk_event_score",
        "research_count",
        "research_score",
        "news_policy_score",
        "news_expansion_score",
        "news_event_score",
    ]:
        if col not in summary:
            summary[col] = 0.0
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0.0)

    summary["policy_score"] = (
        0.35 * summary["keyword_policy_score"]
        + 0.35 * summary["news_policy_score"]
        + 0.30 * summary["market_policy_score"]
    )
    summary["factory_sentiment_score"] = (
        0.45 * summary["expansion_score"] + 0.35 * summary["news_expansion_score"] + 0.20 * summary["keyword_expansion_score"]
    )
    summary["public_logic_score"] = (
        0.22 * summary["fund_flow_score"]
        + 0.18 * summary["research_score"]
        + 0.18 * summary["event_score"]
        + 0.16 * summary["news_event_score"]
        + 0.14 * summary["policy_score"]
        + 0.12 * summary["factory_sentiment_score"]
        + 0.10 * summary["risk_event_score"]
    )
    metrics = {
        "keyword_symbols": float(keywords["code"].nunique() if not keywords.empty else 0),
        "fund_flow_symbols": float(fund_flow["code"].nunique() if not fund_flow.empty else 0),
        "notice_symbols": float(notices["code"].nunique() if not notices.empty else 0),
        "research_symbols": float(research["code"].nunique() if not research.empty else 0),
        "policy_news_rows": float(len(policy_news)),
    }
    return summary, metrics
