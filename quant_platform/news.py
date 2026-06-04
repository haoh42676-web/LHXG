from __future__ import annotations

import concurrent.futures
import time
from datetime import datetime, timedelta

import pandas as pd

from .data_sources import import_akshare
from .settings import CACHE_DIR, NEWS_DIR
from .utils import safe_read_csv, write_csv


POSITIVE_WORDS = (
    "增长",
    "突破",
    "中标",
    "回购",
    "增持",
    "盈利",
    "上调",
    "创新高",
    "签约",
    "利好",
    "扩产",
    "订单",
    "超预期",
)
NEGATIVE_WORDS = (
    "下滑",
    "亏损",
    "减持",
    "处罚",
    "调查",
    "诉讼",
    "风险",
    "暴跌",
    "低于预期",
    "终止",
    "退市",
    "问询",
)


def score_text(text: str) -> float:
    text = str(text)
    pos = sum(1 for word in POSITIVE_WORDS if word in text)
    neg = sum(1 for word in NEGATIVE_WORDS if word in text)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def fetch_stock_news(code: str, refresh: bool = False) -> pd.DataFrame:
    cache_path = NEWS_DIR / f"{code}.csv"
    if not refresh:
        cached = safe_read_csv(cache_path, dtype={"code": str})
        if not cached.empty:
            cached["publish_time"] = pd.to_datetime(cached["publish_time"], errors="coerce")
            return cached

    ak = import_akshare()
    if not hasattr(ak, "stock_news_em"):
        return pd.DataFrame()
    raw = ak.stock_news_em(symbol=str(code).zfill(6))
    if raw.empty:
        return pd.DataFrame()

    df = raw.rename(
        columns={
            "发布时间": "publish_time",
            "新闻标题": "title",
            "新闻内容": "content",
            "新闻链接": "url",
            "文章来源": "source",
        }
    ).copy()
    df["code"] = str(code).zfill(6)
    if "publish_time" in df:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    else:
        df["publish_time"] = pd.NaT
    if "title" not in df:
        df["title"] = ""
    if "content" not in df:
        df["content"] = ""
    if "url" not in df:
        df["url"] = ""
    if "source" not in df:
        df["source"] = "eastmoney"
    df["sentiment"] = (df["title"].fillna("") + " " + df["content"].fillna("")).map(score_text)
    df = df[["code", "publish_time", "title", "content", "source", "url", "sentiment"]].drop_duplicates(
        subset=["code", "title", "url"]
    )
    write_csv(df, cache_path)
    return df


def fetch_news_for_stocks(stocks: pd.DataFrame, refresh: bool = False, max_workers: int = 4) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    def job(code: str) -> pd.DataFrame:
        time.sleep(0.08)
        return fetch_stock_news(code, refresh=refresh)

    codes = stocks["code"].astype(str).str.zfill(6).tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_code = {executor.submit(job, code): code for code in codes}
        for future in concurrent.futures.as_completed(future_to_code):
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=["code", "publish_time", "title", "content", "source", "url", "sentiment"])
    return pd.concat(frames, ignore_index=True)


def fetch_hot_rankings(refresh: bool = False) -> pd.DataFrame:
    cache_path = CACHE_DIR / "hot_rankings.csv"
    if not refresh:
        cached = safe_read_csv(cache_path, dtype={"code": str})
        if not cached.empty:
            return cached

    ak = import_akshare()
    candidates = [
        ("stock_hot_rank_em", {}),
        ("stock_hot_rank_wc", {"date": datetime.now().strftime("%Y%m%d")}),
        ("stock_hot_follow_xq", {"symbol": "最热门"}),
        ("stock_hot_tweet_xq", {"symbol": "最热门"}),
        ("stock_hot_deal_xq", {"symbol": "最热门"}),
    ]
    frames: list[pd.DataFrame] = []
    for func_name, kwargs in candidates:
        func = getattr(ak, func_name, None)
        if func is None:
            continue
        try:
            raw = func(**kwargs)
            if raw is None or raw.empty:
                continue
            df = raw.copy()
            code_col = next((col for col in df.columns if "代码" in str(col) or str(col).lower() in {"code", "symbol"}), None)
            name_col = next((col for col in df.columns if "名称" in str(col) or str(col).lower() == "name"), None)
            if code_col is None:
                continue
            df = df.rename(columns={code_col: "code"})
            if name_col is not None:
                df = df.rename(columns={name_col: "name"})
            df["code"] = df["code"].astype(str).str.extract(r"(\d{6})")[0]
            df = df[df["code"].notna()].copy()
            df["hot_source"] = func_name
            df["hot_rank"] = range(1, len(df) + 1)
            frames.append(df[["code", "name", "hot_source", "hot_rank"]] if "name" in df else df[["code", "hot_source", "hot_rank"]])
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["code", "hot_source", "hot_rank"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code", "hot_source"])
    write_csv(out, cache_path)
    return out


def summarize_news(news: pd.DataFrame, hot_rankings: pd.DataFrame | None = None) -> pd.DataFrame:
    if news.empty:
        summary = pd.DataFrame(columns=["code", "news_count", "recent_news_count", "news_score"])
    else:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
        tmp = news.copy()
        tmp["recent"] = tmp["publish_time"].fillna(pd.Timestamp("1900-01-01")) >= cutoff
        summary = tmp.groupby("code").agg(
            news_count=("title", "count"),
            recent_news_count=("recent", "sum"),
            news_score=("sentiment", "mean"),
        ).reset_index()

    if hot_rankings is not None and not hot_rankings.empty:
        hot = hot_rankings.groupby("code").agg(best_hot_rank=("hot_rank", "min")).reset_index()
        hot["hot_score"] = (101 - hot["best_hot_rank"].clip(upper=100)) / 100
        summary = summary.merge(hot, on="code", how="outer")
    else:
        summary["best_hot_rank"] = pd.NA
        summary["hot_score"] = 0.0

    summary["news_count"] = summary["news_count"].fillna(0)
    summary["recent_news_count"] = summary["recent_news_count"].fillna(0)
    summary["news_score"] = summary["news_score"].fillna(0.0)
    summary["hot_score"] = summary["hot_score"].fillna(0.0)
    return summary
