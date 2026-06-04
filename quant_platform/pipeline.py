from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os

import pandas as pd

from .alternative_data import summarize_alternative_data
from .data_sources import fetch_histories, fetch_top_market_cap_stocks
from .deepseek import analyze_candidates_with_deepseek
from .explain import add_explanations
from .features import FEATURE_COLUMNS, add_technical_features, clean_model_frame, latest_feature_rows
from .model import SUCCESS_TOLERANCE, predict, train_models
from .news import fetch_hot_rankings, fetch_news_for_stocks, summarize_news
from .settings import OUTPUT_DIR
from .utils import write_csv


@dataclass
class PipelineResult:
    predictions: pd.DataFrame
    history: pd.DataFrame
    news: pd.DataFrame
    hot_rankings: pd.DataFrame
    validation: pd.DataFrame
    optimization_log: pd.DataFrame
    metrics: dict[str, float | str]
    generated_at: datetime


ALT_COLUMNS = [
    "news_count",
    "recent_news_count",
    "news_score",
    "hot_score",
    "fund_flow_score",
    "event_score",
    "policy_score",
    "factory_sentiment_score",
    "research_score",
    "public_logic_score",
    "risk_event_score",
]


def merge_event_features(latest: pd.DataFrame, news_df: pd.DataFrame, hot_rankings: pd.DataFrame, stocks: pd.DataFrame, fetch_news: bool, refresh_news: bool) -> tuple[pd.DataFrame, dict[str, float]]:
    news_summary = summarize_news(news_df, hot_rankings)
    latest = latest.merge(news_summary, on="code", how="left")
    alternative_metrics: dict[str, float] = {}
    if fetch_news:
        alternative_summary, alternative_metrics = summarize_alternative_data(stocks, news_df, refresh=refresh_news)
        if not alternative_summary.empty:
            latest = latest.merge(alternative_summary, on="code", how="left")
    for column in ALT_COLUMNS:
        if column not in latest:
            latest[column] = 0.0
        latest[column] = pd.to_numeric(latest[column], errors="coerce").fillna(0.0)
    return latest, alternative_metrics


def build_basis(row: pd.Series) -> str:
    parts = [
        str(row.get("logic_notes", "") or ""),
        str(row.get("ai_review", "") or ""),
    ]
    status = str(row.get("stock_status", "正常") or "正常")
    if status != "正常":
        parts.append(f"状态标记:{status}")
    if row.get("ai_risk_flags"):
        parts.append(f"风险提示:{row.get('ai_risk_flags')}")
    return "；".join(part for part in parts if part)


def run_pipeline(
    top_n: int | None = None,
    lookback_days: int | None = None,
    adjust: str = "qfq",
    fetch_news: bool = True,
    refresh_spot: bool = False,
    refresh_history: bool = False,
    refresh_news: bool = False,
    deepseek_api_key: str | None = None,
    deepseek_model: str = "deepseek-chat",
) -> PipelineResult:
    generated_at = datetime.now()
    lookback_days = lookback_days or int(os.getenv("LOOKBACK_DAYS", "1825"))
    stocks = fetch_top_market_cap_stocks(top_n=top_n, refresh=refresh_spot)
    history, fetch_status = fetch_histories(
        stocks,
        lookback_days=lookback_days,
        adjust=adjust,
        refresh=refresh_history,
    )
    if history.empty:
        raise RuntimeError("未能抓取到历史行情，请检查网络、AKShare 或备用数据源配置")

    feature_history = add_technical_features(history)
    model_frame = clean_model_frame(feature_history)
    bundle = train_models(model_frame)
    latest = latest_feature_rows(feature_history)

    news_df = pd.DataFrame(columns=["code", "publish_time", "title", "content", "source", "url", "sentiment"])
    hot_rankings = pd.DataFrame(columns=["code", "hot_source", "hot_rank"])
    if fetch_news:
        news_df = fetch_news_for_stocks(stocks, refresh=refresh_news)
        hot_rankings = fetch_hot_rankings(refresh=refresh_news)

    latest, alternative_metrics = merge_event_features(latest, news_df, hot_rankings, stocks, fetch_news, refresh_news)
    predictions = add_explanations(predict(bundle, latest))
    try:
        predictions = analyze_candidates_with_deepseek(
            predictions,
            news_df,
            api_key=deepseek_api_key,
            model=deepseek_model,
        )
        deepseek_status = "enabled" if deepseek_api_key else "disabled"
    except Exception as exc:
        predictions["ai_review"] = ""
        predictions["ai_risk_flags"] = ""
        deepseek_status = f"failed: {exc}"

    predictions["risk_rate"] = (100 - predictions["risk_score"]) / 100
    predictions["basis"] = predictions.apply(build_basis, axis=1)
    predictions["latest_close"] = predictions["close"]

    wanted_order = [
        "rank",
        "code",
        "name",
        "exchange",
        "date",
        "latest_close",
        "market_cap",
        "pred_1d",
        "pred_5d",
        "pred_20d",
        "risk_rate",
        "risk_score",
        "risk_level",
        "stock_status",
        "basis",
        "signal",
        "confidence",
        "score",
        "meets_target",
        "main_factors",
        "logic_notes",
        "ai_review",
        "ai_risk_flags",
    ] + ALT_COLUMNS
    ordered = [col for col in wanted_order if col in predictions.columns]
    rest = [col for col in predictions.columns if col not in ordered]
    predictions = predictions[ordered + rest]

    validation = bundle.validation.copy()
    optimization_log = bundle.optimization_log.copy()
    write_csv(predictions, OUTPUT_DIR / "latest_predictions.csv")
    write_csv(validation, OUTPUT_DIR / "latest_validation.csv")
    write_csv(optimization_log, OUTPUT_DIR / "latest_optimization_log.csv")
    feature_history.to_csv(OUTPUT_DIR / "latest_feature_history.csv", index=False, encoding="utf-8-sig")
    if not news_df.empty:
        write_csv(news_df, OUTPUT_DIR / "latest_news.csv")

    metrics: dict[str, float | str] = dict(bundle.metrics)
    metrics.update(
        {
            "stock_pool": float(len(stocks)),
            "history_ok": float(fetch_status.ok),
            "history_failed": float(fetch_status.failed),
            "data_source_failures": float(fetch_status.failed),
            "history_rows": float(len(history)),
            "feature_rows": float(len(model_frame)),
            "news_symbols": float(news_df["code"].nunique() if not news_df.empty else 0),
            "news_rows": float(len(news_df)),
            "hot_rows": float(len(hot_rankings)),
            "deepseek_status": deepseek_status,
            "target_candidates": float(predictions["meets_target"].sum() if "meets_target" in predictions else 0),
            "success_tolerance": SUCCESS_TOLERANCE,
            "model_input_factor_count": float(len(FEATURE_COLUMNS)),
            "failed_symbols_sample": " | ".join(fetch_status.failures[:20]),
        }
    )
    metrics.update(alternative_metrics)

    return PipelineResult(
        predictions=predictions,
        history=feature_history,
        news=news_df,
        hot_rankings=hot_rankings,
        validation=validation,
        optimization_log=optimization_log,
        metrics=metrics,
        generated_at=generated_at,
    )
