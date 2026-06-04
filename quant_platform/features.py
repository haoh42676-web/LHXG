from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "ret_250d",
    "close_to_ma5",
    "close_to_ma20",
    "close_to_ma60",
    "ma5_to_ma20",
    "ma20_to_ma60",
    "ma60_to_ma120",
    "volatility_20d",
    "volatility_60d",
    "volatility_120d",
    "amount_ratio_20d",
    "turnover_rate",
    "max_drawdown_60d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "volume_ratio_20d",
    "price_position_60d",
    "price_position_250d",
    "market_cap_log",
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


def max_drawdown(values: pd.Series) -> float:
    running_max = values.cummax()
    drawdown = values / running_max - 1
    return drawdown.min()


def add_technical_features(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history

    frames: list[pd.DataFrame] = []
    for code, group in history.groupby("code", sort=False):
        df = group.sort_values("date").copy()
        df["ret_1d"] = df["close"].pct_change(1)
        df["ret_5d"] = df["close"].pct_change(5)
        df["ret_10d"] = df["close"].pct_change(10)
        df["ret_20d"] = df["close"].pct_change(20)
        df["ret_60d"] = df["close"].pct_change(60)
        df["ret_120d"] = df["close"].pct_change(120)
        df["ret_250d"] = df["close"].pct_change(250)
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma120"] = df["close"].rolling(120).mean()
        df["close_to_ma5"] = df["close"] / df["ma5"] - 1
        df["close_to_ma20"] = df["close"] / df["ma20"] - 1
        df["close_to_ma60"] = df["close"] / df["ma60"] - 1
        df["ma5_to_ma20"] = df["ma5"] / df["ma20"] - 1
        df["ma20_to_ma60"] = df["ma20"] / df["ma60"] - 1
        df["ma60_to_ma120"] = df["ma60"] / df["ma120"] - 1
        df["volatility_20d"] = df["ret_1d"].rolling(20).std()
        df["volatility_60d"] = df["ret_1d"].rolling(60).std()
        df["volatility_120d"] = df["ret_1d"].rolling(120).std()
        df["amount_ma20"] = df["amount"].rolling(20).mean()
        df["amount_ratio_20d"] = df["amount"] / df["amount_ma20"] - 1
        df["volume_ma20"] = df["volume"].rolling(20).mean()
        df["volume_ratio_20d"] = df["volume"] / df["volume_ma20"] - 1
        df["max_drawdown_60d"] = df["close"].rolling(60).apply(max_drawdown, raw=False)
        low_60 = df["low"].rolling(60).min()
        high_60 = df["high"].rolling(60).max()
        df["price_position_60d"] = (df["close"] - low_60) / (high_60 - low_60)
        low_250 = df["low"].rolling(250).min()
        high_250 = df["high"].rolling(250).max()
        df["price_position_250d"] = (df["close"] - low_250) / (high_250 - low_250)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["market_cap_log"] = np.log(pd.to_numeric(df["market_cap"], errors="coerce").clip(lower=1))
        df["target_1d"] = df["close"].shift(-1) / df["close"] - 1
        df["target_5d"] = df["close"].shift(-5) / df["close"] - 1
        df["target_20d"] = df["close"].shift(-20) / df["close"] - 1
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def latest_feature_rows(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    return features.sort_values("date").groupby("code", as_index=False).tail(1).reset_index(drop=True)


def clean_model_frame(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    for column in FEATURE_COLUMNS:
        if column not in df:
            df[column] = 0.0
    needed = FEATURE_COLUMNS + ["target_1d", "target_5d", "target_20d", "date", "code"]
    df = df[needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=FEATURE_COLUMNS)
