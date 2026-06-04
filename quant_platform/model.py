from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS
from .utils import clamp


MAX_OPTIMIZATION_ATTEMPTS = max(100, int(os.getenv("MAX_OPTIMIZATION_ATTEMPTS", "500")))
SUCCESS_TOLERANCE = 0.005


@dataclass
class ModelBundle:
    model_1d: Pipeline
    model_5d: Pipeline
    model_20d: Pipeline
    metrics: dict[str, float | str]
    thresholds: dict[str, float]
    validation: pd.DataFrame
    optimization_log: pd.DataFrame


def make_model(
    max_iter: int = 160,
    learning_rate: float = 0.055,
    l2_regularization: float = 0.02,
    max_leaf_nodes: int = 31,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    l2_regularization=l2_regularization,
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=42,
                ),
            ),
        ]
    )


def forest_model(kind: str = "extra", max_depth: int | None = 10) -> Pipeline:
    if kind == "rf":
        model = RandomForestRegressor(n_estimators=150, max_depth=max_depth, min_samples_leaf=3, random_state=42, n_jobs=-1)
    else:
        model = ExtraTreesRegressor(n_estimators=200, max_depth=max_depth, min_samples_leaf=3, random_state=42, n_jobs=-1)
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", model)])


def candidate_model_specs(max_attempts: int = MAX_OPTIMIZATION_ATTEMPTS) -> list[tuple[str, tuple[str, dict], tuple[str, dict], tuple[str, dict]]]:
    specs: list[tuple[str, tuple[str, dict], tuple[str, dict], tuple[str, dict]]] = [
        ("hgb_balanced", ("hgb", {"max_iter": 160, "learning_rate": 0.055, "l2_regularization": 0.02, "max_leaf_nodes": 31}), ("hgb", {"max_iter": 180, "learning_rate": 0.05, "l2_regularization": 0.03, "max_leaf_nodes": 31}), ("hgb", {"max_iter": 200, "learning_rate": 0.045, "l2_regularization": 0.04, "max_leaf_nodes": 31})),
        ("hgb_deeper", ("hgb", {"max_iter": 220, "learning_rate": 0.04, "l2_regularization": 0.01, "max_leaf_nodes": 45}), ("hgb", {"max_iter": 240, "learning_rate": 0.035, "l2_regularization": 0.02, "max_leaf_nodes": 45}), ("hgb", {"max_iter": 260, "learning_rate": 0.03, "l2_regularization": 0.03, "max_leaf_nodes": 45})),
        ("hgb_regularized", ("hgb", {"max_iter": 260, "learning_rate": 0.03, "l2_regularization": 0.08, "max_leaf_nodes": 23}), ("hgb", {"max_iter": 280, "learning_rate": 0.028, "l2_regularization": 0.10, "max_leaf_nodes": 23}), ("hgb", {"max_iter": 300, "learning_rate": 0.026, "l2_regularization": 0.12, "max_leaf_nodes": 23})),
        ("extra_trees_10", ("extra", {"max_depth": 10}), ("extra", {"max_depth": 10}), ("extra", {"max_depth": 10})),
        ("extra_trees_14", ("extra", {"max_depth": 14}), ("extra", {"max_depth": 14}), ("extra", {"max_depth": 14})),
        ("random_forest", ("rf", {"max_depth": 10}), ("rf", {"max_depth": 10}), ("rf", {"max_depth": 10})),
    ]
    rng = np.random.default_rng(42)
    while len(specs) < max_attempts:
        idx = len(specs) + 1
        if idx % 8 == 0:
            depth = int(rng.choice([8, 10, 12, 14, 16, 18]))
            specs.append((f"extra_search_{idx}", ("extra", {"max_depth": depth}), ("extra", {"max_depth": depth}), ("extra", {"max_depth": depth})))
            continue
        specs.append(
            (
                f"hgb_search_{idx}",
                ("hgb", random_hgb_params(rng, 100, 260)),
                ("hgb", random_hgb_params(rng, 120, 340)),
                ("hgb", random_hgb_params(rng, 140, 420)),
            )
        )
    return specs[:max_attempts]


def random_hgb_params(rng: np.random.Generator, low_iter: int, high_iter: int) -> dict:
    return {
        "max_iter": int(rng.integers(low_iter, high_iter)),
        "learning_rate": float(rng.uniform(0.015, 0.09)),
        "l2_regularization": float(rng.uniform(0.005, 0.22)),
        "max_leaf_nodes": int(rng.choice([15, 23, 31, 45, 63])),
    }


def build_model(spec: tuple[str, dict]) -> Pipeline:
    kind, params = spec
    if kind == "hgb":
        return make_model(**params)
    if kind == "rf":
        return forest_model("rf", **params)
    return forest_model("extra", **params)


def success_rate(y_true: pd.Series, y_pred: np.ndarray, tolerance: float = SUCCESS_TOLERANCE) -> float:
    mask = y_true.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((np.abs(y_pred[mask.to_numpy()] - y_true[mask].to_numpy()) <= tolerance).mean())


def direction_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    mask = y_true.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask.to_numpy()])).mean())


def validation_score(valid: pd.DataFrame, p1: np.ndarray, p5: np.ndarray, p20: np.ndarray) -> float:
    return float(
        np.nanmean(
            [
                success_rate(valid["target_1d"], p1),
                success_rate(valid["target_5d"], p5),
                success_rate(valid["target_20d"], p20),
            ]
        )
    )


def validation_frame(valid: pd.DataFrame, p1: np.ndarray, p5: np.ndarray, p20: np.ndarray) -> pd.DataFrame:
    out = valid[["date", "code", "target_1d", "target_5d", "target_20d"]].copy()
    out["pred_1d"] = p1
    out["pred_5d"] = p5
    out["pred_20d"] = p20
    out["error_1d"] = (out["pred_1d"] - out["target_1d"]).abs()
    out["error_5d"] = (out["pred_5d"] - out["target_5d"]).abs()
    out["error_20d"] = (out["pred_20d"] - out["target_20d"]).abs()
    out["success_1d"] = out["error_1d"] <= SUCCESS_TOLERANCE
    out["success_5d"] = out["error_5d"] <= SUCCESS_TOLERANCE
    out["success_20d"] = out["error_20d"] <= SUCCESS_TOLERANCE
    return out


def train_models(feature_frame: pd.DataFrame) -> ModelBundle:
    trainable = feature_frame.dropna(subset=["target_1d", "target_5d", "target_20d"]).copy()
    if len(trainable) < 500:
        raise RuntimeError("可训练样本不足，请扩大股票池或历史窗口")

    trainable = trainable.sort_values(["date", "code"])
    split_date = trainable["date"].quantile(0.82)
    train = trainable[trainable["date"] <= split_date]
    valid = trainable[trainable["date"] > split_date]
    if len(valid) < 100:
        split_idx = int(len(trainable) * 0.82)
        train = trainable.iloc[:split_idx]
        valid = trainable.iloc[split_idx:]

    x_train = train[FEATURE_COLUMNS]
    x_valid = valid[FEATURE_COLUMNS]

    best: dict | None = None
    log_rows: list[dict] = []
    for attempt, (name, spec1, spec5, spec20) in enumerate(candidate_model_specs(), start=1):
        try:
            model_1d = build_model(spec1)
            model_5d = build_model(spec5)
            model_20d = build_model(spec20)
            model_1d.fit(x_train, train["target_1d"])
            model_5d.fit(x_train, train["target_5d"])
            model_20d.fit(x_train, train["target_20d"])
            pred_1d = model_1d.predict(x_valid)
            pred_5d = model_5d.predict(x_valid)
            pred_20d = model_20d.predict(x_valid)
            score = validation_score(valid, pred_1d, pred_5d, pred_20d)
            row = {
                "attempt": attempt,
                "model": name,
                "score": score,
                "success_1d": success_rate(valid["target_1d"], pred_1d),
                "success_5d": success_rate(valid["target_5d"], pred_5d),
                "success_20d": success_rate(valid["target_20d"], pred_20d),
                "mae_1d": float(mean_absolute_error(valid["target_1d"], pred_1d)),
                "mae_5d": float(mean_absolute_error(valid["target_5d"], pred_5d)),
                "mae_20d": float(mean_absolute_error(valid["target_20d"], pred_20d)),
            }
            log_rows.append(row)
            if best is None or score > best["score"]:
                best = {
                    **row,
                    "model_1d": model_1d,
                    "model_5d": model_5d,
                    "model_20d": model_20d,
                    "pred_1d": pred_1d,
                    "pred_5d": pred_5d,
                    "pred_20d": pred_20d,
                }
            if attempt >= 100 and score >= 0.95:
                break
        except Exception as exc:
            log_rows.append({"attempt": attempt, "model": name, "score": float("nan"), "error": str(exc)})
            continue

    if best is None:
        raise RuntimeError("所有模型训练均失败")

    validation = validation_frame(valid, best["pred_1d"], best["pred_5d"], best["pred_20d"])
    metrics: dict[str, float | str] = {
        "train_samples": float(len(train)),
        "valid_samples": float(len(valid)),
        "validation_days": float(valid["date"].nunique()),
        "optimization_attempts": float(len(log_rows)),
        "best_model": str(best["model"]),
        "best_validation_score": float(best["score"]),
        "success_1d": float(best["success_1d"]),
        "success_5d": float(best["success_5d"]),
        "success_20d": float(best["success_20d"]),
        "mae_1d": float(best["mae_1d"]),
        "mae_5d": float(best["mae_5d"]),
        "mae_20d": float(best["mae_20d"]),
        "rmse_1d": float(mean_squared_error(valid["target_1d"], best["pred_1d"]) ** 0.5),
        "rmse_5d": float(mean_squared_error(valid["target_5d"], best["pred_5d"]) ** 0.5),
        "rmse_20d": float(mean_squared_error(valid["target_20d"], best["pred_20d"]) ** 0.5),
        "direction_1d": direction_accuracy(valid["target_1d"], best["pred_1d"]),
        "direction_5d": direction_accuracy(valid["target_5d"], best["pred_5d"]),
        "direction_20d": direction_accuracy(valid["target_20d"], best["pred_20d"]),
        "accuracy_gate_95": float(best["success_1d"] >= 0.95 and best["success_5d"] >= 0.95 and best["success_20d"] >= 0.95),
    }
    thresholds = {"risk_score_min": 60.0, "confidence_min": 20.0}
    return ModelBundle(
        model_1d=best["model_1d"],
        model_5d=best["model_5d"],
        model_20d=best["model_20d"],
        metrics=metrics,
        thresholds=thresholds,
        validation=validation,
        optimization_log=pd.DataFrame(log_rows),
    )


def factor_score(row: pd.Series) -> float:
    momentum = (
        0.18 * row.get("ret_1d", 0)
        + 0.24 * row.get("ret_5d", 0)
        + 0.30 * row.get("ret_20d", 0)
        + 0.18 * row.get("ret_60d", 0)
        + 0.10 * row.get("close_to_ma20", 0)
    )
    technical = 0.04 * ((row.get("rsi_14", 50) - 50) / 50) + 0.08 * row.get("macd_hist", 0)
    liquidity = 0.03 * row.get("amount_ratio_20d", 0) + 0.02 * row.get("volume_ratio_20d", 0)
    alternative = (
        0.08 * row.get("fund_flow_score", 0)
        + 0.08 * row.get("event_score", 0)
        + 0.07 * row.get("policy_score", 0)
        + 0.06 * row.get("factory_sentiment_score", 0)
        + 0.05 * row.get("research_score", 0)
        + 0.08 * row.get("public_logic_score", 0)
    )
    risk_penalty = 0.45 * abs(row.get("volatility_20d", 0)) + 0.25 * abs(row.get("max_drawdown_60d", 0))
    return float(momentum + technical + liquidity + alternative - risk_penalty)


def risk_score(row: pd.Series) -> float:
    volatility = abs(row.get("volatility_20d", 0))
    drawdown = abs(row.get("max_drawdown_60d", 0))
    turnover = row.get("turnover_rate", 0)
    amount_ratio = row.get("amount_ratio_20d", 0)
    status = str(row.get("stock_status", "正常"))
    score = 100
    score -= min(volatility * 900, 35)
    score -= min(drawdown * 140, 35)
    score -= min(max(-row.get("risk_event_score", 0), 0) * 18, 18)
    score -= 8 if turnover and turnover > 12 else 0
    score -= 12 if status != "正常" else 0
    score += min(max(amount_ratio, 0) * 8, 8)
    return clamp(float(score), 0, 100)


def risk_level(score: float) -> str:
    if score >= 72:
        return "低风险"
    if score >= 55:
        return "中风险"
    return "高风险"


def label_signal(pred_1d: float, pred_5d: float, pred_20d: float) -> str:
    if pred_1d >= 0.02 and pred_5d >= 0.05 and pred_20d >= 0.15:
        return "强势看多"
    if pred_5d >= 0.03 or pred_20d >= 0.08:
        return "谨慎看多"
    if pred_5d <= -0.03 or pred_20d <= -0.08:
        return "看空"
    return "中性"


def main_factors(row: pd.Series) -> str:
    candidates = [
        ("1日动量", row.get("ret_1d", 0)),
        ("5日动量", row.get("ret_5d", 0)),
        ("20日动量", row.get("ret_20d", 0)),
        ("MA20偏离", row.get("close_to_ma20", 0)),
        ("成交放量", row.get("amount_ratio_20d", 0)),
        ("新闻情绪", row.get("news_score", 0)),
        ("热点排名", row.get("hot_score", 0)),
        ("资金流", row.get("fund_flow_score", 0)),
        ("政策匹配", row.get("policy_score", 0)),
        ("扩产设厂", row.get("factory_sentiment_score", 0)),
        ("公告事件", row.get("event_score", 0)),
        ("研报倾向", row.get("research_score", 0)),
        ("状态风险", -1 if str(row.get("stock_status", "正常")) != "正常" else 0),
    ]
    ranked = sorted(candidates, key=lambda item: abs(0 if pd.isna(item[1]) else item[1]), reverse=True)
    return "、".join(name for name, _ in ranked[:4])


def ensure_alt_columns(out: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "news_score",
        "hot_score",
        "fund_flow_score",
        "event_score",
        "policy_score",
        "factory_sentiment_score",
        "research_score",
        "public_logic_score",
        "risk_event_score",
    ]:
        if column not in out:
            out[column] = 0.0
        out[column] = out[column].fillna(0.0)
    return out


def predict(bundle: ModelBundle, latest: pd.DataFrame) -> pd.DataFrame:
    x = latest[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    out = ensure_alt_columns(latest.copy())
    out["model_pred_1d"] = bundle.model_1d.predict(x)
    out["model_pred_5d"] = bundle.model_5d.predict(x)
    out["model_pred_20d"] = bundle.model_20d.predict(x)
    out["raw_factor_score"] = out.apply(factor_score, axis=1)
    out["risk_score"] = out.apply(risk_score, axis=1)
    out["risk_level"] = out["risk_score"].map(risk_level)
    out["pred_1d"] = out["model_pred_1d"] + 0.002 * out["public_logic_score"] + 0.001 * out["hot_score"]
    out["pred_5d"] = (
        out["model_pred_5d"]
        + 0.004 * out["news_score"]
        + 0.003 * out["hot_score"]
        + 0.006 * out["public_logic_score"]
        + 0.003 * out["factory_sentiment_score"]
        + 0.002 * out["policy_score"]
        + 0.004 * out["risk_event_score"]
    )
    out["pred_20d"] = (
        out["model_pred_20d"]
        + 0.010 * out["news_score"]
        + 0.006 * out["hot_score"]
        + 0.014 * out["public_logic_score"]
        + 0.010 * out["factory_sentiment_score"]
        + 0.008 * out["policy_score"]
        + 0.010 * out["risk_event_score"]
    )
    score_mean = np.nanmean([bundle.metrics.get("success_1d", 0), bundle.metrics.get("success_5d", 0), bundle.metrics.get("success_20d", 0)])
    base_conf = 35 + (score_mean - 0.2) * 55
    out["confidence"] = out.apply(
        lambda row: clamp(
            base_conf + min(abs(row["pred_1d"]) * 160, 8) + min(abs(row["pred_5d"]) * 120, 10) + min(abs(row["pred_20d"]) * 60, 10) - abs(row.get("volatility_20d", 0)) * 120,
            10,
            92,
        ),
        axis=1,
    )
    out["score"] = (
        out["pred_1d"] * 120
        + out["pred_5d"] * 220
        + out["pred_20d"] * 140
        + out["raw_factor_score"] * 45
        + out["public_logic_score"] * 8
        + out["policy_score"] * 5
        + out["factory_sentiment_score"] * 5
        + out["confidence"] * 0.18
        + out["risk_score"] * 0.04
    )
    out["signal"] = [label_signal(a, b, c) for a, b, c in zip(out["pred_1d"], out["pred_5d"], out["pred_20d"])]
    out["meets_target"] = (out["pred_5d"] >= 0.05) & (out["pred_20d"] >= 0.15)
    out["main_factors"] = out.apply(main_factors, axis=1)
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out
