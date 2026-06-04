from __future__ import annotations

import pandas as pd


def factor_direction_notes(row: pd.Series) -> str:
    notes: list[str] = []
    ret_1d = row.get("ret_1d", 0)
    ret_5d = row.get("ret_5d", 0)
    ret_20d = row.get("ret_20d", 0)
    ret_60d = row.get("ret_60d", 0)
    close_to_ma20 = row.get("close_to_ma20", 0)
    ma20_to_ma60 = row.get("ma20_to_ma60", 0)
    amount_ratio = row.get("amount_ratio_20d", 0)
    drawdown = row.get("max_drawdown_60d", 0)
    status = str(row.get("stock_status", "正常"))

    if ret_1d >= 0.03:
        notes.append("1日动量强")
    elif ret_1d <= -0.03:
        notes.append("1日动量弱")
    if ret_5d >= 0.06:
        notes.append("5日动量强")
    elif ret_5d <= -0.06:
        notes.append("5日动量弱")
    if ret_20d >= 0.08:
        notes.append("20日动量强")
    elif ret_20d <= -0.08:
        notes.append("20日动量弱")
    if ret_60d >= 0.15:
        notes.append("60日趋势强")
    elif ret_60d <= -0.15:
        notes.append("60日趋势弱")
    if close_to_ma20 >= 0.04 and ma20_to_ma60 >= 0.02:
        notes.append("均线多头")
    elif close_to_ma20 <= -0.04 and ma20_to_ma60 <= -0.02:
        notes.append("均线空头")
    if amount_ratio >= 0.5:
        notes.append("成交放量")
    elif amount_ratio <= -0.35:
        notes.append("成交收缩")
    if drawdown <= -0.18:
        notes.append("回撤压力大")

    score_notes = [
        ("新闻偏正面", row.get("news_score", 0), 0.25),
        ("热点排名靠前", row.get("hot_score", 0), 0.5),
        ("资金流靠前", row.get("fund_flow_score", 0), 0.65),
        ("公告事件偏正面", row.get("event_score", 0), 0.25),
        ("政策主题匹配", row.get("policy_score", 0), 0.2),
        ("扩产设厂情绪", row.get("factory_sentiment_score", 0), 0.2),
        ("研报倾向正面", row.get("research_score", 0), 0.2),
    ]
    for label, value, threshold in score_notes:
        if value >= threshold:
            notes.append(label)
    if row.get("risk_event_score", 0) <= -0.20:
        notes.append("负面事件压力")
    if status != "正常":
        notes.append(status)
    if not notes:
        notes.append("因子分歧较小")
    return "、".join(notes[:8])


def add_explanations(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out["logic_notes"] = out.apply(factor_direction_notes, axis=1)
    return out
