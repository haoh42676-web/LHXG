from __future__ import annotations

import json
from typing import Any

import pandas as pd
import requests


DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1)
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def analyze_candidates_with_deepseek(
    predictions: pd.DataFrame,
    news: pd.DataFrame,
    api_key: str | None,
    model: str = "deepseek-chat",
    max_candidates: int = 30,
) -> pd.DataFrame:
    out = predictions.copy()
    out["ai_review"] = ""
    out["ai_risk_flags"] = ""
    if not api_key:
        return out

    candidates = out[(out["pred_5d"] >= 0.03) | (out["pred_20d"] >= 0.08)].head(max_candidates)
    if candidates.empty:
        candidates = out.head(min(max_candidates, len(out)))

    news_map: dict[str, list[str]] = {}
    if not news.empty:
        for code, group in news.sort_values("publish_time", ascending=False).groupby("code"):
            news_map[str(code).zfill(6)] = group["title"].dropna().astype(str).head(10).tolist()

    payload_rows = []
    for _, row in candidates.iterrows():
        code = str(row["code"]).zfill(6)
        payload_rows.append(
            {
                "code": code,
                "name": row.get("name", ""),
                "pred_1d": round(float(row.get("pred_1d", 0)), 4),
                "pred_5d": round(float(row.get("pred_5d", 0)), 4),
                "pred_20d": round(float(row.get("pred_20d", 0)), 4),
                "risk_score": round(float(row.get("risk_score", 0)), 2),
                "risk_level": row.get("risk_level", ""),
                "stock_status": row.get("stock_status", ""),
                "logic_notes": row.get("logic_notes", ""),
                "fund_flow_score": round(float(row.get("fund_flow_score", 0)), 4),
                "policy_score": round(float(row.get("policy_score", 0)), 4),
                "factory_sentiment_score": round(float(row.get("factory_sentiment_score", 0)), 4),
                "event_score": round(float(row.get("event_score", 0)), 4),
                "research_score": round(float(row.get("research_score", 0)), 4),
                "risk_event_score": round(float(row.get("risk_event_score", 0)), 4),
                "news_titles": news_map.get(code, []),
            }
        )

    prompt = (
        "你是A股量化研究助手。只基于给定量化结果、公开事件评分和新闻标题复核，不要编造外部事实。"
        "重点关注政策、资金、扩产设厂、订单公告、负面风险、市场情绪是否与量化结论一致。"
        "返回JSON对象，格式为 {\"items\":[{\"code\":\"000001\",\"ai_review\":\"不超过40字\",\"ai_risk_flags\":\"不超过30字\"}]}。"
        "数据如下："
        + json.dumps(payload_rows, ensure_ascii=False)
    )
    response = requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "你只输出严格JSON，不输出Markdown。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 2200,
            "response_format": {"type": "json_object"},
        },
        timeout=90,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if isinstance(parsed, dict):
        parsed = parsed.get("items") or parsed.get("data") or []
    reviews = {str(item.get("code", "")).zfill(6): item for item in parsed if isinstance(item, dict)}

    for idx, row in out.iterrows():
        code = str(row["code"]).zfill(6)
        item = reviews.get(code)
        if item:
            out.at[idx, "ai_review"] = str(item.get("ai_review", ""))
            out.at[idx, "ai_risk_flags"] = str(item.get("ai_risk_flags", ""))
    return out
