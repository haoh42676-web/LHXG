from __future__ import annotations

import concurrent.futures
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd

from .settings import CACHE_DIR, HISTORY_DIR
from .utils import safe_read_csv, to_number, write_csv


EXCHANGE_PREFIX = {
    "SH": ("600", "601", "603", "605", "688", "689"),
    "SZ": ("000", "001", "002", "003", "300", "301"),
}


@dataclass
class FetchStatus:
    ok: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)

    def add_failure(self, symbol: str, error: Exception | str) -> None:
        self.failed += 1
        self.failures.append(f"{symbol}: {error}")


def import_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("缺少 akshare，请先运行 pip install -r requirements.txt") from exc
    return ak


def infer_exchange(code: str) -> str | None:
    code = str(code).zfill(6)
    for exchange, prefixes in EXCHANGE_PREFIX.items():
        if code.startswith(prefixes):
            return exchange
    return None


def tx_symbol(code: str) -> str:
    exchange = infer_exchange(code)
    if exchange == "SH":
        return f"sh{str(code).zfill(6)}"
    if exchange == "SZ":
        return f"sz{str(code).zfill(6)}"
    return str(code).zfill(6)


def normalize_spot(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    rename_map = {
        "代码": "code",
        "名称": "name",
        "最新价": "latest_price",
        "涨跌额": "change",
        "涨跌幅": "pct_change",
        "总市值": "market_cap",
        "流通市值": "float_market_cap",
        "换手率": "turnover_rate",
        "成交额": "amount",
        "成交量": "volume",
        "买入": "bid",
        "卖出": "ask",
        "昨收": "prev_close",
        "今开": "open",
        "最高": "high",
        "最低": "low",
        "时间戳": "quote_time",
    }
    df = raw.rename(columns=rename_map).copy()
    if "code" not in df or "name" not in df:
        raise RuntimeError(f"{source} 行情缺少代码或名称字段")
    df["code"] = df["code"].astype(str).str.extract(r"(\d{6})")[0].str.zfill(6)
    df = df[df["code"].notna()].copy()
    df["exchange"] = df["code"].map(infer_exchange)
    df = df[df["exchange"].notna()].copy()
    for column in [
        "latest_price",
        "change",
        "pct_change",
        "market_cap",
        "float_market_cap",
        "turnover_rate",
        "amount",
        "volume",
        "bid",
        "ask",
        "prev_close",
        "open",
        "high",
        "low",
    ]:
        if column in df:
            df[column] = to_number(df[column])
    if "market_cap" not in df:
        df["market_cap"] = pd.NA
    if "float_market_cap" not in df:
        df["float_market_cap"] = df["market_cap"]
    if "turnover_rate" not in df:
        df["turnover_rate"] = 0.0
    if "amount" not in df:
        df["amount"] = 0.0
    if "volume" not in df:
        df["volume"] = 0.0
    df["pool_source"] = source
    return df.drop_duplicates(subset=["code"]).reset_index(drop=True)


def stock_status_from_spot(row: pd.Series) -> str:
    flags: list[str] = []
    name = str(row.get("name", ""))
    latest = row.get("latest_price")
    amount = row.get("amount")
    volume = row.get("volume")
    if "ST" in name.upper() or "退" in name:
        flags.append("ST/退市风险")
    if pd.isna(latest) or float(latest or 0) <= 0 or (float(amount or 0) <= 0 and float(volume or 0) <= 0):
        flags.append("停牌或无成交")
    return "、".join(flags) if flags else "正常"


def fetch_all_a_stocks(refresh: bool = False) -> pd.DataFrame:
    cache_path = CACHE_DIR / "spot_all.csv"
    if not refresh:
        cached = safe_read_csv(cache_path, dtype={"code": str})
        if not cached.empty and len(cached) > 1000:
            return cached

    ak = import_akshare()
    errors: list[str] = []
    raw = pd.DataFrame()
    source = "eastmoney"
    try:
        raw = ak.stock_zh_a_spot_em()
        source = "eastmoney"
    except Exception as exc:
        errors.append(f"eastmoney spot failed: {exc}")
        try:
            raw = ak.stock_zh_a_spot()
            source = "sina"
        except Exception as fallback_exc:
            errors.append(f"sina spot failed: {fallback_exc}")

    if raw.empty:
        raise RuntimeError("未能抓取沪深 A 股股票池；" + "；".join(errors))

    df = normalize_spot(raw, source=source)
    if df["market_cap"].notna().any():
        df = df.sort_values("market_cap", ascending=False, na_position="last")
    else:
        df = df.sort_values("amount", ascending=False, na_position="last")
        df["market_cap"] = df["amount"].fillna(0)
        df["float_market_cap"] = df["market_cap"]
    df["pool_rank"] = range(1, len(df) + 1)
    df["stock_status"] = df.apply(stock_status_from_spot, axis=1)
    write_csv(df, cache_path)
    write_csv(df, CACHE_DIR / "spot.csv")
    return df


def fetch_top_market_cap_stocks(top_n: int | None = 500, refresh: bool = False) -> pd.DataFrame:
    stocks = fetch_all_a_stocks(refresh=refresh)
    if top_n is None or top_n <= 0 or top_n >= len(stocks):
        return stocks.reset_index(drop=True)
    return stocks.head(top_n).reset_index(drop=True)


def fetch_stock_history(
    code: str,
    start_date: date,
    end_date: date,
    adjust: str = "qfq",
    refresh: bool = False,
) -> pd.DataFrame:
    safe_adjust = adjust or "none"
    cache_path = HISTORY_DIR / f"{str(code).zfill(6)}_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{safe_adjust}.csv"
    if not refresh:
        cached = safe_read_csv(cache_path, dtype={"code": str})
        if not cached.empty:
            cached["date"] = pd.to_datetime(cached["date"])
            return cached

    ak = import_akshare()
    errors: list[str] = []
    raw = pd.DataFrame()
    source = "eastmoney"
    for attempt in range(3):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=str(code).zfill(6),
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust=adjust,
            )
            source = "eastmoney"
            if not raw.empty:
                break
        except Exception as exc:
            errors.append(f"eastmoney#{attempt + 1}: {exc}")
            time.sleep(0.3 * (attempt + 1))

    if raw.empty:
        try:
            raw = ak.stock_zh_a_hist_tx(
                symbol=tx_symbol(code),
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust=adjust,
                timeout=20,
            )
            source = "tencent"
        except Exception as exc:
            errors.append(f"tencent: {exc}")
    if raw.empty:
        try:
            import efinance as ef

            raw = ef.stock.get_quote_history(
                str(code).zfill(6),
                beg=start_date.strftime("%Y%m%d"),
                end=end_date.strftime("%Y%m%d"),
                klt=101,
                fqt=1 if adjust == "qfq" else 0,
                suppress_error=True,
            )
            source = "efinance"
        except Exception as exc:
            errors.append(f"efinance: {exc}")
    if raw.empty:
        try:
            raw = fetch_baostock_history(code, start_date, end_date, adjust)
            source = "baostock"
        except Exception as exc:
            errors.append(f"baostock: {exc}")
    if raw.empty:
        try:
            raw = fetch_tushare_history(code, start_date, end_date, adjust)
            source = "tushare"
        except Exception as exc:
            errors.append(f"tushare: {exc}")
    if raw.empty:
        raise RuntimeError("; ".join(errors))
    if raw.empty:
        return pd.DataFrame()

    df = raw.rename(
        columns={
            "日期": "date",
            "日期 ": "date",
            "股票代码": "code",
            "股票名称": "name",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover_rate",
            "date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
            "amount": "amount",
            "turn": "turnover_rate",
            "pctChg": "pct_change",
        }
    ).copy()
    df["code"] = str(code).zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    for missing in ["volume", "amount", "amplitude", "pct_change", "change", "turnover_rate"]:
        if missing not in df:
            df[missing] = 0.0
    for column in ["open", "close", "high", "low", "volume", "amount", "amplitude", "pct_change", "change", "turnover_rate"]:
        if column in df:
            df[column] = to_number(df[column])
    df["history_source"] = source
    df = df.dropna(subset=["date", "open", "close", "high", "low"]).sort_values("date").reset_index(drop=True)
    write_csv(df, cache_path)
    return df


def bs_code(code: str) -> str:
    exchange = infer_exchange(code)
    prefix = "sh" if exchange == "SH" else "sz"
    return f"{prefix}.{str(code).zfill(6)}"


def fetch_baostock_history(code: str, start_date: date, end_date: date, adjust: str = "qfq") -> pd.DataFrame:
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    try:
        adjustflag = "2" if adjust == "qfq" else "3" if adjust == "hfq" else "1"
        rs = bs.query_history_k_data_plus(
            bs_code(code),
            "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag=adjustflag,
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            raise RuntimeError(rs.error_msg)
        return pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()


def fetch_tushare_history(code: str, start_date: date, end_date: date, adjust: str = "qfq") -> pd.DataFrame:
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        return pd.DataFrame()
    import tushare as ts

    ts.set_token(token)
    pro = ts.pro_api()
    exchange = infer_exchange(code)
    suffix = ".SH" if exchange == "SH" else ".SZ"
    ts_code = f"{str(code).zfill(6)}{suffix}"
    df = ts.pro_bar(
        ts_code=ts_code,
        adj=adjust if adjust in {"qfq", "hfq"} else None,
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
    )
    if df is None or df.empty:
        return pd.DataFrame()
    return df.rename(columns={"trade_date": "date", "vol": "volume", "pct_chg": "pct_change"})


def history_status(hist: pd.DataFrame, lookback_days: int) -> str:
    flags: list[str] = []
    if hist.empty:
        return "历史数据缺失"
    if len(hist) < min(120, lookback_days * 0.25):
        flags.append("上市不足一年或样本过少")
    expected_min = max(60, int(lookback_days / 365 * 180))
    if len(hist) < expected_min:
        flags.append("数据缺口较大")
    latest = hist["date"].max()
    if pd.notna(latest) and latest.date() < datetime.now().date() - timedelta(days=14):
        flags.append("近期停牌或行情滞后")
    return "、".join(flags) if flags else "历史正常"


def fetch_histories(
    stocks: pd.DataFrame,
    lookback_days: int = 1825,
    adjust: str = "qfq",
    refresh: bool = False,
    max_workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, FetchStatus]:
    end = datetime.now().date()
    start = end - timedelta(days=lookback_days)
    status = FetchStatus()
    frames: list[pd.DataFrame] = []
    workers = max_workers or int(os.getenv("HISTORY_MAX_WORKERS", "8"))

    def job(row: pd.Series) -> pd.DataFrame:
        time.sleep(0.04)
        hist = fetch_stock_history(row["code"], start, end, adjust=adjust, refresh=refresh)
        if hist.empty:
            return hist
        hist["name"] = row.get("name")
        hist["exchange"] = row.get("exchange")
        hist["market_cap"] = row.get("market_cap")
        hist["pool_rank"] = row.get("pool_rank")
        hist["pool_source"] = row.get("pool_source")
        spot_status = str(row.get("stock_status", "正常"))
        h_status = history_status(hist, lookback_days)
        if spot_status == "正常" and h_status == "历史正常":
            combined = "正常"
        else:
            combined = "、".join(flag for flag in [spot_status if spot_status != "正常" else "", h_status if h_status != "历史正常" else ""] if flag)
        hist["stock_status"] = combined
        return hist

    rows = [row for _, row in stocks.iterrows()]
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_code = {executor.submit(job, row): row["code"] for row in rows}
        for future in concurrent.futures.as_completed(future_to_code):
            code = future_to_code[future]
            try:
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
                    status.ok += 1
                else:
                    status.add_failure(code, "empty history")
            except Exception as exc:
                status.add_failure(code, exc)
            done += 1
            if progress is not None:
                progress(done, len(rows))

    if not frames:
        return pd.DataFrame(), status
    return pd.concat(frames, ignore_index=True), status
