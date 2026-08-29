"""Local Parquet stock-history provider (A-share candles exported from PostgreSQL).

数据来自 openclaw_paper_trade 的 `~/python/backtest/data/`（凌晨由 PostgreSQL 导出）：

    candles/{year}.parquet              code, date, open, high, low, close, volume, amount
    factors/adjustment_factors.parquet  code, date, qfq_factor, hfq_factor

复权语义与 openclaw 生产代码一致：前复权价 = 原始价 / qfq_factor，
volume 按同因子缩放以保持成交额一致；缺失因子按 1.0 处理（即不复权）。

只服务 A 股（6 位纯数字 code）。港股/美股、hfq 请求、数据未覆盖到请求
结束日、或 pyarrow 不可用时抛错，由上层降级到 tickflow 等实时源。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from integrations.data_source_format import (
    MARKET_OVERVIEW_INDICES,
    STOCK_HIST_COLUMNS,
    apply_qfq_factors,
    data_covers_end,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("PARQUET_DATA_DIR", str(Path.home() / "python" / "backtest" / "data")))


def fetch_index_parquet(symbol: str, days: int) -> pd.DataFrame:
    """从本地 Parquet 读取指数历史日线，返回标准列 DataFrame。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("parquet pyarrow missing") from None

    path = DATA_DIR / "index" / "index_daily.parquet"
    if not path.exists():
        raise RuntimeError("parquet index file missing")

    code = str(symbol).split(".", 1)[0].strip()
    table = pq.read_table(
        path,
        filters=[("index_code", "==", code), ("k_period", "==", "day")],
        columns=["date", "open", "high", "low", "close", "volume", "amount"],
    )
    if not table.num_rows:
        raise RuntimeError(f"parquet index {symbol} empty")

    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    if df.empty:
        raise RuntimeError(f"parquet index {symbol} empty")

    max_date = str(df["date"].max()).replace("-", "")
    today_str = pd.Timestamp.now().strftime("%Y%m%d")
    if not data_covers_end(max_date, today_str):
        raise RuntimeError(f"parquet index stale max={df['date'].max()}")

    return df.sort_values("date").tail(days * 2).reset_index(drop=True)


def fetch_market_overview_parquet(requested: str = "") -> dict:
    """从 index_daily.parquet 提取大盘指数截面与涨跌家数。"""
    df = _read_index_daily_parquet()
    target_end = requested or pd.Timestamp.now().strftime("%Y%m%d")
    max_available = str(df["date"].max()).replace("-", "")
    if not data_covers_end(max_available, target_end):
        raise RuntimeError(f"parquet index stale max={df['date'].max()}")

    req_iso = f"{requested[:4]}-{requested[4:6]}-{requested[6:]}" if requested else ""
    df_sub = df[df["date"] <= req_iso] if req_iso else df
    if df_sub.empty:
        raise RuntimeError("parquet index empty for requested date")

    actual_date = str(df_sub["date"].max())
    df_actual = df_sub[df_sub["date"] == actual_date].drop_duplicates("index_code", keep="last")
    indices = _build_parquet_indices_summary(df, df_actual, actual_date)
    res = {
        "indices": indices,
        "source": "parquet",
        "requested_date": requested,
        "trade_date": actual_date,
    }
    breadth = _build_parquet_breadth(df_actual, actual_date)
    if breadth:
        res["breadth"] = breadth
    return res


def _read_index_daily_parquet() -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("parquet pyarrow missing") from None

    path = DATA_DIR / "index" / "index_daily.parquet"
    if not path.exists():
        raise RuntimeError("parquet index file missing")

    cols = ["index_code", "date", "open", "high", "low", "close", "volume", "amount", "up_count", "down_count"]
    table = pq.read_table(path, filters=[("k_period", "==", "day")], columns=cols)
    if not table.num_rows:
        raise RuntimeError("parquet index empty")

    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "volume", "amount", "up_count", "down_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_parquet_indices_summary(df: pd.DataFrame, df_actual: pd.DataFrame, actual_date: str) -> dict:
    indices = {}
    for ts_code, name in MARKET_OVERVIEW_INDICES.items():
        code = ts_code.split(".", 1)[0]
        rows = df_actual[df_actual["index_code"] == code]
        if rows.empty:
            indices[name] = {"error": "no data"}
            continue
        row = rows.iloc[0]
        hist = df[(df["index_code"] == code) & (df["date"] < actual_date)]
        pct_chg = 0.0
        if not hist.empty:
            prev = float(hist.sort_values("date").iloc[-1]["close"])
            if prev > 0:
                pct_chg = round((float(row["close"]) / prev - 1.0) * 100.0, 2)

        indices[name] = {
            "ts_code": ts_code,
            "trade_date": actual_date,
            "close": round(float(row["close"]), 2),
            "pct_chg": pct_chg,
            "vol": int(float(row["volume"] or 0)),
            "amount": round(float(row["amount"] or 0), 2),
        }
    return indices


def _build_parquet_breadth(df_actual: pd.DataFrame, actual_date: str) -> dict | None:
    breadth_rows = df_actual[df_actual["up_count"].notna() & (df_actual["up_count"] > 0)]
    if breadth_rows.empty:
        return None
    b_row = breadth_rows.iloc[0]
    up = int(b_row["up_count"])
    down = int(b_row["down_count"])
    tot = up + down
    return {
        "trade_date": actual_date,
        "sample_size": tot,
        "up_count": up,
        "down_count": down,
        "flat_count": 0,
        "up_ratio_pct": round(up / tot * 100.0, 2) if tot else None,
        "median_pct_chg": None,
        "average_pct_chg": None,
        "note": "flat_count not tracked in parquet summary",
    }


def fetch_stock_parquet(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """从本地 Parquet 读取 A 股日线，返回 STOCK_HIST_COLUMNS 中文列。

    start/end 为 YYYYMMDD 字符串。adjust 仅支持 ""(raw)/qfq；hfq 抛错降级。
    """
    if not _is_a_share_symbol(symbol):
        raise RuntimeError("parquet unsupported symbol")
    if str(adjust or "").strip().lower() == "hfq":
        raise RuntimeError("parquet hfq unsupported")

    candles = _read_candles(symbol)
    if candles.empty:
        raise RuntimeError("parquet empty")
    # stale 检查基于全量最新日期：数据源未覆盖到请求结束日时降级实时源。
    max_date = str(candles["date"].max()).replace("-", "")
    if not data_covers_end(max_date, end):
        raise RuntimeError(f"parquet stale max={candles['date'].max()}")

    candles = _slice_by_window(candles, start, end)
    if candles.empty:
        raise RuntimeError("parquet empty window")

    if str(adjust or "").strip().lower() == "qfq":
        candles = _apply_qfq(candles, symbol)

    return _to_hist_columns(candles)


def _is_a_share_symbol(symbol: str) -> bool:
    code = str(symbol or "").strip()
    return len(code) == 6 and code.isdigit()


def _read_candles(symbol: str) -> pd.DataFrame:
    """读取该股全部年份 K 线（stale 判断与窗口切片由调用方做）。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("parquet pyarrow missing") from None

    candles_dir = DATA_DIR / "candles"
    if not candles_dir.is_dir():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(candles_dir.glob("*.parquet")):
        table = pq.read_table(path, filters=[("code", "==", symbol)])
        if table.num_rows:
            frames.append(table.to_pandas())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.sort_values("date").reset_index(drop=True)


def _slice_by_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """按请求窗口过滤（YYYYMMDD），范围外数据不得返回。"""
    start_s = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    end_s = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    out = df[(df["date"] >= start_s) & (df["date"] <= end_s)]
    return out.sort_values("date").reset_index(drop=True)


def _apply_qfq(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return apply_qfq_factors(df, _read_factors(symbol))


def _read_factors(symbol: str) -> pd.DataFrame | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    path = DATA_DIR / "factors" / "adjustment_factors.parquet"
    if not path.exists():
        return None
    table = pq.read_table(path, filters=[("code", "==", symbol)])
    if table.num_rows == 0:
        return None
    out = table.to_pandas()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out[["date", "qfq_factor"]]


def _to_hist_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "日期": df["date"],
        "开盘": pd.to_numeric(df.get("open"), errors="coerce"),
        "最高": pd.to_numeric(df.get("high"), errors="coerce"),
        "最低": pd.to_numeric(df.get("low"), errors="coerce"),
        "收盘": pd.to_numeric(df.get("close"), errors="coerce"),
        "成交量": pd.to_numeric(df.get("volume"), errors="coerce"),
        "成交额": pd.to_numeric(df.get("amount"), errors="coerce") if "amount" in df.columns else pd.NA,
        "涨跌幅": pd.NA,
        "换手率": pd.NA,
        "振幅": pd.NA,
    }
    return pd.DataFrame(columns, index=df.index).reindex(columns=list(STOCK_HIST_COLUMNS))
