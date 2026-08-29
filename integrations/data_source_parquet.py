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

from integrations.data_source_format import STOCK_HIST_COLUMNS

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("PARQUET_DATA_DIR", str(Path.home() / "python" / "backtest" / "data")))


def fetch_stock_parquet(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """从本地 Parquet 读取 A 股日线，返回 STOCK_HIST_COLUMNS 中文列。

    start/end 为 YYYYMMDD 字符串。adjust 仅支持 ""(raw)/qfq；hfq 抛错降级。
    """
    if not _is_a_share_symbol(symbol):
        raise RuntimeError("parquet unsupported symbol")
    if str(adjust or "").strip().lower() == "hfq":
        raise RuntimeError("parquet hfq unsupported")

    candles = _read_candles(symbol, start, end)
    if candles.empty:
        raise RuntimeError("parquet empty")
    max_date = str(candles["date"].max()).replace("-", "")
    if max_date < end:
        raise RuntimeError(f"parquet stale max={candles['date'].max()}")

    if str(adjust or "").strip().lower() == "qfq":
        candles = _apply_qfq(candles, symbol)

    return _to_hist_columns(candles)


def _is_a_share_symbol(symbol: str) -> bool:
    code = str(symbol or "").strip()
    return len(code) == 6 and code.isdigit()


def _read_candles(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("parquet pyarrow missing") from None

    start_year = int(start[:4])
    end_year = int(end[:4])
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        path = DATA_DIR / "candles" / f"{year}.parquet"
        if not path.exists():
            continue
        table = pq.read_table(path, filters=[("code", "==", symbol)])
        if table.num_rows:
            frames.append(table.to_pandas())
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.sort_values("date").reset_index(drop=True)


def _apply_qfq(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    factors = _read_factors(symbol)
    if factors is None or factors.empty:
        return df
    left = df.copy()
    left["_key"] = pd.to_datetime(left["date"], errors="coerce")
    right = factors.copy()
    right["_key"] = pd.to_datetime(right["date"], errors="coerce")
    merged = pd.merge_asof(
        left.sort_values("_key"),
        right.sort_values("_key"),
        on="_key",
        direction="backward",
    )
    factor = merged["qfq_factor"].ffill().bfill().fillna(1.0).replace(0, 1.0)
    out = df.copy().reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce") / factor
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce") * factor
    return out


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
