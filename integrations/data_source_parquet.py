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

    candles = _read_candles(symbol)
    if candles.empty:
        raise RuntimeError("parquet empty")
    # stale 检查基于全量最新日期：数据源未覆盖到请求结束日时降级实时源。
    max_date = str(candles["date"].max()).replace("-", "")
    if max_date < end:
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
    # 因子按 _key 显式对齐，不依赖 merge_asof 后的位置顺序。
    factor_by_key = merged["qfq_factor"].ffill().bfill().fillna(1.0).replace(0, 1.0)
    factor_by_key = factor_by_key.set_axis(merged["_key"].tolist())
    out = df.copy().reset_index(drop=True)
    out["_key"] = pd.to_datetime(out["date"], errors="coerce")
    factor = out["_key"].map(factor_by_key)
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce") / factor
    # 成交量按因子放大保持成交额一致；amount 不动——复权量×复权价=原成交额，本身就是对账口径。
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
