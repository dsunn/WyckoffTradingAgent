"""PostgreSQL stock-history provider (A-share daily candles from market database).

直连 openclaw 的 `market` 库（192.168.10.117:5432/market，经 PG* 环境变量配置）：

    stock_raw_data             code, date, open, high, low, close, volume, amount (k_period='day')
    stock_adjustment_factors   code, date, qfq_factor, hfq_factor

复权语义与 parquet provider / openclaw 生产代码一致：前复权价 = 原始价 / qfq_factor，
volume 按同因子缩放以保持成交额一致；缺失因子按 1.0 处理（即不复权）。

只服务 A 股（6 位纯数字 code）。港股/美股、hfq 请求、数据未覆盖到请求结束日、
PG 不可达或 psycopg2 不可用时抛错，由上层降级到 tickflow 等实时源。
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from integrations.data_source_format import STOCK_HIST_COLUMNS

logger = logging.getLogger(__name__)

PG_HOST = os.getenv("PGHOST", "192.168.10.117")
PG_PORT = int(os.getenv("PGPORT", "5432"))
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASSWORD = os.getenv("PGPASSWORD", "")
PG_DATABASE = os.getenv("PGDATABASE", "market")


def _pg_password() -> str:
    """运行时读 PGPASSWORD（模块 import 时读会让测试/改环境变量失效）。"""
    return os.getenv("PGPASSWORD", "").strip()


def fetch_stock_postgres(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """从 market 库读取 A 股日线，返回 STOCK_HIST_COLUMNS 中文列。

    start/end 为 YYYYMMDD 字符串。adjust 仅支持 ""(raw)/qfq；hfq 抛错降级。
    """
    if not _is_a_share_symbol(symbol):
        raise RuntimeError("postgres unsupported symbol")
    if str(adjust or "").strip().lower() == "hfq":
        raise RuntimeError("postgres hfq unsupported")
    if not _pg_password():
        raise RuntimeError("postgres unconfigured")

    candles = _read_candles(symbol)
    if candles.empty:
        raise RuntimeError("postgres empty")
    max_date = str(candles["date"].max()).replace("-", "")
    if max_date < end:
        raise RuntimeError(f"postgres stale max={candles['date'].max()}")

    candles = _slice_by_window(candles, start, end)
    if candles.empty:
        raise RuntimeError("postgres empty window")

    if str(adjust or "").strip().lower() == "qfq":
        candles = _apply_qfq(candles, symbol)

    return _to_hist_columns(candles)


def _is_a_share_symbol(symbol: str) -> bool:
    code = str(symbol or "").strip()
    return len(code) == 6 and code.isdigit()


def _connect():
    import psycopg2

    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=_pg_password(),
        dbname=PG_DATABASE,
        connect_timeout=5,
    )


def _read_candles(symbol: str) -> pd.DataFrame:
    try:
        conn = _connect()
    except ImportError:
        raise RuntimeError("postgres psycopg2 missing") from None
    except Exception as exc:
        raise RuntimeError(f"postgres connect {type(exc).__name__}") from None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, open, high, low, close, volume, amount
                FROM stock_raw_data
                WHERE code = %s AND k_period = 'day'
                ORDER BY date
                """,
                (symbol,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    for col in ("open", "high", "low", "close", "volume", "amount"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.sort_values("date").reset_index(drop=True)


def _read_factors(symbol: str) -> pd.DataFrame | None:
    try:
        conn = _connect()
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, qfq_factor
                FROM stock_adjustment_factors
                WHERE code = %s
                ORDER BY date
                """,
                (symbol,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    out = pd.DataFrame(rows, columns=["date", "qfq_factor"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


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
    # PG numeric 列经 psycopg2 读出是 Decimal，先转 float 再参与除法。
    factor_by_key = merged["qfq_factor"].astype(float).ffill().bfill().fillna(1.0).replace(0, 1.0)
    factor_by_key = factor_by_key.set_axis(merged["_key"].tolist())
    out = df.copy().reset_index(drop=True)
    out["_key"] = pd.to_datetime(out["date"], errors="coerce")
    factor = out["_key"].map(factor_by_key)
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce") / factor
    # 成交量按因子放大保持成交额一致；amount 不动——复权量×复权价=原成交额，本身就是对账口径。
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce") * factor
    return out


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
