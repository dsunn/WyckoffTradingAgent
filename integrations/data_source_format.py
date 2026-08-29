"""Shared stock-history formatting helpers."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

import pandas as pd


def compact_error(exc: Exception, max_len: int = 120) -> str:
    msg = str(exc or "").strip().replace("\n", " ")
    msg = re.sub(r"\s+", " ", msg)
    if len(msg) > max_len:
        msg = msg[: max_len - 3] + "..."
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


STOCK_HIST_COLUMNS = (
    "日期",
    "开盘",
    "最高",
    "最低",
    "收盘",
    "成交量",
    "成交额",
    "涨跌幅",
    "换手率",
    "振幅",
)
SH_PREFIXES = (
    "600",
    "601",
    "603",
    "605",
    "688",
    "510",
    "511",
    "512",
    "513",
    "515",
    "516",
    "518",
    "560",
    "561",
    "562",
    "563",
)


def to_ts_code(symbol: str) -> str:
    code = str(symbol).strip()
    if "." in code:
        return code
    return f"{code}.SH" if code.startswith(SH_PREFIXES) else f"{code}.SZ"


def tag_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df.attrs["source"] = source
    return df


def apply_qfq_factors(df: pd.DataFrame, factors: pd.DataFrame | None) -> pd.DataFrame:
    """按复权因子前复权 OHLCV：价÷因子、量×因子；amount 不动。

    与 openclaw 生产语义一致。factors 为 None/空时原样返回（即不复权）。
    PG numeric 列经驱动读出是 Decimal，先转 float 再参与除法。
    """
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


def hist_date_text(value: str | date) -> str:
    return value.strftime("%Y%m%d") if isinstance(value, date) else str(value).replace("-", "")


def normalize_efinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "日期" not in work.columns:
        for column in work.columns:
            if str(column).endswith("日期") or "日期" in str(column):
                work.rename(columns={column: "日期"}, inplace=True)
                break
    for standard in STOCK_HIST_COLUMNS:
        _rename_prefixed_column(work, standard)
    for column in ["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "涨跌幅"]:
        if column not in work.columns:
            raise RuntimeError(f"efinance missing column {column}")
    for column in ["换手率", "振幅"]:
        if column not in work.columns:
            work = work.assign(**{column: pd.NA})
    work["日期"] = pd.to_datetime(work["日期"]).dt.strftime("%Y-%m-%d")
    return work[list(STOCK_HIST_COLUMNS)].copy()


def tickflow_daily_window(start: str, end: str) -> tuple[date, date, int, int]:
    try:
        start_d = datetime.strptime(start, "%Y%m%d").date()
        end_d = datetime.strptime(end, "%Y%m%d").date()
    except Exception as exc:
        raise RuntimeError(f"tickflow date parse failed: {start}..{end}") from exc
    if end_d < start_d:
        raise RuntimeError(f"tickflow invalid range: {start}..{end}")
    cn_tz = timezone(timedelta(hours=8))
    start_dt = datetime.combine(start_d, datetime.min.time(), tzinfo=cn_tz)
    end_dt = datetime.combine(end_d + timedelta(days=1), datetime.min.time(), tzinfo=cn_tz) - timedelta(milliseconds=1)
    return start_d, end_d, int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)


def tickflow_daily_count(start_d: date, end_d: date, max_count: int) -> int:
    day_span = (end_d - start_d).days + 1
    return min(max(day_span * 2 + 16, 64), max(int(max_count), 64))


def tickflow_adjust_mode(adjust: str) -> str:
    adjust_norm = str(adjust or "").strip().lower()
    mapped = {
        "": "none",
        "none": "none",
        "qfq": "forward",
        "forward": "forward",
        "hfq": "backward",
        "backward": "backward",
    }
    return mapped.get(adjust_norm, "forward")


def tickflow_daily_frame(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    out = df[(df["date"] >= start_d.isoformat()) & (df["date"] <= end_d.isoformat())].copy()
    if out.empty:
        raise RuntimeError("tickflow empty in range")
    close = pd.to_numeric(out.get("close"), errors="coerce")
    prev_close = pd.to_numeric(out.get("prev_close"), errors="coerce")
    prev_ref = prev_close.where(prev_close > 0)
    if prev_ref.notna().sum() == 0:
        prev_ref = close.shift(1)
    result = pd.DataFrame(
        {
            "日期": out["date"],
            "开盘": pd.to_numeric(out.get("open"), errors="coerce"),
            "最高": pd.to_numeric(out.get("high"), errors="coerce"),
            "最低": pd.to_numeric(out.get("low"), errors="coerce"),
            "收盘": close,
            "成交量": pd.to_numeric(out.get("volume"), errors="coerce"),
            "成交额": pd.to_numeric(out.get("amount"), errors="coerce"),
            "涨跌幅": (close / prev_ref - 1.0) * 100.0,
            "换手率": pd.NA,
            "振幅": _tickflow_amplitude(out, prev_ref),
        }
    )
    return result[list(STOCK_HIST_COLUMNS)].copy()


def _rename_prefixed_column(df: pd.DataFrame, standard: str) -> None:
    if standard in df.columns:
        return
    for column in df.columns:
        if str(column).startswith(standard):
            df.rename(columns={column: standard}, inplace=True)
            return


def _tickflow_amplitude(df: pd.DataFrame, prev_ref: pd.Series) -> pd.Series:
    high = pd.to_numeric(df.get("high"), errors="coerce")
    low = pd.to_numeric(df.get("low"), errors="coerce")
    return (high - low) / prev_ref * 100.0
