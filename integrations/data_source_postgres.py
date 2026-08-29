"""PostgreSQL stock-history provider (A-share daily candles from market database).

直连 openclaw 的 `market` 库（默认 192.168.10.117:5432/market，经 PG* 环境变量配置）：

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
import threading
import time

import pandas as pd

from integrations.data_source_format import STOCK_HIST_COLUMNS, apply_qfq_factors

logger = logging.getLogger(__name__)

_PG_DEFAULTS = {
    "host": "192.168.10.117",
    "port": "5432",
    "user": "postgres",
    "database": "market",
}

# 连接池：按 (host, port, user, database) 复用单个连接，避免批量扫描时连接耗尽。
_CONN_POOL: dict[tuple, tuple] = {}
_CONN_LOCK = threading.Lock()
_CONN_IDLE_TIMEOUT_SECONDS = 300.0


def _pg_config() -> dict[str, str]:
    """PG 连接配置：环境变量 > ~/.wyckoff/wyckoff.json 的 pg_data_source 段 > 默认值。

    运行时读取（模块 import 时读会让测试/改配置失效）。config 文件路径与
    项目本地配置一致，Windows 桌面端无需设置环境变量即可配 PG 数据源。
    """
    file_cfg = _pg_config_file()
    out: dict[str, str] = {}
    for key, env, default in _pg_env_bindings():
        env_value = os.getenv(env, "").strip()
        file_value = str(file_cfg.get(key, "") or "").strip()
        out[key] = env_value or file_value or default
    return out


def _pg_config_file() -> dict[str, str]:
    try:
        from integrations.local_auth import load_config

        data = load_config()
        return dict(data.get("pg_data_source", {}) or {})
    except Exception:
        return {}


def _save_pg_config(cfg: dict[str, str]) -> None:
    """把 PG 连接写入 ~/.wyckoff/wyckoff.json 的 pg_data_source 段（供桌面端调用）。"""
    from integrations.local_auth import save_config_key

    save_config_key("pg_data_source", {key: str(cfg.get(key, "") or "") for key, _env, _default in _pg_env_bindings()})


def _pg_env_bindings():
    return [
        ("host", "PGHOST", _PG_DEFAULTS["host"]),
        ("port", "PGPORT", _PG_DEFAULTS["port"]),
        ("user", "PGUSER", _PG_DEFAULTS["user"]),
        ("password", "PGPASSWORD", ""),
        ("database", "PGDATABASE", _PG_DEFAULTS["database"]),
    ]


def _connect():
    import psycopg2

    cfg = _pg_config()
    key = (cfg["host"], cfg["port"], cfg["user"], cfg["database"])
    now = time.monotonic()
    with _CONN_LOCK:
        pooled = _CONN_POOL.get(key)
        if pooled and now - pooled[1] < _CONN_IDLE_TIMEOUT_SECONDS:
            return pooled[0]
        conn = psycopg2.connect(
            host=cfg["host"],
            port=int(cfg["port"]),
            user=cfg["user"],
            password=cfg["password"],
            dbname=cfg["database"],
            connect_timeout=5,
        )
        _CONN_POOL[key] = (conn, now)
        return conn


def _release(conn) -> None:
    # 池化连接不真正关闭；查询失败时才丢弃。
    if conn is None:
        return
    with _CONN_LOCK:
        for _key, (pooled, _ts) in list(_CONN_POOL.items()):
            if pooled is conn:
                return
    conn.close()


def _discard(conn) -> None:
    with _CONN_LOCK:
        for key, (pooled, _ts) in list(_CONN_POOL.items()):
            if pooled is conn:
                del _CONN_POOL[key]
                break
    try:
        conn.close()
    except Exception:
        pass


def fetch_stock_postgres(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """从 market 库读取 A 股日线，返回 STOCK_HIST_COLUMNS 中文列。

    start/end 为 YYYYMMDD 字符串。adjust 仅支持 ""(raw)/qfq；hfq 抛错降级。
    """
    if not _is_a_share_symbol(symbol):
        raise RuntimeError("postgres unsupported symbol")
    if str(adjust or "").strip().lower() == "hfq":
        raise RuntimeError("postgres hfq unsupported")
    if not _pg_config()["password"]:
        raise RuntimeError("postgres unconfigured")

    max_date = _latest_date(symbol)
    if not max_date:
        raise RuntimeError("postgres empty")
    if max_date.replace("-", "") < end:
        raise RuntimeError(f"postgres stale max={max_date}")

    candles = _read_candles(symbol, start, end)
    if candles.empty:
        raise RuntimeError("postgres empty window")

    if str(adjust or "").strip().lower() == "qfq":
        candles = apply_qfq_factors(candles, _read_factors(symbol))

    return _to_hist_columns(candles)


def _is_a_share_symbol(symbol: str) -> bool:
    code = str(symbol or "").strip()
    return len(code) == 6 and code.isdigit()


def _latest_date(symbol: str) -> str:
    """该股全量最新日期（YYYY-MM-DD）；stale 判断用，与请求窗口无关。"""
    rows = _query(
        symbol,
        "SELECT MAX(date)::text FROM stock_raw_data WHERE code = %s AND k_period = 'day'",
    )
    if not rows or not rows[0][0]:
        return ""
    return str(rows[0][0])[:10]


def _read_candles(symbol: str, start: str, end: str) -> pd.DataFrame:
    rows = _query(
        symbol,
        """
        SELECT date::text, open, high, low, close, volume, amount
        FROM stock_raw_data
        WHERE code = %s AND k_period = 'day'
          AND date >= %s AND date <= %s
        ORDER BY date
        """,
        (f"{start[:4]}-{start[4:6]}-{start[6:]}", f"{end[:4]}-{end[4:6]}-{end[6:]}"),
    )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    for col in ("open", "high", "low", "close", "volume", "amount"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["date"] = out["date"].astype(str).str[:10]
    return out.sort_values("date").reset_index(drop=True)


def _read_factors(symbol: str) -> pd.DataFrame | None:
    try:
        rows = _query(
            symbol,
            "SELECT date::text, qfq_factor FROM stock_adjustment_factors WHERE code = %s ORDER BY date",
        )
    except Exception as exc:
        # 因子读失败就静默回退 raw，会得到标成 qfq 实为 raw 的数据——必须留痕。
        logger.warning("postgres factor read failed: %s: %s", symbol, exc)
        return None
    if not rows:
        return None
    out = pd.DataFrame(rows, columns=["date", "qfq_factor"])
    out["date"] = out["date"].astype(str).str[:10]
    return out


def _query(symbol: str, sql: str, params: tuple | None = None) -> list[tuple]:
    try:
        conn = _connect()
    except ImportError:
        raise RuntimeError("postgres psycopg2 missing") from None
    except Exception as exc:
        raise RuntimeError(f"postgres connect {type(exc).__name__}") from None
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol,) if params is None else (symbol, *params))
            return cur.fetchall()
    except Exception:
        _discard(conn)
        raise
    finally:
        _release(conn)


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
