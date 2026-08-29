from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from agents.tool_context import ToolContext, ensure_tushare_token, get_credential
from utils.safe import safe_float

logger = logging.getLogger(__name__)

from integrations.data_source_format import (
    MARKET_HISTORY_ALIASES,
    MARKET_HISTORY_INDEXES,
    MARKET_OVERVIEW_INDICES,
)


def get_market_overview(
    trade_date: str = "", include_breadth: bool = False, tool_context: ToolContext | None = None
) -> dict:
    try:
        errors: list[str] = []
        requested = _normalize_trade_date(trade_date)
        parquet_result = _fetch_parquet_index_overview(errors, requested)
        if parquet_result:
            return parquet_result
        pg_result = _fetch_pg_index_overview(errors, requested, include_breadth or bool(requested))
        if pg_result:
            return pg_result
        tushare_result = _fetch_tushare_overview(tool_context, errors, requested, include_breadth or bool(requested))
        if tushare_result:
            return tushare_result
        if requested:
            return {"error": "无法获取指定日期市场数据", "requested_date": requested, "details": "; ".join(errors)}
        akshare_result = _fetch_akshare_overview(errors)
        if akshare_result:
            return {
                "indices": akshare_result,
                "source": "akshare",
                "trade_date": "",
                "freshness": "unknown",
                "warning": "AkShare 实时截面不提供可验证交易日期，不得视为今日已确认数据",
            }
        return {"error": "无法获取大盘数据", "details": "; ".join(errors) if errors else "unknown"}
    except Exception as e:
        logger.exception("get_market_overview error")
        return {"error": str(e)}


def _normalize_trade_date(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    raise ValueError("trade_date 必须是 YYYY-MM-DD 或 YYYYMMDD")


def _fetch_parquet_index_overview(errors: list[str], requested: str) -> dict | None:
    """从本地 index_daily.parquet 读指数日线及大盘水温。"""
    try:
        from utils.env import env_flag

        if env_flag("DATA_SOURCE_DISABLE_PARQUET"):
            errors.append("parquet: disabled_by_env")
            return None
        import integrations.data_source_parquet as parquet_provider

        return parquet_provider.fetch_market_overview_parquet(requested)
    except Exception as exc:
        errors.append(f"parquet: {exc}")
        return None


def _fetch_pg_index_overview(errors: list[str], requested: str, include_breadth: bool) -> dict | None:
    """从本地 PG market 库读指数日线（index_raw_data），产出与 tushare 同构结果。

    大盘数据优先走本地库（无需 TUSHARE_TOKEN/网络），tushare 作为兜底。
    代码映射：tushare 的 000001.SH 对应 PG 的 index_code 000001 + market SH。
    """
    try:
        import integrations.data_source_postgres as pg_provider

        if not pg_provider._pg_config()["password"]:
            errors.append("postgres: unconfigured")
            return None
        end_d = datetime.strptime(requested, "%Y%m%d").date() if requested else date.today()
        indices: dict[str, dict] = {}
        for ts_code, name in MARKET_OVERVIEW_INDICES.items():
            code, market = ts_code.split(".")
            rows = pg_provider._query(
                code,
                """
                SELECT date::text, open, high, low, close, volume, amount
                FROM index_raw_data
                WHERE index_code = %s AND market = %s AND k_period = 'day' AND date <= %s
                ORDER BY date DESC
                LIMIT 1
                """,
                (market, end_d.isoformat()),
            )
            if not rows:
                indices[name] = {"error": "no data"}
                continue
            row = rows[0]
            indices[name] = {
                "ts_code": ts_code,
                "trade_date": str(row[0])[:10],
                "close": round(float(row[4]), 2),
                "pct_chg": 0.0,  # 涨跌幅用前一交易日收盘补算
                "vol": int(float(row[5])),
                "amount": round(float(row[6]), 2),
            }
        if not indices:
            return None
        _fill_index_pct_changes(pg_provider, indices)
        actual_dates = [entry.get("trade_date", "") for entry in indices.values() if entry.get("trade_date")]
        actual_date = max(actual_dates) if actual_dates else ""
        result = {
            "indices": indices,
            "source": "postgres",
            "requested_date": requested,
            "trade_date": actual_date,
        }
        if include_breadth and actual_date:
            result["breadth"] = _pg_market_breadth(pg_provider, actual_date)
        return result
    except Exception as exc:
        errors.append(f"postgres: {exc}")
        return None


def _fill_index_pct_changes(pg_provider, indices: dict[str, dict]) -> None:
    """用前一交易日收盘补算各指数涨跌幅（PG 无 pct_chg 列）。"""
    for ts_code, name in MARKET_OVERVIEW_INDICES.items():
        entry = indices.get(name)
        if not entry or "error" in entry:
            continue
        code, market = ts_code.split(".")
        rows = pg_provider._query(
            code,
            """
            SELECT close FROM index_raw_data
            WHERE index_code = %s AND market = %s AND k_period = 'day' AND date < %s
            ORDER BY date DESC LIMIT 1
            """,
            (market, entry["trade_date"]),
        )
        if rows:
            prev = float(rows[0][0])
            if prev > 0:
                entry["pct_chg"] = round((float(entry["close"]) / prev - 1.0) * 100.0, 2)


def _pg_market_breadth(pg_provider, trade_date: str) -> dict:
    """从 index_raw_data 的 up_count/down_count 算 breadth（数据源维护的真实统计）。

    000001.SH 是沪市口径、399001.SZ 是深市口径，两市无重叠，相加即全市场。
    相比从 stock_raw_data 全表算（5 千万行扫描），零计算成本且口径权威。
    """
    rows = pg_provider._query(
        "",
        """
        SELECT index_code, up_count, down_count FROM index_raw_data
        WHERE k_period = 'day' AND date::date = %s
          AND index_code IN ('000001', '399001') AND up_count IS NOT NULL
        """,
        (trade_date,),
        use_symbol=False,
    )
    if not rows:
        return {"error": "指定交易日无指数涨跌家数", "trade_date": trade_date}
    up = sum(int(r[1] or 0) for r in rows)
    down = sum(int(r[2] or 0) for r in rows)
    total = up + down
    return {
        "trade_date": trade_date,
        "sample_size": total,
        "up_count": up,
        "down_count": down,
        "flat_count": 0,
        "up_ratio_pct": round(up / total * 100.0, 2) if total else None,
        "median_pct_chg": None,
        "average_pct_chg": None,
    }


def _fetch_tushare_overview(
    tool_context: ToolContext | None, errors: list[str], requested: str, include_breadth: bool
) -> dict | None:
    try:
        ensure_tushare_token(tool_context)
        from integrations.tushare_client import get_pro

        pro = get_pro()
        if pro is None:
            errors.append("tushare: token 未配置或 client 不可用")
            return None
        end = datetime.strptime(requested, "%Y%m%d").date() if requested else date.today()
        end_date = end.strftime("%Y%m%d")
        start_date = (end - timedelta(days=10)).strftime("%Y%m%d")
        indices = _tushare_index_rows(pro, start_date, end_date)
        if not indices:
            return None
        actual_dates = [str(row.get("trade_date") or "") for row in indices.values() if row.get("trade_date")]
        if not actual_dates:
            return None
        actual_date = max(actual_dates)
        result = {
            "indices": indices,
            "source": "tushare",
            "requested_date": requested or end_date,
            "trade_date": actual_date,
        }
        if include_breadth and actual_date:
            result["breadth"] = _tushare_market_breadth(pro, actual_date)
        return result
    except Exception as e:
        errors.append(f"tushare: {e}")
        return None


def _tushare_market_breadth(pro, trade_date: str) -> dict:
    df = pro.daily(trade_date=trade_date)
    if df is None or df.empty or "pct_chg" not in df.columns:
        return {"error": "指定交易日无全市日线截面", "trade_date": trade_date}
    import pandas as pd

    changes = pd.to_numeric(df["pct_chg"], errors="coerce").dropna()
    up = int((changes > 0).sum())
    down = int((changes < 0).sum())
    flat = int((changes == 0).sum())
    total = int(len(changes))
    return {
        "trade_date": trade_date,
        "sample_size": total,
        "up_count": up,
        "down_count": down,
        "flat_count": flat,
        "up_ratio_pct": round(up / total * 100.0, 2) if total else None,
        "median_pct_chg": round(float(changes.median()), 2) if total else None,
        "average_pct_chg": round(float(changes.mean()), 2) if total else None,
        "up_5pct_count": int((changes >= 5).sum()),
        "down_5pct_count": int((changes <= -5).sum()),
    }


def _tushare_index_rows(pro, start_date: str, end_date: str) -> dict[str, dict]:
    result = {}
    for ts_code, name in MARKET_OVERVIEW_INDICES.items():
        try:
            df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                result[name] = _tushare_latest_row(ts_code, df)
        except Exception as e:
            result[name] = {"error": str(e)}
    return result


def _tushare_latest_row(ts_code: str, df) -> dict:
    latest = df.sort_values("trade_date").iloc[-1]
    return {
        "ts_code": ts_code,
        "trade_date": str(latest.get("trade_date", "")),
        "close": round(float(latest.get("close", 0)), 2),
        "pct_chg": round(float(latest.get("pct_chg", 0)), 2),
        "vol": int(latest.get("vol", 0)),
        "amount": round(float(latest.get("amount", 0)), 2),
    }


def _fetch_akshare_overview(errors: list[str]) -> dict[str, dict] | None:
    try:
        import akshare as ak

        spot = ak.stock_zh_index_spot_em()
        if spot is None or spot.empty:
            errors.append("akshare: stock_zh_index_spot_em 返回空")
            return None
        columns = _akshare_columns(spot)
        if not columns["code"]:
            errors.append("akshare: 缺少指数代码列")
            return None
        result = _akshare_index_rows(spot, columns)
        if result:
            return result
        errors.append("akshare: 目标指数未命中")
        return None
    except Exception as e:
        errors.append(f"akshare: {e}")
        return None


def _akshare_columns(spot) -> dict[str, str]:
    return {
        "code": _first_column(spot, ("代码", "指数代码")),
        "name": _first_column(spot, ("名称", "指数名称")),
        "close": _first_column(spot, ("最新价", "最新")),
        "pct": _first_column(spot, ("涨跌幅", "涨跌幅(%)")),
        "vol": _first_column(spot, ("成交量",)),
        "amount": _first_column(spot, ("成交额",)),
    }


def _first_column(df, candidates: tuple[str, ...]) -> str:
    return next((col for col in candidates if col in df.columns), "")


def _akshare_index_rows(spot, columns: dict[str, str]) -> dict[str, dict]:
    code_to_ts = {symbol.split(".", 1)[0]: symbol for symbol in MARKET_OVERVIEW_INDICES}
    result: dict[str, dict] = {}
    for _, row in spot.iterrows():
        code = "".join(ch for ch in str(row.get(columns["code"], "") or "").strip() if ch.isdigit())[-6:]
        if code not in code_to_ts:
            continue
        ts_code = code_to_ts[code]
        name = str(row.get(columns["name"], "") or "").strip() or MARKET_OVERVIEW_INDICES[ts_code]
        result[name] = _akshare_latest_row(ts_code, row, columns)
    return result


def _akshare_latest_row(ts_code: str, row, columns: dict[str, str]) -> dict:
    return {
        "ts_code": ts_code,
        "trade_date": "",
        "close": round(safe_float(row.get(columns["close"], 0) if columns["close"] else 0), 2),
        "pct_chg": round(safe_float(row.get(columns["pct"], 0) if columns["pct"] else 0), 2),
        "vol": int(safe_float(row.get(columns["vol"], 0) if columns["vol"] else 0)),
        "amount": round(safe_float(row.get(columns["amount"], 0) if columns["amount"] else 0), 2),
    }


def resolve_market_history_index(index: str) -> tuple[str, str, str]:
    raw = str(index or "sse").strip()
    key = MARKET_HISTORY_ALIASES.get(raw, MARKET_HISTORY_ALIASES.get(raw.lower(), raw.lower()))
    if key in MARKET_HISTORY_INDEXES:
        symbol, name = MARKET_HISTORY_INDEXES[key]
        return key, symbol, name
    code = raw.upper()
    for item_key, (symbol, name) in MARKET_HISTORY_INDEXES.items():
        if code in {symbol, symbol.split(".", 1)[0]}:
            return item_key, symbol, name
    symbol, name = MARKET_HISTORY_INDEXES["sse"]
    return "sse", symbol, name


def json_float(value: Any, digits: int = 2) -> float | None:
    out = safe_float(value, None)
    return None if out is None else round(out, digits)


def prepare_market_history_frame(df: Any, days: int) -> Any:
    import pandas as pd

    out = df.copy()
    for col in ("open", "high", "low", "close", "volume", "amount", "prev_close"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "pct_chg" not in out.columns:
        basis = out["prev_close"] if "prev_close" in out.columns else out["close"].shift(1)
        out["pct_chg"] = (out["close"] / basis - 1.0) * 100.0
    return _finalize_market_history_frame(out, days)


def _finalize_market_history_frame(df, days: int):
    import pandas as pd

    cols = ["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    for col in ("open", "high", "low", "close", "volume", "amount", "pct_chg"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out = df.dropna(subset=["date", "close"]).sort_values("date").tail(days)
    return out[cols].reset_index(drop=True)


def _fetch_index_history_from_parquet(symbol: str, days: int):
    """从本地 index_daily.parquet 读指数历史日线，返回 DataFrame；不可用时返回 None。"""
    try:
        from utils.env import env_flag

        if env_flag("DATA_SOURCE_DISABLE_PARQUET"):
            return None
        import integrations.data_source_parquet as parquet_provider

        return parquet_provider.fetch_index_parquet(symbol, days)
    except Exception as exc:
        logger.debug("parquet index history unavailable: %s", exc)
        return None


def _fetch_index_history_from_pg(symbol: str, days: int):
    """从本地 PG index_raw_data 读指数历史日线，返回 DataFrame；不可用时返回 None。"""
    try:
        import pandas as pd

        import integrations.data_source_postgres as pg_provider

        if not pg_provider._pg_config()["password"]:
            return None
        code, market = str(symbol).split(".", 1)
        rows = pg_provider._query(
            code,
            """
            SELECT date::text, open, high, low, close, volume, amount
            FROM index_raw_data
            WHERE index_code = %s AND market = %s AND k_period = 'day'
            ORDER BY date DESC
            LIMIT %s
            """,
            (market.upper(), int(days) * 2),
        )
        if not rows:
            return None
        out = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
        for col in ("open", "high", "low", "close", "volume", "amount"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out["date"] = out["date"].astype(str).str[:10]
        # PG 偶发同日多行（不同来源重复导入），按日期去重取最后一条。
        out = out.drop_duplicates("date", keep="last")
        return out.sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def fetch_market_history_frame(symbol: str, days: int, tool_context: ToolContext | None) -> tuple[Any, str, list[str]]:
    errors: list[str] = []
    parquet_frame = _fetch_index_history_from_parquet(symbol, days)
    if parquet_frame is not None:
        return parquet_frame, "parquet", errors
    pg_frame = _fetch_index_history_from_pg(symbol, days)
    if pg_frame is not None:
        return pg_frame, "postgres", errors
    api_key = get_credential(tool_context, "tickflow_api_key", "TICKFLOW_API_KEY")
    if api_key:
        try:
            from integrations.tickflow_client import TickFlowClient

            client = TickFlowClient(api_key=api_key)
            return client.get_klines(symbol, period="1d", count=days, adjust="none"), "tickflow", errors
        except Exception as e:
            errors.append(f"tickflow: {e}")
    else:
        errors.append("tickflow: TICKFLOW_API_KEY 未配置")
    return _fetch_market_history_fallback(symbol, days, tool_context, errors)


def _fetch_market_history_fallback(
    symbol: str, days: int, tool_context: ToolContext | None, errors: list[str]
) -> tuple[Any, str, list[str]]:
    try:
        ensure_tushare_token(tool_context)
        from integrations.index_data_source import fetch_index_hist

        end = date.today()
        start = end - timedelta(days=int(days * 2.4) + 30)
        return fetch_index_hist(symbol, start, end), "tushare/akshare", errors
    except Exception as e:
        errors.append(f"tushare/akshare: {e}")
    raise RuntimeError("; ".join(errors))


def market_history_summary(df: Any) -> dict[str, Any]:
    close = df["close"]
    volume = df["volume"]
    latest = df.iloc[-1]
    tail20 = df.tail(min(len(df), 20))
    prior = df.iloc[:-20] if len(df) > 20 else df.iloc[:0]
    return {
        "latest_date": str(latest["date"]),
        "latest_close": json_float(latest["close"]),
        "latest_pct_chg": json_float(latest["pct_chg"]),
        "period_return_pct": json_float((float(close.iloc[-1]) / float(close.iloc[0]) - 1.0) * 100.0),
        "recent_20d_return_pct": json_float(
            (float(tail20["close"].iloc[-1]) / float(tail20["close"].iloc[0]) - 1.0) * 100.0
        ),
        "latest_volume_ratio_20d": json_float(float(latest["volume"]) / float(tail20["volume"].mean())),
        "recent_20d_volume_vs_prior": json_float(_recent_volume_ratio(tail20, prior)),
        "max_drawdown_pct": json_float(((close / close.cummax()) - 1.0).mul(100).min()),
        "up_days": int((df["pct_chg"] > 0).sum()),
        "down_days": int((df["pct_chg"] < 0).sum()),
        "price_up_volume_up_days": int(((df["pct_chg"] > 0) & (volume > volume.shift(1))).sum()),
        "price_down_volume_up_days": int(((df["pct_chg"] < 0) & (volume > volume.shift(1))).sum()),
    }


def _recent_volume_ratio(tail20, prior) -> float | None:
    prior_volume = prior["volume"].mean() if len(prior) else None
    return float(tail20["volume"].mean()) / float(prior_volume) if prior_volume else None


def market_history_rows(df: Any) -> list[dict[str, Any]]:
    return [
        {
            "date": str(row.get("date", "")),
            "open": json_float(row.get("open")),
            "high": json_float(row.get("high")),
            "low": json_float(row.get("low")),
            "close": json_float(row.get("close")),
            "pct_chg": json_float(row.get("pct_chg")),
            "volume": json_float(row.get("volume"), 0),
        }
        for row in df.to_dict("records")
    ]


def get_market_history(days: int = 100, index: str = "sse", tool_context: ToolContext | None = None) -> dict:
    try:
        requested_days = max(1, min(int(days or 100), 320))
        lookback = max(20, requested_days)
        key, symbol, name = resolve_market_history_index(index)
        raw, source, errors = fetch_market_history_frame(symbol, lookback, tool_context)
        df = prepare_market_history_frame(raw, lookback).tail(requested_days).reset_index(drop=True)
        if df.empty:
            return {"error": f"{name} {symbol} 没有可用历史 K 线", "source": source}
        return {
            "ok": True,
            "index": {"key": key, "symbol": symbol, "name": name},
            "requested_days": requested_days,
            "returned_days": int(len(df)),
            "source": source,
            "fallback_errors": errors,
            "summary": market_history_summary(df),
            "rows": market_history_rows(df),
        }
    except Exception as e:
        logger.exception("get_market_history error")
        return {"error": str(e)}
