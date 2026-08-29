# Copyright (c) 2024-2026 youngcan. All Rights Reserved.
# 本代码仅供个人学习研究使用，未经授权不得用于商业目的。
# 商业授权请联系作者支付授权费用。

"""Unified stock-history facade over concrete vendor providers."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

import integrations.data_source_akshare
import integrations.data_source_baostock
import integrations.data_source_efinance
import integrations.data_source_format
import integrations.data_source_parquet
import integrations.data_source_postgres
import integrations.data_source_tickflow
import integrations.data_source_tushare
from integrations.index_data_source import fetch_index_akshare as fetch_index_akshare
from integrations.index_data_source import fetch_index_hist as fetch_index_hist
from integrations.spot_snapshot import fetch_stock_spot_snapshot as fetch_stock_spot_snapshot
from integrations.spot_snapshot import load_spot_snapshot_map as load_spot_snapshot_map
from integrations.tickflow_notice import (
    TICKFLOW_LIMIT_HINT,
    TICKFLOW_UPGRADE_URL,
    is_tickflow_rate_limited_error,
    record_tickflow_limit_event,
)
from utils.env import env_flag as _env_flag

logger = logging.getLogger(__name__)
StockFetcher = Callable[["StockHistFetchContext"], pd.DataFrame | None]


@dataclass
class StockHistFetchContext:
    symbol: str
    start_s: str
    end_s: str
    adjust: str
    failed_details: list[str] = field(default_factory=list)
    tickflow_limit_notices: list[str] = field(default_factory=list)
    tickflow_failed: bool = False


def fetch_stock_hist(
    symbol: str,
    start: str | date,
    end: str | date,
    adjust: Literal["", "qfq", "hfq"] = "qfq",
) -> pd.DataFrame:
    ctx = StockHistFetchContext(
        symbol=str(symbol).strip(),
        start_s=integrations.data_source_format.hist_date_text(start),
        end_s=integrations.data_source_format.hist_date_text(end),
        adjust=str(adjust or ""),
    )
    for fetcher in _stock_fetchers():
        out = fetcher(ctx)
        if out is not None:
            return out
    raise _stock_hist_failure(ctx)


def _stock_fetchers() -> tuple[StockFetcher, ...]:
    return (
        _try_parquet,
        _try_postgres,
        _try_tickflow,
        _try_tushare,
        _try_akshare,
        _try_baostock,
        _try_efinance,
    )


def _try_parquet(ctx: StockHistFetchContext) -> pd.DataFrame | None:
    if _env_flag("DATA_SOURCE_DISABLE_PARQUET"):
        ctx.failed_details.append("parquet=disabled_by_env")
        return None
    try:
        df = integrations.data_source_parquet.fetch_stock_parquet(ctx.symbol, ctx.start_s, ctx.end_s, ctx.adjust)
        return _fallback_output(ctx, df, "parquet")
    except Exception as exc:
        ctx.failed_details.append(f"parquet={integrations.data_source_format.compact_error(exc)}")
        _debug_source_fail("parquet", exc)
        return None


def _try_postgres(ctx: StockHistFetchContext) -> pd.DataFrame | None:
    if _env_flag("DATA_SOURCE_DISABLE_POSTGRES"):
        ctx.failed_details.append("postgres=disabled_by_env")
        return None
    try:
        df = integrations.data_source_postgres.fetch_stock_postgres(ctx.symbol, ctx.start_s, ctx.end_s, ctx.adjust)
        return _fallback_output(ctx, df, "postgres")
    except Exception as exc:
        ctx.failed_details.append(f"postgres={integrations.data_source_format.compact_error(exc)}")
        _debug_source_fail("postgres", exc)
        return None


def _try_tickflow(ctx: StockHistFetchContext) -> pd.DataFrame | None:
    if _env_flag("DATA_SOURCE_DISABLE_TICKFLOW"):
        ctx.failed_details.append("tickflow=disabled_by_env")
        return None
    if not os.getenv("TICKFLOW_API_KEY", "").strip():
        ctx.failed_details.append("tickflow=unconfigured")
        return None
    try:
        df = integrations.data_source_tickflow.fetch_stock_tickflow(ctx.symbol, ctx.start_s, ctx.end_s, ctx.adjust)
        return integrations.data_source_format.tag_source(df, "tickflow")
    except Exception as exc:
        _record_tickflow_failure(ctx, exc)
        return None


def _try_tushare(ctx: StockHistFetchContext) -> pd.DataFrame | None:
    try:
        df = integrations.data_source_tushare.fetch_stock_tushare(ctx.symbol, ctx.start_s, ctx.end_s)
        return _fallback_output(ctx, df, "tushare")
    except Exception as exc:
        detail = "token_missing" if str(exc) == "token_missing" else integrations.data_source_format.compact_error(exc)
        ctx.failed_details.append(f"tushare={detail}")
        _debug_source_fail("tushare", exc)
        return None


def _try_akshare(ctx: StockHistFetchContext) -> pd.DataFrame | None:
    if _env_flag("DATA_SOURCE_DISABLE_AKSHARE"):
        ctx.failed_details.append("akshare=disabled_by_env")
        return None
    try:
        df = integrations.data_source_akshare.fetch_stock_akshare(ctx.symbol, ctx.start_s, ctx.end_s, ctx.adjust)
        return _fallback_output(ctx, df, "akshare")
    except Exception as exc:
        ctx.failed_details.append(f"akshare={integrations.data_source_format.compact_error(exc)}")
        _debug_source_fail("akshare", exc)
        return None


def _try_baostock(ctx: StockHistFetchContext) -> pd.DataFrame | None:
    if _env_flag("DATA_SOURCE_DISABLE_BAOSTOCK"):
        ctx.failed_details.append("baostock=disabled_by_env")
        return None
    circuit_open, circuit_note = integrations.data_source_baostock.baostock_circuit_state()
    if circuit_open:
        ctx.failed_details.append(f"baostock={circuit_note or 'circuit_open'}")
        return None
    try:
        df = integrations.data_source_baostock.fetch_stock_baostock(ctx.symbol, ctx.start_s, ctx.end_s)
        integrations.data_source_baostock.baostock_mark_success()
        return _fallback_output(ctx, df, "baostock")
    except Exception as exc:
        detail = integrations.data_source_format.compact_error(exc)
        integrations.data_source_baostock.baostock_mark_failure(detail, debug_enabled=_debug_enabled())
        ctx.failed_details.append(f"baostock={detail}")
        _debug_source_fail("baostock", exc)
        return None


def _try_efinance(ctx: StockHistFetchContext) -> pd.DataFrame | None:
    if _env_flag("DATA_SOURCE_DISABLE_EFINANCE"):
        ctx.failed_details.append("efinance=disabled_by_env")
        return None
    try:
        df = integrations.data_source_efinance.fetch_stock_efinance(ctx.symbol, ctx.start_s, ctx.end_s)
        return _fallback_output(ctx, df, "efinance")
    except Exception as exc:
        ctx.failed_details.append(f"efinance={integrations.data_source_format.compact_error(exc)}")
        _debug_source_fail("efinance", exc)
        return None


def _record_tickflow_failure(ctx: StockHistFetchContext, exc: Exception) -> None:
    ctx.tickflow_failed = True
    detail = integrations.data_source_format.compact_error(exc)
    ctx.failed_details.append(f"tickflow={detail}")
    _debug_source_fail("tickflow", exc)
    if not is_tickflow_rate_limited_error(exc):
        logger.info(
            "tickflow failed, falling back: symbol=%s, range=%s..%s, err=%s", ctx.symbol, ctx.start_s, ctx.end_s, detail
        )
        return
    record_tickflow_limit_event(exc)
    ctx.tickflow_limit_notices.append(TICKFLOW_LIMIT_HINT)
    ctx.failed_details.append(f"tickflow_limit_hint={TICKFLOW_LIMIT_HINT}")
    logger.warning(
        "tickflow rate limited: symbol=%s, range=%s..%s, fallback_chain=tushare->akshare->baostock->efinance",
        ctx.symbol,
        ctx.start_s,
        ctx.end_s,
    )


def _fallback_output(ctx: StockHistFetchContext, df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = integrations.data_source_tickflow.attach_tickflow_limit_notices(df, ctx.tickflow_limit_notices)
    out = integrations.data_source_format.tag_source(out, source)
    if ctx.tickflow_failed:
        logger.info(
            "fallback hit: symbol=%s, source=%s, tickflow_limit_hint=%s",
            ctx.symbol,
            source,
            bool(ctx.tickflow_limit_notices),
        )
    return out


def _stock_hist_failure(ctx: StockHistFetchContext) -> RuntimeError:
    detail_suffix = f" 失败详情：{'；'.join(ctx.failed_details[:4])}。" if ctx.failed_details else ""
    hint_suffix = _build_datasource_hint(ctx.failed_details)
    return RuntimeError(
        f"数据拉取全线失败 [标:{ctx.symbol}, 范围:{ctx.start_s}..{ctx.end_s}, 复权:{ctx.adjust}]：已按顺序尝试 "
        f"parquet→postgres→tickflow→tushare→akshare→baostock→efinance，均无可用 K 线数据。请检查该标的是否已退市或处于长期停牌期。"
        f"{detail_suffix}{hint_suffix}"
    )


def _build_datasource_hint(failed_details: list[str]) -> str:
    hint = _network_hint_from_details(failed_details)
    has_tickflow = bool(os.getenv("TICKFLOW_API_KEY", "").strip())
    from integrations.tushare_client import has_tushare_token

    has_tushare = has_tushare_token()
    if not has_tickflow and not has_tushare:
        return f" 请配置数据源：{TICKFLOW_UPGRADE_URL}"
    if has_tushare and not has_tickflow:
        return f" Tushare 数据权限不足，可购买 TickFlow 获取更稳定的数据源：{TICKFLOW_UPGRADE_URL}"
    return f" 诊断提示：{hint}" if hint else ""


def _network_hint_from_details(details: list[str]) -> str:
    blob = " ".join(details).lower()
    if any(key in blob for key in ("nameresolutionerror", "failed to resolve", "getaddrinfo failed")):
        return "疑似 DNS/网络异常，请检查代理、DNS、系统防火墙或公司网络策略。"
    if "ssl" in blob or "certificate" in blob:
        return "疑似 SSL/证书链异常，请检查系统证书与 Python requests/certifi 环境。"
    if "remotedisconnected" in blob or "remote end closed connection" in blob:
        return "疑似上游行情源瞬时断连，可稍后重试；服务端已支持自动重试。"
    if "permission denied" in blob and "efinance" in blob:
        return "部署环境对 site-packages 为只读，efinance 本地缓存写入失败。"
    return ""


def _debug_enabled() -> bool:
    return _env_flag("DATA_SOURCE_DEBUG")


def _debug_source_fail(source: str, exc: Exception) -> None:
    if _debug_enabled():
        logger.debug("%s failed: %s: %s", source, type(exc).__name__, exc)
