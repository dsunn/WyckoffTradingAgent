"""data_source 中 tickflow 优先链路测试。"""

from __future__ import annotations

import pandas as pd
import pytest

import integrations.data_source as ds


def _sample_cn_hist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "日期": "2026-04-18",
                "开盘": 10.0,
                "最高": 10.5,
                "最低": 9.9,
                "收盘": 10.3,
                "成交量": 1000000.0,
                "成交额": 10000000.0,
                "涨跌幅": 1.2,
                "换手率": pd.NA,
                "振幅": 2.3,
            }
        ]
    )


def _disable_other_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_SOURCE_DISABLE_AKSHARE", "1")
    monkeypatch.setenv("DATA_SOURCE_DISABLE_BAOSTOCK", "1")
    monkeypatch.setenv("DATA_SOURCE_DISABLE_EFINANCE", "1")
    monkeypatch.setenv("DATA_SOURCE_DISABLE_PARQUET", "1")
    monkeypatch.setenv("DATA_SOURCE_DISABLE_POSTGRES", "1")
    monkeypatch.delenv("DATA_SOURCE_DISABLE_TICKFLOW", raising=False)


def test_fetch_stock_hist_prefers_parquet_over_tickflow(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """本地 parquet 数据可用时优先于 tickflow；tickflow 不应被调用。"""
    pytest.importorskip("pyarrow")
    import integrations.data_source_parquet as parquet_provider

    candles = tmp_path / "candles"
    candles.mkdir()
    df = pd.DataFrame(
        [
            ("600519", "2026-04-15", 10.0, 10.5, 9.9, 10.3, 1000),
            ("600519", "2026-04-16", 10.3, 10.8, 10.1, 10.6, 1100),
            ("600519", "2026-04-17", 10.6, 11.0, 10.4, 10.9, 1200),
            ("600519", "2026-04-18", 10.9, 11.2, 10.6, 11.0, 1300),
        ],
        columns=["code", "date", "open", "high", "low", "close", "volume"],
    )
    df.to_parquet(candles / "2026.parquet", index=False)
    monkeypatch.setattr(parquet_provider, "DATA_DIR", tmp_path)

    def _raise_tickflow_if_called(*args, **kwargs):
        raise RuntimeError("should_not_call")

    monkeypatch.setattr("integrations.data_source_tickflow.fetch_stock_tickflow", _raise_tickflow_if_called)
    monkeypatch.setenv("DATA_SOURCE_DISABLE_AKSHARE", "1")
    monkeypatch.setenv("DATA_SOURCE_DISABLE_BAOSTOCK", "1")
    monkeypatch.setenv("DATA_SOURCE_DISABLE_EFINANCE", "1")
    monkeypatch.delenv("DATA_SOURCE_DISABLE_TICKFLOW", raising=False)
    monkeypatch.delenv("DATA_SOURCE_DISABLE_PARQUET", raising=False)
    monkeypatch.setenv("DATA_SOURCE_DISABLE_POSTGRES", "1")

    out = ds.fetch_stock_hist("600519", "2026-04-10", "2026-04-18", adjust="qfq")
    assert out.attrs.get("source") == "parquet"
    assert out.iloc[-1]["日期"] == "2026-04-18"


def test_fetch_stock_hist_prefers_tickflow_when_both_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_other_fallbacks(monkeypatch)
    monkeypatch.setenv("TICKFLOW_API_KEY", "dummy")

    def _raise_tushare_if_called(*args, **kwargs):
        raise RuntimeError("should_not_call")

    monkeypatch.setattr("integrations.data_source_tushare.fetch_stock_tushare", _raise_tushare_if_called)
    monkeypatch.setattr(
        "integrations.data_source_tickflow.fetch_stock_tickflow", lambda *args, **kwargs: _sample_cn_hist()
    )

    out = ds.fetch_stock_hist("600519", "2026-04-10", "2026-04-18", adjust="qfq")
    assert not out.empty
    assert out.attrs.get("source") == "tickflow"
    assert out.iloc[0]["日期"] == "2026-04-18"


def test_fetch_stock_hist_falls_back_to_tushare_when_tickflow_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_other_fallbacks(monkeypatch)
    monkeypatch.setenv("TICKFLOW_API_KEY", "dummy")

    def _raise_tickflow(*args, **kwargs):
        raise RuntimeError("tickflow timeout")

    monkeypatch.setattr("integrations.data_source_tickflow.fetch_stock_tickflow", _raise_tickflow)
    monkeypatch.setattr(
        "integrations.data_source_tushare.fetch_stock_tushare", lambda *args, **kwargs: _sample_cn_hist()
    )

    out = ds.fetch_stock_hist("600519", "2026-04-10", "2026-04-18", adjust="qfq")
    assert not out.empty
    assert out.attrs.get("source") == "tushare"


def test_fetch_stock_hist_keeps_limit_hint_when_tickflow_rate_limited_and_fallback_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_other_fallbacks(monkeypatch)
    monkeypatch.setenv("TICKFLOW_API_KEY", "dummy")

    def _rate_limited(*args, **kwargs):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr("integrations.data_source_tickflow.fetch_stock_tickflow", _rate_limited)
    monkeypatch.setattr(
        "integrations.data_source_tushare.fetch_stock_tushare", lambda *args, **kwargs: _sample_cn_hist()
    )

    out = ds.fetch_stock_hist("600519", "2026-04-10", "2026-04-18", adjust="qfq")
    assert out.attrs.get("source") == "tushare"
    assert "触发数据源限制，升级数据源：" in str(out.attrs.get("tickflow_limit_hint", ""))


def test_fetch_stock_hist_error_message_contains_tickflow_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_other_fallbacks(monkeypatch)
    monkeypatch.delenv("TICKFLOW_API_KEY", raising=False)
    monkeypatch.setattr("integrations.tushare_client.get_pro", lambda: None)

    with pytest.raises(RuntimeError) as exc:
        ds.fetch_stock_hist("000001", "2026-04-10", "2026-04-18", adjust="qfq")
    assert "parquet→postgres→tickflow→tushare→akshare→baostock→efinance" in str(exc.value)
