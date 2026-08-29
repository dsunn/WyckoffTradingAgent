from __future__ import annotations

import pandas as pd
import pytest

import integrations.data_source_parquet as provider

pyarrow = pytest.importorskip("pyarrow")


@pytest.fixture()
def parquet_dir(tmp_path, monkeypatch) -> object:
    """构造临时 parquet 数据目录（candles + factors），并指向 provider。"""
    candles = tmp_path / "candles"
    factors = tmp_path / "factors"
    candles.mkdir()
    factors.mkdir()

    # 三年日线：600519 完整，000001 只到 2025（模拟 stale）
    rows = [
        ("600519", "2024-12-30", 10.0, 11.0, 9.5, 10.5, 1000),
        ("600519", "2024-12-31", 10.5, 11.5, 10.0, 11.0, 1200),
        ("600519", "2025-01-02", 11.0, 12.0, 10.5, 11.5, 1100),
        ("600519", "2025-12-31", 15.0, 16.0, 14.5, 15.5, 2000),
        ("600519", "2026-01-05", 15.5, 16.5, 15.0, 16.0, 2100),
        ("600519", "2026-01-31", 16.0, 17.0, 15.5, 16.5, 2200),
        ("000001", "2025-12-31", 5.0, 5.5, 4.8, 5.2, 500),
    ]
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume"])
    for year in ("2024", "2025", "2026"):
        year_df = df[df["date"].str.startswith(year)].reset_index(drop=True)
        year_df.to_parquet(candles / f"{year}.parquet", index=False)

    # 复权因子：600519 在 2025-01-01 除权，qfq_factor=2.0
    fac = pd.DataFrame(
        {
            "code": ["600519", "600519"],
            "date": pd.to_datetime(["2024-01-01", "2025-01-01"]),
            "qfq_factor": [2.0, 1.0],
            "hfq_factor": [1.0, 2.0],
        }
    )
    fac.to_parquet(factors / "adjustment_factors.parquet", index=False)

    monkeypatch.setattr(provider, "DATA_DIR", tmp_path)
    return tmp_path


def test_raw_reads_full_range(parquet_dir) -> None:
    df = provider.fetch_stock_parquet("600519", "20241201", "20260131", "")
    assert list(df.columns) == list(provider.STOCK_HIST_COLUMNS)
    assert len(df) == 6
    assert df.iloc[0]["日期"] == "2024-12-30"
    assert df.iloc[0]["开盘"] == 10.0
    assert df.iloc[0]["收盘"] == 10.5
    # amount 缺失 → NA，不报错
    assert pd.isna(df.iloc[0]["成交额"])


def test_filters_to_requested_date_range(parquet_dir) -> None:
    """请求窗口外的行不得返回（review P2 回归）。"""
    df = provider.fetch_stock_parquet("600519", "20250101", "20250131", "")
    assert len(df) == 1
    assert df.iloc[0]["日期"] == "2025-01-02"
    # 跨年窗口：2024 尾 + 2025 头，不含 2025 尾
    cross = provider.fetch_stock_parquet("600519", "20241201", "20250110", "")
    assert cross["日期"].tolist() == ["2024-12-30", "2024-12-31", "2025-01-02"]


def test_amount_passthrough_when_populated(tmp_path, monkeypatch) -> None:
    """asharedb 补填 amount 后，provider 应透传真值而非 NA。"""
    candles = tmp_path / "candles"
    candles.mkdir()
    df = pd.DataFrame(
        [("600519", "2026-02-02", 10.0, 10.5, 9.9, 10.3, 1000, 123456789.0)],
        columns=["code", "date", "open", "high", "low", "close", "volume", "amount"],
    )
    df.to_parquet(candles / "2026.parquet", index=False)
    monkeypatch.setattr(provider, "DATA_DIR", tmp_path)

    out = provider.fetch_stock_parquet("600519", "20260201", "20260202", "")
    assert out.iloc[0]["成交额"] == pytest.approx(123456789.0)


def test_qfq_applies_factor(parquet_dir) -> None:
    df = provider.fetch_stock_parquet("600519", "20241201", "20250131", "qfq")
    # 2024 年价格 / 2.0，2025 年起因子 1.0
    pre = df[df["日期"] <= "2024-12-31"]
    post = df[df["日期"] >= "2025-01-02"]
    assert pre.iloc[0]["收盘"] == pytest.approx(10.5 / 2.0)
    assert post.iloc[0]["收盘"] == pytest.approx(11.5)


def test_qfq_volume_scaled(parquet_dir) -> None:
    df = provider.fetch_stock_parquet("600519", "20241201", "20250131", "qfq")
    pre = df[df["日期"] <= "2024-12-31"]
    assert pre.iloc[0]["成交量"] == pytest.approx(1000 * 2.0)


def test_stale_data_raises(parquet_dir) -> None:
    # 000001 只到 2025-12-31，请求 2026 会 stale
    with pytest.raises(RuntimeError, match="parquet stale"):
        provider.fetch_stock_parquet("000001", "20250101", "20260131", "")


def test_unsupported_symbol_raises(parquet_dir) -> None:
    for symbol in ("00700.HK", "AAPL.US", "600519X"):
        with pytest.raises(RuntimeError, match="parquet unsupported symbol"):
            provider.fetch_stock_parquet(symbol, "20250101", "20260131", "")


def test_hfq_unsupported_raises(parquet_dir) -> None:
    with pytest.raises(RuntimeError, match="parquet hfq unsupported"):
        provider.fetch_stock_parquet("600519", "20250101", "20260131", "hfq")


def test_missing_year_file_returns_empty(parquet_dir) -> None:
    # 请求 2027 年（无该年份文件）→ stale（全量最新 2026-01-31 < 2027 请求）
    with pytest.raises(RuntimeError, match="parquet stale"):
        provider.fetch_stock_parquet("600519", "20270101", "20271231", "")


def test_missing_pyarrow_degrades(tmp_path, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("no pyarrow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(provider, "DATA_DIR", tmp_path)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="pyarrow missing"):
        provider.fetch_stock_parquet("600519", "20250101", "20260131", "")


def test_fetch_index_parquet_success(tmp_path, monkeypatch) -> None:
    idx_dir = tmp_path / "index"
    idx_dir.mkdir()
    df = pd.DataFrame(
        [
            ("000001", "SH", "2026-01-02", 3000.0, 3050.0, 2980.0, 3020.0, 100000, 10000000.0, 1500, 500, "day"),
            ("000001", "SH", "2026-01-05", 3020.0, 3080.0, 3010.0, 3060.0, 120000, 12000000.0, 1600, 400, "day"),
        ],
        columns=[
            "index_code",
            "market",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "up_count",
            "down_count",
            "k_period",
        ],
    )
    df.to_parquet(idx_dir / "index_daily.parquet", index=False)
    monkeypatch.setattr(provider, "DATA_DIR", tmp_path)
    # Mock today_str to avoid staleness trigger
    monkeypatch.setattr("pandas.Timestamp.now", lambda: pd.Timestamp("2026-01-05"))

    res = provider.fetch_index_parquet("000001.SH", days=5)
    assert len(res) == 2
    assert res.iloc[-1]["date"] == "2026-01-05"
    assert res.iloc[-1]["close"] == 3060.0


def test_fetch_index_parquet_stale(tmp_path, monkeypatch) -> None:
    idx_dir = tmp_path / "index"
    idx_dir.mkdir()
    df = pd.DataFrame(
        [
            ("000001", "SH", "2025-01-02", 3000.0, 3050.0, 2980.0, 3000.0, 100000, 10000000.0, 1000, 1000, "day"),
        ],
        columns=[
            "index_code",
            "market",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "up_count",
            "down_count",
            "k_period",
        ],
    )
    df.to_parquet(idx_dir / "index_daily.parquet", index=False)
    monkeypatch.setattr(provider, "DATA_DIR", tmp_path)
    monkeypatch.setattr("pandas.Timestamp.now", lambda: pd.Timestamp("2026-01-05"))

    with pytest.raises(RuntimeError, match="parquet index stale"):
        provider.fetch_index_parquet("000001.SH", days=5)


def test_fetch_index_parquet_missing_pyarrow(tmp_path, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("no pyarrow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(provider, "DATA_DIR", tmp_path)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="pyarrow missing"):
        provider.fetch_index_parquet("000001.SH", days=5)


def test_fetch_market_overview_parquet_success(tmp_path, monkeypatch) -> None:
    idx_dir = tmp_path / "index"
    idx_dir.mkdir()
    df = pd.DataFrame(
        [
            ("000001", "SH", "2026-01-02", 3000.0, 3050.0, 2980.0, 3000.0, 100000, 10000000.0, 1000, 1000, "day"),
            ("000001", "SH", "2026-01-05", 3000.0, 3080.0, 3000.0, 3060.0, 120000, 12000000.0, 1600, 400, "day"),
            ("399001", "SZ", "2026-01-05", 9000.0, 9100.0, 8950.0, 9050.0, 150000, 15000000.0, None, None, "day"),
        ],
        columns=[
            "index_code",
            "market",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "up_count",
            "down_count",
            "k_period",
        ],
    )
    df.to_parquet(idx_dir / "index_daily.parquet", index=False)
    monkeypatch.setattr(provider, "DATA_DIR", tmp_path)

    overview = provider.fetch_market_overview_parquet("20260105")
    assert overview["source"] == "parquet"
    assert overview["trade_date"] == "2026-01-05"
    assert "上证指数" in overview["indices"]
    sh = overview["indices"]["上证指数"]
    assert sh["close"] == 3060.0
    assert sh["pct_chg"] == pytest.approx(2.0)
    assert overview["breadth"]["up_count"] == 1600
    assert overview["breadth"]["down_count"] == 400
