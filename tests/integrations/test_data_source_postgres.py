from __future__ import annotations

import pandas as pd
import pytest

import integrations.data_source_postgres as provider


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def cursor(self):
        return _FakeCursor(self._rows)

    def close(self):
        self.closed = True


def _fake_connect(rows):
    import integrations.data_source_postgres as mod

    def _connect(**kwargs):
        return _FakeConn(rows)

    mod._connect = _connect  # noqa: SLF001


def _rows(dates: list[str]) -> list[tuple]:
    """SELECT 的 7 列：date, open, high, low, close, volume, amount（不含 code）。"""
    return [
        (
            pd.Timestamp(d),
            float(10 + i),
            float(10.5 + i),
            float(9.8 + i),
            float(10.2 + i),
            float(1000 * (i + 1)),
            float(12345.0 * (i + 1)),
        )
        for i, d in enumerate(dates)
    ]


@pytest.fixture()
def pg_env(monkeypatch):
    monkeypatch.setenv("PGHOST", "testhost")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGUSER", "testuser")
    monkeypatch.setenv("PGPASSWORD", "testpass")
    monkeypatch.setenv("PGDATABASE", "market")


def test_raw_reads_full_range(pg_env, monkeypatch) -> None:
    rows = _rows(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-31"])
    _fake_connect(rows)

    df = provider.fetch_stock_postgres("600519", "20241201", "20250131", "")
    assert list(df.columns) == list(provider.STOCK_HIST_COLUMNS)
    assert len(df) == 4
    assert df.iloc[0]["日期"] == "2024-12-30"
    assert df.iloc[0]["收盘"] == pytest.approx(10.2)
    assert df.iloc[0]["成交额"] == pytest.approx(12345.0)


def test_filters_to_requested_date_range(pg_env, monkeypatch) -> None:
    rows = _rows(["2024-12-30", "2024-12-31", "2025-01-02", "2025-12-31"])
    _fake_connect(rows)

    df = provider.fetch_stock_postgres("600519", "20250101", "20250131", "")
    assert len(df) == 1
    assert df.iloc[0]["日期"] == "2025-01-02"


def test_qfq_applies_factor(pg_env, monkeypatch) -> None:
    rows = _rows(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-31"])
    _fake_connect(rows)

    # 复权因子：2025-01-01 前 2.0，之后 1.0
    factor_rows = [
        (pd.Timestamp("2024-01-01"), 2.0),
        (pd.Timestamp("2025-01-01"), 1.0),
    ]

    class _FactoredCursor(_FakeCursor):
        def __init__(self, rows, factor_rows):
            super().__init__(rows)
            self._factor_rows = factor_rows

        def execute(self, sql, params=None):
            self._rows = self._factor_rows if "stock_adjustment_factors" in str(sql) else self._rows

    class _FactoredConnect(_FakeConn):
        def cursor(self):
            return _FactoredCursor(rows, factor_rows)

    def _connect(**kwargs):
        return _FactoredConnect(rows)

    monkeypatch.setattr(provider, "_connect", _connect)

    df = provider.fetch_stock_postgres("600519", "20241201", "20250131", "qfq")
    pre = df[df["日期"] <= "2024-12-31"]
    post = df[df["日期"] >= "2025-01-02"]
    assert pre.iloc[0]["收盘"] == pytest.approx(10.2 / 2.0)
    assert post.iloc[0]["收盘"] == pytest.approx(12.2)


def test_stale_data_raises(pg_env, monkeypatch) -> None:
    rows = _rows(["2025-12-31"])
    _fake_connect(rows)
    with pytest.raises(RuntimeError, match="postgres stale"):
        provider.fetch_stock_postgres("000001", "20250101", "20260131", "")


def test_unsupported_symbol_raises(pg_env) -> None:
    for symbol in ("00700.HK", "AAPL.US"):
        with pytest.raises(RuntimeError, match="postgres unsupported symbol"):
            provider.fetch_stock_postgres(symbol, "20250101", "20260131", "")


def test_hfq_unsupported_raises(pg_env) -> None:
    with pytest.raises(RuntimeError, match="postgres hfq unsupported"):
        provider.fetch_stock_postgres("600519", "20250101", "20260131", "hfq")


def test_unconfigured_raises(monkeypatch) -> None:
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.setattr(provider, "PG_PASSWORD", "")
    with pytest.raises(RuntimeError, match="postgres unconfigured"):
        provider.fetch_stock_postgres("600519", "20250101", "20260131", "")
