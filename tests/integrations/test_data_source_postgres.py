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
    def __init__(self, rows, sql_routes=None):
        self._rows = rows
        self._sql_routes = sql_routes or {}
        self.closed = False

    def cursor(self):
        return _FakeCursor(self._rows)

    def close(self):
        self.closed = True


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


def _factor_rows() -> list[tuple]:
    return [
        (pd.Timestamp("2024-01-01"), 2.0),
        (pd.Timestamp("2025-01-01"), 1.0),
    ]


def _install_fake_connect(monkeypatch, rows, factor_rows=None, latest="2025-01-31"):
    """mock _query：按 SQL 内容路由到蜡烛行 / 因子行 / 最新日期。"""
    calls: list[str] = []

    def fake_query(symbol, sql, params=None):
        calls.append(sql)
        if "MAX(date)" in sql:
            return [(latest,)]
        if "stock_adjustment_factors" in sql:
            return list(factor_rows) if factor_rows else []
        return list(rows)

    monkeypatch.setattr(provider, "_query", fake_query)
    return calls


@pytest.fixture()
def pg_env(monkeypatch):
    monkeypatch.setenv("PGHOST", "testhost")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGUSER", "testuser")
    monkeypatch.setenv("PGPASSWORD", "testpass")
    monkeypatch.setenv("PGDATABASE", "market")


def test_raw_reads_full_range(pg_env, monkeypatch) -> None:
    rows = _rows(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-31"])
    _install_fake_connect(monkeypatch, rows)

    df = provider.fetch_stock_postgres("600519", "20241201", "20250131", "")
    assert list(df.columns) == list(provider.STOCK_HIST_COLUMNS)
    assert len(df) == 4
    assert df.iloc[0]["日期"] == "2024-12-30"
    assert df.iloc[0]["收盘"] == pytest.approx(10.2)
    assert df.iloc[0]["成交额"] == pytest.approx(12345.0)


def test_filters_to_requested_date_range(pg_env, monkeypatch) -> None:
    """SQL 层过滤（P2 修复）：provider 依赖 _query 已按窗口过滤。"""
    rows = _rows(["2024-12-30", "2024-12-31", "2025-01-02", "2025-12-31"])

    def fake_query(symbol, sql, params=None):
        if "MAX(date)" in sql:
            return [("2025-12-31",)]
        if "stock_adjustment_factors" in sql:
            return []
        start_s, end_s = params[0], params[1]
        return [r for r in rows if start_s <= str(r[0])[:10] <= end_s]

    monkeypatch.setattr(provider, "_query", fake_query)

    df = provider.fetch_stock_postgres("600519", "20250101", "20250131", "")
    assert len(df) == 1
    assert df.iloc[0]["日期"] == "2025-01-02"


def test_sql_filters_by_date_window(pg_env, monkeypatch) -> None:
    """SQL 层按请求窗口过滤，不拉全量（review P2）。"""
    rows = _rows(["2024-12-30", "2025-12-31"])
    calls = _install_fake_connect(monkeypatch, rows)

    provider.fetch_stock_postgres("600519", "20250101", "20250131", "")
    candle_sql = next(s for s in calls if "MAX(date)" not in s and "stock_adjustment_factors" not in s)
    assert "date >= %s AND date <= %s" in candle_sql


def test_qfq_applies_factor(pg_env, monkeypatch) -> None:
    rows = _rows(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-31"])
    _install_fake_connect(monkeypatch, rows, factor_rows=_factor_rows())

    df = provider.fetch_stock_postgres("600519", "20241201", "20250131", "qfq")
    pre = df[df["日期"] <= "2024-12-31"]
    post = df[df["日期"] >= "2025-01-02"]
    assert pre.iloc[0]["收盘"] == pytest.approx(10.2 / 2.0)
    assert post.iloc[0]["收盘"] == pytest.approx(12.2)


def test_stale_data_raises(pg_env, monkeypatch) -> None:
    rows = _rows(["2025-12-31"])
    _install_fake_connect(monkeypatch, rows, latest="2025-12-31")
    with pytest.raises(RuntimeError, match="postgres stale"):
        provider.fetch_stock_postgres("000001", "20250101", "20260130", "")


def test_weekend_end_not_stale(pg_env, monkeypatch) -> None:
    """周六请求：数据到周五即视为覆盖，不降级（真实场景 2026-08-29 周六）。"""
    rows = _rows(["2026-08-28"])
    _install_fake_connect(monkeypatch, rows, latest="2026-08-28")
    df = provider.fetch_stock_postgres("600519", "20260801", "20260829", "")
    assert len(df) == 1
    assert df.iloc[0]["日期"] == "2026-08-28"


def test_unsupported_symbol_raises(pg_env) -> None:
    for symbol in ("00700.HK", "AAPL.US"):
        with pytest.raises(RuntimeError, match="postgres unsupported symbol"):
            provider.fetch_stock_postgres(symbol, "20250101", "20260131", "")


def test_hfq_unsupported_raises(pg_env) -> None:
    with pytest.raises(RuntimeError, match="postgres hfq unsupported"):
        provider.fetch_stock_postgres("600519", "20250101", "20260131", "hfq")


def test_unconfigured_raises(monkeypatch) -> None:
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.setattr(provider, "_pg_config_file", lambda: {})
    with pytest.raises(RuntimeError, match="postgres unconfigured"):
        provider.fetch_stock_postgres("600519", "20250101", "20260131", "")


def test_connection_pool_reuses_connection(pg_env, monkeypatch) -> None:
    """同一配置下多次查询复用同一连接（review P1）。"""
    import integrations.data_source_postgres as mod

    mod._CONN_POOL.clear()
    conns = []

    def fake_psycopg2_connect(**kwargs):
        conn = _FakeConn([])
        conns.append(conn)
        return conn

    monkeypatch.setattr("psycopg2.connect", fake_psycopg2_connect)

    c1 = mod._connect()
    c2 = mod._connect()
    assert c1 is c2, "应复用同一连接"
    assert len(conns) == 1


def test_psycopg2_missing_raises(pg_env, monkeypatch) -> None:
    """psycopg2 缺失 → 明确报错（review P3 错误路径）。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psycopg2":
            raise ImportError("no psycopg2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="postgres psycopg2 missing"):
        provider._query("600519", "SELECT 1")


def test_empty_result_raises(pg_env, monkeypatch) -> None:
    """库中无该股 → postgres empty（review P3 错误路径）。"""
    _install_fake_connect(monkeypatch, [], latest="")
    with pytest.raises(RuntimeError, match="postgres empty"):
        provider.fetch_stock_postgres("600519", "20250101", "20250131", "")


def test_factor_read_failure_logs_warning(pg_env, monkeypatch, caplog) -> None:
    """因子读失败必须留痕（review P7），回退 raw 而非静默。"""
    rows = _rows(["2025-01-02", "2025-01-31"])

    def fake_query(symbol, sql, params=None):
        if "MAX(date)" in sql:
            return [("2025-01-31",)]
        if "stock_adjustment_factors" in sql:
            raise RuntimeError("boom")
        return list(rows)

    monkeypatch.setattr(provider, "_query", fake_query)
    import logging

    with caplog.at_level(logging.WARNING, logger="integrations.data_source_postgres"):
        df = provider.fetch_stock_postgres("600519", "20250101", "20250131", "qfq")
    assert any("factor read failed" in r.message for r in caplog.records)
    assert len(df) == 2


def test_pg_config_prefers_file_over_default(monkeypatch) -> None:
    """无环境变量时从 wyckoff.json 的 pg_data_source 段读（Windows 桌面端场景）。"""
    for key in ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"):
        monkeypatch.delenv(key, raising=False)

    def fake_load_config():
        return {
            "pg_data_source": {
                "host": "file-host",
                "port": "15432",
                "user": "file-user",
                "password": "file-pass",
                "database": "file-db",
            }
        }

    monkeypatch.setattr("integrations.local_auth.load_config", fake_load_config)
    cfg = provider._pg_config()
    assert cfg["host"] == "file-host"
    assert cfg["port"] == "15432"
    assert cfg["password"] == "file-pass"
    assert cfg["database"] == "file-db"


def test_pg_config_env_overrides_file(monkeypatch) -> None:
    """环境变量优先于 config 文件。"""
    monkeypatch.setenv("PGHOST", "env-host")

    def fake_load_config():
        return {"pg_data_source": {"host": "file-host"}}

    monkeypatch.setattr("integrations.local_auth.load_config", fake_load_config)
    assert provider._pg_config()["host"] == "env-host"


def test_pg_config_empty_env_overrides_file(monkeypatch) -> None:
    """显式空 env 也赢过 config 文件（review P2）。"""
    monkeypatch.setenv("PGHOST", "")
    monkeypatch.delenv("PGPORT", raising=False)

    def fake_load_config():
        return {"pg_data_source": {"host": "file-host", "port": "15432"}}

    monkeypatch.setattr("integrations.local_auth.load_config", fake_load_config)
    cfg = provider._pg_config()
    assert cfg["host"] == provider._PG_DEFAULTS["host"], "空 env 应回退默认而非 config"
    assert cfg["port"] == "15432", "未设 env 的键才用 config"


def test_pg_config_file_corrupt_logs_debug(monkeypatch, caplog) -> None:
    """config 文件损坏时留 debug 日志（review P2）。"""
    for key in ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"):
        monkeypatch.delenv(key, raising=False)
    import logging

    def fake_load_config():
        raise RuntimeError("corrupted")

    monkeypatch.setattr("integrations.local_auth.load_config", fake_load_config)
    with caplog.at_level(logging.DEBUG, logger="integrations.data_source_postgres"):
        cfg = provider._pg_config()
    assert cfg["host"] == provider._PG_DEFAULTS["host"], "损坏时回退默认"
    assert any("pg config file read failed" in r.message for r in caplog.records)


def test_pool_key_includes_password(pg_env, monkeypatch) -> None:
    """换密码后不复用旧认证连接（review P2）。"""
    import integrations.data_source_postgres as mod

    mod._CONN_POOL.clear()
    conns = []

    def fake_psycopg2_connect(**kwargs):
        conn = _FakeConn([])
        conns.append(conn)
        return conn

    monkeypatch.setattr("psycopg2.connect", fake_psycopg2_connect)
    monkeypatch.setenv("PGPASSWORD", "pass-1")
    c1 = mod._connect()
    monkeypatch.setenv("PGPASSWORD", "pass-2")
    c2 = mod._connect()
    assert c1 is not c2, "密码不同应新建连接"
    assert len(conns) == 2


def test_idle_expired_connection_closed(pg_env, monkeypatch) -> None:
    """空闲超时替换旧连接时显式关闭（review P2）。"""
    import integrations.data_source_postgres as mod

    mod._CONN_POOL.clear()
    conns = []

    def fake_psycopg2_connect(**kwargs):
        conn = _FakeConn([])
        conns.append(conn)
        return conn

    monkeypatch.setattr("psycopg2.connect", fake_psycopg2_connect)
    monkeypatch.setattr(mod, "_CONN_IDLE_TIMEOUT_SECONDS", -1.0)  # 强制立即过期

    c1 = mod._connect()
    c2 = mod._connect()
    assert c1 is not c2, "过期后应新建连接"
    assert c1.closed is True, "旧连接应被显式关闭"
    assert len(conns) == 2


def test_pg_index_overview_builds_indices(monkeypatch) -> None:
    """PG 指数大盘：mock _query 返回指数行，验证结构与 pct_chg 计算。"""
    import agents.market_tools as mt

    calls: list[str] = []

    def fake_query(symbol, sql, params=None, *, use_symbol=True):
        calls.append(sql)
        if "index_raw_data" in sql and "date < %s" in sql:
            # 前一日收盘查询（注意 date <= 也含 "date <" 子串，须精确匹配）
            return [(3800.0,)]
        if "index_raw_data" in sql:
            # 最新行：date, open, high, low, close, volume, amount
            return [("2026-08-28", 3900.0, 3955.0, 3890.0, 3952.18, 510581645, 970365140992.0)]
        return []  # breadth 不测

    monkeypatch.setattr("integrations.data_source_postgres._query", fake_query)
    monkeypatch.setattr("integrations.data_source_postgres._pg_config", lambda: {"password": "x"})

    result = mt._fetch_pg_index_overview([], "20260828", False)
    assert result is not None
    assert result["source"] == "postgres"
    assert result["trade_date"] == "2026-08-28"
    sh = result["indices"]["上证指数"]
    assert sh["close"] == 3952.18
    assert sh["pct_chg"] == pytest.approx((3952.18 / 3800.0 - 1.0) * 100.0, abs=0.01)


def test_pg_index_overview_unconfigured(monkeypatch) -> None:
    """PG 未配置时返回 None 并记录错误。"""
    import agents.market_tools as mt

    monkeypatch.setattr("integrations.data_source_postgres._pg_config", lambda: {"password": ""})
    errors: list[str] = []
    result = mt._fetch_pg_index_overview(errors, "20260828", False)
    assert result is None
    assert any("unconfigured" in e for e in errors)


def test_resolve_star_index_000680() -> None:
    """科创综指 000680 解析（含别名）。"""
    import agents.market_tools as mt

    key, symbol, name = mt.resolve_market_history_index("科创综指")
    assert (key, symbol) == ("star", "000680.SH")
    assert name == "科创综指"
    key2, symbol2, _ = mt.resolve_market_history_index("科创")
    assert (key2, symbol2) == ("star", "000680.SH")


def test_fetch_index_history_from_pg_dedups_dates(monkeypatch) -> None:
    """PG 指数历史去重：同日多行只保留最后一条。"""
    import agents.market_tools as mt

    def fake_query(symbol, sql, params=None, *, use_symbol=True):
        if "index_raw_data" in sql:
            return [
                ("2026-08-28", 1966.18, 2000.58, 1942.17, 1986.69, 45318682, 1e10),
                ("2026-08-28", 1966.18, 2000.58, 1942.17, 1942.91, 45318682, 1e10),
                ("2026-08-27", 1950.0, 1990.0, 1940.0, 1974.0, 40000000, 9e9),
            ]
        return []

    monkeypatch.setattr("integrations.data_source_postgres._query", fake_query)
    monkeypatch.setattr("integrations.data_source_postgres._pg_config", lambda: {"password": "x"})

    df = mt._fetch_index_history_from_pg("000680.SH", 10)
    assert df is not None
    assert len(df) == 2
    assert list(df["date"]) == ["2026-08-27", "2026-08-28"]
    assert df.iloc[-1]["close"] == 1942.91  # keep="last" 保留后一条
