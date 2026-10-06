"""
src.data.signal_db — PostgreSQL database for ml-service2.0 signal storage
==========================================================================

Primary storage: PostgreSQL (TimescaleDB) at localhost:5445
  Container: ml-service-postgres (docker-compose.full-stack.yml)
  Database:  mlservice | User: mlservice | Port: 5445

Tables
------
  sessions         — one row per trading day
  signals          — immutable signal log (generated_at, score, direction)
  positions        — mutable intraday state (entry/exit price, WIN/LOSS)
  price_snapshots  — mark-to-market history every ~2 minutes

Connection
----------
  DATABASE_URL env var (fallback: postgresql://mlservice:mlservice_dev@localhost:5445/mlservice)
  ThreadedConnectionPool (min=1, max=10) for safe multi-thread access

Common queries
--------------
  -- Today's open positions
  SELECT * FROM positions WHERE session_date='2026-10-07' AND status='OPEN';

  -- Rolling win rate
  SELECT session_date,
         SUM((status='SETTLED_WIN')::int) wins,
         COUNT(*) total,
         ROUND(SUM((status='SETTLED_WIN')::int)*100.0/COUNT(*),1) win_pct,
         ROUND(AVG(final_return_pct)::numeric,3) mean_ret
  FROM positions WHERE status!='OPEN'
  GROUP BY session_date ORDER BY session_date DESC;

  -- Best symbols
  SELECT symbol, COUNT(*) trades, ROUND(AVG(final_return_pct)::numeric,3) avg_ret
  FROM positions WHERE status='SETTLED_WIN'
  GROUP BY symbol HAVING COUNT(*)>=3 ORDER BY avg_ret DESC;
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.pool
import psycopg2.extras   # RealDictCursor

_DEFAULT_URL = "postgresql://mlservice:mlservice_dev@localhost:5445/mlservice"
DATABASE_URL: str = os.getenv("ML_DATABASE_URL", _DEFAULT_URL)

_SCHEMA_SQL = """
-- Sessions
CREATE TABLE IF NOT EXISTS sessions (
    session_date     TEXT PRIMARY KEY,
    nifty_open       REAL,
    nifty_close      REAL,
    nifty_chg_pct    REAL,
    n_scored         INTEGER DEFAULT 0,
    n_long           INTEGER DEFAULT 0,
    n_short          INTEGER DEFAULT 0,
    model_version    TEXT,
    created_at       TEXT NOT NULL
);

-- Signals (immutable)
CREATE TABLE IF NOT EXISTS signals (
    signal_id        TEXT PRIMARY KEY,
    session_date     TEXT NOT NULL REFERENCES sessions(session_date),
    symbol           TEXT NOT NULL,
    generated_at     TEXT NOT NULL,
    data_date        TEXT,
    direction        INTEGER NOT NULL CHECK (direction IN (-1, 1)),
    score            REAL NOT NULL,
    conviction       TEXT,
    rank             INTEGER,
    cost_bps         REAL DEFAULT 7.26,
    nifty_at_signal  REAL,
    model_version    TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_session   ON signals(session_date);
CREATE INDEX IF NOT EXISTS idx_signals_sym_sess  ON signals(symbol, session_date);

-- Positions (mutable)
CREATE TABLE IF NOT EXISTS positions (
    signal_id        TEXT PRIMARY KEY REFERENCES signals(signal_id),
    session_date     TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    direction        INTEGER NOT NULL,
    entry_time       TEXT,
    entry_price      REAL,
    exit_time        TEXT,
    exit_price       REAL,
    last_price       REAL,
    last_price_time  TEXT,
    unrealized_pct   REAL,
    status           TEXT NOT NULL DEFAULT 'OPEN'
                         CHECK (status IN ('OPEN','SETTLED_WIN','SETTLED_LOSS','EXPIRED')),
    settled_at       TEXT,
    final_return_pct REAL,
    outcome          TEXT
);
CREATE INDEX IF NOT EXISTS idx_pos_session        ON positions(session_date);
CREATE INDEX IF NOT EXISTS idx_pos_sess_status    ON positions(session_date, status);
CREATE INDEX IF NOT EXISTS idx_pos_symbol         ON positions(symbol);

-- Price snapshots (MTM history every ~2 min)
CREATE TABLE IF NOT EXISTS price_snapshots (
    id               BIGSERIAL PRIMARY KEY,
    session_date     TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    snapshot_time    TEXT NOT NULL,
    price            REAL NOT NULL,
    unrealized_pct   REAL
);
CREATE INDEX IF NOT EXISTS idx_snap_sess_sym ON price_snapshots(session_date, symbol);
"""

_lock = threading.Lock()


class SignalDB:
    """
    Thread-safe PostgreSQL wrapper for ml-service2.0 signal storage.

    Uses ThreadedConnectionPool (min=1, max=10).
    All write methods are atomic (autocommit=False, explicit commit/rollback).
    """

    def __init__(self, dsn: str = DATABASE_URL) -> None:
        self._dsn  = dsn
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1, maxconn=10, dsn=dsn
        )
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_SQL)

    @contextmanager
    def _connect(self):
        conn = self._pool.getconn()
        conn.autocommit = False
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _dict_rows(self, conn, sql: str, params=()) -> list[dict]:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self._pool.closeall()

    # ── Session ───────────────────────────────────────────────────────────────

    def upsert_session(
        self,
        session_date: str,
        n_scored: int = 0,
        n_long: int = 0,
        n_short: int = 0,
        nifty_chg_pct: float | None = None,
        nifty_ltp: float | None = None,
        model_version: str | None = None,
    ) -> None:
        with _lock, self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sessions(session_date, nifty_close, nifty_chg_pct,
                        n_scored, n_long, n_short, model_version, created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(session_date) DO UPDATE SET
                        nifty_close    = COALESCE(EXCLUDED.nifty_close, sessions.nifty_close),
                        nifty_chg_pct  = COALESCE(EXCLUDED.nifty_chg_pct, sessions.nifty_chg_pct),
                        n_scored       = EXCLUDED.n_scored,
                        n_long         = EXCLUDED.n_long,
                        n_short        = EXCLUDED.n_short,
                        model_version  = COALESCE(EXCLUDED.model_version, sessions.model_version)
                """, (session_date, nifty_ltp, nifty_chg_pct,
                      n_scored, n_long, n_short, model_version,
                      datetime.now(tz=timezone.utc).isoformat()))

    # ── Signals ───────────────────────────────────────────────────────────────

    def insert_signal(self, sig: dict) -> bool:
        """Returns True if inserted, False if already exists."""
        with _lock, self._connect() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        INSERT INTO signals(signal_id,session_date,symbol,generated_at,
                            data_date,direction,score,conviction,rank,cost_bps,
                            nifty_at_signal,model_version)
                        VALUES(%(signal_id)s,%(session_date)s,%(symbol)s,%(generated_at)s,
                            %(data_date)s,%(direction)s,%(score)s,%(conviction)s,%(rank)s,
                            %(cost_bps)s,%(nifty_at_signal)s,%(model_version)s)
                    """, sig)
                    return True
                except psycopg2.IntegrityError:
                    conn.rollback()
                    return False

    def insert_signals_batch(self, signals: list[dict]) -> int:
        """Batch insert, skip duplicates. Returns count inserted."""
        inserted = 0
        with _lock, self._connect() as conn:
            with conn.cursor() as cur:
                for sig in signals:
                    cur.execute("""
                        INSERT INTO signals(signal_id,session_date,symbol,generated_at,
                            data_date,direction,score,conviction,rank,cost_bps,
                            nifty_at_signal,model_version)
                        VALUES(%(signal_id)s,%(session_date)s,%(symbol)s,%(generated_at)s,
                            %(data_date)s,%(direction)s,%(score)s,%(conviction)s,%(rank)s,
                            %(cost_bps)s,%(nifty_at_signal)s,%(model_version)s)
                        ON CONFLICT(signal_id) DO NOTHING
                    """, sig)
                    inserted += cur.rowcount
        return inserted

    # ── Positions ─────────────────────────────────────────────────────────────

    def upsert_position(self, pos: dict) -> None:
        with _lock, self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO positions(signal_id,session_date,symbol,direction,
                        entry_time,entry_price,exit_time,exit_price,last_price,
                        last_price_time,unrealized_pct,status,settled_at,
                        final_return_pct,outcome)
                    VALUES(%(signal_id)s,%(session_date)s,%(symbol)s,%(direction)s,
                        %(entry_time)s,%(entry_price)s,%(exit_time)s,%(exit_price)s,
                        %(last_price)s,%(last_price_time)s,%(unrealized_pct)s,%(status)s,
                        %(settled_at)s,%(final_return_pct)s,%(outcome)s)
                    ON CONFLICT(signal_id) DO UPDATE SET
                        entry_price     = COALESCE(EXCLUDED.entry_price, positions.entry_price),
                        exit_time       = COALESCE(EXCLUDED.exit_time, positions.exit_time),
                        exit_price      = COALESCE(EXCLUDED.exit_price, positions.exit_price),
                        last_price      = COALESCE(EXCLUDED.last_price, positions.last_price),
                        last_price_time = COALESCE(EXCLUDED.last_price_time, positions.last_price_time),
                        unrealized_pct  = COALESCE(EXCLUDED.unrealized_pct, positions.unrealized_pct),
                        status          = EXCLUDED.status,
                        settled_at      = COALESCE(EXCLUDED.settled_at, positions.settled_at),
                        final_return_pct= COALESCE(EXCLUDED.final_return_pct, positions.final_return_pct),
                        outcome         = COALESCE(EXCLUDED.outcome, positions.outcome)
                """, pos)

    def update_prices_batch(
        self,
        session_date: str,
        price_updates: list[tuple],  # (last_price, unrealized_pct, last_price_time, symbol, session_date)
    ) -> int:
        with _lock, self._connect() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, """
                    UPDATE positions
                    SET last_price=%s, unrealized_pct=%s, last_price_time=%s
                    WHERE symbol=%s AND session_date=%s AND status='OPEN'
                """, price_updates)
                return cur.rowcount

    def settle_position(self, signal_id: str, pos: dict) -> None:
        with _lock, self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE positions SET
                        exit_time=%s, exit_price=%s, status=%s, settled_at=%s,
                        final_return_pct=%s, outcome=%s, unrealized_pct=%s
                    WHERE signal_id=%s
                """, (pos.get("exit_time"), pos.get("exit_price"), pos.get("status"),
                      pos.get("settled_at"), pos.get("final_return_pct"),
                      pos.get("outcome"), pos.get("final_return_pct"), signal_id))

    # ── Price Snapshots ───────────────────────────────────────────────────────

    def insert_snapshot_batch(self, session_date: str, snapshots: list[dict]) -> None:
        with _lock, self._connect() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, """
                    INSERT INTO price_snapshots(session_date,symbol,snapshot_time,price,unrealized_pct)
                    VALUES(%(session_date)s,%(symbol)s,%(snapshot_time)s,%(price)s,%(unrealized_pct)s)
                """, snapshots)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_open_positions(self, session_date: str) -> list[dict]:
        with self._connect() as conn:
            return self._dict_rows(conn, """
                SELECT p.*, s.score, s.conviction, s.rank, s.generated_at
                FROM positions p JOIN signals s USING (signal_id)
                WHERE p.session_date=%s AND p.status='OPEN'
                ORDER BY s.score DESC
            """, (session_date,))

    def get_session_summary(self, session_date: str) -> dict | None:
        with self._connect() as conn:
            rows = self._dict_rows(conn, """
                SELECT
                    %s::text                                           AS session_date,
                    COUNT(*)                                           AS total,
                    SUM((p.status='SETTLED_WIN')::int)                 AS wins,
                    SUM((p.status='SETTLED_LOSS')::int)                AS losses,
                    SUM((p.status='OPEN')::int)                        AS open,
                    ROUND(AVG(p.final_return_pct)::numeric,4)          AS mean_return,
                    ROUND(SUM((p.status='SETTLED_WIN')::int)*100.0 /
                        NULLIF(SUM((p.status IN ('SETTLED_WIN','SETTLED_LOSS'))::int),0),1)
                                                                       AS win_pct
                FROM positions p WHERE p.session_date=%s
            """, (session_date, session_date))
        return rows[0] if rows else None

    def get_historical_performance(self, days: int = 30) -> list[dict]:
        with self._connect() as conn:
            return self._dict_rows(conn, """
                SELECT
                    p.session_date,
                    COUNT(*)                                          AS total_signals,
                    SUM((p.status='SETTLED_WIN')::int)               AS wins,
                    SUM((p.status='SETTLED_LOSS')::int)              AS losses,
                    ROUND(SUM((p.status='SETTLED_WIN')::int)*100.0 /
                        NULLIF(SUM((p.status IN ('SETTLED_WIN','SETTLED_LOSS'))::int),0),1)
                                                                      AS win_pct,
                    ROUND(AVG(p.final_return_pct)::numeric,4)         AS mean_return_pct,
                    ses.nifty_chg_pct
                FROM positions p
                JOIN sessions ses ON p.session_date=ses.session_date
                WHERE p.status NOT IN ('OPEN','EXPIRED')
                GROUP BY p.session_date, ses.nifty_chg_pct
                ORDER BY p.session_date DESC
                LIMIT %s
            """, (days,))

    def get_symbol_performance(self, symbol: str) -> list[dict]:
        with self._connect() as conn:
            return self._dict_rows(conn, """
                SELECT p.session_date, p.direction, s.score, s.conviction,
                       p.entry_price, p.exit_price, p.final_return_pct, p.status
                FROM positions p JOIN signals s USING (signal_id)
                WHERE p.symbol=%s AND p.status NOT IN ('OPEN','EXPIRED')
                ORDER BY p.session_date DESC
            """, (symbol,))

    def get_conviction_stats(self) -> list[dict]:
        with self._connect() as conn:
            return self._dict_rows(conn, """
                SELECT
                    s.conviction,
                    COUNT(*)                                         AS trades,
                    ROUND(AVG(p.final_return_pct)::numeric,4)        AS mean_return,
                    ROUND(SUM((p.status='SETTLED_WIN')::int)*100.0
                        /COUNT(*),1)                                 AS win_pct
                FROM positions p JOIN signals s USING (signal_id)
                WHERE p.status NOT IN ('OPEN','EXPIRED')
                  AND s.conviction IS NOT NULL
                GROUP BY s.conviction
                ORDER BY mean_return DESC
            """)

    def signal_exists(self, signal_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM signals WHERE signal_id=%s", (signal_id,))
                return cur.fetchone() is not None

    def session_exists(self, session_date: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM sessions WHERE session_date=%s", (session_date,))
                return cur.fetchone() is not None

    def db_stats(self) -> dict:
        with self._connect() as conn:
            stats: dict[str, Any] = {}
            with conn.cursor() as cur:
                for table in ("sessions", "signals", "positions", "price_snapshots"):
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cur.fetchone()[0]
                cur.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) AS db_size
                """)
                stats["db_size"] = cur.fetchone()[0]
        stats["db_url"] = self._dsn.split("@")[-1]  # hide credentials
        return stats

    def execute_query(self, sql: str, params=()) -> list[dict]:
        """Run any read-only SQL and return rows as dicts."""
        with self._connect() as conn:
            return self._dict_rows(conn, sql, params)


# ── Module-level singleton ────────────────────────────────────────────────────
# Lazily created on first access; reused across SignalLedger instances.
_db_instance: "SignalDB | None" = None
_db_lock = threading.Lock()


def get_db(dsn: str = DATABASE_URL) -> "SignalDB":
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = SignalDB(dsn)
    return _db_instance


# Re-export DB_PATH compat alias (no longer a path — kept for import compat)
DB_PATH = DATABASE_URL
