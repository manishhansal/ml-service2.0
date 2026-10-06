"""
src.data.signal_db — Dedicated SQLite database for ml-service2.0 signals
=========================================================================

Stores the complete signal lifecycle for every intraday session:
  sessions      — one row per trading day
  signals       — immutable: one row per (symbol, session_date), created at open
  positions     — mutable: entry/exit/outcome, updated throughout the day
  price_snapshots — mark-to-market history every ~2 minutes

Database location:  data/ml_signals.db   (gitignored alongside parquets)

Quick reference — common queries:
  # Today's open positions
  SELECT * FROM positions WHERE session_date='2026-10-06' AND status='OPEN';

  # Win rate by session
  SELECT session_date,
         SUM(status='SETTLED_WIN') AS wins,
         COUNT(*) AS total,
         ROUND(SUM(status='SETTLED_WIN')*100.0/COUNT(*),1) AS win_pct,
         ROUND(AVG(final_return_pct),3) AS mean_ret
  FROM positions WHERE status!='OPEN'
  GROUP BY session_date ORDER BY session_date DESC;

  # Best performing symbols historically
  SELECT symbol, COUNT(*) AS trades,
         ROUND(AVG(final_return_pct),3) AS avg_ret,
         SUM(status='SETTLED_WIN') AS wins
  FROM positions WHERE status NOT IN ('OPEN','EXPIRED')
  GROUP BY symbol HAVING trades>=5 ORDER BY avg_ret DESC LIMIT 20;

  # Unrealized P&L right now
  SELECT symbol, direction, entry_price, last_price, unrealized_pct
  FROM positions WHERE status='OPEN' ORDER BY unrealized_pct DESC;

  # Conviction grade performance
  SELECT s.conviction, COUNT(*) AS trades,
         ROUND(AVG(p.final_return_pct),3) AS mean_ret,
         ROUND(SUM(p.status='SETTLED_WIN')*100.0/COUNT(*),1) AS win_pct
  FROM positions p JOIN signals s USING (signal_id)
  WHERE p.status NOT IN ('OPEN','EXPIRED')
  GROUP BY s.conviction ORDER BY mean_ret DESC;
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_BASE = Path(__file__).parent.parent.parent
DB_PATH = _BASE / "data" / "ml_signals.db"

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

-- ── Sessions ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    session_date     TEXT PRIMARY KEY,        -- 'YYYY-MM-DD'
    nifty_open       REAL,
    nifty_close      REAL,
    nifty_chg_pct    REAL,
    n_scored         INTEGER DEFAULT 0,
    n_long           INTEGER DEFAULT 0,
    n_short          INTEGER DEFAULT 0,
    model_version    TEXT,
    created_at       TEXT NOT NULL
);

-- ── Signals (immutable — written once at session open) ────────────────────────
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
CREATE INDEX IF NOT EXISTS idx_signals_session  ON signals(session_date);
CREATE INDEX IF NOT EXISTS idx_signals_sym_sess ON signals(symbol, session_date);

-- ── Positions (mutable — entry/exit/outcome, updated throughout day) ──────────
CREATE TABLE IF NOT EXISTS positions (
    signal_id        TEXT PRIMARY KEY REFERENCES signals(signal_id),
    session_date     TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    direction        INTEGER NOT NULL,
    entry_time       TEXT,                  -- ISO UTC when first recorded
    entry_price      REAL,
    exit_time        TEXT,                  -- ISO UTC at settlement
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
CREATE INDEX IF NOT EXISTS idx_pos_session       ON positions(session_date);
CREATE INDEX IF NOT EXISTS idx_pos_session_status ON positions(session_date, status);
CREATE INDEX IF NOT EXISTS idx_pos_symbol        ON positions(symbol);

-- ── Price snapshots (MTM history, ~285 symbols × every 2 min) ────────────────
CREATE TABLE IF NOT EXISTS price_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
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
    Thread-safe SQLite wrapper for ml-service2.0 signal storage.

    Uses WAL mode for concurrent reads + single writer.
    All write methods are atomic (single transaction).
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

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
            conn.execute("""
                INSERT INTO sessions(session_date,nifty_close,nifty_chg_pct,
                    n_scored,n_long,n_short,model_version,created_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(session_date) DO UPDATE SET
                    nifty_close=excluded.nifty_close,
                    nifty_chg_pct=excluded.nifty_chg_pct,
                    n_scored=excluded.n_scored,
                    n_long=excluded.n_long,
                    n_short=excluded.n_short,
                    model_version=COALESCE(excluded.model_version, model_version)
            """, (session_date, nifty_ltp, nifty_chg_pct,
                  n_scored, n_long, n_short, model_version,
                  datetime.now(tz=timezone.utc).isoformat()))

    # ── Signals ───────────────────────────────────────────────────────────────

    def insert_signal(self, sig: dict) -> bool:
        """Insert one signal. Returns True if inserted, False if already exists."""
        with _lock, self._connect() as conn:
            try:
                conn.execute("""
                    INSERT INTO signals(signal_id,session_date,symbol,generated_at,
                        data_date,direction,score,conviction,rank,cost_bps,
                        nifty_at_signal,model_version)
                    VALUES(:signal_id,:session_date,:symbol,:generated_at,
                        :data_date,:direction,:score,:conviction,:rank,:cost_bps,
                        :nifty_at_signal,:model_version)
                """, sig)
                return True
            except sqlite3.IntegrityError:
                return False

    def insert_signals_batch(self, signals: list[dict]) -> int:
        """Insert many signals. Returns count inserted."""
        inserted = 0
        with _lock, self._connect() as conn:
            for sig in signals:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO signals(signal_id,session_date,symbol,
                            generated_at,data_date,direction,score,conviction,rank,
                            cost_bps,nifty_at_signal,model_version)
                        VALUES(:signal_id,:session_date,:symbol,:generated_at,
                            :data_date,:direction,:score,:conviction,:rank,
                            :cost_bps,:nifty_at_signal,:model_version)
                    """, sig)
                    if conn.rowcount > 0:
                        inserted += 1
                except sqlite3.IntegrityError:
                    pass
        return inserted

    # ── Positions ─────────────────────────────────────────────────────────────

    def upsert_position(self, pos: dict) -> None:
        with _lock, self._connect() as conn:
            conn.execute("""
                INSERT INTO positions(signal_id,session_date,symbol,direction,
                    entry_time,entry_price,exit_time,exit_price,last_price,
                    last_price_time,unrealized_pct,status,settled_at,
                    final_return_pct,outcome)
                VALUES(:signal_id,:session_date,:symbol,:direction,
                    :entry_time,:entry_price,:exit_time,:exit_price,:last_price,
                    :last_price_time,:unrealized_pct,:status,:settled_at,
                    :final_return_pct,:outcome)
                ON CONFLICT(signal_id) DO UPDATE SET
                    entry_price=COALESCE(:entry_price, entry_price),
                    exit_time=COALESCE(:exit_time, exit_time),
                    exit_price=COALESCE(:exit_price, exit_price),
                    last_price=COALESCE(:last_price, last_price),
                    last_price_time=COALESCE(:last_price_time, last_price_time),
                    unrealized_pct=COALESCE(:unrealized_pct, unrealized_pct),
                    status=:status,
                    settled_at=COALESCE(:settled_at, settled_at),
                    final_return_pct=COALESCE(:final_return_pct, final_return_pct),
                    outcome=COALESCE(:outcome, outcome)
            """, pos)

    def update_prices_batch(
        self,
        session_date: str,
        price_updates: list[tuple[float, float | None, str, str, str]],
    ) -> int:
        """Bulk update last_price + unrealized_pct for open positions.
        price_updates: [(last_price, unrealized_pct, last_price_time, symbol, session_date)]
        """
        with _lock, self._connect() as conn:
            conn.executemany("""
                UPDATE positions
                SET last_price=?, unrealized_pct=?, last_price_time=?
                WHERE symbol=? AND session_date=? AND status='OPEN'
            """, price_updates)
            return conn.rowcount

    def settle_position(self, signal_id: str, pos: dict) -> None:
        with _lock, self._connect() as conn:
            conn.execute("""
                UPDATE positions SET
                    exit_time=:exit_time, exit_price=:exit_price,
                    status=:status, settled_at=:settled_at,
                    final_return_pct=:final_return_pct, outcome=:outcome,
                    unrealized_pct=:final_return_pct
                WHERE signal_id=:signal_id
            """, {**pos, "signal_id": signal_id})

    # ── Price Snapshots ───────────────────────────────────────────────────────

    def insert_snapshot_batch(self, session_date: str, snapshots: list[dict]) -> None:
        with _lock, self._connect() as conn:
            conn.executemany("""
                INSERT INTO price_snapshots(session_date,symbol,snapshot_time,price,unrealized_pct)
                VALUES(:session_date,:symbol,:snapshot_time,:price,:unrealized_pct)
            """, snapshots)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_open_positions(self, session_date: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT p.*, s.score, s.conviction, s.rank, s.generated_at
                FROM positions p JOIN signals s USING (signal_id)
                WHERE p.session_date=? AND p.status='OPEN'
                ORDER BY s.score DESC
            """, (session_date,)).fetchall()
        return [dict(r) for r in rows]

    def get_session_summary(self, session_date: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    p.session_date,
                    COUNT(*) AS total,
                    SUM(p.status='SETTLED_WIN')   AS wins,
                    SUM(p.status='SETTLED_LOSS')  AS losses,
                    SUM(p.status='OPEN')          AS open,
                    ROUND(AVG(p.final_return_pct),4) AS mean_return,
                    ROUND(SUM(p.status='SETTLED_WIN')*100.0 /
                        NULLIF(SUM(p.status IN ('SETTLED_WIN','SETTLED_LOSS')),0),1) AS win_pct
                FROM positions p
                WHERE p.session_date=?
            """, (session_date,)).fetchone()
        return dict(row) if row else None

    def get_historical_performance(self, days: int = 30) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT
                    p.session_date,
                    COUNT(*) AS total_signals,
                    SUM(p.status='SETTLED_WIN')   AS wins,
                    SUM(p.status='SETTLED_LOSS')  AS losses,
                    ROUND(SUM(p.status='SETTLED_WIN')*100.0 /
                        NULLIF(SUM(p.status IN ('SETTLED_WIN','SETTLED_LOSS')),0),1) AS win_pct,
                    ROUND(AVG(p.final_return_pct),4) AS mean_return_pct,
                    ses.nifty_chg_pct
                FROM positions p
                JOIN sessions ses ON p.session_date=ses.session_date
                WHERE p.status NOT IN ('OPEN','EXPIRED')
                GROUP BY p.session_date
                ORDER BY p.session_date DESC
                LIMIT ?
            """, (days,)).fetchall()
        return [dict(r) for r in rows]

    def get_symbol_performance(self, symbol: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT p.session_date, p.direction, s.score, s.conviction,
                       p.entry_price, p.exit_price, p.final_return_pct, p.status
                FROM positions p JOIN signals s USING (signal_id)
                WHERE p.symbol=? AND p.status NOT IN ('OPEN','EXPIRED')
                ORDER BY p.session_date DESC
            """, (symbol,)).fetchall()
        return [dict(r) for r in rows]

    def get_conviction_stats(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT
                    s.conviction,
                    COUNT(*) AS trades,
                    ROUND(AVG(p.final_return_pct),4) AS mean_return,
                    ROUND(SUM(p.status='SETTLED_WIN')*100.0/COUNT(*),1) AS win_pct
                FROM positions p JOIN signals s USING (signal_id)
                WHERE p.status NOT IN ('OPEN','EXPIRED')
                  AND s.conviction IS NOT NULL
                GROUP BY s.conviction
                ORDER BY mean_return DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def session_exists(self, session_date: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_date=?", (session_date,)
            ).fetchone()
        return row is not None

    def signal_exists(self, signal_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM signals WHERE signal_id=?", (signal_id,)
            ).fetchone()
        return row is not None

    def db_stats(self) -> dict:
        with self._connect() as conn:
            stats = {}
            for table in ("sessions", "signals", "positions", "price_snapshots"):
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count
            db_size = self._path.stat().st_size if self._path.exists() else 0
        return {**stats, "db_size_kb": round(db_size / 1024, 1), "db_path": str(self._path)}
