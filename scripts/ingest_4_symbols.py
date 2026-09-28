#!/usr/bin/env python3
"""Ingest 4 specific symbols from cached JSON files."""
import json, pandas as pd
from datetime import datetime, timezone
from pathlib import Path

PARQUET_DIR = Path("data/1d/1d")
SYMBOLS = ["PETRONET", "SHRIRAMFIN", "SOLARINDS", "TCS"]
TMPDIR = Path("/tmp")

for sym in SYMBOLS:
    fp = TMPDIR / f"{sym}.json"
    if not fp.exists():
        print(f"{sym}: no cached file")
        continue
    try:
        d = json.loads(fp.read_text())
        bars = d.get("data", []) if isinstance(d, dict) else (d if isinstance(d, list) else [])
        if not bars:
            print(f"{sym}: empty response")
            continue

        pf = PARQUET_DIR / f"{sym}.parquet"
        if pf.exists():
            df_ex = pd.read_parquet(pf)
            if df_ex.index.tz is None:
                df_ex.index = df_ex.index.tz_localize("UTC")
            last = df_ex.index.max()
        else:
            df_ex = None
            last = pd.Timestamp("2020-01-01", tz="UTC")

        new = [b for b in bars if datetime.fromtimestamp(int(b.get("time", 0)), tz=timezone.utc) > last.to_pydatetime()]
        if not new:
            print(f"{sym}: no new bars (last={last.date()})")
            continue

        rows = []
        for b in new:
            t = datetime.fromtimestamp(int(b["time"]), tz=timezone.utc)
            rows.append({"open": b.get("open"), "high": b.get("high"), "low": b.get("low"),
                         "close": b.get("close"), "volume": b.get("volume")})

        df_new = pd.DataFrame(rows, index=pd.DatetimeIndex(
            [datetime.fromtimestamp(int(b["time"]), tz=timezone.utc) for b in new]))
        df_new.index.name = None

        if df_ex is not None:
            common = [c for c in df_ex.columns if c in df_new.columns]
            pd.concat([df_ex[common], df_new[common]]).to_parquet(pf)
        else:
            df_new.to_parquet(pf)
        latest = datetime.fromtimestamp(int(new[-1]["time"]), tz=timezone.utc).date()
        print(f"{sym}: +{len(new)} bars → last={latest}")
    except Exception as e:
        print(f"{sym}: error — {e}")
