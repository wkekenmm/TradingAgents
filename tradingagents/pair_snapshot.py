"""Point-in-time safe multi-asset snapshot for relative-flow analysis.

This module deliberately does *not* reproduce the Sigma Monitor's beta,
residual, or sigma model.  It only provides synchronized market observations
and simple cumulative relative returns.  The canonical model can consume this
snapshot without risking definition drift between repositories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv


DEFAULT_SYMBOLS = ("SOXX", "QQQ", "QLD", "SPY", "IEF")
DEFAULT_HORIZONS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class PairSnapshotConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    horizons: tuple[int, ...] = DEFAULT_HORIZONS


def _clean_symbol_frame(symbol: str, as_of_date: str) -> pd.DataFrame:
    """Return settled Date/Close rows through ``as_of_date`` for one symbol."""
    frame = load_ohlcv(symbol, as_of_date, fill_gaps=False)
    if frame.empty or "Date" not in frame.columns or "Close" not in frame.columns:
        raise ValueError(f"{symbol}: usable Date/Close data unavailable")

    out = frame[["Date", "Close"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"])
    out = out.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    return out.rename(columns={"Close": symbol}).set_index("Date")


def _aligned_closes(symbols: Iterable[str], as_of_date: str) -> pd.DataFrame:
    """Inner-join settled closes so every observation is from the same session."""
    frames = [_clean_symbol_frame(symbol, as_of_date) for symbol in symbols]
    aligned = pd.concat(frames, axis=1, join="inner").dropna().sort_index()
    if aligned.empty:
        raise ValueError("No common settled session across requested symbols")
    return aligned


def _cum_return(closes: pd.Series, horizon: int) -> float:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(closes) <= horizon:
        raise ValueError(
            f"Need at least {horizon + 1} common sessions; got {len(closes)}"
        )
    return float(closes.iloc[-1] / closes.iloc[-(horizon + 1)] - 1.0)


def build_pair_snapshot(
    as_of_date: str,
    *,
    config: PairSnapshotConfig | None = None,
) -> dict:
    """Build a deterministic, point-in-time-safe relative-flow snapshot.

    Returns decimal returns (0.01 == 1%).  Relative values are simple return
    differences in decimal percentage points, not a trading recommendation.

    The module intentionally excludes beta/residual/sigma calculations.  Those
    remain owned by the Sigma Monitor's canonical model.
    """
    cfg = config or PairSnapshotConfig()
    symbols = tuple(symbol.upper() for symbol in cfg.symbols)
    horizons = tuple(sorted(set(int(h) for h in cfg.horizons)))

    required = {"SOXX", "QQQ", "QLD", "SPY", "IEF"}
    missing = required.difference(symbols)
    if missing:
        raise ValueError(f"Required symbols missing: {sorted(missing)}")

    aligned = _aligned_closes(symbols, as_of_date)
    max_horizon = max(horizons)
    if len(aligned) <= max_horizon:
        raise ValueError(
            f"Need at least {max_horizon + 1} common sessions; got {len(aligned)}"
        )

    returns: dict[str, dict[str, float]] = {}
    relative: dict[str, dict[str, float]] = {}

    for horizon in horizons:
        label = f"{horizon}d"
        by_symbol = {
            symbol: _cum_return(aligned[symbol], horizon) for symbol in symbols
        }
        returns[label] = by_symbol
        relative[label] = {
            # Semiconductor versus executable 2x Nasdaq hedge proxy.
            "soxx_minus_qld": by_symbol["SOXX"] - by_symbol["QLD"],
            # Semiconductor versus unlevered Nasdaq benchmark.
            "soxx_minus_qqq": by_symbol["SOXX"] - by_symbol["QQQ"],
            # Nasdaq risk versus broad equity.
            "qld_minus_spy": by_symbol["QLD"] - by_symbol["SPY"],
            "qqq_minus_spy": by_symbol["QQQ"] - by_symbol["SPY"],
            # Broad equity versus intermediate Treasuries: coarse risk extraction proxy.
            "spy_minus_ief": by_symbol["SPY"] - by_symbol["IEF"],
        }

    latest = aligned.iloc[-1]
    return {
        "schema_version": 1,
        "requested_as_of": str(pd.Timestamp(as_of_date).date()),
        "latest_common_session": str(aligned.index[-1].date()),
        "common_sessions": int(len(aligned)),
        "symbols": list(symbols),
        "latest_close": {symbol: float(latest[symbol]) for symbol in symbols},
        "returns": returns,
        "relative": relative,
        "definitions": {
            "return_unit": "decimal; 0.01 = 1%",
            "relative_unit": "decimal return difference",
            "alignment": "inner join of settled sessions across all symbols",
            "look_ahead": "load_ohlcv filters rows after requested_as_of",
            "model_boundary": "beta/residual/sigma intentionally excluded; Sigma Monitor remains canonical",
        },
    }
