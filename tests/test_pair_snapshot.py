import pandas as pd
import pytest

import tradingagents.pair_snapshot as ps


def _frame(multiplier: float) -> pd.DataFrame:
    dates = pd.bdate_range("2026-09-30", periods=25)
    closes = [100.0 * multiplier + i * multiplier for i in range(25)]
    return pd.DataFrame({"Date": dates, "Close": closes})


def test_pair_snapshot_aligns_common_sessions_and_computes_spreads(monkeypatch):
    multipliers = {
        "SOXX": 1.0,
        "QQQ": 2.0,
        "QLD": 3.0,
        "SPY": 4.0,
        "IEF": 5.0,
    }

    def fake_load(symbol, as_of_date, fill_gaps=False):
        assert as_of_date == "2026-11-02"
        assert fill_gaps is False
        return _frame(multipliers[symbol])

    monkeypatch.setattr(ps, "load_ohlcv", fake_load)
    snap = ps.build_pair_snapshot("2026-11-02")

    assert snap["schema_version"] == 1
    assert snap["requested_as_of"] == "2026-11-02"
    assert snap["common_sessions"] == 25
    assert set(snap["returns"]) == {"1d", "3d", "5d", "10d", "20d"}

    one = snap["returns"]["1d"]
    rel = snap["relative"]["1d"]
    assert rel["soxx_minus_qld"] == pytest.approx(one["SOXX"] - one["QLD"])
    assert rel["qqq_minus_spy"] == pytest.approx(one["QQQ"] - one["SPY"])
    assert rel["spy_minus_ief"] == pytest.approx(one["SPY"] - one["IEF"])


def test_pair_snapshot_uses_only_common_sessions(monkeypatch):
    base = _frame(1.0)

    def fake_load(symbol, as_of_date, fill_gaps=False):
        frame = base.copy()
        if symbol == "QLD":
            frame = frame.drop(index=10).reset_index(drop=True)
        return frame

    monkeypatch.setattr(ps, "load_ohlcv", fake_load)
    snap = ps.build_pair_snapshot("2026-11-02")
    assert snap["common_sessions"] == 24


def test_pair_snapshot_does_not_reimplement_sigma_model(monkeypatch):
    monkeypatch.setattr(ps, "load_ohlcv", lambda *args, **kwargs: _frame(1.0))
    snap = ps.build_pair_snapshot("2026-11-02")
    assert "beta" not in snap
    assert "sigma" not in snap
    assert "residual" not in snap
    assert "Sigma Monitor remains canonical" in snap["definitions"]["model_boundary"]


def test_pair_snapshot_requires_full_symbol_set():
    cfg = ps.PairSnapshotConfig(symbols=("SOXX", "QQQ", "QLD", "SPY"))
    with pytest.raises(ValueError, match="IEF"):
        ps.build_pair_snapshot("2026-11-02", config=cfg)
