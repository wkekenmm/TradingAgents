"""Export the PIT-safe pair snapshot as a portable JSON evidence artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.pair_snapshot import build_pair_snapshot


SCHEMA_NAME = "tradingagents_pair_snapshot"
SCHEMA_VERSION = 1


def build_export_payload(
    as_of_date: str,
    *,
    source_repo: str = "wkekenmm/TradingAgents",
    source_ref: str | None = None,
) -> dict:
    snapshot = build_pair_snapshot(as_of_date)
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo": source_repo,
            "ref": source_ref,
            "role": "external_market_evidence_only",
        },
        "snapshot": snapshot,
        "contract": {
            "consumer_must_not_treat_as_model_output": True,
            "canonical_model_owner": "wkekenmm/semi_shock_reversal",
            "excluded_model_fields": ["beta", "residual", "sigma", "model_gap"],
        },
    }


def export_pair_snapshot(
    as_of_date: str,
    output: str | Path,
    *,
    source_repo: str = "wkekenmm/TradingAgents",
    source_ref: str | None = None,
) -> Path:
    payload = build_export_payload(
        as_of_date,
        source_repo=source_repo,
        source_ref=source_ref,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, dest="as_of_date")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-repo", default="wkekenmm/TradingAgents")
    parser.add_argument("--source-ref", default=None)
    args = parser.parse_args()

    path = export_pair_snapshot(
        args.as_of_date,
        args.output,
        source_repo=args.source_repo,
        source_ref=args.source_ref,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
