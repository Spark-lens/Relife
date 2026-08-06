from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from .market_data import load_market_data
from .snapshot import build_market_snapshot
from .sources import discover_latest, parse_source


def build_snapshot(payload: dict) -> dict:
    sample = json.loads(Path(payload["samplePath"]).read_text(encoding="utf-8"))
    watchlist = payload.get("watchlist", {"categories": []})
    snapshot = {"generatedAt": datetime.now(timezone.utc).isoformat(), "markets": {}, "errors": [], "cache": {"stale": False, "lastSuccessAt": None}}
    for market in ("us", "cn"):
        source = payload.get("sources", {}).get(market, {"mode": "sample"})
        if source.get("mode") == "sample":
            market_snapshot = copy.deepcopy(sample["markets"][market])
            market_snapshot["source"] = {"mode": "sample", "label": "示例数据"}
            snapshot["markets"][market] = market_snapshot
            continue
        path = discover_latest(Path(source["directory"]), market)
        transactions = parse_source(path, market)
        symbols = {item["symbol"] for item in transactions if item["symbol"] and item["kind"] in {"buy", "sell", "dividend"}}
        symbols.update(
            symbol["symbol"]
            for category in watchlist.get("categories", [])
            for symbol in category.get("symbols", [])
            if symbol.get("market") == market
        )
        closes, benchmark, calendar, errors = load_market_data(market, sorted(symbols))
        market_snapshot = build_market_snapshot(
            market, transactions, closes, benchmark, calendar, errors,
            source={"mode": "directory", "label": path.name, "file": str(path)},
        )
        snapshot["markets"][market] = market_snapshot
        snapshot["errors"].extend({"market": market, **error} for error in errors)
    snapshot["cache"]["lastSuccessAt"] = snapshot["generatedAt"]
    return snapshot
