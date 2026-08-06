from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from relife_engine.service import build_snapshot

HEADERS = ["Symbol", "Side", "Qty", "Fill Price", "Commission", "Closing Time"]


class ServiceTests(unittest.TestCase):
    def test_one_real_market_uses_latest_file_and_other_market_stays_sample(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample = root / "sample.json"
            sample.write_text(json.dumps({"markets": {"us": {"market": "us"}, "cn": {"market": "cn", "source": {"mode": "sample"}}}}), encoding="utf-8")
            for stamp, symbol in (("2026-07-01", "OLD"), ("2026-08-01", "DEMO")):
                path = root / f"tradingview_full_latest_{stamp}.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=HEADERS)
                    writer.writeheader()
                    writer.writerow({"Symbol": symbol, "Side": "Deposit", "Qty": "100", "Fill Price": "", "Commission": "", "Closing Time": "2026-08-01 09:00:00"})
                    writer.writerow({"Symbol": symbol, "Side": "Buy", "Qty": "1", "Fill Price": "10", "Commission": "0", "Closing Time": "2026-08-01 09:01:00"})
            provider_result = (
                {"DEMO": {"2026-08-01": Decimal("11")}, "SPY": {"2026-08-01": Decimal("500")}},
                {"2026-08-01": Decimal("6000")}, [], [],
            )
            with patch("relife_engine.service.load_market_data", return_value=provider_result) as provider:
                result = build_snapshot({
                    "samplePath": str(sample),
                    "sources": {"us": {"mode": "directory", "directory": str(root)}, "cn": {"mode": "sample"}},
                    "watchlist": {"categories": [{"symbols": [{"market": "us", "symbol": "SPY"}]}]},
                })
            self.assertEqual(result["markets"]["us"]["source"]["label"], "tradingview_full_latest_2026-08-01.csv")
            self.assertEqual(result["markets"]["us"]["holdings"][0]["symbol"], "DEMO")
            self.assertEqual(result["markets"]["cn"]["source"]["label"], "示例数据")
            self.assertEqual(provider.call_args.args, ("us", ["DEMO", "SPY"]))


if __name__ == "__main__":
    unittest.main()
