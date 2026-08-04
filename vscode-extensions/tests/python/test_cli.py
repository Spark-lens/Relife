from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "python" / "relife_cli.py"
HEADERS = ["Symbol", "Side", "Qty", "Fill Price", "Commission", "Closing Time"]


class CliTests(unittest.TestCase):
    def run_cli(self, payload: dict) -> tuple[int, dict]:
        process = subprocess.run(
            [sys.executable, str(CLI)], input=json.dumps(payload), text=True,
            capture_output=True, check=False,
        )
        return process.returncode, json.loads(process.stdout)

    def test_validate_source_json_contract(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tradingview_full_latest_2026-08-01.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=HEADERS)
                writer.writeheader()
                writer.writerow({"Symbol": "DEMO", "Side": "Deposit", "Qty": "100", "Fill Price": "", "Commission": "", "Closing Time": "2026-08-01 09:00:00"})
            code, response = self.run_cli({"command": "validate-source", "market": "us", "path": str(path)})
            self.assertEqual(code, 0)
            self.assertTrue(response["ok"])
            self.assertEqual(response["result"]["recordCount"], 1)

    def test_build_snapshot_can_mix_bundled_sample_markets(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            sample = Path(folder) / "sample.json"
            sample.write_text(json.dumps({"generatedAt": "demo", "markets": {"us": {"market": "us"}, "cn": {"market": "cn"}}}), encoding="utf-8")
            code, response = self.run_cli({
                "command": "build-snapshot", "samplePath": str(sample),
                "sources": {"us": {"mode": "sample"}, "cn": {"mode": "sample"}},
                "watchlist": {"categories": []},
            })
            self.assertEqual(code, 0)
            self.assertEqual(response["result"]["markets"]["us"]["source"]["label"], "示例数据")
            self.assertEqual(response["result"]["markets"]["cn"]["source"]["label"], "示例数据")


if __name__ == "__main__":
    unittest.main()
