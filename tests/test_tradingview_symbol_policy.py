import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tradingview_sync.py"
MAP_PATH = REPO_ROOT / "data" / "templates" / "tradingview" / "symbol_map.json"
SPEC = importlib.util.spec_from_file_location("tradingview_sync", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def transaction(symbol: str):
    return MODULE.NormalizedTransaction(
        trade_date=date(2026, 11, 3),
        action="Buy",
        symbol_raw=symbol,
        description="",
        quantity="1",
        price="10",
        commission="0",
        amount="",
        broker="charles_schwab",
    )


class SymbolPolicyTests(unittest.TestCase):
    def test_unknown_symbol_is_rejected_without_an_explicit_exchange(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            map_path = Path(temp_dir) / "symbol_map.json"
            map_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NEW"):
                MODULE.ensure_symbol_map_for_transactions(
                    [], {}, map_path, [transaction("NEW")], default_exchange=""
                )
            self.assertEqual(json.loads(map_path.read_text(encoding="utf-8")), {})

    def test_explicit_exchange_can_opt_into_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            map_path = Path(temp_dir) / "symbol_map.json"
            map_path.write_text("{}\n", encoding="utf-8")
            symbol_map, additions = MODULE.ensure_symbol_map_for_transactions(
                [], {}, map_path, [transaction("NEW")], default_exchange="NYSE"
            )
            self.assertEqual(symbol_map["NEW"], "NYSE:NEW")
            self.assertEqual(additions, {"NEW": "NYSE:NEW"})
            self.assertEqual(
                json.loads(map_path.read_text(encoding="utf-8")),
                {"NEW": "NYSE:NEW"},
            )

    def test_symbol_map_rejects_a_bare_ticker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            map_path = Path(temp_dir) / "symbol_map.json"
            map_path.write_text('{"BAD": "NASDAQ"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "交易所"):
                MODULE.load_symbol_overrides(map_path)

    def test_audited_america_listings_have_explicit_exchange_prefixes(self):
        symbol_map = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        expected = {
            "BOXX": "CBOE:BOXX",
            "BRKB": "NYSE:BRK.B",
            "HSBC": "NYSE:HSBC",
            "SPYI": "CBOE:SPYI",
            "XBI": "AMEX:XBI",
        }
        for raw_symbol, tv_symbol in expected.items():
            self.assertEqual(symbol_map.get(raw_symbol), tv_symbol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
