import importlib.util
import sys
import unittest
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tradingview_sync.py"
SPEC = importlib.util.spec_from_file_location("tradingview_sync_cutoff", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def transaction(trade_date: date):
    return MODULE.NormalizedTransaction(
        trade_date=trade_date,
        action="Buy",
        symbol_raw="TEST",
        description="",
        quantity="1",
        price="10",
        commission="0",
        amount="",
        broker="charles_schwab",
    )


class CutoffSelectionTests(unittest.TestCase):
    def test_transactions_on_the_cutoff_date_are_candidates(self):
        cutoff = date(2026, 11, 4)
        transactions = [
            transaction(date(2026, 11, 3)),
            transaction(cutoff),
        ]

        selected = MODULE.select_candidate_transactions(transactions, cutoff)

        self.assertEqual([tx.trade_date for tx in selected], [cutoff])


if __name__ == "__main__":
    unittest.main(verbosity=2)
