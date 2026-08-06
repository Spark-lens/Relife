from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from relife_engine.market_data import fetch_with_retry  # noqa: E402


class FetchWithRetryTests(unittest.TestCase):
    def test_success_returns_without_retry(self) -> None:
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return "ok"

        self.assertEqual(fetch_with_retry(fn), "ok")
        self.assertEqual(calls["n"], 1)

    def test_permanent_error_not_retried(self) -> None:
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise ValueError("无历史行情：该标的已退市")

        with self.assertRaises(ValueError):
            fetch_with_retry(fn)
        self.assertEqual(calls["n"], 1)

    def test_network_error_retried_up_to_three_times(self) -> None:
        calls = {"n": 0}
        permanent = ValueError("404 Not Found")

        def fn():
            calls["n"] += 1
            raise permanent

        with patch("relife_engine.market_data._is_permanent_error", return_value=False), \
             patch("relife_engine.market_data._is_retryable_error", return_value=True):
            with self.assertRaises(ValueError):
                fetch_with_retry(fn, retries=3)
        self.assertEqual(calls["n"], 3)

    def test_unknown_error_not_retried(self) -> None:
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise RuntimeError("unexpected boom")

        with patch("relife_engine.market_data._is_permanent_error", return_value=False), \
             patch("relife_engine.market_data._is_retryable_error", return_value=False):
            with self.assertRaises(RuntimeError):
                fetch_with_retry(fn)
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
