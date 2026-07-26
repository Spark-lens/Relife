import unittest

from portfolio_dashboard.schema import validate_dashboard_payload


class DashboardSchemaTest(unittest.TestCase):
    def test_requires_both_isolated_markets(self) -> None:
        payload = {
            "generatedAt": "2026-07-17T00:00:00+08:00",
            "markets": {},
        }

        with self.assertRaisesRegex(ValueError, "markets.us"):
            validate_dashboard_payload(payload)

    def test_accepts_complete_market_contracts(self) -> None:
        market = {
            "currency": "USD",
            "asOf": "2026-07-16",
            "summary": {},
            "performance": [],
            "benchmarks": [],
            "groups": [],
            "transactions": [],
            "dividends": [],
            "dividendMonths": [],
        }

        validate_dashboard_payload(
            {
                "generatedAt": "2026-07-17T00:00:00+08:00",
                "markets": {"us": market, "cn": {**market, "currency": "CNY"}},
            }
        )


if __name__ == "__main__":
    unittest.main()
