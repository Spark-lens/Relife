# Bollinger Band Reversion Strategy

Paper-trading strategy for `QQQ`, `QQQI`, `BRK.B`, `YANG`, `SQQQ`, and `SOXS`.

## Rules

- Buy when any daily, weekly, or monthly close is below the lower Bollinger Band.
- Buy size is 1% of Schwab net liquidation, rounded down to whole shares.
- Sell when the monthly close is above the monthly upper Bollinger Band.
- Sell size is 20% of the current position, at least 1 share and never more than the current position.
- Buys require `cash_available_without_margin`; margin buying power is intentionally ignored.

## Running

```bash
source ~/.zshrc >/dev/null 2>&1
conda activate istorm_rag_gpu
export ALPHA_VANTAGE_API_KEY=...
export RELIFE_SCHWAB_NET_LIQUIDATION=100000
export RELIFE_SCHWAB_CASH_AVAILABLE_WITHOUT_MARGIN=5000
python strategies/bollinger_band_reversion/run.py --once --mode paper
```

Add positions for sell sizing with environment variables such as:

```bash
export RELIFE_POSITION_QQQ=10
export RELIFE_POSITION_BRK_B=2
```

Use `--dry-run` to fetch data and evaluate orders without writing `state.sqlite`.

## Data Sources

Alpha Vantage is the primary provider because it can return `BBANDS` directly for
`daily`, `weekly`, and `monthly` intervals. `yfinance` and `AKShare` are optional
fallbacks; if they are not installed and the fallback path is reached, the run
records a provider error and fails closed for that symbol/timeframe.
