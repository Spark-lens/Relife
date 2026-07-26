#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_dashboard.generator import ProviderBundle, generate_dashboard
from portfolio_dashboard.market_data import AksharePriceProvider, YahooPriceProvider
from portfolio_dashboard.parsers import latest_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取最新交易记录并生成投资组合仪表盘数据。"
    )
    parser.add_argument("--us-transactions", type=Path)
    parser.add_argument("--cn-transactions", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/data/portfolio.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    us_path = args.us_transactions or latest_source(
        "tradingview_full_latest_*.csv",
        REPOSITORY_ROOT / "data" / "tradingview",
    )
    cn_path = args.cn_transactions or latest_source(
        "交割单_*.csv",
        REPOSITORY_ROOT / "data" / "transactions" / "yinhe",
    )
    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    payload = generate_dashboard(
        us_path=us_path,
        cn_path=cn_path,
        output_path=output_path,
        providers=ProviderBundle(
            us=YahooPriceProvider(),
            cn=AksharePriceProvider(),
        ),
        generated_at=datetime.now(ZoneInfo("Asia/Shanghai")),
    )
    print("投资组合仪表盘数据已更新")
    print(f"- 美股交易：{us_path}")
    print(f"- A 股交易：{cn_path}")
    print(f"- 输出：{output_path}")
    print(f"- 美股行情截止：{payload['markets']['us']['asOf']}")
    print(f"- A 股行情截止：{payload['markets']['cn']['asOf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
