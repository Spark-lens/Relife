#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from portfolio_viewer.portfolio_dashboard.market_data import (
    AksharePriceProvider,
    YahooPriceProvider,
)
from portfolio_viewer.portfolio_dashboard.watchlist import check_watchlist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 Relife 观察列表的布林下轨候选信号。")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "data" / "watchlist.json",
        help="观察列表 JSON 路径。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        payload = check_watchlist(
            config,
            providers={"us": YahooPriceProvider(), "cn": AksharePriceProvider()},
            checked_at=datetime.now(ZoneInfo("Asia/Shanghai")),
        )
    except Exception as exc:
        payload = {
            "checkedAt": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "results": [],
            "alerts": [],
            "errors": [{"symbol": None, "timeframe": None, "message": str(exc)}],
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 1
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
