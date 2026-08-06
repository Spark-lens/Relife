#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from relife_engine.service import build_snapshot
from relife_engine.sources import discover_latest, validate_source


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        command = payload.get("command")
        if command == "validate-source":
            target = Path(payload["path"])
            if target.is_dir():
                target = discover_latest(target, payload["market"])
            result = validate_source(target, payload["market"])
        elif command == "build-snapshot":
            result = build_snapshot(payload)
        else:
            raise ValueError(f"不支持的命令：{command!r}")
        json.dump({"ok": True, "result": result}, sys.stdout, ensure_ascii=False)
        return 0
    except Exception as exc:
        json.dump({"ok": False, "error": {"message": str(exc), "type": type(exc).__name__}}, sys.stdout, ensure_ascii=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
