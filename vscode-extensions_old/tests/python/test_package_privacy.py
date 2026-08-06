from __future__ import annotations

import json
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VSIX = ROOT / "clannad0710.relife-0.0.1.vsix"


class PackagePrivacyTests(unittest.TestCase):
    def test_vsix_contains_only_new_extension_and_synthetic_sample(self) -> None:
        self.assertTrue(VSIX.is_file())
        with zipfile.ZipFile(VSIX) as archive:
            names = archive.namelist()
            blocked_names = ("portfolio_viewer", "data/transactions", "__pycache__", ".pyc", ".csv")
            self.assertFalse([name for name in names if any(blocked in name for blocked in blocked_names)])
            self.assertNotIn("extension/snapshot.json", names)
            package = json.loads(archive.read("extension/package.json"))
            self.assertEqual((package["publisher"], package["name"], package["version"]), ("clannad0710", "relife", "0.0.1"))
            sample = json.loads(archive.read("extension/resources/sample/portfolio-snapshot.json"))
            self.assertEqual({sample["markets"][market]["source"]["mode"] for market in ("us", "cn")}, {"sample"})
            searchable = b"\n".join(archive.read(name) for name in names if name.startswith("extension/") and not name.endswith("/"))
            for secret in (b"/mnt/d/workspace", b"C:\\Users\\", b"ALPHA_VANTAGE_API_KEY", b"tradingview_full_latest_2026-07-29"):
                self.assertNotIn(secret, searchable)


if __name__ == "__main__":
    unittest.main()
