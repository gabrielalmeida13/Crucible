#!/usr/bin/env python3
"""One-time bulk download of SEC EDGAR companyfacts.zip and CIK mapping.

Run once before Phase 2 backtesting. The ZIP is ~1.5 GB compressed (~8 GB extracted).
Individual CIK JSON files land in data/raw/edgar/companyfacts/.

Usage:
    python scripts/download_edgar_bulk.py

Environment variables (set in .env):
    EDGAR_USER_AGENT — required; SEC blocks requests without a valid agent string.
                       Example: "Crucible yourname@example.com"
    EDGAR_DATA_DIR   — optional; defaults to data/raw/edgar
"""

from __future__ import annotations

import json
import logging
import os
import sys
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_CIK_MAPPING_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANYFACTS_URL = (
    "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
)
_CHUNK_SIZE = 8 * 1024 * 1024        # 8 MB streaming chunks
_LOG_EVERY_BYTES = 100 * 1024 * 1024  # log progress every 100 MB


def _headers() -> dict[str, str]:
    """Build SEC-compliant request headers. Raises if EDGAR_USER_AGENT is missing."""
    agent = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if not agent:
        raise ValueError(
            "EDGAR_USER_AGENT is not set in .env.\n"
            "The SEC requires a descriptive User-Agent string.\n"
            "Example: EDGAR_USER_AGENT=Crucible yourname@example.com"
        )
    return {"User-Agent": agent, "Accept-Encoding": "gzip, deflate"}


def download_cik_mapping(edgar_dir: Path) -> Path:
    """Download SEC company_tickers.json → edgar_dir/cik_mapping.json."""
    out = edgar_dir / "cik_mapping.json"
    if out.exists():
        log.info("CIK mapping already present at %s — skipping", out)
        return out

    log.info("Downloading CIK mapping from %s", _CIK_MAPPING_URL)
    resp = requests.get(_CIK_MAPPING_URL, headers=_headers(), timeout=30)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    data = json.loads(resp.content)
    log.info("CIK mapping saved → %s (%d companies)", out, len(data))
    return out


def download_companyfacts_zip(edgar_dir: Path) -> Path:
    """Stream-download companyfacts.zip to edgar_dir/, logging every 100 MB."""
    zip_path = edgar_dir / "companyfacts.zip"
    if zip_path.exists():
        size_mb = zip_path.stat().st_size // (1024 * 1024)
        log.info(
            "companyfacts.zip already present (%d MB) at %s — skipping download",
            size_mb,
            zip_path,
        )
        return zip_path

    log.info(
        "Streaming %s — this may take several minutes (~1.5 GB)", _COMPANYFACTS_URL
    )
    with requests.get(
        _COMPANYFACTS_URL, headers=_headers(), stream=True, timeout=600
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        next_log = _LOG_EVERY_BYTES
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_log:
                        pct = (
                            f"{100 * downloaded / total:.0f}%"
                            if total
                            else f"{downloaded / 1e6:.0f} MB"
                        )
                        log.info("  Downloaded %s …", pct)
                        next_log += _LOG_EVERY_BYTES

    size_mb = zip_path.stat().st_size // (1024 * 1024)
    log.info("companyfacts.zip saved → %s (%d MB)", zip_path, size_mb)
    return zip_path


def extract_companyfacts(zip_path: Path, dest_dir: Path) -> None:
    """Extract all CIK*.json files from companyfacts.zip into dest_dir."""
    existing = list(dest_dir.glob("CIK*.json"))
    if existing:
        log.info(
            "companyfacts already extracted (%d files in %s) — skipping",
            len(existing),
            dest_dir,
        )
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting companyfacts.zip → %s", dest_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        total = len(names)
        log.info("  Found %d JSON files in archive", total)
        for i, name in enumerate(names, 1):
            zf.extract(name, dest_dir)
            if i % 10_000 == 0:
                log.info(
                    "  Extracted %d / %d (%.0f%%) …", i, total, 100 * i / total
                )

    final_count = len(list(dest_dir.glob("CIK*.json")))
    log.info("Extraction complete — %d files in %s", final_count, dest_dir)


def main() -> None:
    edgar_dir = Path(os.environ.get("EDGAR_DATA_DIR", ROOT / "data" / "raw" / "edgar"))
    edgar_dir.mkdir(parents=True, exist_ok=True)
    companyfacts_dir = edgar_dir / "companyfacts"

    download_cik_mapping(edgar_dir)
    zip_path = download_companyfacts_zip(edgar_dir)
    extract_companyfacts(zip_path, companyfacts_dir)

    log.info("EDGAR bulk download complete.")
    log.info("  CIK mapping  : %s", edgar_dir / "cik_mapping.json")
    log.info("  Facts dir    : %s", companyfacts_dir)
    log.info("Next step: run scripts/run_scan.py or scripts/run_backtest.py")


if __name__ == "__main__":
    main()
