"""Entry point for the monthly scan pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Resolve project root so the script works from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from crucible.config import load_config  # noqa: E402  (after dotenv load)
from crucible.logging_setup import configure_logging  # noqa: E402


def main() -> None:
    """Run the full monthly scan pipeline."""
    config = load_config()
    configure_logging(config.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Crucible scan starting — universe=%s", config.universe)
    logger.warning(
        "Pipeline not yet implemented (Phase 0). "
        "No data will be fetched or scored."
    )
    logger.info("Crucible scan finished.")


if __name__ == "__main__":
    sys.exit(main())
