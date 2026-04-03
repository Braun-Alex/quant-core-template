"""
Main entry point for the Quant-Core-Template engine.
Utilizes standard logging library for production-grade event tracking.
"""
import sys
import logging

from core.wallet import WalletManager


# Logging configuration: Timestamp | Level | Message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Engine initialization started")

    try:
        wallet = WalletManager.from_env()
        logger.info(f"Wallet loaded: {wallet.address}")
    except Exception as err:
        logger.error(f"❌ Critical error ❌: could not load wallet from environment: {err}")
        sys.exit(1)

    logger.info("Engine is now running")


if __name__ == "__main__":
    main()
