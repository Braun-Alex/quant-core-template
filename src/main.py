"""
Main entry point for the Quant-Core-Template engine.
Utilizes standard logging library for production-grade event tracking.
"""

import os
import logging
from dotenv import load_dotenv

# Logging configuration: Timestamp | Level | Message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def initialize_engine():
    """
    Load environment variables and prepare configuration.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    config = {
        "api_key": api_key,
    }
    return config


def main():
    logger.info("Engine initialization started")

    try:
        config = initialize_engine()
        openai_api_key = config["api_key"]

        if openai_api_key:
            # Masking API key for logs to prevent accidental exposure
            masked_key = f"{openai_api_key[:6]}****"
            logger.info(f"OpenAI service: key detected "
                        f"and loaded ({masked_key})")
        else:
            # Non-blocking warning
            logger.warning(
                "OpenAI Service: API key is missing. "
                "AI features will be unavailable."
            )

        logger.info("Engine is now running")

    except Exception as error:
        logger.critical(f"Critical failure during "
                        f"startup: {error}", exc_info=True)


if __name__ == "__main__":
    main()
