import os
from dotenv import load_dotenv
import logging
from termcolor import colored

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    """
    uv run install.py
    nohup uv run install.py > no_git_oic/install.log 2>&1 &
    """
    # fmt: off
    os.makedirs("no_git_oic", exist_ok=True)
    logger.info(colored(f"Env install successfullly!", "green"))
    # fmt: on
