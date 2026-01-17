import logging
from datetime import datetime, timezone

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=(level or "INFO").upper(),
        format="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
    )
    logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()
