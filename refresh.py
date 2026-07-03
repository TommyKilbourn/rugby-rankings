r"""
refresh.py
==========
One-shot pipeline refresh, intended for scheduled/automated runs:

    1. update_data.py  -> pull any newly played internationals from ESPN
    2. build_site.py   -> rebuild the dashboard (site/index.html)

Writes a timestamped line to logs/refresh.log and exits non-zero on failure so
a scheduler can surface problems. Run with the project venv:

    .\.venv\Scripts\python.exe refresh.py
"""
import os
import sys
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG = os.path.join(LOG_DIR, "refresh.log")


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    log("refresh start")
    try:
        import update_data
        import build_site
        update_data.main()
        build_site.main()
        log("refresh OK")
    except Exception:
        log("refresh FAILED\n" + traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
