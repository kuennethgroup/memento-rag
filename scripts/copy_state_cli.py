#!/usr/bin/env python3
# scripts/copy_state_cli.py: is a copy of the Memento pipes enabled? (memento/library/copies.py). Exit 0: enabled; 1: not
# (installed, configured and active are all needed); 2: the database cannot be read. services.sh starts the second
# copy's gap updater only on 0.
#   ( set -a; . config/.env; set +a; .venv/bin/python scripts/copy_state_cli.py --function-id memento_query_polymer )
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--function-id", required=True)
    ap.add_argument("--role", default="query", choices=("query", "ingestion"))
    args = ap.parse_args(argv)
    from sqlalchemy import create_engine
    from memento.library import copies as cs
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("DATABASE_URL is not set")
        return 2
    state = cs.copy_state(create_engine(url), args.function_id, args.role)
    print(f"{args.function_id}: {'enabled' if state.enabled else 'configured' if state.configured else 'not set up'}"
          + (f" ({state.reason})" if state.reason else ""))
    return 0 if state.enabled else 1


if __name__ == "__main__":
    sys.exit(main())
