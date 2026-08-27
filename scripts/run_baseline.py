"""
scripts/run_baseline.py — thin CLI wrapper around inference.main().

Kept as a separate entrypoint for convenience (`python scripts/run_baseline.py`)
without duplicating the Gemini-agent baseline logic, which lives in
inference.py (the file required by the OpenEnv validator at repo root).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import main

if __name__ == "__main__":
    main()
