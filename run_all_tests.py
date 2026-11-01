"""Run every lesson's test suite and print one summary table.

Each lesson is self-contained and several share module names (fakes.py, test_agent.py, ...), so they cannot run in one
pytest session. This script gives each lesson its own process, from inside its own folder.

    python run_all_tests.py             # all lessons
    python run_all_tests.py 04 09       # only lessons whose folder name contains "04" or "09"

The tests are offline: no API key, no network, no cost.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def main(filters: list[str]) -> int:
    root = Path(__file__).resolve().parent
    lessons = sorted(p for p in root.glob("lesson-*") if p.is_dir() and (not filters or any(f in p.name for f in filters)))
    if not lessons:
        print("no lessons matched")
        return 2
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    rows, total, all_ok = [], 0, True
    for lesson in lessons:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                              cwd=lesson, capture_output=True, text=True, env=env)
        last = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", last)) else 0
        ok = proc.returncode == 0
        all_ok &= ok
        total += passed
        rows.append((lesson.name, "PASS" if ok else "FAIL", last))
        if not ok:
            print(proc.stdout[-2000:], proc.stderr[-1000:], sep="\n")
    width = max(len(r[0]) for r in rows)
    for name, status, last in rows:
        print(f"{status}  {name:<{width}}  {last}")
    print(f"\n{total} tests passed across {len(rows)} lessons" if all_ok else "\nSOME LESSONS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
