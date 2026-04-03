import os
import sys
import trace
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET_FILE = (ROOT / "app" / "main.py").resolve()


def discover_and_run():
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def main():
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    tracer = trace.Trace(count=True, trace=False)
    result = tracer.runfunc(discover_and_run)
    counts = tracer.results().counts
    executable = set(trace._find_executable_linenos(str(TARGET_FILE)))

    executed = {
        lineno
        for (filename, lineno), count in counts.items()
        if Path(filename).resolve() == TARGET_FILE and count > 0
    }

    covered = len(executable & executed)
    total = len(executable)
    percent = 100.0 if total == 0 else covered / total * 100
    missed = sorted(executable - executed)

    print()
    print(f"Coverage for {TARGET_FILE}: {covered}/{total} lines ({percent:.2f}%)")
    if missed:
        print("Missed lines:", ", ".join(str(line) for line in missed))

    return 0 if result.wasSuccessful() and percent == 100.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
