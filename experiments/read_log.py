"""Read last N lines of a log file."""
import sys

logfile = sys.argv[1] if len(sys.argv) > 1 else "logs/remaining_experiments.log"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30

try:
    with open(logfile, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print("---LAST {} LINES---".format(n))
    for line in lines[-n:]:
        print(line.rstrip())
except FileNotFoundError:
    print(f"File not found: {logfile}")
except Exception as e:
    print(f"Error: {e}")
