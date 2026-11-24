import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGETS = sorted(ROOT.glob("fig*.py"))


def main():
    for path in TARGETS:
        print(f"[run_all] {path.name}")
        runpy.run_path(str(path), run_name="__main__")
    print(f"[run_all] done -> {len(TARGETS)} figures")


main()
