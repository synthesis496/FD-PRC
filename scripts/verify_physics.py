import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tests import test_physics

TESTS = [
    ("Limit Cycle Bounded", test_physics.test_limit_cycle_bounded_orbit),
    ("Dissipation Forgets", test_physics.test_dissipation_forgets),
    ("Mirror Involution", test_physics.test_mirror_involution),
    ("Fibonacci Convergence", test_physics.test_fibonacci_convergence),
    ("Healing Converges", test_physics.test_healing_converges),
]


def run_test(name, test_fn):
    try:
        test_fn()
        return f"{name:<25} | PASS"
    except AssertionError:
        return f"{name:<25} | FAIL"


def main():
    print(f"{'Physics Check':<25} | Status")
    print("-" * 35)
    results = [run_test(name, fn) for name, fn in TESTS]
    for res in results:
        print(res)


main()
