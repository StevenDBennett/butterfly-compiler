"""
butterfly_compiler.py — The Butterfly Compiler

This file is a thin wrapper that imports the modular package `butterfly`
and runs the integrated test suite.

Command-line interface:
    python butterfly_compiler.py                    # run tests only
    python butterfly_compiler.py --benchmark        # run benchmarks only
    python butterfly_compiler.py --all              # run tests then benchmarks
    python butterfly_compiler.py --discover         # discover solvable seeds
"""

import argparse

from butterfly import print_discovered_seeds
from butterfly.tests.test_all import run_benchmarks, run_tests


def main():
    parser = argparse.ArgumentParser(
        description="Butterfly Compiler v1.0",
        epilog="See AGENTS.md for project details.",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks only"
    )
    parser.add_argument("--all", action="store_true", help="Run tests then benchmarks")
    parser.add_argument(
        "--discover", action="store_true", help="Discover butterfly-compilable seeds"
    )
    args = parser.parse_args()

    if args.discover:
        print("=" * 70)
        print("  Butterfly Compiler v1.0 — Seed Discovery")
        print("=" * 70)
        print_discovered_seeds()
        return

    if args.benchmark:
        run_benchmarks()
    elif args.all:
        run_tests()
        print("\n" + "=" * 70)
        run_benchmarks()
    else:
        run_tests()


if __name__ == "__main__":
    main()
