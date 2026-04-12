#!/usr/bin/env python3
"""
Test runner script for VRT project.

This script provides convenient ways to run tests with proper environment setup.
"""
import argparse
import os
import subprocess
import sys


def run_pytest(args):
    """Run pytest with given arguments."""
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "-v",
    ])
    cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, env=env, cwd=project_root)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run VRT tests")
    parser.add_argument("test_path", nargs="*", default=["tests/"], help="Test paths to run (default: all tests)")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--smoke", action="store_true", help="Run only smoke tests")
    parser.add_argument("--e2e", action="store_true", help="Run only end-to-end tests")
    parser.add_argument("--models", action="store_true", help="Run only model-related tests")
    parser.add_argument("--flows", action="store_true", help="Run only optical flow tests")
    parser.add_argument("--vrt", action="store_true", help="Run only VRT-specific tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--no-capture", action="store_true", help="Don't capture output (useful for debugging)")

    args = parser.parse_args()

    pytest_args = []

    if args.integration:
        pytest_args.extend(["-m", "integration"])
    elif args.unit:
        pytest_args.extend(["-m", "unit"])
    elif args.smoke:
        pytest_args.extend(["-m", "smoke"])
    elif args.e2e:
        pytest_args.extend(["-m", "e2e"])
    elif args.models:
        pytest_args.extend(["tests/models/"])
    elif args.flows:
        pytest_args.extend(["-k", "flow"])
    elif args.vrt:
        pytest_args.extend(["-k", "vrt"])

    if args.coverage:
        pytest_args.extend([
            "--cov=.",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
        ])

    if args.no_capture:
        pytest_args.append("-s")

    pytest_args.extend(args.test_path)

    exit_code = run_pytest(pytest_args)

    print("\n" + "=" * 60)
    if exit_code == 0:
        print("All tests passed.")
    else:
        print(f"Tests failed with exit code: {exit_code}")
        print("Try running with --no-capture for more detailed output.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
