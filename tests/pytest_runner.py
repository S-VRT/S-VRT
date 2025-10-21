#!/usr/bin/env python3
"""
Pytest runner with custom configuration.

Usage:
    python tests/pytest_runner.py              # Run all tests
    python tests/pytest_runner.py unit         # Run only unit tests
    python tests/pytest_runner.py integration  # Run only integration tests
    python tests/pytest_runner.py benchmark    # Run only benchmarks
"""
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_tests(category=None):
    """Run pytest with specified category."""
    
    cmd = ["pytest", "tests/"]
    
    if category == "unit":
        cmd = ["pytest", "tests/unit/", "-v"]
        print("Running unit tests...")
    elif category == "integration":
        cmd = ["pytest", "tests/integration/", "-v"]
        print("Running integration tests...")
    elif category == "benchmark":
        cmd = ["pytest", "tests/benchmark/", "-v"]
        print("Running benchmarks...")
    elif category == "quick":
        cmd = ["pytest", "tests/unit/", "tests/integration/", "-v"]
        print("Running quick tests (unit + integration)...")
    elif category == "all":
        cmd = ["pytest", "tests/", "-v"]
        print("Running all tests...")
    else:
        cmd = ["pytest", "tests/", "-v"]
        print("Running all tests...")
    
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def main():
    """Main entry point."""
    category = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    print("="*70)
    print("VRT+Spike Video Deblurring - Test Runner")
    print("="*70)
    
    exit_code = run_tests(category)
    
    if exit_code == 0:
        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ Some tests failed")
        print("="*70)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

