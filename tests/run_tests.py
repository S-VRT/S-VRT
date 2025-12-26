#!/usr/bin/env python3
"""
Test runner script for VRT project.

This script provides convenient ways to run tests with proper environment setup.
"""
import subprocess
import sys
import os
import argparse


def run_pytest(args):
    """Run pytest with given arguments."""
    # Set up environment
    env = os.environ.copy()
    # Set PYTHONPATH to project root (parent of tests directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env['PYTHONPATH'] = project_root + ':' + env.get('PYTHONPATH', '')

    # Build pytest command
    cmd = [sys.executable, '-m', 'pytest']

    # Add common pytest options
    cmd.extend([
        '--tb=short',  # Shorter traceback format
        '--strict-markers',  # Strict marker validation
        '-v',  # Verbose output
    ])

    # Add user arguments
    cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    # Run pytest from project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(cmd, env=env, cwd=project_root)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run VRT tests')
    parser.add_argument('test_path', nargs='*', default=['tests/'],
                       help='Test paths to run (default: all tests)')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--unit', action='store_true',
                       help='Run only unit tests')
    parser.add_argument('--smoke', action='store_true',
                       help='Run only smoke tests')
    parser.add_argument('--models', action='store_true',
                       help='Run only model-related tests')
    parser.add_argument('--flows', action='store_true',
                       help='Run only optical flow tests')
    parser.add_argument('--vrt', action='store_true',
                       help='Run only VRT-specific tests')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage reporting')
    parser.add_argument('--no-capture', action='store_true',
                       help='Don\'t capture output (useful for debugging)')

    args = parser.parse_args()

    pytest_args = []

    # Handle special test categories
    if args.integration:
        pytest_args.extend(['-m', 'integration'])
    elif args.unit:
        pytest_args.extend(['-m', 'unit'])
    elif args.smoke:
        pytest_args.extend(['-k', 'smoke'])
    elif args.models:
        pytest_args.extend(['tests/models/'])
    elif args.flows:
        pytest_args.extend(['-k', 'flow'])
    elif args.vrt:
        pytest_args.extend(['-k', 'vrt'])

    # Add coverage if requested
    if args.coverage:
        pytest_args.extend([
            '--cov=.',
            '--cov-report=html:htmlcov',
            '--cov-report=term-missing'
        ])

    # Add output capture control
    if args.no_capture:
        pytest_args.extend(['-s'])

    # Add test paths
    pytest_args.extend(args.test_path)

    # Run tests
    exit_code = run_pytest(pytest_args)

    # Print summary
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code: {exit_code}")
        print("\nTry running with --no-capture for more detailed output.")

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
