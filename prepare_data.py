#!/usr/bin/env python
"""Data preparation entry (stub)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
try:
    from mmvrt.config.parser import build_argparser, load_config  # type: ignore
    from mmvrt.tools.prepare_data import prepare  # type: ignore
except Exception:
    sys.path.insert(0, str(ROOT / "src"))
    from config.parser import build_argparser, load_config  # noqa: E402
    from tools.prepare_data import prepare  # noqa: E402

# If a packaged prepare() is not available (partial refactor), provide a safe stub
if "prepare" not in globals() or prepare is None:  # type: ignore[name-defined]
    def prepare(cfg):
        """Fallback no-op prepare: print informational message and return."""
        print("prepare_data: no packaged prepare() found; skipping data preparation (noop).")
        return


def main():
    parser = build_argparser()
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    prepare(cfg)


if __name__ == "__main__":
    main()


