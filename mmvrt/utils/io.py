from pathlib import Path
from typing import List
import os


def scandir(root: str, suffix: str = "png", recursive: bool = True, full_path: bool = False) -> List[str]:
    """Scan directory for files with given suffix. Returns sorted list."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.' + suffix.lower()):
                rel = Path(dirpath) / fn
                matches.append(str(rel) if full_path else str(rel.relative_to(root_path)))
        if not recursive:
            break
    matches.sort()
    return matches




