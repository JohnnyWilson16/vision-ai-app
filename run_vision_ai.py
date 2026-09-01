#!/usr/bin/env python3
"""
Vision AI Runner Entrypoint
Author: Johnny Wilson Dougherty
"""

import sys
from pathlib import Path

# Ensure src/ is on python path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from vision_ai.cli import main

if __name__ == "__main__":
    main()
