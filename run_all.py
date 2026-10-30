"""One command to regenerate every figure and table from the raw data."""

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for script in ("plot_ftir.py", "classify_ftir.py"):
        subprocess.run([sys.executable, str(here / script)], cwd=here, check=True)
    print("Finished. See figures/ and results/.")
