import sys
from pathlib import Path

# Ensure the project root (one level up from tests/) is on sys.path so imports work
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
