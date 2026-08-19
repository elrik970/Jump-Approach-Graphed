from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

PATHS = cfg["paths"]
DETECTION = cfg["detection"]
ATHLETE = cfg["athlete"]
TRACKING = cfg["tracking"]