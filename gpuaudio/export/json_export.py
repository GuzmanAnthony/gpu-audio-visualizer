import json
from pathlib import Path
from typing import Dict


def write_feature_bundle_json(bundle: Dict, path: Path) -> None:
    path.write_text(json.dumps(bundle, indent=2))
