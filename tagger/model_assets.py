from __future__ import annotations

import os
from pathlib import Path


def resolve_local_asset_pair(model_key: str, model_filename: str, tags_filename: str) -> tuple[Path, Path] | None:
    model_root = os.environ.get("WD14_MODEL_ROOT")
    if not model_root:
        return None

    base_dir = Path(model_root) / model_key
    model_path = base_dir / model_filename
    tags_path = base_dir / tags_filename
    if model_path.is_file() and tags_path.is_file():
        return model_path, tags_path
    return None
