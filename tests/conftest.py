import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train a tiny model once into a temp dir (does not touch real artifacts)."""
    from src.train import run
    base = tmp_path_factory.mktemp("proj")
    art, data = base / "artifacts", base / "data"
    os.environ["ARTIFACT_DIR"] = str(art)
    os.environ["LOG_DIR"] = str(base / "logs")
    run(n_admissions=6000, fast=True, data_dir=data, artifact_dir=art)
    return art
