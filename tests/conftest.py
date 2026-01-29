"""Pytest fixtures for LeRobot to RLDS converter tests."""

import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def v21_sample_dir(fixtures_dir: Path) -> Path:
    """Return the path to the v2.1 sample dataset."""
    return fixtures_dir / "v21_sample"


@pytest.fixture
def v30_sample_dir(fixtures_dir: Path) -> Path:
    """Return the path to the v3.0 sample dataset."""
    return fixtures_dir / "v30_sample"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Return a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
