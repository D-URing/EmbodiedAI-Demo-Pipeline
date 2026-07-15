from __future__ import annotations

from pathlib import Path

import pytest

from embodied_demo.config import compose_yaml
from embodied_demo.errors import ConfigurationError


def test_extends_cycle_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="cycle"):
        compose_yaml(first)
