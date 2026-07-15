from __future__ import annotations

from embodied_demo.cli import main


def test_export_schema_cli(tmp_path, capsys) -> None:
    exit_code = main(["export-schema", "--output-dir", str(tmp_path)])
    capsys.readouterr()
    assert exit_code == 0
    assert (tmp_path / "episode_result.schema.json").is_file()
    assert (tmp_path / "training_evidence.schema.json").is_file()
    assert (tmp_path / "inference_evidence.schema.json").is_file()
