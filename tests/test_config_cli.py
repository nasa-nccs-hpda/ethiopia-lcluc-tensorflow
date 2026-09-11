import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ethiopia_lcluc_tensorflow.utils.config import composite_config, load_config, resolved_config_file


def test_overrides_environment_and_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ETHIOPIA_OUTPUT", str(tmp_path / "results"))
    path = tmp_path / "run.yaml"
    path.write_text('output_dir: ${oc.env:ETHIOPIA_OUTPUT}\nstart_year: 2009\n')
    config = load_config(path, ["start_year=2017", "nested.value=3"])
    with resolved_config_file(config) as filename:
        assert OmegaConf.load(filename).output_dir == str(tmp_path / "results")
        assert "oc.env" not in Path(filename).read_text()
        assert OmegaConf.load(filename).nested.value == 3
    assert not Path(filename).exists()
    assert OmegaConf.load(path).start_year == 2009
    with pytest.raises(ValueError, match="KEY=VALUE"):
        load_config(path, ["no-equals"])


def test_legacy_composite_alias_defaults_and_years(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text('output_dir: output\ngrid_path: grid.gpkg\nstart_year: 2009\nend_year: 2016\n')
    config = composite_config(path)
    assert config.grid_filename == "grid.gpkg"
    assert config.filter_months is False
    assert config.overwrite_tifs is False
    with pytest.raises(ValueError, match="inclusive"):
        composite_config(path, ["end_year=2008"])
    with pytest.raises(ValueError, match="not implemented"):
        composite_config(path, ["post_process_combine=true"])


@pytest.mark.parametrize("module", ["landcover_cnn_pipeline_cli", "landcover_composite_pipeline_cli",
                                    "landcover_rf_pipeline_cli", "validation_cli"])
def test_cli_help_without_gpu_dependencies(module):
    result = subprocess.run([sys.executable, "-m", "ethiopia_lcluc_tensorflow.view." + module,
                             "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_composite_requires_tiles_before_loading_gpu_stack():
    result = subprocess.run([sys.executable, "-m",
                             "ethiopia_lcluc_tensorflow.view.landcover_composite_pipeline_cli",
                             "-c", "unused.yaml", "-s", "composite"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "--tiles-filename" in result.stderr


def test_portable_examples_resolve_without_site_paths(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    for key in ("OUTPUT", "IMAGES", "PREDICTIONS", "CLOUDMASKS", "GRID"):
        monkeypatch.setenv("ETHIOPIA_" + key, str(tmp_path / key.lower()))
    cnn = load_config(root / "examples/cnn.yaml")
    composite = composite_config(root / "examples/composite.yaml")
    for config in (cnn, composite):
        resolved = OmegaConf.to_yaml(config, resolve=True)
        assert "/explore/" not in resolved
        assert "/lscratch/" not in resolved
        assert str(tmp_path) in resolved
    assert composite.post_process_combine is False
