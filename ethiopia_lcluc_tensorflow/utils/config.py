"""Load YAML with environment interpolation and explicit CLI overrides."""
from omegaconf import OmegaConf


def load_config(filename, overrides=()):
    config = OmegaConf.load(filename)
    if not OmegaConf.is_dict(config):
        raise ValueError("Configuration must be a YAML mapping.")
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected KEY=VALUE override, got {override!r}")
    return OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))


def composite_config(filename, overrides=()):
    config = load_config(filename, overrides)
    if not config.get("grid_filename") and config.get("grid_path"):
        config.grid_filename = config.grid_path
    defaults = {
        "filter_months": False, "filter_name": "all-months", "remove_months": [],
        "soil_moisture_qa": False, "fill_value": 255, "burn_area_value": 15,
        "calculate_mode_composite": True, "calculate_nobservations": True,
        "calculate_binary_stats": False, "calculate_confidence": False,
        "confidence_metrics": [], "overwrite_tifs": False,
        "overwrite_zarrs": False, "post_process_combine": False,
    }
    config = OmegaConf.merge(defaults, config)
    for field in ("output_dir", "grid_filename", "start_year", "end_year"):
        if config.get(field) is None:
            raise ValueError(f"Missing composite configuration field: {field}")
    if int(config.start_year) > int(config.end_year):
        raise ValueError("start_year must be less than or equal to end_year (inclusive).")
    if config.post_process_combine:
        raise ValueError("post_process_combine is not implemented; set it to false.")
    return config


from contextlib import contextmanager
import tempfile
from pathlib import Path


@contextmanager
def resolved_config_file(config):
    """Materialize overrides/environment variables for upstream file-based APIs.

    Relative paths remain relative to the process working directory.
    """
    with tempfile.TemporaryDirectory(prefix="ethiopia-config-") as directory:
        filename = Path(directory) / "config.yaml"
        OmegaConf.save(config, filename, resolve=True)
        yield str(filename)
