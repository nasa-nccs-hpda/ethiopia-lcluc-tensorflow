import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from ethiopia_lcluc_tensorflow.model.validation import reference_codes, validate_points


def raster(tmp_path, values, name="map.tif", crs="EPSG:3857", nodata=255):
    path = tmp_path / name
    data = np.array([values], dtype="uint8")
    with rasterio.open(path, "w", driver="GTiff", height=1, width=len(values),
                       count=1, dtype="uint8", crs=crs, nodata=nodata,
                       transform=from_origin(0, 1000, 1000, 1000)) as src:
        src.write(data, 1)
    return str(path)


def reference(tmp_path, codes, xs=None):
    xs = xs if xs is not None else [500 + 1000 * i for i in range(len(codes))]
    # Store reference in a different CRS from the predictions.
    points = gpd.GeoDataFrame({"val_class": codes},
                             geometry=[Point(x, 500) for x in xs], crs="EPSG:3857")
    path = tmp_path / "reference.gpkg"
    points.to_crs("EPSG:4326").to_file(path, driver="GPKG")
    return str(path)


def test_metrics_reprojection_crop_zero_masks_and_outside(tmp_path):
    image = raster(tmp_path, [0, 1, 255, 7, 4])
    points = reference(tmp_path, ["0", "2", "0", "3", "4", "0"])
    report = validate_points(points, [image], tmp_path / "result", ignore_values=[7])
    assert report["evaluated_count"] == 3
    assert report["excluded_count"] == 3
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["balanced_accuracy"] == pytest.approx(2 / 3)
    assert report["confusion_matrix"][2][1] == 1
    saved = gpd.read_file(tmp_path / "result/points.gpkg")
    assert saved.prediction.iloc[0] == 0
    assert saved.prediction.isna().sum() == 3
    assert saved.crs.to_epsg() == 4326
    assert json.loads((tmp_path / "result/metrics.json").read_text())["point_count"] == 6


def test_overlap_requires_explicit_policy(tmp_path):
    a = raster(tmp_path, [0], "a.tif")
    b = raster(tmp_path, [1], "b.tif")
    points = reference(tmp_path, ["0"])
    with pytest.raises(ValueError, match="Overlapping"):
        validate_points(points, [b, a], tmp_path / "error")
    first = validate_points(points, [b, a], tmp_path / "first", overlap="first")
    last = validate_points(points, [a, b], tmp_path / "last", overlap="last")
    assert first["accuracy"] == 1
    assert last["accuracy"] == 0
    assert first["overlap_point_count"] == 1


def test_masked_overlap_does_not_replace_valid_crop(tmp_path):
    a = raster(tmp_path, [0], "a.tif")
    b = raster(tmp_path, [255], "b.tif")
    report = validate_points(reference(tmp_path, [0]), [a, b], tmp_path / "out")
    assert report["accuracy"] == 1
    assert report["overlap_point_count"] == 0


def test_unknown_classes_and_missing_globs_fail(tmp_path):
    points = reference(tmp_path, [0])
    image = raster(tmp_path, [7])
    with pytest.raises(ValueError, match="Unexpected raster class"):
        validate_points(points, [image], tmp_path / "out")
    with pytest.raises(ValueError, match="No prediction rasters match"):
        validate_points(points, [image, str(tmp_path / "missing*.tif")], tmp_path / "out")
    with pytest.raises(ValueError, match="Unknown reference class"):
        reference_codes(["forest"], {0: "Crop"})


def test_reference_mapping_handles_numeric_zero_and_label_aliases():
    assert reference_codes([0, "0", " 3 ", "tree/shrub", "built-up"],
                           {0: "Crop", 1: "Tree", 3: "Built"}).tolist() == [0, 0, 3, 1, 3]
    assert reference_codes(["forest"], {9: "Forest"}, {"forest": 9}).tolist() == [9]


def test_no_coverage_missing_column_and_overwrite(tmp_path):
    points = reference(tmp_path, [0])
    image = raster(tmp_path, [255])
    with pytest.raises(ValueError, match="No reference points"):
        validate_points(points, [image], tmp_path / "out")
    image = raster(tmp_path, [0])
    with pytest.raises(ValueError, match="Reference column"):
        validate_points(points, [image], tmp_path / "out", label_column="missing")
    validate_points(points, [image], tmp_path / "out")
    with pytest.raises(FileExistsError):
        validate_points(points, [image], tmp_path / "out")
    validate_points(points, [image], tmp_path / "out", overwrite=True)


def test_raster_internal_mask_and_missing_crs(tmp_path):
    points = reference(tmp_path, [0])
    image = raster(tmp_path, [0], nodata=None)
    with rasterio.open(image, "r+") as src:
        src.write_mask(np.zeros((1, 1), dtype="uint8"))
    with pytest.raises(ValueError, match="No reference points"):
        validate_points(points, [image], tmp_path / "out")
    image = raster(tmp_path, [0], name="no-crs.tif", crs=None)
    with pytest.raises(ValueError, match="CRS"):
        validate_points(points, [image], tmp_path / "out")


def test_cnn_validate_only_uses_yaml_paths_without_tensorflow(tmp_path):
    import subprocess
    import sys
    points = reference(tmp_path, [0])
    image = raster(tmp_path, [0])
    config = tmp_path / 'validate.yaml'
    config.write_text(f'validation_database: {points}\nvalidation_predictions: {image}\n'
                      f'validation_output_dir: {tmp_path / "out"}\n')
    result = subprocess.run([sys.executable, '-m',
                             'ethiopia_lcluc_tensorflow.view.landcover_cnn_pipeline_cli',
                             '-c', str(config), '-s', 'validate'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'out/metrics.json').exists()
