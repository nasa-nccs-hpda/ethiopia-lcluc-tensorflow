"""Point-based validation of categorical rasters without TensorFlow or a GPU.

Overlapping predictions require an explicit selection policy. NoData and
out-of-bounds points are excluded from metrics but retained in the output.
"""
import json
from glob import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASS_NAMES = {0: "Crop", 1: "Tree / Shrub", 2: "Grass", 3: "Built", 4: "Water"}
LABEL_CODES = {
    "crop": 0, "shrub": 1, "tree/shrub": 1, "tree / shrub": 1,
    "grass": 2, "builtup": 3, "built-up": 3, "built": 3,
    "buitlup": 3, "water": 4,
}


def reference_codes(values, class_names, label_map=None):
    mapping = LABEL_CODES if label_map is None else label_map
    mapping = {str(k).strip().lower(): int(v) for k, v in mapping.items()}
    codes = []
    for value in values:
        text = str(value).strip().lower()
        try:
            number = float(text)
            code = int(number) if np.isfinite(number) and number.is_integer() else None
        except ValueError:
            code = mapping.get(text)
        if code not in class_names:
            raise ValueError(f"Unknown reference class {value!r}; provide a label map/class names.")
        codes.append(code)
    return np.array(codes, dtype=int)


def validate_points(reference, predictions, output_dir, label_column="val_class",
                    label_map=None, class_names=None, ignore_values=(),
                    overlap="error", band=1, overwrite=False):
    """Sample rasters at point geometries and write points.gpkg and metrics.json.

    Raster paths/globs are sorted and deduplicated. `first`/`last` overlap
    policies refer to that order, not acquisition time or quality.
    """
    class_names = CLASS_NAMES if class_names is None else {int(k): v for k, v in class_names.items()}
    if not class_names:
        raise ValueError("class_names must not be empty.")
    if overlap not in {"error", "first", "last"}:
        raise ValueError("overlap must be error, first, or last.")
    if band < 1:
        raise ValueError("Raster band numbers start at 1.")
    patterns = [predictions] if isinstance(predictions, str) else predictions
    paths = set()
    for pattern in patterns:
        matches = glob(str(Path(pattern).expanduser()), recursive=True)
        if not matches:
            raise ValueError(f"No prediction rasters match: {pattern}")
        paths.update(str(Path(p).resolve()) for p in matches)
    paths = sorted(paths)
    if not paths:
        raise ValueError("At least one prediction raster is required.")
    output = Path(output_dir).expanduser().resolve()
    targets = [output / "points.gpkg", output / "metrics.json"]
    if any(p.exists() for p in targets) and not overwrite:
        raise FileExistsError(f"Validation outputs already exist in {output}; use --overwrite.")
    if str(targets[0]) == str(Path(reference).resolve()) or any(str(p) in paths for p in targets):
        raise ValueError("Output must not replace an input dataset.")

    points = gpd.read_file(reference).reset_index(drop=True)
    if points.empty or points.crs is None:
        raise ValueError("Reference points must be nonempty and have a CRS.")
    if (points.geometry.isna().any() or points.geometry.is_empty.any()
            or not points.geom_type.eq("Point").all()):
        raise ValueError("Reference geometries must all be nonempty Points.")
    if label_column not in points.columns:
        raise ValueError(f"Reference column {label_column!r} not found: {list(points.columns)}")
    truth = reference_codes(points[label_column], class_names, label_map)
    predicted = np.full(len(points), np.nan)
    filenames = np.full(len(points), "", dtype=object)
    coverage = np.zeros(len(points), dtype=int)
    raster_metadata = []

    for filename in paths:
        with rasterio.open(filename) as src:
            if src.crs is None or band > src.count:
                raise ValueError(f"Raster must have a CRS and band {band}: {filename}")
            projected = points.to_crs(src.crs)
            coords = list(zip(projected.geometry.x, projected.geometry.y))
            if not np.isfinite(coords).all():
                raise ValueError(f"Point reprojection produced nonfinite coordinates: {filename}")
            rows, cols = rasterio.transform.rowcol(src.transform, *zip(*coords))
            inside = (np.array(rows) >= 0) & (np.array(rows) < src.height)
            inside &= (np.array(cols) >= 0) & (np.array(cols) < src.width)
            for index, sample in zip(np.flatnonzero(inside), src.sample(
                    [coords[i] for i in np.flatnonzero(inside)], indexes=band, masked=True)):
                if np.ma.getmaskarray(sample)[0]:
                    continue
                value = float(sample[0])
                if not np.isfinite(value) or value in ignore_values:
                    continue
                if not value.is_integer() or int(value) not in class_names:
                    raise ValueError(f"Unexpected raster class {value} in {filename}; "
                                     "set class names or --ignore-values explicitly.")
                coverage[index] += 1
                if coverage[index] > 1:
                    if overlap == "error":
                        raise ValueError(f"Overlapping predictions at reference row {index}; "
                                         "select --overlap first or last explicitly.")
                    if overlap == "first":
                        continue
                predicted[index] = int(value)
                filenames[index] = filename
            raster_metadata.append({
                "path": filename, "crs": str(src.crs), "nodata": str(src.nodata),
                "band": band, "shape": list(src.shape), "transform": list(src.transform),
                "size_bytes": Path(filename).stat().st_size,
            })

    valid = np.isfinite(predicted)
    if not valid.any():
        raise ValueError("No reference points have valid predictions; check CRS, coverage, and masks.")
    labels = sorted(class_names)
    y_true, y_pred = truth[valid], predicted[valid].astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = matrix.sum(axis=1)
    recalls = np.divide(matrix.diagonal(), support, out=np.zeros(len(labels)), where=support > 0)
    report = {
        "reference": str(Path(reference).resolve()), "label_column": label_column,
        "class_names": class_names, "label_map": LABEL_CODES if label_map is None else label_map,
        "ignore_values": list(ignore_values), "overlap_policy": overlap,
        "point_count": len(points), "evaluated_count": int(valid.sum()),
        "excluded_count": int((~valid).sum()), "overlap_point_count": int((coverage > 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(recalls[support > 0].mean()),
        "confusion_matrix_labels": labels, "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": "rows=reference, columns=prediction",
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=[class_names[k] for k in labels],
            output_dict=True, zero_division=0),
        "rasters": raster_metadata,
    }
    points["reference_class"] = truth
    points["prediction"] = predicted
    points["prediction_file"] = filenames
    points["valid_predictions"] = coverage
    output.mkdir(parents=True, exist_ok=True)
    # All data checks finish before replacing an existing output.
    if targets[0].exists():
        targets[0].unlink()
    points.to_file(targets[0], layer="validation", driver="GPKG")
    targets[1].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
