"""Validate categorical rasters against reference points."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Point GeoPackage or shapefile")
    parser.add_argument("--predictions", nargs="+", required=True, help="Raster paths or quoted globs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="val_class")
    parser.add_argument("--label-map", help="JSON object mapping reference labels to integer codes")
    parser.add_argument("--class-names", help="JSON object mapping integer codes to names")
    parser.add_argument("--ignore-values", nargs="*", type=int, default=[])
    parser.add_argument("--overlap", choices=["error", "first", "last"], default="error")
    parser.add_argument("--band", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    from ethiopia_lcluc_tensorflow.model.validation import validate_points
    try:
        report = validate_points(
            args.reference, args.predictions, args.output_dir,
            label_column=args.label_column,
            label_map=json.loads(Path(args.label_map).read_text()) if args.label_map else None,
            class_names=json.loads(Path(args.class_names).read_text()) if args.class_names else None,
            ignore_values=args.ignore_values, overlap=args.overlap,
            band=args.band, overwrite=args.overwrite)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Evaluated {report['evaluated_count']}/{report['point_count']} points; "
          f"accuracy={report['accuracy']:.4f}, balanced_accuracy={report['balanced_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
