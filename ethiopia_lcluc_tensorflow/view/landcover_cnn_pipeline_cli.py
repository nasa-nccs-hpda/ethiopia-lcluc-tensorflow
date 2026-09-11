"""CNN preprocessing, training, prediction, and reference-point validation."""
import argparse
import logging
import time

from ethiopia_lcluc_tensorflow.utils.config import load_config, resolved_config_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config-file", required=True)
    parser.add_argument("-d", "--data-csv")
    parser.add_argument("-vd", "--validation-database")
    parser.add_argument("--validation-predictions", nargs="+")
    parser.add_argument("--validation-output-dir")
    parser.add_argument("--label-column", default="val_class")
    parser.add_argument("--ignore-values", nargs="*", type=int, default=[])
    parser.add_argument("--overlap", choices=["error", "first", "last"], default="error")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a YAML setting; may be repeated")
    parser.add_argument("-s", "--step", nargs="+", required=True,
                        choices=["preprocess", "train", "predict", "validate"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    timer = time.time()
    config = load_config(args.config_file, args.set)
    if "preprocess" in args.step and not args.data_csv:
        parser.error("--data-csv is required for preprocessing")
    if "validate" in args.step:
        reference = args.validation_database or config.get("validation_database")
        predictions = args.validation_predictions or config.get("validation_predictions")
        output = args.validation_output_dir or config.get("validation_output_dir")
        if not reference or not predictions or not output:
            parser.error("validation requires a database, prediction paths/globs, and output directory")
    if any(step in args.step for step in ("preprocess", "train", "predict")):
        # Validation and --help do not require the TensorFlow/GPU stack.
        from ethiopia_lcluc_tensorflow.model.pipelines.landcover_pipeline import LandCoverPipeline
        # Evaluation settings are handled here, not by the upstream CNN schema.
        cnn_config = config.copy()
        for key in ("validation_predictions", "validation_output_dir"):
            cnn_config.pop(key, None)
        with resolved_config_file(cnn_config) as filename:
            pipeline = LandCoverPipeline(filename, args.data_csv)
            if "preprocess" in args.step:
                pipeline.preprocess(enable_multiprocessing=True)
            if "train" in args.step:
                pipeline.train()
            if "predict" in args.step:
                pipeline.predict()
    if "validate" in args.step:
        from ethiopia_lcluc_tensorflow.model.validation import validate_points
        report = validate_points(reference, predictions, output,
                                 label_column=args.label_column, ignore_values=args.ignore_values,
                                 overlap=args.overlap, overwrite=args.overwrite)
        logging.info("Validation accuracy: %.4f", report["accuracy"])
    logging.info("Took %.2f min.", (time.time() - timer) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
