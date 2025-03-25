import sys
import time
import logging
import argparse
from ethiopia_lcluc_tensorflow.model.pipelines.landcover_composite_pipeline \
    import LandCoverCompositePipeline


# -----------------------------------------------------------------------------
# main
#
# python landcover_composite_pipeline_cli.py -c config.yaml
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to generate Senegal composite.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file (YAML)')
    parser.add_argument('-t',
                        '--tiles-filename',
                        dest='tiles_filename',
                        type=str,
                        required=False,
                        help='Filename with tiles to process')
    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=[
                            'build_footprints',
                            'extract_metadata',
                            'composite'],
                        choices=[
                            'build_footprints',
                            'extract_metadata',
                            'composite'])
    args = parser.parse_args()

    # Setup timer to monitor script execution time
    timer = time.time()

    # setup pipeline object
    pipeline = LandCoverCompositePipeline(args.config_file)

    # Regression CHM pipeline steps
    if "build_footprints" in args.pipeline_step:
        pipeline.build_footprints()
    if "extract_metadata" in args.pipeline_step:
        pipeline.extract_metadata()
    if "composite" in args.pipeline_step:
        pipeline.composite(args.tiles_filename)

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
