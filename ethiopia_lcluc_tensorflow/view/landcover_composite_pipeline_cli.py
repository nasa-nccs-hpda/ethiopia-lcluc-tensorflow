import sys
import time
import logging
import argparse


# -----------------------------------------------------------------------------
# main
#
# python landcover_composite_pipeline_cli.py -c config.yaml
# -----------------------------------------------------------------------------
def main(argv=None):

    # Process command-line args.
    desc = 'Use this application to generate Amhara composites.'
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
                        nargs='+',
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
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='Override a YAML setting; may be repeated')
    args = parser.parse_args(argv)
    if 'composite' in args.pipeline_step and not args.tiles_filename:
        parser.error('--tiles-filename is required for the composite step')
    from ethiopia_lcluc_tensorflow.model.pipelines.landcover_composite_pipeline import LandCoverCompositePipeline

    # Setup timer to monitor script execution time
    timer = time.time()

    # setup pipeline object
    pipeline = LandCoverCompositePipeline(args.config_file, args.set)

    # Compositing pipeline steps
    if "build_footprints" in args.pipeline_step:
        pipeline.build_footprints()
    if "extract_metadata" in args.pipeline_step:
        pipeline.extract_metadata()
    if "composite" in args.pipeline_step:
        if args.tiles_filename is None:
            sys.exit(
                'ERROR: You need to provide --tiles-filename with ' +
                'the compositing step of this pipeline.'
            )
        pipeline.composite(args.tiles_filename)

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
