#!/bin/bash
#SBATCH -t05-00:00:00 -c20 --mem-per-cpu=20G -G1 -J ethiopia --export=ALL -q ilab
module load singularity

# Run tasks sequentially without ‘&’
# TODO: need to doublecheck that this works fine
# srun -G1 -n1 singularity exec \
#    --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney",PROJ_LIB="/usr/share/proj" \
#    --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects \
#    /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 \
#    python $NOBACKUP/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py \
#    -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v2.yaml \
#    -s predict

# -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v2.yaml \
# -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short.yaml \
# -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v2.yaml

srun -G1 -n1 singularity exec \
    --env PYTHONPATH="/explore/nobackup/people/$USER/development/ethiopia-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney" \
    --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,/explore/nobackup/projects/hls,$NOBACKUP,/lscratch,/explore/nobackup/people \
    /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 \
    python /explore/nobackup/people/$USER/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py \
    -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v12.yaml \
    -s predict

# this worked out fine
#-c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v4_all.yaml \
# singularity exec --env PYTHONPATH="/explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow:/explore/nobackup/people/$USER/development/tensorflow-caney" --nv -B /explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,/explore/nobackup/projects/hls,$NOBACKUP,/lscratch,/explore/nobackup/people /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v2.yaml -s predict