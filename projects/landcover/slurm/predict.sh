#!/bin/bash
#SBATCH -t05-00:00:00 -c10 --mem-per-cpu=10240 -G1 -J ethiopia --export=ALL --nodelist gpu015
module load singularity
#conda activate ilab
#export PYTHONPATH="/adapt/nobackup/people/jacaraba/development/tensorflow-caney"

# Run tasks sequentially without ‘&’
srun -G1 -n1 singularity exec \
    --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney",PROJ_LIB="/usr/share/proj" \
    --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects \
    /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 \
    python $NOBACKUP/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py \
    -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short.yaml \
    -s predict
