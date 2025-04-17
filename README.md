# Ethiopia LCLUC

Ethiopia LCLUC using WorldView imagery

[![DOI](https://zenodo.org/badge/527702332.svg)](https://zenodo.org/badge/latestdoi/527702332)


## Objectives

- LCLUC utilizing random forest algorithm
- LCLUC utilizing XGBoost algorithm
- LCLUC utilizing CNN algorithm
- LCLUC utilizing CNN ensemble algorithm

## Data Catalog

```bash
- Project Location: /explore/nobackup/projects/ilab/projects/Ethiopia/LCLUC_Ethiopia
- Full Domain Data Location: /adapt/nobackup/people/mwooten3/Ethiopia_Woubet/VHR
- Gonji Subset Data Location: /adapt/nobackup/people/walemu/NASA_NPP/CRPld_Map_Pred_and_Forec/EVHR/Gonji_Subset/5-toas
```

## Structure of this Repository

This repository takes care of preprocessing, training, inference, and compositing
of WorldView imagery for Ethiopia. The different steps are guided by pipelines. There
are two main pipelines available in this repository:

- Land Cover: generates GeoTIFF predictions of land cover outputs
- Compositing: takes the outputs from the Land Cover pipeline and generates multi-year composites

## Explore/ADAPT Basic Information

1. SSH to ADAPT Login

```bash
ssh adaptlogin.nccs.nasa.gov
```

2. SSH to GPU Login

```bash
ssh gpulogin1
```

3. Clone above-shrubs repository

Clone the github:

```bash
git clone https://github.com/nasa-nccs-hpda/ethiopia-lcluc-tensorflow.git
```

4. Accessing the container

To download a clean version of the container, run the following command:

```bash
singularity build --sandbox /lscratch/$USER/container/ethiopia-lcluc-tensorflow docker://nasanccs/ethiopia-lcluc-tensorflow:latest
```

An already downloaded version of the container is location in the Explore HPC cluster under:

```bash
/explore/nobackup/projects/ilab/containers/ethiopia-lcluc-tensorflow.2025.04
```

## Workflow Documentation

### Land Cover Outputs Generation

TBD

### Cloud Masking Outputs Generation

NOTE: these instructions need to be updated with the new vhr-cloudmask software
developed by the team. Overall example to run cloud masking:

```bash
for i in {0..64}; do sbatch --mem-per-cpu=10240 -G1 -c10 -q ilab -t05-00:00:00 -J clouds --wrap="singularity exec --env PYTHONPATH=\"/explore/nobackup/people/jacaraba/development/vhr-cloudmask:/explore/nobackup/people/jacaraba/development/tensorflow-caney\" --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif python /explore/nobackup/people/jacaraba/development/vhr-cloudmask/vhr_cloudmask/view/cloudmask_cnn_pipeline_cli.py -o '/explore/nobackup/projects/3sl/products/cloudmask/v2' -r '/explore/nobackup/projects/hls/EVHR/Amhara-MS/*-toa.tif' -s predict"; done
```

Additional example on how to run cloud masking:

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-cloudmask:/explore/nobackup/people/jacaraba/development/tensorflow-caney" --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif python /explore/nobackup/people/jacaraba/development/vhr-cloudmask/vhr_cloudmask/view/cloudmask_cnn_pipeline_cli.py -o '/explore/nobackup/projects/3sl/products/cloudmask/v2' -r '/explore/nobackup/projects/hls/EVHR/Amhara-MS/*-toa.tif' -s predict
```

### Compositing

After we generate predictions for the entire study area, we need to proceed to create composites. Below you will find the documentation to perform the compositing steps. 
This pipeline has 3 main steps:

1. Build footprints
2. Extract metadata
3. Build composite

Below you will find examples on how to run each one of these.

#### Build footprints

To generate the initial footprints that match the tiles with the grid information (including an example output):

```bash
[jacaraba@gpu011 ~]$ singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-composite:/explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects,/panfs/ccds02/nobackup/projects /lscratch/jacaraba/container/tensorflow-caney python /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_composite_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/composite/configs/dev/composite_ethiopia_epoch1.yaml -s build_footprints
WARNING: underlay of /etc/localtime required more than 50 (116) bind mounts
WARNING: underlay of /usr/bin/nvidia-smi required more than 50 (666) bind mounts
15:4: not a valid test operator:
15:4: not a valid test operator: 12.6
21:4: not a valid test operator: (
21:4: not a valid test operator: 570.124.06
2025-03-25 14:12:22; INFO; Output logs sent to: /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/2009.2015/2009.2015.log
2025-03-25 14:12:22; INFO; Created output dir: /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/2009.2015
2025-03-25 14:12:22; INFO; Saved config file to: /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/2009.2015
2025-03-25 14:12:22; INFO; Output logs sent to: /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/2009.2015/build_footprints.log
2025-03-25 14:12:22; INFO; Building footprints
2025-03-25 14:12:22; INFO; Found 2144 tifs to process.
2025-03-25 14:12:43; INFO; Created 2,144 records
2025-03-25 14:12:43; INFO; Saved /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/Amhara_M1BS_griddedToa.gpkg.
2025-03-25 14:12:43; INFO; Adding base fields to /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/Amhara_M1BS_griddedToa.gpkg
2025-03-25 14:12:43; INFO; Adding xml_path
2025-03-25 14:12:43; INFO; Adding strip_id
2025-03-25 14:12:43; INFO; Adding sensor
2025-03-25 14:12:43; INFO; Adding spec_type
2025-03-25 14:12:43; INFO; Adding catalog_id
2025-03-25 14:12:43; INFO; Adding date
2025-03-25 14:12:43; INFO; Adding year
2025-03-25 14:12:43; INFO; Adding month
2025-03-25 14:12:43; INFO; Adding day
2025-03-25 14:12:43; INFO; Adding acq_time
2025-03-25 14:13:03; INFO; Created 2,144 records
2025-03-25 14:13:04; INFO; Adding acquisition geom from data to /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/Amhara_M1BS_griddedToa.gpkg
2025-03-25 14:14:45; INFO; Created 2,144 records
2025-03-25 14:14:45; INFO; Updated /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/Amhara_M1BS_griddedToa.gpkg.
2025-03-25 14:14:45; INFO; Adding region
2025-03-25 14:14:46; INFO; Adding grid metadata
2025-03-25 14:14:47; INFO; Created 39,367 records
2025-03-25 14:14:47; INFO; Updated /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/Amhara_M1BS_griddedToa.gpkg.
2025-03-25 14:14:47; INFO; Took 2.42 min.
```

#### Extract metadata

To generate the shapefile with metadata for each strip (including an example output):

```bash
[jacaraba@gpu011 ~]$ singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-composite:/explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects,/panfs/ccds02/nobackup/projects /lscratch/jacaraba/container/tensorflow-caney python /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_composite_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/composite/configs/dev/composite_ethiopia_epoch1.yaml -s extract_metadata
WARNING: underlay of /etc/localtime required more than 50 (116) bind mounts
WARNING: underlay of /usr/bin/nvidia-smi required more than 50 (666) bind mounts
15:4: not a valid test operator:
15:4: not a valid test operator: 12.6
21:4: not a valid test operator: (
21:4: not a valid test operator: 570.124.06
2025-03-25 14:16:02; INFO; Output logs sent to: /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/2009.2015/2009.2015.log
2025-03-25 14:16:02; INFO; Created output dir: /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/2009.2015
2025-03-25 14:16:02; INFO; Saved config file to: /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/2009.2015
2025-03-25 14:16:02; INFO; Output logs sent to: /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/2009.2015/extract_metadata.log
2025-03-25 14:16:03; INFO; Reading footprint file /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/Amhara_M1BS_griddedToa.gpkg
2025-03-25 14:16:03; INFO; Processing the following metrics: all_touched
2025-03-25 14:16:04; INFO; Created 39,367 records
2025-03-25 14:16:05; INFO; Saved /explore/nobackup/projects/ilab/scratch/jacaraba/ethiopia/cnn_landcover_composite/Amhara-1.0/Amhara_M1BS_griddedToa_metadata.gpkg
2025-03-25 14:16:05; INFO; Took 0.04 min.
```

#### Compositing

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/vhr-composite:/explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects,/panfs/ccds02/nobackup/projects /lscratch/jacaraba/container/tensorflow-caney python /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_composite_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/composite/configs/dev/composite_ethiopia_epoch1.yaml -t /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/composite/configs/dev/test_1_tiles.txt -s composite
```

To run multiple tiles:

```bash
for i in {0..78}; do sbatch --mem-per-cpu=10240 -G1 -c10 -q ilab -t05-00:00:00 -J clouds --wrap="singularity exec --env PYTHONPATH=\"/explore/nobackup/people/jacaraba/development/vhr-composite-jordan-edits:/explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow\" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects,/panfs/ccds02/nobackup/projects /explore/nobackup/projects/ilab/containers/tensorflow-caney-2023.05 python /explore/nobackup/people/jacaraba/development/vhr-composite-jordan-edits/examples/ethiopia/landcover_composite_pipeline.py -c /explore/nobackup/people/jacaraba/development/vhr-composite-jordan-edits/examples/ethiopia/composite_ethiopia_epoch1.yaml -t /explore/nobackup/people/jacaraba/development/vhr-composite-jordan-edits/examples/ethiopia/output_tiles_${i}.txt -s composite"; done
```

## Legacy Documentation to Fix

### Debugging

```bash
/explore/nobackup/people/walemu/NASA_NPP/CRPld_Map_Pred_and_Forec/Training_data/training_data_with_without_ParcelsShape/WV02_20150101_M1BS_103001003CD72200-toa_BD.tif,/explore/nobackup/people/walemu/NASA_NPP/CRPld_Map_Pred_and_Forec/Training_data/training_data_with_without_ParcelsShape/Bahir_Dar_merged_6class_WV02_20150101_M1BS_103001003CD72200-toa.tif,3000


singularity exec --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney",PROJ_LIB="/usr/share/proj" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python $NOBACKUP/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/local_standardization_256_crop_4band_short_tversky.yaml -d /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/landcover_ethiopia.csv -s preprocess train predict


singularity exec --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney",PROJ_LIB="/usr/share/proj" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python $NOBACKUP/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v2.yaml -d /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/landcover_ethiopia.csv -s preprocess train predict


singularity exec --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney",PROJ_LIB="/usr/share/proj" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python $NOBACKUP/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/local_standardization_256_tree_4band_short_tversky.yaml -d /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/landcover_ethiopia.csv -s preprocess train predict


singularity exec --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney",PROJ_LIB="/usr/share/proj" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python $NOBACKUP/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/land_cover_otcb_cas-wcas_global-std_50TS_4band-tversky.yaml -d /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/land_cover_512_otcb_50TS_cas-wcas.csv -s preprocess train predict
```

## Validation


```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney",PROJ_LIB="/usr/share/proj" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python $NOBACKUP/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short.yaml -d /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/landcover_ethiopia.csv -s validate

rio clip --like /explore/nobackup/people/walemu/NASA_NPP/CRPld_Map_Pred_and_Forec/Training_data/training_data_with_without_ParcelsShape/Gondar_Zunia_merged3_WV02_20130318_M1BS_10300100215E6300_toa.tif /explore/nobackup/projects/hls/EVHR/Amhara-MS/WV02_20130318_M1BS_10300100215E6300-toa.tif WV02_20130318_M1BS_10300100215E6300_toa_ext_fixed.tif

singularity exec --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney:$NOBACKUP/development/senegal-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v7.yaml -d /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/landcover_ethiopia_v7.csv -s preprocess

singularity exec --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney:$NOBACKUP/development/senegal-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v4_all.yaml -d /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/landcover_ethiopia_v4_all.csv -s preprocess
```


```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney:$NOBACKUP/development/senegal-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v8.yaml -d /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/landcover_ethiopia_v8.csv -s preprocess train predict
```

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney:$NOBACKUP/development/senegal-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python /explore/nobackup/people/$USER/development/senegal-lcluc-tensorflow/senegal_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v12.yaml -d /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/landcover_ethiopia_v8.csv -s predict
```


```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/ethiopia-lcluc-tensorflow:$NOBACKUP/development/tensorflow-caney:$NOBACKUP/development/senegal-lcluc-tensorflow" --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 python /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/2023-06-27/global_standardization_256_crop_4band_short-v12.yaml -vd /explore/nobackup/people/walemu/NASA_NPP/CRPld_Map_Pred_and_Forec/Training_data/Validation_data/GT_points_AGU2023_12_2010.shp -s validate
```

## Tutorial

```bash
singularity exec --env PYTHONPATH="/explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow:/explore/nobackup/people/jacaraba/development/tensorflow-caney:/explore/nobackup/people/jacaraba/development/senegal-lcluc-tensorflow" --nv -B /explore/nobackup/people/jacaraba,$NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif python /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/ethiopia_lcluc_tensorflow/view/landcover_cnn_pipeline_cli.py -c /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/tutorial/ethiopia_normalized_256_builtup_4band.yaml -d /explore/nobackup/people/jacaraba/development/ethiopia-lcluc-tensorflow/projects/landcover/config/experiments/tutorial/ethiopia_normalized_256_builtup_4band.csv -s preprocess
```
