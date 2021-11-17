# Random Forest Procedures

- Scripts location:
- Notebooks:

## Preprocess

LOCAL

```bash
python rf_pipeline.py --step preprocess \
                      --data-csv config/test-input.csv \
                      --train-csv /Users/jacaraba/Desktop/development/ilab/ethiopia-lcluc/adapt-data/random_forest/rf-data.csv \
                      --bands CB B G Y R RE NIR1 NIR2 --n-classes 6
```

## Train

```bash
cd /adapt/nobackup/projects/ilab/projects/Ethiopia/LCLUC_Ethiopia/ethiopia-lcluc/projects/land_cover/random_forest
python rf_pipeline_gpu.py --step train \
                      --train-csv /adapt/nobackup/projects/ilab/projects/Ethiopia/LCLUC_Ethiopia/random_forest/train_data_ethiopia_v2.csv \
                      --train-size 0.80 --seed 22 --n-trees 200 --max-features log2 \
                      --output-model /adapt/nobackup/projects/ilab/projects/Ethiopia/LCLUC_Ethiopia/random_forest/rf-ethiopia-8band.pkl
```

## Results

0.5502083897590637 - 20, log2



python cnn_pipeline.py -c ../config/config.yaml -d ../config/ethiopia-land-cover-dataset-rf-2021-10-22.csv -s preprocess


python rf_pipeline_gpu.py --step predict --gpu \
    --bands CB B G Y R RE NIR1 NIR2 --window-size 8192 \
    --output-model /adapt/nobackup/projects/ilab/projects/Ethiopia/LCLUC_Ethiopia/random_forest/rf-ethiopia-8band.pkl \
    --output-dir /adapt/nobackup/projects/ilab/projects/Ethiopia/LCLUC_Ethiopia/random_forest/predictions \
    --rasters '/adapt/nobackup/projects/ilab/projects/Ethiopia/LCLUC_Ethiopia/data/images/*.tif'