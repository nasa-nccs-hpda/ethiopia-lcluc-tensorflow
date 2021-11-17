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
cd /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/projects/land_cover/random_forest
python rf_pipeline_gpu.py --step train \
                      --train-csv /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc-data/data/random_forest/train_data.csv \
                      --train-size 0.90 --seed 22 --n-trees 40 --max-features log2 \
                      --output-model /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc-data/data/random_forest/rf-ethiopia-8band.pkl
```

## Results

0.5502083897590637 - 20, log2