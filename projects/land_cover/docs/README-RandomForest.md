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
python rf_pipeline.py --step train \
                      --train-csv /Users/jacaraba/Desktop/development/ilab/ethiopia-lcluc/adapt-data/random_forest/rf-data.csv \
                      --bands CB B G Y R RE NIR1 NIR2 --test-size 0.20 --seed 22 --n-trees 20 --max-features log2 \
                      --output-model /Users/jacaraba/Desktop/development/ilab/ethiopia-lcluc/adapt-data/random_forest/rf-ethiopia-8band.pkl
```
