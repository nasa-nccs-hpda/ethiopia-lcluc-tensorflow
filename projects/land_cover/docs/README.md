# Ethipia LCLUC - Land Cover

## Data Location

Project Location: /att/gpfsfs/atrepo01/ILAB/projects/Ethiopia
Data Location: /att/nobackup/walemu/NASA_NPP/CRPld_Map_Pred_and_Forec/EVHR/Gonji_Subset/5-toas
Data Location Local: /Users/jacaraba/Desktop/development/ilab/ethiopia-lcluc/adapt-data

## Training Data

- Non-continuous
  - Gonji_Kolela_All_2_SymDiff_SortLat_01_prj_selectS_ras: non-continuous labels, 1-7 classes of fields.
  - What year was used to create this raster?
  - Respective shape files are location in the directory

- Continuous
  - Gonji_Kolela_All_2_SymDiff_SortLat_01_prj_v3r1m.tif: 1 meter resolution, classes 1-6
  - Gonji_Kolela_All_2_SymDiff_SortLat_01_prj_v3r20cm.tif: 20 cm resolution
  - Gonji_Kolela_All_2_SymDiff_SortLat_01_prj_v3.shp: shapefile
  - Year 2016
  - Comment: It’s two version, 1m and 20cm resolution raster. You can use whichever is appropriate for you.
    The land uses/cover are for the 2016 year. Generally, there is no much change from year to year for few years.
    But with longer time gap from 2016 (say 2020), croplands may encroach to shrublands, barelands, and forestlands.
    In this regard, we may encounter an issue in the image classification, since our input WorldView images are mainly in 2010.
    Not sure, if we should also consider other similar images.

    The project people exclude road and river networks from the people’s landholdings data, when they digitize the aerial photos.
    So, these networks are assigned with noData in the raster. I hope that will not create an issue in the classification.

## Matching Data

Based on Woubet's assertion that data does not change much in a span of 4 years, we can extract training points from:

- WV01_20141220_P1BS_1020010037B2E200-toa.tif: 1 meter
- WV03_20180114_M1BS_1040010037AF5F00-toa.tif: 2 meter

Which means we can only used M1BS for training since we would need indices for P1BS images. 

- Could we get more data?
- Do we want to explore P1BS imagery?

## Questions

- Weekly meetings? What days are you available?
- Can we get data from 2016?
- Are you sure that is a good comparison with other dates lines 2014 or 2018
- Where did you get the training from?

## Problems

- Lack of data and labels
- WV03_20180114_M1BS_1040010037AF5F00-toa.tif is just a small set of the data
- we need a lot more data for the CNN to work
- we need validation data for the AGU presentation
- Since we do not have matching labels for each image, WV02_20100215_M1BS_10300100043A6100-toa-clipped cannot be used for training
or validation because of the presence of the cloud

## Random Forest Run

### Preprocess

LOCAL

```bash
python rf_pipeline.py --step preprocess \
                      --data-csv config/test-input.csv \
                      --train-csv /Users/jacaraba/Desktop/development/ilab/ethiopia-lcluc/adapt-data/random_forest/rf-data.csv \
                      --bands CB B G Y R RE NIR1 NIR2 --n-classes 6
```

### Train

```bash
python rf_pipeline.py --step train \
                      --train-csv /Users/jacaraba/Desktop/development/ilab/ethiopia-lcluc/adapt-data/random_forest/rf-data.csv \
                      --bands CB B G Y R RE NIR1 NIR2 --test-size 0.20 --seed 22 --n-trees 20 --max-features log2 \
                      --output-model /Users/jacaraba/Desktop/development/ilab/ethiopia-lcluc/adapt-data/random_forest/rf-ethiopia-8band.pkl
```