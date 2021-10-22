# Ethipia LCLUC - Land Cover

Objectives:

- LCLUC utilizing random forest algorithm.
- LCLUC utilizing CNN algorithm.

## Data Location

- Project Location: /att/gpfsfs/atrepo01/ILAB/projects/Ethiopia
- Data Location: /att/nobackup/walemu/NASA_NPP/CRPld_Map_Pred_and_Forec/EVHR/Gonji_Subset/5-toas
- Data Location Local: /Users/jacaraba/Desktop/development/ilab/ethiopia-lcluc/adapt-data

## Classes to Identify

```bash
Land use/cover codes
0 = Crops
1 = Shrub
3 = Forest
4 = Grass
5 = Bare
6 = Water
```

## Challenges

- Lack of data and labels
  - We need to wait for an office to provide the labels
  - A single set of labels for the entire set of dates
- We need a lot more data for the CNN to work
- we need validation data for the AGU presentation
- Small data, one of the images has cloud presence
- No-data values in between the image makes it a challenge to generate CNN training

## Training Data

- Non-continuous
  - Gonji_Kolela_All_2_SymDiff_SortLat_01_prj_selectS_ras: non-continuous labels, 1-7 classes of fields.
  - What year was used to create this raster?
  - Respective shape files are location in the directory

- Continuous (preferred from training)
  - Gonji_Kolela_All_2_SymDiff_SortLat_01_prj_v3r1m.tif: 1 meter resolution, classes 1-6
  - Gonji_Kolela_All_2_SymDiff_SortLat_01_prj_v3r20cm.tif: 20 cm resolution
  - Gonji_Kolela_All_2_SymDiff_SortLat_01_prj_v3.shp: shapefile
  - Year 2016, can use labels with matching data of two years of difference

  ```bash
  It’s two version, 1m and 20cm resolution raster. You can use whichever is appropriate for you.
  The land uses/cover are for the 2016 year. Generally, there is no much change from year to year for few years.
  But with longer time gap from 2016 (say 2020), croplands may encroach to shrublands, barelands, and forestlands.
  In this regard, we may encounter an issue in the image classification, since our input WorldView images are mainly in 2010.
  Not sure, if we should also consider other similar images.
  The project people exclude road and river networks from the people’s landholdings data, when they digitize the aerial photos.
  So, these networks are assigned with noData in the raster. I hope that will not create an issue in the classification.
  ```

  - Based on Woubet's assertion that data does not change much in a span of 2 years, we can extract training points from:
    - WV01_20141220_P1BS_1020010037B2E200-toa.tif: 1 meter
    - WV03_20180114_M1BS_1040010037AF5F00-toa.tif: 2 meter

Which means we can only use M1BS for training since we would need indices for P1BS images, and we do not have the appropiate bands.

## Questions

- Weekly meetings? What days are you available?
  - Friday's 13:00-14:00
- Where did you get the labels from?
  - Office from Ethiopia, land administration, to get the data through personal communication
  - Very hard to get labeled data, communication takes a very long time - 2-3 months
  - LIFT - Brithish government covers the problem, more than 9 years
  - Aerial photos, spatial resolution? cm's
- Can we get data from 2016?
  - In the works with Maggie's work.
- Other data sources?
  - Active microwave data as a compliment
- Do we want to explore P1BS imagery?
  - We do not have the matching data to create indices from it

## Action Items

- We need more labeled data


## Priorities & Action Items

### 2021-10-22

- Woubet
  - AGU Presentation
  - Where can we get images for the region?
  - some training data on the existing ones that we have
  - make the training data match multi-spectral 2m resolution
- Jordan
  - EDA Notebook with ADAPT data
  - Matching container
  - Random Forest Notebook
