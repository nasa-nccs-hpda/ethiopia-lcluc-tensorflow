# Dataset Creation

## Training Labels

The training classes are as follow:

```bash
0=crop
1=shrub
2=forest
3=bare
4=settlement
5=grass
```

North:

```
- 1 crop
- 2 forest
- 3 grass
- 4 settlement
- 5 shrub
```


The training images are:

- Gonji_north_training_data_Prj_noHollow_naMerge_5class_codedFinal.tif: has the precense of 5 classes
- Gonji_Kolela_All_2_SymDiff_SortLat_01_05_prjc_noNA_6class_codedFinal.tif: has the precense of 6 classes

Move the data to /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/

We are aiming to segment 6 classes. Grass is not included in the 5 class image.

## Training Data

The training data used must be between from 2014, 2015, and 2016; since the training data was created on
2015 or 2016. We could use 2017 for validation and so forth. The original data is located under
/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS.

The selected data includes:

/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20141024_M1BS_1040010003457200-toa.tif - not in the training
/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV02_20150101_M1BS_103001003CD72200-toa.tif - not in the training
/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20150209_M1BS_104001000717B500-toa.tif - not in the training
/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20160923_M1BS_1040010021454200-toa.tif - not in the training
/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV02_20180216_M1BS_1030010079D74900-toa.tif - not in the training
/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20180221_M1BS_10400100366E1C00-toa.tif - small piece but cloudy, bad

Used in training:

/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20141024_M1BS_1040010003ACA100-toa.tif - small piece
/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20141118_M1BS_1040010004D47B00-toa.tif - one piece of class 5 image
/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20141207_M1BS_10400100053C0600-toa.tif - one piece of class 5 image
/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV02_20180221_M1BS_1030010078802100-toa.tif - small piece of training, cloudy
/att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20180114_M1BS_1040010037AF5F00-toa.tif - small piece

## Clip each one of these images to match AOI

### /att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20141024_M1BS_1040010003ACA100-toa.tif

```bash
rio clip /att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20141024_M1BS_1040010003ACA100-toa.tif /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20141024_M1BS_1040010003ACA100-toa_Gonji_5class.tif --like /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/Gonji_north_training_data_Prj_noHollow_naMerge_5class_codedFinal.tif
```

Then for additional match:

```bash
rio clip /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/Gonji_north_training_data_Prj_noHollow_naMerge_5class_codedFinal.tif /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/WV03_20141024_M1BS_1040010003ACA100-toa_Gonji_5class_label.tif --like /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20141024_M1BS_1040010003ACA100-toa_Gonji_5class.tif
```

### /att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20141118_M1BS_1040010004D47B00-toa.tif

```bash
rio clip /att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20141118_M1BS_1040010004D47B00-toa.tif /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20141118_M1BS_1040010004D47B00-toa_Gonji_5class.tif --like /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/Gonji_north_training_data_Prj_noHollow_naMerge_5class_codedFinal.tif
```

Then for additional match:

```bash
rio clip /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/Gonji_north_training_data_Prj_noHollow_naMerge_5class_codedFinal.tif /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/WV03_20141118_M1BS_1040010004D47B00-toa_Gonji_5class_label.tif --like /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20141118_M1BS_1040010004D47B00-toa_Gonji_5class.tif
```


### /att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20141207_M1BS_10400100053C0600-toa.tif

```bash
rio clip /att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20141207_M1BS_10400100053C0600-toa.tif /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20141207_M1BS_10400100053C0600-toa_Gonji_5class.tif --like /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/Gonji_north_training_data_Prj_noHollow_naMerge_5class_codedFinal.tif
```

Then for additional match:

```bash
rio clip /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/Gonji_north_training_data_Prj_noHollow_naMerge_5class_codedFinal.tif /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/WV03_20141207_M1BS_10400100053C0600-toa_Gonji_5class_label.tif --like /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20141207_M1BS_10400100053C0600-toa_Gonji_5class.tif
```

### /att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20180114_M1BS_1040010037AF5F00-toa.tif

```bash
rio clip /att/nobackup/mwooten3/Ethiopia_Woubet/VHR/M1BS/WV03_20180114_M1BS_1040010037AF5F00-toa.tif /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20180114_M1BS_1040010037AF5F00-toa_Gonji_6class.tif --like /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/Gonji_Kolela_All_2_SymDiff_SortLat_01_05_prjc_noNA_6class_codedFinal.tif
```

Then for additional match:

```bash
rio clip /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/Gonji_Kolela_All_2_SymDiff_SortLat_01_05_prjc_noNA_6class_codedFinal.tif /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/WV03_20180114_M1BS_1040010037AF5F00-toa_Gonji_6class_label.tif --like /att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20180114_M1BS_1040010037AF5F00-toa_Gonji_6class.tif
```

## Final Training Dataset

/att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20141024_M1BS_1040010003ACA100-toa_Gonji_5class.tif
/att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/WV03_20141024_M1BS_1040010003ACA100-toa_Gonji_5class_label.tif
[0,1,3,4]

/att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20141118_M1BS_1040010004D47B00-toa_Gonji_5class.tif
/att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/WV03_20141118_M1BS_1040010004D47B00-toa_Gonji_5class_label.tif
[0,1,2,3,4]

/att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20141207_M1BS_10400100053C0600-toa_Gonji_5class.tif
/att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/WV03_20141207_M1BS_10400100053C0600-toa_Gonji_5class_label.tif
[0,1,2,3,4]

/att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/images/WV03_20180114_M1BS_1040010037AF5F00-toa_Gonji_6class.tif
/att/pubrepo/ILAB/projects/Ethiopia/ethiopia-lcluc/data/labels/WV03_20180114_M1BS_1040010037AF5F00-toa_Gonji_6class_label.tif
[0,1,2,3,5]


## Dataset Details

dataset_metadata = {
    'WV03_20141024_M1BS_1040010003ACA100-toa_Gonji_5class.tif': {
        '0': 15408517,
        '1': 4131660,
        '2': 0,
        '3': 830,
        '4': 481725,
        '5': 0,
    },
    'WV03_20141118_M1BS_1040010004D47B00-toa_Gonji_5class.tif': {
        '0': 47905479,
        '1': 14533773,
        '2': 147723,
        '3': 206861,
        '4': 1715572,
        '5': 0,
    },
    'WV03_20141207_M1BS_10400100053C0600-toa_Gonji_5class.tif': {
        '0': 46115343,
        '1': 14364116,
        '2': 147093,
        '3': 206861,
        '4': 1612382,
        '5': 0,
    },
    'WV03_20180114_M1BS_1040010037AF5F00-toa_Gonji_6class.tif': {
        '0': 5756895,
        '1': 249971,
        '2': 221769,
        '3': 137988,
        '4': 0,
        '5': 570779,
    },
}