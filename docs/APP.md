# Amhara Land Cover Explorer

**[Open the public application](https://gsfc-dsg.projects.earthengine.app/view/ethiopia-lcluc)**

Explore the products directly in your browser. The steps below describe running or updating the source code.

### Run and update the app

1. Open the [Earth Engine Code Editor](https://code.earthengine.google.com/) with an account and project that can read the assets below.
2. Copy [app/ethiopia-lcluc.js](../app/ethiopia-lcluc.js) into a script, save it, and run it.
3. Check the layers and validation-point popup in the Code Editor.
4. To update a hosted app, publish the saved script through Earth Engine's Apps interface. Editing this repository does not update a deployed app automatically. Ensure that the app itself can read the required assets; see the [Earth Engine Apps documentation](https://developers.google.com/earth-engine/guides/apps).

The app uses Earth Engine's `ee`, `ui`, and `Map` globals. It is not a standalone browser or Node.js application.

### Explore the products

The study boundary and **GSFC LCLU 2017–2024** are enabled initially. Other products are available through checkboxes:

- GSFC 2 m land cover for **2009–2016**, **2018–2022**, and **2017–2024**.
- Observation counts for the same three periods.
- Aligned/reclassified Digital Earth Africa Cropland 2019, ESA WorldCover 2020, ESRI Land Cover 2020, GLAD 2020, and Google Dynamic World 2020.
- Meta Canopy Height 1 m and the 2026 validation/reference points.

Comparison rasters sit beneath the GSFC land-cover layers. Enable one comparison at a time and reduce the **LCLU 2017–2024** opacity or turn off the overlying land-cover layers to see it. The shared five-class legend explicitly covers GSFC, ESA, ESRI, GLAD, and Dynamic World. Cropland extent, observation count, canopy height, and reference points have separate legends.

Enable **Validation / Reference Points 2026**, then click a point to open a bottom-left popup with its class name, numeric `val_class`, and original `Land_Use` label. The nearest point within an eight-pixel click tolerance is highlighted in yellow. Close dismisses the popup; disabling the points through the control panel also clears the selection.

### Classes and NoData

These codes describe the final five-class app products. Historical training datasets and six-class model configurations can use different schemes; check the specific experiment before reusing labels.

| Code | Class | Display color |
| --- | --- | --- |
| 0 | Crop | `#ffaa00` |
| 1 | Tree / Shrub | `#267300` |
| 2 | Grass | `#ffffbe` |
| 3 | Built | `#730000` |
| 4 | Water | `#0070ff` |

The app applies the following class values and masks. The interpretation of codes 0–4 is supported by reference-point sampling; the comparison GeoTIFFs do not contain embedded class names.

| Comparison product | Displayed values | Masked values in the app |
| --- | --- | --- |
| Digital Earth Africa Cropland 2019 | 0 = crop | 255 (non-crop/declared NoData) |
| ESA WorldCover 2020 | 0–4 | −128 |
| ESRI Land Cover 2020 | 0–4 | 7 and 15 |
| GLAD 2020 | 0–4 | 15 |
| Google Dynamic World 2020 | 0–4 | 15 |

ESRI code 7 is treated as NoData for display, in addition to the file's declared NoData value of 15. These masks do not modify the source GeoTIFFs. Crop code 0 remains visible.

### Earth Engine assets

The app reads the following assets under `projects/gsfc-dsg/assets/`:

| Product | Asset name |
| --- | --- |
| LCLU 2009–2016 | `Amhara_LCLU_5class_2009_2016_2m_native_cog_clean_cog` |
| LCLU 2018–2022 | `Amhara_LCLU_5class_2018_2022_2m_native_cog_clean_cog` |
| LCLU 2017–2024 | `Amhara_LCLU_5class_2017_2024_2m_native_cog_clean_cog` |
| Observations 2009–2016 | `Amhara_nobservations_2009_2016_2m_native_cog_clean_cog` |
| Observations 2018–2022 | `Amhara_nobservations_2018_2022_2m_native_cog_clean_cog` |
| Observations 2017–2024 | `Amhara_nobservations_2017_2024_2m_native_cog_clean_cog` |
| Digital Earth Africa | `DigitalEarthAfrica_crop_mask_2019_Amhara_LCLUcrop0_nonCrop255` |
| ESA WorldCover | `ESA_WorldCover_10m_2020_v100_Amhara_reclass` |
| ESRI | `ESRI_LULC_36P37P_2020_Amhara_reclass` |
| GLAD | `GLAD2020_Amhara_reclass` |
| Dynamic World | `Google_DynamicWorld_LULC_2020_mode_2_reclass` |
| Validation points | `Amhara_validation_points_2026` |

Additional dependencies are the boundary asset `projects/ee-jacaraba-ethiopia/assets/boundaries/Amhara_Study_Area_Boundary_4buf10km_EPSG_GEE`, canopy-height collection `projects/sat-io/open-datasets/facebook/meta-canopy-height`, and palette module `users/gena/packages:palettes`.

The comparison images use their first band. Validation points are loaded as a FeatureCollection and filtered to the study boundary.

