# Ethiopia LCLUC

Ethiopia LCLUC using WorldView imagery

[![DOI](https://zenodo.org/badge/527702332.svg)](https://zenodo.org/badge/latestdoi/527702332)

## Objectives

- LCLUC utilizing random forest algorithm
- LCLUC utilizing XGBoost algorithm
- LCLUC utilizing CNN algorithm

## Data Catalog

- Project Location: /explore/nobackup/projects/ilab/projects/Ethiopia/LCLUC_Ethiopia
- Full Domain Data Location: /adapt/nobackup/people/mwooten3/Ethiopia_Woubet/VHR
- Gonji Subset Data Location: /adapt/nobackup/people/walemu/NASA_NPP/CRPld_Map_Pred_and_Forec/EVHR/Gonji_Subset/5-toas


## Container

```bash
module load singularity
singularity build --sandbox nccs-lcluc docker://gitlab.nccs.nasa.gov:5050/nccs-lcluc/alaska-lcluc/nccs-lcluc
```

## Using the container

```bash
singularity shell -B /att,/lscratch/jacaraba:/tmp --nv nccs-lcluc
source activate rapids
```

singularity shell --nv -B /lscratch,/explore/nobackup/projects/ilab,/explore/nobackup/projects/3sl,/explore/nobackup/people /lscratch/jacaraba/container/tensorflow-caney
export PYTHONPATH="/explore/nobackup/people/jacaraba/development/tensorflow-caney:/explore/nobackup/people/jacaraba/development/ethiopia_lcluc_tensorflow"