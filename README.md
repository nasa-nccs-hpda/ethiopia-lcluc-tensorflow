# Ethiopia LCLUC

Ethiopia LCLUC

## Data Directory

/adapt/nobackup/projects/ilab/projects/Ethiopia/LCLUC_Ethiopia

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