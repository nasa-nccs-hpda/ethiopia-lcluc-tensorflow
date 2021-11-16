# Ethiopia LCLUC

Ethiopia LCLUC

## Container

```bash
module load singularity
singularity build --sandbox nccs-lcluc docker://gitlab.nccs.nasa.gov:5050/nccs-lcluc/alaska-lcluc/nccs-lcluc
```

## Using the container

```bash
singularity shell -B /att --nv nccs-lcluc
source activate rapids
```