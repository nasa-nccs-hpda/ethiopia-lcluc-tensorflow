#!/usr/bin/env bash
#SBATCH --job-name=ethiopia-predict
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail

# Submit from the repository root:
# sbatch --export=ALL,ETHIOPIA_CONTAINER=/path/to/image.sif slurm/predict.sh /path/to/run.yaml
# Override resource directives and partition/account with sbatch arguments.
: "${ETHIOPIA_CONTAINER:?Set ETHIOPIA_CONTAINER to a compatible container image}"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG.yaml [additional CNN CLI arguments]" >&2
  exit 2
fi
ethiopia_config=$1
shift
ethiopia_repo=${ETHIOPIA_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
# Additional data mounts and source checkouts can be passed through these variables.
ethiopia_binds="$ethiopia_repo:$ethiopia_repo${ETHIOPIA_BINDS:+,$ETHIOPIA_BINDS}"
ethiopia_pythonpath="$ethiopia_repo${ETHIOPIA_PYTHONPATH:+:$ETHIOPIA_PYTHONPATH}"

srun singularity exec --nv --bind "$ethiopia_binds" --pwd "$ethiopia_repo" \
  --env "PYTHONPATH=$ethiopia_pythonpath" "$ETHIOPIA_CONTAINER" \
  python -m ethiopia_lcluc_tensorflow.view.landcover_cnn_pipeline_cli \
  -c "$ethiopia_config" -s predict "$@"
