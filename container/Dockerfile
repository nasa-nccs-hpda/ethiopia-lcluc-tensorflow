# Modified from https://github.com/NVIDIA-Merlin/Merlin/blob/main/docker/dockerfile.torch
# Additional changes to include Earth science libraries

# Arguments to pass to the image
ARG CUDA_VER=11.2
ARG TORCH_VER=
ARG RAPIDS_VER=21.10
ARG LINUX_VER=ubuntu20.04
ARG FROM_IMAGE=nvcr.io/nvidia/rapidsai/rapidsai

# Import RAPIDS container as the BASE Image (cuda base image)
# Ubuntu 20.08 image with CUDA 11.0 support. 
# runtime container includes examples notebooks and jupyterhub server
# which we do not need.
#FROM ${FROM_IMAGE}:${RAPIDS_VER}-cuda${CUDA_VER}-base-${LINUX_VER}
FROM rapidsai/rapidsai:cuda11.2-base-ubuntu20.04-py3.8

# Adding ENV variables
ENV RAPIDS_DIR=/rapids

# Ubuntu needs noninteractive to be forced
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV DEBIAN_FRONTEND noninteractive
ENV FORCE_CUDA="1"

# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
# Set a fixed model cache directory.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Install Jax with CUDA support.
RUN source activate rapids \
    # && pip install torch==1.7.1+cu110 torchvision -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html \
    && pip install pytorch-toolbelt tifffile rasterio opencv-python webcolors Pillow seaborn \
        rioxarray xgboost pytorch-lightning lightning-bolts omegaconf \
    && pip cache purge

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
