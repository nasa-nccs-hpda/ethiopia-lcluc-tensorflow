# Additional changes to include Earth science libraries

# Arguments to pass to the image
ARG CUDA_VER=11.0
ARG TORCH_VER=
ARG RAPIDS_VER=21.10
ARG LINUX_VER=ubuntu20.04
ARG FROM_IMAGE=nvcr.io/nvidia/rapidsai/rapidsai

# Import RAPIDS container as the BASE Image (cuda base image)
# Ubuntu 20.08 image with CUDA 11.0 support. 
# runtime container includes examples notebooks and jupyterhub server
# which we do not need.
FROM ${FROM_IMAGE}:${RAPIDS_VER}-cuda${CUDA_VER}-base-${LINUX_VER}

# Adding ENV variables
ENV RAPIDS_DIR=/rapids

# Install Jax with CUDA support.
RUN source activate rapids \
    && pip install torch==1.7.1+cu110 torchvision -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install pytorch-toolbelt tifffile rasterio opencv-python webcolors Pillow seaborn \
        rioxarray xgboost pytorch-lightning \
    && pip cache purge

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
