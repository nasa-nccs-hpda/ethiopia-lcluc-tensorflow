# Supply a tested base tag/digest for a publication build.
ARG BASE_IMAGE=nasanccs/vhr-cloudmask:latest
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/nasa-nccs-hpda/ethiopia-lcluc-tensorflow"

WORKDIR /opt/ethiopia-lcluc-tensorflow
COPY pyproject.toml MANIFEST.in README.md ./
COPY ethiopia_lcluc_tensorflow ./ethiopia_lcluc_tensorflow
COPY examples ./examples
COPY app ./app
COPY docs ./docs
COPY slurm ./slurm
RUN python -m pip install --no-cache-dir '.[validation]'

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["ethiopia-cnn", "--help"]
