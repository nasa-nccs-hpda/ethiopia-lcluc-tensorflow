FROM nasanccs/vhr-cloudmask:latest
LABEL maintainer="jordan.a.caraballo-vega@nasa.gov"

ENV TFC_VERSION=0.1.1
ENV ETHIOPIA_LCLUC_TENSORFLOW_VERSION=main

RUN pip install git+https://github.com/nasa-nccs-hpda/ethiopia-lcluc-tensorflow.git@main

HEALTHCHECK NONE
ENTRYPOINT [""]
