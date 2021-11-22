FROM ubuntu:20.04

LABEL maintainer="jordan.a.caraballo-vega@nasa.gov"

LABEL BENCHMARK NASA-SPEC2601OS.Ubuntu20 v1.0
LABEL OS Ubuntu
LABEL OS_MAJOR_VERS 20

ENV DEBIAN_FRONTEND noninteractive

# Install Ansible, run baseline playbook
# RUN sed -i 's/http:\/\/archive/http:\/\/us.archive/g' /etc/apt/sources.list && \
RUN apt-get update && apt-get -y dist-upgrade && \
    apt-get -y install python3-pip python3-apt libpam-pwquality git && \
    pip3 install ansible && \
    git clone https://gitlab.nccs.nasa.gov/nccs-swg/nasa-linux-baseline-ansible.git /baseline && \
    cd /baseline && \
    ansible-playbook -i localhost, -c local playbook.yml -e 'ansible_python_interpreter=/usr/bin/python3' && \
    pip3 uninstall ansible -y && \
    apt-get -y purge python3-pip python3-apt && apt-get -y autoremove && \
    rm -rf /var/cache/apt && \
    rm -rf /baseline

HEALTHCHECK NONE

CMD ["/bin/bash"]
