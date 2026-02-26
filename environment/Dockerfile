# hash:sha256:7dedcf7c4bf5a563ff9be1e0aba0bc8176c8eeedb053fc0534446603f0e0fe01
FROM registry.codeocean.com/codeocean/py-r:python3.12.8-R4.4.2-JupyterLab4.3.5-RStudiorstudio-server-2024.12.1-563-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        nilearn==0.13.0 \
    && conda clean -ya

COPY postInstall /
RUN /postInstall