FROM python:3.8-buster as base
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=GMT
RUN mkdir -p /home/src \
    && ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils software-properties-common build-essential bash-completion ca-certificates \
        htop vim gawk telnet tmux git tig screen wget curl cmake unzip gcc ffmpeg jq miller \
        locales locales-all dirmngr gnupg2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV HOME="/root"
ENV PIP_NO_CACHE_DIR=on
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=30
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
WORKDIR /home/src

FROM base as dev
RUN git clone http://github.com//sisinflab/elliot.git \
    && cd elliot \
    && pip install -e . \
    && pip install protobuf==3.19.0
COPY experiments/elliot.py elliot/elliot.py
WORKDIR /home/src/elliot
