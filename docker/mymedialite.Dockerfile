FROM ubuntu:20.04 as base
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
RUN git clone https://github.com/zenogantner/MyMediaLite.git \
    && gpg --homedir /tmp --no-default-keyring --keyring /usr/share/keyrings/mono-official-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF \
    && echo "deb [signed-by=/usr/share/keyrings/mono-official-archive-keyring.gpg] https://download.mono-project.com/repo/ubuntu stable-focal main" | tee /etc/apt/sources.list.d/mono-official-stable.list \
    && apt update \
    && apt install -y --no-install-recommends mono-devel \
    && cd MyMediaLite \
    && make
WORKDIR /home/src/MyMediaLite
CMD ["bash"]
