FROM python:3.10-buster as base
WORKDIR /home/src
ENV TZ=GMT
RUN mkdir -p /home/src \
    && ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && apt-get update || true \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-utils software-properties-common build-essential bash-completion ca-certificates \
        htop vim gawk telnet tmux git tig screen wget curl cmake unzip gcc ffmpeg jq miller \
        locales locales-all dirmngr gnupg2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

FROM base as dev
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
RUN pip install poetry==1.7.1 --no-cache-dir
COPY poetry.lock pyproject.toml Makefile /home/src/
RUN make install
COPY . .
CMD ["bash"]
