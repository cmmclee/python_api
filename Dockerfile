#!/usr/bin/env bash
# Use Ubuntu:16.04 image as parent image
FROM ubuntu:16.04

EXPOSE 5000
VOLUME /root/volume
USER root

# Modify apt-get to aliyun mirror
RUN sed -i 's/archive.ubuntu/mirrors.aliyun/g' /etc/apt/sources.list
RUN apt-get update

# Clone the docker-opencv-python repository
RUN apt-get -y install git
RUN git clone https://github.com/cmmclee/python_api.git /docker-opencv-python
WORKDIR /docker-opencv-python

## Modify timezone to GTM+8
ENV TZ=Asia/Shanghai
RUN apt-get -y install tzdata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Modify locale
RUN apt-get -y install locales
RUN locale-gen en_US.UTF-8
RUN echo "LANG=\"en_US.UTF-8\"" > /etc/default/locale && \
    echo "LANGUAGE=\"en_US:en\"" >> /etc/default/locale && \
    echo "LC_ALL=\"en_US.UTF-8\"" >> /etc/default/locale

# Install necessary library
RUN apt-get -y install apt-utils \
    lib32z1 libglib2.0-0 libsm6 libxrender1 \
    libxext6 libice6 libxt6 libfontconfig1 libcups2

# Install python3
RUN  apt-get install -y python3

# Install pip
RUN apt-get install -y wget vim
RUN wget -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN python3 /tmp/get-pip.py

# Modify pip mirror
RUN mkdir -p /root/.pip
RUN echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url=http://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf && \
    echo "[install]" >> /root/.pip/pip.conf && \
    echo "trusted-host=mirrors.aliyun.com" >> /root/.pip/pip.conf

# Install necessary python-library
RUN pip install --upgrade pip
RUN pip install flask numpy scipy opencv-python tensorflow keras

# Make startup run file
CMD python3 app.py

