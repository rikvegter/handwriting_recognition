FROM ubuntu:20.04
ARG DEBIAN_FRONTEND="noninteractive" 
ENV TZ="Europe/Amsterdam"
COPY . /hwr
WORKDIR /hwr

# Install required packages
RUN \
 echo "***** Installing basic dependencies *****" && \
 apt-get update && \
 apt-get install -y \
     libssl-dev openssl git wget build-essential mergerfs

# Compile and install python 3.8.7
RUN \
 echo "***** Compiling Python 3.8.7 *****" && \
 echo "deb-src http://archive.ubuntu.com/ubuntu/ focal main" >> /etc/apt/sources.list && \
 apt-get update && \
 apt-get build-dep -y python3 && \
 apt-get install -y gdb lcov libbz2-dev libffi-dev \
      libgdbm-dev liblzma-dev libncurses5-dev libreadline6-dev \
      libsqlite3-dev libssl-dev lzma lzma-dev tk-dev uuid-dev zlib1g-dev && \
 wget https://www.python.org/ftp/python/3.8.7/Python-3.8.7.tgz && \
 tar -xzvf Python-3.8.7.tgz && \
 cd Python-3.8.7 && \
 ./configure --with-ensurepip=install && \
 make && \
 make install

# Set up our project
RUN \
 echo "***** Setting up project  *****" && \
 python3 -m pip install --upgrade pip && \
 python3 -m pip install -r requirements.txt

# Cleanup
RUN \
 apt-get purge --auto-remove -y \
        build-essential libssl-dev openssl git wget && \
 apt-get clean && \
 rm -rf \
        /tmp/* \
        /var/lib/apt/lists/* \
        /var/tmp/* \
        $HOME/.cache

ENV MOUNT_DIR=/files
ENV RUN_DIR=/hwr_run
ENV CONTAINER_DIR=/hwr

# Setup mergerfs command
RUN \
 echo '#!/bin/sh' > /mergerfs.sh && \
 echo '/usr/bin/mergerfs "$MOUNT_DIR":"$CONTAINER_DIR=NC" "$RUN_DIR" -o defaults' >> /mergerfs.sh && \
 chmod +x /mergerfs.sh && \
 mkdir /hwr_run

CMD bash /mergerfs.sh && /python3 /hwr_run/main.py

