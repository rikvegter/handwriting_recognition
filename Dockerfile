FROM ubuntu:20.04
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="Europe/Amsterdam"
ENV NUMBA_CACHE_DIR="/tmp/numba"
ENV MPLCONFIGDIR="/tmp/matplotlib"
COPY . /hwr
WORKDIR /hwr

# Install required packages
RUN \
 echo "***** Installing basic dependencies *****" && \
 apt-get update && \
 apt-get install -y \
     libssl-dev openssl python3 python3-pip python3-distutils python3-setuptools && \
 echo "***** Setting up project  *****" && \
 python3 -m pip install --upgrade pip && \
 python3 -m pip install -r /hwr/requirements.txt && \
 apt-get purge --auto-remove -y libssl-dev openssl && \
 apt-get clean && \
 rm -rf \
        /tmp/* \
        /var/lib/apt/lists/* \
        /var/tmp/* \
        $HOME/.cache

CMD python3 /hwr/main.py -i "/input" -o "/output/results" && python3 /hwr/style/scripts/classify_style.py "/input/" "/hwr/style/classifier/" "/output/results/"

