# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.8    (apt)
# pytorch       1.6 (pip)
# ==================================================================
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

# ==================================================================
# tools
# ------------------------------------------------------------------
RUN apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
	    nano \
        libx11-dev \
        fish \
        libsparsehash-dev \
        software-properties-common \
        libsm6 \
        libxext6 \
        libxrender-dev

# ==================================================================
# python
# ------------------------------------------------------------------
WORKDIR /PDV
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3-distutils \
        python3-apt \
        python3-pip \
        python3-setuptools
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip --no-cache-dir install --upgrade -r requirements.txt

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# Install cmake v3.13.2
RUN apt-get purge -y cmake && \
    mkdir /root/temp && \
    cd /root/temp && \
    wget https://cmake.org/files/v3.13/cmake-3.13.2.tar.gz && \
    tar -xzvf cmake-3.13.2.tar.gz && \
    cd cmake-3.13.2 && \
    bash ./bootstrap && \
    make && \
    make install && \
    cmake --version && \
    rm -rf /root/temp

WORKDIR /root

# Install Boost geometry
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.68.0/source/boost_1_68_0.tar.gz
RUN tar xzvf boost_1_68_0.tar.gz
RUN cp -r ./boost_1_68_0/boost /usr/include
RUN rm -rf ./boost_1_68_0
RUN rm -rf ./boost_1_68_0.tar.gz

# Install spconv v1.1
RUN git clone https://github.com/traveller59/spconv.git
RUN cd ./spconv && git checkout abf0acf30f5526ea93e687e3f424f62d9cd8313a && git submodule update --init --recursive && SPCONV_FORCE_BUILD_CUDA=1 python setup.py bdist_wheel
RUN python -m pip install /root/spconv/dist/spconv-*-cp38-cp38-linux_x86_64.whl && \
    rm -rf /root/spconv
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages/spconv:${LD_LIBRARY_PATH}"

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# nvidia runtime
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04 \
 /usr/lib/x86_64-linux-gnu \
 /usr/lib/x86_64-linux-gnu

COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04 \
 /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
 /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN echo '/usr/local/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
 ldconfig && \
 echo '/usr/$LIB/libGL.so.1' >> /etc/ld.so.preload && \
 echo '/usr/$LIB/libEGL.so.1' >> /etc/ld.so.preload

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# ==================================================================
# PDV Framework
# ------------------------------------------------------------------
WORKDIR /PDV
COPY pcdet pcdet
COPY setup.py setup.py
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV PYTHONPATH="/usr/lib/python3.8/site-packages/:${PYTHONPATH}"
RUN python setup.py develop --install-dir=/usr/local/lib/python3.8/dist-packages/
RUN mkdir checkpoints && mkdir data && mkdir output && mkdir tests && mkdir tools

# Tensorflow 2.2 doesn't work with CUDA 10.2
# Symlink hack to get it working: https://github.com/tensorflow/tensorflow/issues/38194
RUN ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudart.so.10.2 /usr/lib/x86_64-linux-gnu/libcudart.so.10.1

# Tensorflow 2.2 needs libcudnn.so.7
# Map existing libcudnn.so.8: https://github.com/tensorflow/tensorflow/issues/20271#issuecomment-643296453
# Also add path to LD_LIBRARY_PATH so tensorflow can find it
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so.7
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
