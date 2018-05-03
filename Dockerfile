FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
# FROM ubuntu:16.04
MAINTAINER Christian Schroeder de Witt

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*


#Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger

RUN mkdir /install
WORKDIR /install

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------

# RUN git clone https://github.com/csarofeen/pytorch /install/pytorch && cd /install/pytorch 
# RUN pip3 install numpy pyyaml mkl setuptools cffi
# RUN apt-get install -y cmake gcc 
# RUN cd /install/pytorch && python3 setup.py install
#RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision snakeviz

#### -------------------------------------------------------------------
#### install tensorflow
#### -------------------------------------------------------------------
RUN pip3 install tensorflow-gpu
#RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp34-cp34m-linux_x86_64.whl


#### -------------------------------------------------------------------
#### install MongoDB (for Sacred)
#### -------------------------------------------------------------------

# Install pymongo
RUN pip3 install pymongo

#### -------------------------------------------------------------------
#### install pysc2 (from Mika fork)
#### -------------------------------------------------------------------

RUN git clone https://github.com/samvelyan/pysc2.git /install/pysc2
RUN pip3 install /install/pysc2/

#### -------------------------------------------------------------------
#### install Sacred (from OxWhirl fork)
#### -------------------------------------------------------------------

RUN pip3 install setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### install OpenBW (from OxWhirl fork)
#### -------------------------------------------------------------------

#RUN apt-get install -y libsdl2-dev libsdl2-2.0
#RUN git clone https://github.com/oxwhirl/openbw /install/openbw-git
#RUN git clone https://github.com/oxwhirl/bwapi /install/bwapi-git
#RUN cd /install/bwapi-git && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENBW_DIR=/install/openbw-git -
# DOPENBW_ENABLE_UI=1 -DCMAKE_INSTALL_PREFIX=/install && make

#RUN apt-get install -y libzstd-dev
#RUN wget https://github.com/zeromq/libzmq/releases/download/v4.2.2/zeromq-4.2.2.tar.gz
#RUN tar xvzf zeromq-4.2.2.tar.gz
#RUN sudo apt-get update && sudo apt-get install -y libtool pkg-config build-essential autoconf automake uuid-dev
#RUN cd zeromq-4.2.2 && ./configure && sudo make -j 12 install && sudo ldconfig && ldconfig -p | grep zmq

RUN apt-get install -y libzstd1 libzstd1-dev zstd libzmq-dev
RUN pip3 install pybind11
RUN git clone https://github.com/oxwhirl/TorchCraft.git torchcraft
RUN cd torchcraft && git submodule update --init --recursive && pip3 install .

#### -------------------------------------------------------------------
#### final steps
#### -------------------------------------------------------------------
RUN pip3 install pygame

#### -------------------------------------------------------------------
#### Plotting tools
#### -------------------------------------------------------------------

# RUN apt-get -y install ipython ipython-notebook
RUN pip3 install statsmodels pandas seaborn
RUN mkdir /pool && echo "export PATH=$PATH:'/pool/pool'" >> ~/.bashrc
# RUN cd /pool && git clone https://github.com/oxwhirl/pool.git pool-repo &&  ln -s pool-repo/pool && git submodule update --init --recursive

RUN pip3 install cloudpickle ruamel.yaml

EXPOSE 8888


WORKDIR /pymarl
# RUN echo "mongod --fork --logpath /var/log/mongod.log" >> ~/.bashrc
#CMD ["mongod", "--fork", "--logpath", "/var/log/mongod.log"]
# EXPOSE 27017
# EXPOSE 28017

# CMD service mongod start && tail -F /dev/null
