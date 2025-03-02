#!/bin/bash

cd ~
mkdir PNP
cd ./PNP

python3 -m venv .venv
source ./.venv/bin/activate

git clone --recursive -b yhh-gpu-dev git@github.com:PNP-team/GPU-FHE.git
cd GPU-FHE
pip install -r requirements.txt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
USE_DISTRIBUTED=0 USE_MKLDNN=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 USE_NINJA=OFF USE_ROCM=0 python3 setup.py develop --install-dir=~/torch/
mkdir ./torch/fhe/data
mkdir ./torch/fhe/example/data
cd ../

git clone --recursive git@github.com:openfheorg/openfhe-development.git
cd openfhe-development
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/PNP/openfhe ..
make -j
make install
cd ../../

pip install pybind11
export CMAKE_PREFIX_PATH=$(pip show pybind11 | grep Location | awk '{print $2}'):$CMAKE_PREFIX_PATH

git clone --recursive git@github.com:jizhuoran/openfhe-python.git
cd openfhe-python
mkdir build
cd build
export CMAKE_PREFIX_PATH=~/PNP/openfhe:$CMAKE_PREFIX_PATH
cmake ..
make -j
cp openfhe.cpython*.so ~/PNP/GPU-FHE/torch/fhe/client/
cd ../../