# Readme

## Installation

This guide explains how to set up the GPU-FHE framework based on PyTorch along with OpenFHE and the customized OpenFHE Python bindings. Follow these steps to configure your development environment:

1. **Create a Directory for Development Files**  
   Start by creating a directory named `PNP` where all your project-related files will be stored.

    ```bash
    cd ~
    mkdir PNP
    cd PNP
    ```

2. **Set Up the Virtual Environment (venv)**  
   Create and activate a virtual environment. This step is mandatory, but you can place the `.venv` directory anywhere in your home folder. **(The `.venv` folder is assumed to be placed inside the PNP directory in this guide.)**

    ```bash
    python3 -m venv .venv        # Create a virtual environment (optional, can be placed anywhere)
    source .venv/bin/activate    # Activate the virtual environment
    ```

3. **Install GPU-FHE**  
   Clone the GPU-FHE repository, install the required dependencies: 

    ```bash
    git clone --recursive -b yhh-gpu-dev git@github.com:PNP-team/GPU-FHE.git
    cd GPU-FHE
    pip install -r requirements.txt
    ```

   Set up the CUDA environment variables:

    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    export PATH=$PATH:/usr/local/cuda/bin
    export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
    ```

   Build and install GPU-FHE. Create the necessary data folders.

    ```bash
    USE_DISTRIBUTED=1 USE_NCCL=1 BUILD_TEST=0 USE_NINJA=OFF USE_ROCM=0 python3 setup.py develop --install-dir=~/torch/
    mkdir ./torch/fhe/data
    mkdir ./torch/fhe/example/data
    cd ../
    ```
   
4. **Install OpenFHE (Local Installation)**  
   Clone and build OpenFHE:

    ```bash
    git clone --recursive git@github.com:openfheorg/openfhe-development.git
    cd openfhe-development
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=~/PNP/openfhe ..
    make -j
    make install
    cd ../../
    ```

5. **Install OpenFHE-Python (Custom Version)**  
   Install pybind11 and then build the openfhe-python bindings with the following steps:

    ```bash
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
    ```
