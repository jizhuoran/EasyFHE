# Homomorphic Encryption Example with ResNet-20 on CIFAR-10

This repository provides an example implementation of homomorphic encryption applied to a ResNet-20 model trained on the CIFAR-10 dataset. The design and implementation closely follow the methodology described in the paper "[Low-Complexity Deep Convolutional Neural Networks on Fully Homomorphic Encryption Using Multiplexed Parallel Convolutions](https://proceedings.mlr.press/v162/lee22e/lee22e.pdf)", which details the packing strategy and algorithm design.

## Setup and Prerequisites

1. **Environment Configuration:**  
   Set the system environment variable `DATA_DIR` to the path of a directory where the dataset and pre-encoded weights will be stored.

2. **Download Required Files:**  
   Download the prepared CIFAR-10 dataset and pre-encoded weights from [this link](https://1drv.ms/f/c/bf37f4266c3f52d0/EudeJ2juTltFvAnRS8yypz0BVMYR65X7sQvEyCXleme8gQ?e=paaZNk).  

   Place both the dataset and the weights into the directory specified by `DATA_DIR`.

## Running the Example

Once the prerequisites are completed, run the example by executing the following command in the repository’s root directory:

```bash
python3 fhe-mp-cnn.py
```

### First Run Considerations

- **Context Generation:**  
  On the first run, EasyFHE will generate the corresponding cryptographic context, which will be saved to the `DATA_DIR` directory. This process can take several minutes on high-end machines.

- **Subsequent Runs:**  
  For later executions, EasyFHE will load the pre-generated context from the file, resulting in a significantly faster startup time.

## Implementation Details

The implementation in `fhe-mp-cnn.py` is directly based on the techniques described in "[Low-Complexity Deep Convolutional Neural Networks on Fully Homomorphic Encryption Using Multiplexed Parallel Convolutions](https://proceedings.mlr.press/v162/lee22e/lee22e.pdf)". Although this implementation demonstrates the potential of homomorphic encryption for deep learning, the current design may not be optimal for GPU acceleration. We welcome contributions that propose and implement more efficient algorithmic designs.

## Performance

TBW

## Project Team

The resnet example is developed and actively maintained by:
- [Kanyu Ye](https://github.com/kanyuYe)
- [Honghui You](https://github.com/youhonghui)
Contributions from the broader community are welcome and greatly appreciated.



## how to install fhe-mp-cnn

note that `set(CMAKE_PREFIX_PATH "~/PNP/seal-modified-3.6.6-install" ${CMAKE_PREFIX_PATH})` is added in CMakeLists.txt,
and the path should be aligned with the following installation steps.

```bash
# install seal-3.6.6 modified
git clone --recursive git@github.com:snu-ccl/FHE-MP-CNN.git
cd cnn_ckks/cpu-ckks/single-key/seal-modified-3.6.6
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/PNP/seal-modified-3.6.6-install ..
make -j
make install

# build fhe-mp-cnn
cd ~/PNP/FHE-MP-CNN/cnn_ckks
mkdir build
cd build
cmake ..
make -j
```









## Dependencies
The GMP library and the NTL library is needed for the Remez algorithm.
OpenMP library is needed for the multi-threaded execution of cnn.
All programs for building the SEAL library is needed.
This source code has been developed and checked in Ubuntu-20.04.
384GB RAM is required to test one image. (not exact hard limit but encouraged)
512GB RAM is required to test 50 images simultaneously with multi-threading. (not exact hard limit but encouraged)

## Regarding SEAL library
We use Microsoft SEAL library version 3.6.6 for RNS-CKKS homomorphic encryption scheme. Since the original SEAL library version 3.6.6 is not bootstrapping-friendly, we modified the SEAL library so that the bootstrapping operation of the RNS-CKKS scheme can be implemented. You should build and install the modified SEAL library in "cnn_ckks/cpu-ckks/single-key/seal-modified-3.6.6" if you want to build our homomorphic ResNet CNN source code. Specifically, you can build and install the modified SEAL library by the following commands.

```PowerShell
cd cnn_ckks/cpu-ckks/single-key/seal-modified-3.6.6
cmake -S . -B build
cmake --build build
cmake --install build
```

## Building cnn_ckks 
The executable file can be built by the following commands.

```PowerShell
cd cnn_ckks
cmake -S . -B build
cd build
make
```

The build outputs including executable is stored in "cnn_ckks/build" directory, and the executable file is named as "cnn".

## Executing the executable file
There are some additional parameters when executing the "cnn" file, and the form of the command should be as follows.

```PowerShell
cd cnn_ckks/build
./cnn (LAYER NUMBER) (DATASET NUMBER) (START IMAGE) (END IMAGE)
```

(LAYTER NUMBER) : the number of layers in cnn
(DATASET NUMBER) : the type of dataset (10: CIFAR-10, 100: CIFAR-100)
(START IMAGE) : the label number of the first image you want to infer
(END IMAGE) : the label number of the last image you want to infer

For example, if you want to perform ResNet-110 for images in CIFAR-10 test dataset with label number 6, 7, 8, 9, and 10, you may want to execute the "cnn" file as follows.

```PowerShell
cd cnn_ckks/build
./cnn 110 10 6 10 
```

If you want to perform ResNet-32 for only an image in CIFAR-100 with label number 4, you may want to execute the "cnn" file as follows.

```PowerShell
cd cnn_ckks/build
./cnn 32 100 4 4 
```

Supported layers : CIFAR-10 - 20, 32, 44, 56, 110 / CIFAR-100 - 32
Supported image label numbers : 0 ~ 9999

## Checking results
Text files for various intermediate and final results are generated in the result directory in the root directory of our supplementary file.
The following text files are generated.

resnet(LAYER NUMBER)_cifar(DATASET NUMBER)_image(IMAGE NUMBER).txt: This file includes information about the running time, the remaining level, and the scaling factor for each procedure. Also, it partly shows the decrypted values when each layer is terminated and shows the resultant decrypted values. Finally, it shows the inference result with the correct image label and total running time. This type of file is generated for each input image.

resnet(LAYER NUMBER)_cifar(DATASET NUMBER)_label_(START IMAGE)_(END IMAGE): This file includes information about the inference results with the correct image label for all images.
