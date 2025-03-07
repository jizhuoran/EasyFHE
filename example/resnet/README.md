# Homomorphic Encryption Example with ResNet-20 on CIFAR-10

This repository provides an example implementation of homomorphic encryption applied to a ResNet-20 model trained on the CIFAR-10 dataset. The design and implementation closely follow the methodology described in the paper "[Encrypted Image Classification with Low Memory Footprint using Fully Homomorphic Encryption](https://eprint.iacr.org/2024/460.pdf)", which details the packing strategy and algorithm design.

## Setup and Prerequisites

1. **Environment Configuration:**  
   Set the system environment variable `DATA_DIR` to the path of a directory where the dataset and pre-encoded weights will be stored.

2. **Download Required Files:**  
   Download the prepared CIFAR-10 dataset and pre-encoded weights from [this link](https://1drv.ms/f/c/bf37f4266c3f52d0/EudeJ2juTltFvAnRS8yypz0BVMYR65X7sQvEyCXleme8gQ?e=paaZNk).  

   Place both the dataset and the weights into the directory specified by `DATA_DIR`.

## Running the Example

Once the prerequisites are completed, run the example by executing the following command in the repository’s root directory:

```bash
python3 resnet20.py
```

### First Run Considerations

- **Context Generation:**  
  On the first run, EasyFHE will generate the corresponding cryptographic context, which will be saved to the `DATA_DIR` directory. This process can take several minutes on high-end machines.

- **Subsequent Runs:**  
  For later executions, EasyFHE will load the pre-generated context from the file, resulting in a significantly faster startup time.

## Implementation Details

The implementation in `resnet20.py` is directly based on the techniques described in "[Encrypted Image Classification with Low Memory Footprint using Fully Homomorphic Encryption](https://eprint.iacr.org/2024/460.pdf)". Although this implementation demonstrates the potential of homomorphic encryption for deep learning, the current design may not be optimal for GPU acceleration. We welcome contributions that propose and implement more efficient algorithmic designs.

## Performance

TBW

## Project Team

The resnet example is developed and actively maintained by:
- [Honghui You](https://github.com/Eyxxxxx)
- [Kanyu Ye](https://github.com/kanyuYe)
- [Zhuoran Ji](https://github.com/jizhuoran)

Contributions from the broader community are welcome and greatly appreciated.

