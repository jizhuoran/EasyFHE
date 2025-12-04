# Homomorphic Encryption Example with Logistic Regression Training

This repository demonstrates an example implementation of homomorphic encryption techniques integrated into deep learning. It features logistic regression training while leveraging the methodology described in the paper “[Secure logistic regression based on homomorphic encryption: Design and evaluation](https://eprint.iacr.org/2018/074.pdf)”. This paper details the packing strategy and algorithm design that form the foundation of this implementation.

## Overview

This project illustrates how homomorphic encryption can be applied in deep learning workflows. Although the showcased model is based on the ResNet-20 architecture applied to the CIFAR-10 dataset, the underlying encryption strategies are derived from secure logistic regression techniques. While the example demonstrates the potential of homomorphic encryption in a deep learning context, the current design may not be optimized for GPU acceleration, and further improvements are welcome.

## Setup and Prerequisites

1. **Environment Configuration:**  
   Set the system environment variable `DATA_DIR` to the path of a directory where the encrypted input data will be stored. Within that directory, create a `helr/encData` subdirectory.

2. **Dependencies:**  
   Ensure you have Python 3 installed. Check the repository's dependency file (if provided) for additional required packages.

## Running the Example

After completing the setup, navigate to the repository’s root directory and run the example with the following command:

```bash
python3 helr.py
```

### First Run Considerations

- **Context Generation:**  
  On the first execution, EasyFHE will generate the necessary cryptographic context and save it to the `DATA_DIR` directory. Note that this process may take several minutes, even on high-end machines.

- **Subsequent Runs:**  
  For later executions, EasyFHE will load the pre-generated context from disk, resulting in a substantially faster startup time.

## Implementation Details

The implementation in `helr.py` directly adapts techniques from “[Secure logistic regression based on homomorphic encryption: Design and evaluation](https://eprint.iacr.org/2018/074.pdf)”. Although the example showcases the integration of homomorphic encryption within a deep learning model, it is a starting point—improvements such as efficient GPU acceleration and further algorithmic optimizations are welcome.

## Performance

*Performance metrics and benchmarks will be updated in future releases.*

## Project Team

The ResNet example is actively developed and maintained by:

- [Yusi Chen](https://github.com/chenyusii)
- [Honghui You](https://github.com/youhonghui)
- [Zhuoran Ji](https://github.com/jizhuoran)

Contributions and community feedback are always greatly appreciated.

