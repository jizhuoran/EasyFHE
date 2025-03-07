# EasyFHE Bootstrap Example

This repository provides a bootstrap example for EasyFHE, a cryptographic framework designed for efficient homomorphic encryption. The example demonstrates the initialization and execution of the cryptographic context required for EasyFHE.

## Setup and Prerequisites

1. **Environment Configuration:**  
   Prior to running the example, ensure that the system environment variable `DATA_DIR` is set to the directory path where the cryptographic context will be stored.

## Running the Example

After completing the setup, execute the following command in the repository's root directory:

```bash
python3 bootstrap.py
```

### First Run Considerations

- **Context Generation:**  
  On the initial execution, EasyFHE will generate the required cryptographic context. This context will be stored in the directory specified by `DATA_DIR`. Note that this generation process may take several minutes, even on high-end systems.

- **Subsequent Runs:**  
  On later executions, EasyFHE will load the pre-generated context from the file, which significantly reduces the startup time.

## Performance

*To be written.* (TBW)

## Project Team

The bootstrap example is developed and actively maintained by:
- [Honghui You](https://github.com/Eyxxxxx)
- [Zhuoran Ji](https://github.com/jizhuoran)

Contributions from the broader community are welcome and greatly appreciated.

