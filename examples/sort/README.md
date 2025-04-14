This repo contains source code for sorting benchmark. 

Prerequisite:
- GPU-FHE
- OpenFHE--catslab version

1. Run the project

Run:
```bash
cd examples/sort/src
python3 ./sort.py
```

2. Change the hyperparameters

- Length of the input array could be easily changed by input a different number into the **Sorting** function. **Sorting** function will then perform sorting on an array with the length of your input.

- Hyperparameters for non-linear function and FHE scheme should be kept untouched unless you are familiar with FHE, openFHE and GPU-FHE.

---

original repo: https://github.com/FHE-Applications/FHE-Applications/tree/master/dev/CKKS-App/Sorting
