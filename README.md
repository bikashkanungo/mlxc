# Demonstration of various ML-based XC functionals
This repository contains the .ptc model files for the NNLDA, NNGGA, and NNGGA-UEG XC functionals (see `models/` directory). It also contains straightforward demonstration on how to use the files. The `demo/demo_NNLDA.py` provides the demonstration on how to use the NNLDA functional. The `demo/demo_NNGGA.py` provides the demonstration for both NNGGA and NNGGA-UEG functionals (one only needs to supply the appropriate .ptc file for NNGGA and NNGGA-UEG inside it). The `bench` directory contains aggregated benchmarks for the NN-based functionals.
## Requirements
1. Numpy
2. Torch
3. Libxc with python bindings (https://libxc.gitlab.io/installation/)
