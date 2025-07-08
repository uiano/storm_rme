# <span style="color:black">STORM</span>: <span style="color:blue">S</span>patial <span style="color:blue">T</span>ransf<span style="color:blue">O</span>rmer for <span style="color:blue">R</span>adio <span style="color:blue">M</span>ap Estimation

This repository contains the implementation of STORM and the code to reproduce the numerical experiments of the paper ["Spatial Transformer for Radio Map Estimation"](https://arxiv.org/abs/2411.01211), by Pham Q. Viet and Daniel Romero. The paper was presented in the International Conference on Communications (ICC), 2025.



## Overview
- Python version: you may need Python 3.12 or later.

- The experiments (code) to train and evaluate STORM are in `experiments/transformer_experiments.py`.

- Training and testing data are available inside folder `data`. We will work with three datasets, referred to as USRP, Gradiant, and ray-tracing; for more information, see ["Radio Map Estimation: Empirical Validation and Analysis"](https://arxiv.org/pdf/2310.11036).

## Radio map estimation
1. Create training datasets:
    - You can use Experiments 1000, 1005, or 1010 to create a training dataset for STORM.
    - Experiment 1015 uses ray-tracing data to create a training set for other DNN benchmarks.

The obtained training datasets will be stored inside folder `output/datasets`.

2. To train STORM with USRP, Gradiant, ray-tracing data, use Experiments 2000, 2005, and 2010, respectively.

3. MC 

## Active sensing
- Experiment
- Experiment
