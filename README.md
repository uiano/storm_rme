# <span style="color:black">STORM</span>: <span style="color:blue">S</span>patial <span style="color:blue">T</span>ransf<span style="color:blue">O</span>rmer for <span style="color:blue">R</span>adio <span style="color:blue">M</span>ap Estimation

This repository contains the implementation of STORM and the code to reproduce the numerical experiments of the paper ["Spatial Transformer for Radio Map Estimation"](https://arxiv.org/abs/2411.01211) by Pham Q. Viet and Daniel Romero. The paper was presented in the International Conference on Communications (ICC), 2025.



## Overview
- Python version: you may need Python 3.12 or later.

- The experiments (code) to train and evaluate STORM are in `experiments/transformer_experiments.py`.

- Training and testing data are available inside folder `data`. We will work with three datasets, referred to as USRP, Gradiant, and ray-tracing; for more information, see ["Radio Map Estimation: Empirical Validation and Analysis"](https://arxiv.org/pdf/2310.11036).



## Set up
After cloning the repository, do the following steps.
1. Install gsim:
    `git submodule`
    `git submodule update`
    `bash gsim/install.sh`

2. In `gsim_conf.py`, change `module_name = "experiments.example_experiments"` to `module_name = "experiments.transformer_experiments"`.

3. An experiment, e.g. `experiment_1000` can be run as follows: 
    `python run_experiment 1000`

4. This step is optional as it is to download the trained weights for the benchmark estimators. The following folders should be downloaded to `output/trained_estimators`:
    - USRP: [Updating link]()
    - Gradiant: [Updating link]()
    - Ray-tracing: [Updating link]()

## Radio map estimation
1. Create training datasets:
    - You can use Experiments 1000, 1005, or 1010 to create a training dataset for STORM.
    - Experiment 1015 creates a training set with ray-tracing data for other DNN benchmarks.

The obtained training datasets will be stored inside folder `output/datasets`.

2. To train STORM on USRP, Gradiant, and ray-tracing data, run Experiments 2000, 2005, and 2010, respectively.

3. Numerical results:
    - Run Experiment `3112` to obtain Figure 3.
    - Run Experiment `3105` to obtain Figure 4.
    - Run Experiment `3107` to obtain Figure 5.

## Active sensing
- Run Experiments `4000` and `4010` to obtain Figure 6.

Contact:
    - Viet Pham: viet.q.pham@uia.no
    - Daniel Romero: daniel.romero@uia.no
