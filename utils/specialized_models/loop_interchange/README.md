## Overview

This project, entitled: `A Deep Learning Model for Loop Interchange`, presents a Deep Learning based model dedicated to predict the best loop interchange instance for a Tiramisu program given as input. Loop Interchange is a code optimization transformation that aims to permute two loops in the program in order to gain better locality, parallelism and overall better execution.

Through the scripts provided in this repository, as well as the different datasets, it is possible to reproduce the model by retraining it using the provided datasets, by performing the paper’s tests well on both the test set and the benchmarks.

This tool can be considered as a module for compilers’ auto-schedulers, from which we state: Tiramisu, Halide & Pluto.

## Requirements

### Hardware
* This model can be run on either CPU or NVIDIA GPU. It is recommended to use a GPU for faster execution.
* No specialized hardware is needed to run any of the scripts.

### Software
* Linux distributions are recommended, but other operating system are Ok.
* Python 3.8.
* Pip (https://pip.pypa.io/en/stable/installation/).
* Libraries: numpy 1.20.1, pandas 1.2.4, torch 1.9.1+cu111, tqdm 4.59.0, matplotlib 3.3.4 (the version numbers for these libraries are for reference in case of compatibility issues only).

For instance, if you have an Nvidia GPU, you can install the previous dependencies using pip as follows:

	pip install numpy pandas tqdm matplotlib
	pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

## Getting Started
Cloning this project can be done using the following instructions:

	git clone https://github.com/Tiramisu-Compiler/tiramisu.git
	cd tiramisu/utils/specialized_models/loop_interchange/

Dataset: https://drive.google.com/drive/folders/1inZjtVcQtU5eyMF7D80gqO5NS1ToiQ6D.

First create a folder called 'datasets' in 

	tiramisu/utils/specialized_models/loop_interchange/datasets

Then, the content of the `datasets` folder in the drive (5 files) should be downloaded and copied to the `datasets` folder created previously.

To change the execution from GPU to CPU and vice-versa, the lines 4 and 5 from the training and test scripts (presented later) should be changed ('cpu' or 'cuda:X' where X is the number of the target node).

## Repository content

### Scripts
The model can be accessed and tested through 3 scripts:

1) Model_training.py: It loads the data, builds the model and trains it. No input is required, provided that all dataset files are in the indicated folder (datasets) located in the same location as this script. By default, a 5-best model is going to be created. However, it can be changed in the Utils file. The execution of this script outputs the model in a pickle format (with a default name that can be changed). Moreover, It outputs throughout epochs the loss values that the model is getting in both the training and the validation set. Finally, the accuracy of the resulting model on both sets is computed by the end. To run it, use the command: `python model_training.py`.

2) Model_tests.py: This script aims to reproduce the results shown in the paper: Both the accuracy of the model, and its search performances once integrated in Tiramisu’s auto-scheduler. It uses the default name of the pickle model (produced by the precedent script, can be changed) to run these tests. It outputs: the results of the tests (on both the synthetic test set and the benchmarks): the accuracy and the search performance. Moreover, it outputs a text file presenting the results for the search performance.  To run it, use the command: `python model_testing.py`.

3) Utils.py: Helper functions for both the training and the different tests. It is going to be called internally by the other scripts.


### Datasets:

In the datasets folder (after downloading the files from the drive link), 5 different pickle/json files can be found:

* 2 Training and 2 Test sets: For compatibility reasons, and in order to reuse the old Tiramisu programs datasets, our model is capable of inputting both the old and the new Tiramisu datasets, as there is slight change in their structure. Which explains the existence of 4 files instead of 2.
* The benchmarks file: It contains different programs, of different sizes, used in different domains: linear algebra (matmul, for matrix multiplication; and jacobi (1D and 2D) and seidel2d, for solving linear systems with the Jacobi and the Gauss-Seidel method respectively), image processing (blur for blurring images) and simulation domain (heat2d and heat3d, for heat propagation simulations in 2D and 3D spaces, respectively).

## Scripts Execution Expectations: 

* Approximated time to install the needed environment: a few minutes
* Approximated time to run/use the scripts and (if wanted) to reproduce the results.
1) The training script should take approximately 110 minutes if run on a GPU. The number of epochs (NB_EPOCHS) can be changed to reduce this execution time. A trained model is provided in this repository, making this step skippable.
2) The test script should take a few minutes at most (mainly to read the data). The results are shown in the console, with the search performance tables exported in a csv files as well.

