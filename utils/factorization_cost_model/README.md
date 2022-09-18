# Factorization cost model
Train a recursive lstm architecture on Tiramisu schedules.


## Installation  
Here are the steps of the installtion:  
1. Install pytorch. Note that the repo was tested only with the version 1.10.2 of pytorch.
2. Install the required packages using the command:  
```python
pip install -r requirements.txt
```  

## Configuring the repository
All of the main scripts use Hydra for configuration management. To configure the repository, copy the `conf/conf.yml.example` file to `conf/conf.yml` as follows:  
```bash
# After navigating to the root directory of this repo
cp conf/conf.yml.example conf/conf.yml
```
Modify the configuration according to your personal need.
While using one of the following script files, you can override any of the configurations in the conf file. For example to modify the batch size to 512 for training, use the following command. The parameter should be included with its section name.  
```
python generate_dataset.py data_generation.batch_size=512
```

## Generating the dataset
To generate the dataset, run the python script `generate_dataset.py` (after configuring the repository):  
```bash
python generate_dataset.py
```

## Training the model
To run the training, run the bash script `run.sh` with the GPU number to run the training on (after configuring the repository and generatoing the dataset):  
```bash
bash run.sh [num] # replace [num] with a GPU number
```

## Using wandb for visualization
The repository allows to use Weights and Biases for visualization. To enable it, set the `use_wandb` parameter to `True`, after logging into wandb from command line. The project name should be specified. This name does not have to already exist in wandb. During training, the progress can be found on wandb 

## Evaluation of the trained model
To evaluate the trained model, run the python script `evaluate_model.py` (after configuring the repository and generatoing the dataset):  
```bash
python evaluate_model.py
```