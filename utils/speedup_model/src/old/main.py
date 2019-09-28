from data_loader import *
from model import *
from model_bn import *

from fastai.basic_data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np

import fastai as fai 

from fire import Fire
import matplotlib.pyplot as plt
import pickle
import seaborn as sns 


def train_dev_split(dataset, batch_size, num_workers, maxsize, split_factor=10, seed=42):
    indices = np.random.RandomState(seed=seed).permutation(maxsize)

    val_indices, train_indices = indices[:maxsize//split_factor], indices[maxsize//split_factor:]

    train_dl = DataLoader(DatasetFromHdf5(dataset, maxsize=len(train_indices)), 
                        batch_size=batch_size,
                        sampler=SubsetRandomSampler(train_indices),
                         num_workers=num_workers)

    val_dl = DataLoader(DatasetFromHdf5(dataset, maxsize=len(val_indices)), 
                        batch_size=batch_size, 
                        sampler=SubsetRandomSampler(val_indices),
                         num_workers=num_workers)


    return train_dl, val_dl




def main(batch_size=2048, num_epochs=400, 
            num_workers=8, algorithm='adam',
             maxsize=50000, new=True, dataset='data/speedup_dataset.h5', 
             batch_norm=False, filename='data/results.pkl',lr=0.001):
    
   
    train_dl, val_dl = train_dev_split(dataset, batch_size, num_workers, maxsize)

    input_size = train_dl.dataset.X.shape[1]
    output_size = train_dl.dataset.Y.shape[1]

    model_name = "model " + algorithm
    if batch_norm:
        model_name += " batch_norm"

    model = load_model(input_size, output_size, model_name=model_name,filename=filename, batch_norm=batch_norm)

    criterion = nn.MSELoss()
    optimizer= optim.Adam(model.parameters(), lr=0.01)

    if algorithm != 'adam':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    dl = {'train':train_dl, 'val': val_dl}
 
    model, losses = train_model(model, criterion, optimizer, dl, num_epochs)

    #pickle results
    save_results(model_name, model, losses)
    
    #plot_results(losses)




if __name__ == '__main__':
    Fire()







