from data_loader import *
from model import *
from model_bn import *

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 

import fastai as fai 
from fastai.basic_data import DataLoader

from fire import Fire
import matplotlib.pyplot as plt
import pickle

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

    db = fai.basic_data.DataBunch(train_dl, val_dl)

    input_size = train_dl.dataset.X.shape[1]
    output_size = train_dl.dataset.Y.shape[1]

    model_name = "model " + algorithm
    model= None

    if batch_norm:
        model_name += " batch_norm"
        model = Model_BN(input_size, output_size)
    else:
        model = Model(input_size, output_size)



    criterion = nn.MSELoss()
    
    l = fai.Learner(db, model, loss_func=criterion)

    if algorithm == 'SGD':
        l.opt_func = optim.SGD
        dl = {'train':train_dl, 'val': val_dl}
    
        model, losses = train_model(model, criterion, optimizer, dl, num_epochs)



if __name__=='__main__':
    Fire()