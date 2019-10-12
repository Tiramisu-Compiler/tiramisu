#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import environ

environ['optimizer'] = 'Adam'
environ['num_workers']= '8'
environ['batch_size']= str(8192)
environ['n_epochs']= '1000'
environ['batch_norm']= 'True'
environ['loss_func']='MAPE'
environ['layers'] = '600 350 200 180'
environ['dropouts'] = '0.3 '*4
environ['log'] = 'False'
environ['weight_decay'] = '0.01'
environ['cuda_device'] ='cuda:0'
environ['dataset'] = 'speedup_dataset.pkl'


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [40, 30]
import fastai as fai
from fastai.basic_data import DataLoader
import numpy as np
from torch.utils.data import SubsetRandomSampler
import torch 
import pandas as pd
import seaborn as sns
from os import environ
from torch import optim
from bisect import bisect_left
import time
from tqdm import tqdm
from fastai.callbacks import EarlyStoppingCallback
from functools import partial
from fastai import train
import random

from src.data.dataset import *
from src.model.model_bn import *
optimizer = environ.get('optimizer', 'Adam')
num_workers= int(environ.get('num_workers', '8'))
batch_size=int(environ.get('batch_size', '2048'))
n_epochs=int(environ.get('n_epochs', '500'))
batch_norm = environ.get('batch_norm', 'True') == 'True'
dataset= environ.get('dataset', 'data/speedup_dataset2.pkl')
loss_func = environ.get('loss_func', 'MSE')
log = environ.get('log', 'True') == 'True'
wd = float(environ.get('weight_decay', '0.01'))
cuda_device = environ.get('cuda_device', 'cuda:0')

layers_sizes = list(map(int, environ.get('layers', '300 200 120 80 30').split()))
drops = list(map(float, environ.get('dropouts', '0.2 0.2 0.1 0.1 0.1').split()))
device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
print(device)
def train_model(model, criterion, optimizer, dataloader, num_epochs=100):
    since = time.time()
    
    losses = []
    train_loss = 0
    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  
            else:
                model.eval()

            running_loss = 0.0
           
            # Iterate over data.
            for inputs, labels in tqdm(dataloader[phase], total=len(dataloader[phase])):       
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  
                    assert outputs.shape == labels.shape

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                       # print(loss.item())
                        
            
                # statistics
                running_loss += loss.item()     
                
                #running_corrects += torch.sum((outputs.data - labels.data) < e)/inputs.shape[0]

            epoch_loss = running_loss / len(dataloader[phase])
            
            print('{} Loss: {:.4f}'.format(
               phase, epoch_loss))

            
            if phase == 'val':
                losses.append((train_loss, epoch_loss))
            else:
                train_loss = epoch_loss

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    
    return losses



def get_results_df(dl, model, log=False):
    df = pd.DataFrame()

    indices = dl.sampler.indices
    
    inputs, targets = dl.dataset[indices]
    names = [dl.dataset.programs[dl.dataset.restricted_program_indexes[i]].name for i in indices]
    sched_names = [dl.dataset.schedules[i].name for i in indices]
    
    inputs = torch.Tensor(inputs)
    
    model.eval()
    preds = model(inputs.to(device))

    interchange, tile, unroll = zip(*[dl.dataset.schedules[index].binary_repr for index in indices])

    preds = preds.cpu().detach().numpy().reshape((-1,))
    targets = targets.reshape((-1,))
    
    if log:
        preds = np.exp(preds*dl.dataset.std + dl.dataset.mean)
        targets = np.exp(targets*dl.dataset.std + dl.dataset.mean)

    df['index'] = indices
    df['name'] = names
    df['sched_name'] = sched_names
    df['prediction'] = preds
    df['target'] = targets
    df['abs_diff'] = np.abs(preds - targets)
    df['APE'] = np.abs(df.target - df.prediction)/df.target * 100
    df['SMAPE'] = 100*np.abs(df.target - df.prediction)/((np.abs(df.target) + np.abs(df.prediction))/2)
    
    df['interchange'] = interchange
    df['tile'] = tile
    df['unroll'] = unroll
    
    return df

def train_dev_split(dataset, batch_size, num_workers, log=False, seed=42):
    
    test_size = validation_size = 2000
    ds = DatasetFromPkl(dataset, maxsize=None, log=log)
    
    indices = range(len(ds))
    test_indices, val_indices, train_indices = indices[:test_size], \
                                                indices[test_size:test_size+validation_size], \
                                               indices[test_size+validation_size:]

    train_dl = DataLoader(ds, batch_size=batch_size,
                        sampler=SubsetRandomSampler(train_indices),
                         num_workers=num_workers)

    val_dl = DataLoader(ds, batch_size=batch_size, 
                        sampler=SubsetRandomSampler(val_indices),
                         num_workers=num_workers)
    
    test_dl = DataLoader(ds, batch_size=batch_size, 
                         sampler=SubsetRandomSampler(test_indices),
                         num_workers=num_workers)
    
    
    return train_dl, val_dl, test_dl

def train_dev_split2(dataset1,dataset2, batch_size, num_workers, log=False, seed=42):
    
    test_size = validation_size = 10000
    ds = DatasetFromPkl2(dataset1,dataset2, maxsize=None, log=log)
    
    indices = list(range(len(ds)))
    
    random.shuffle(indices) 
    test_indices, val_indices, train_indices = indices[:test_size], \
                                                indices[test_size:test_size+validation_size], \
                                               indices[test_size+validation_size:]

    train_dl = DataLoader(ds, batch_size=batch_size,
                        sampler=SubsetRandomSampler(train_indices),
                         num_workers=num_workers)

    val_dl = DataLoader(ds, batch_size=batch_size, 
                        sampler=SubsetRandomSampler(val_indices),
                         num_workers=num_workers)
    
    test_dl = DataLoader(ds, batch_size=batch_size, 
                         sampler=SubsetRandomSampler(test_indices),
                         num_workers=num_workers)
    
    
    return train_dl, val_dl, test_dl
def mape_criterion(inputs, targets):
    eps = 1e-5
    return 100*torch.mean(torch.abs(targets - inputs)/(targets+eps))


def smape_criterion(inputs, targets):
    return 100*torch.mean(torch.abs(targets - inputs)/((torch.abs(targets)+torch.abs(inputs))/2))


def rmse_criterion(inputs, targets):
    return torch.sqrt(nn.MSELoss()(inputs, targets))
    
    

def get_data_with_names(dl):
    dataset = dl.dataset
    names, X, Y = zip(*[(dataset.get_sched_name(index), *dataset[index]) for index in dl.sampler.indices])
    
    return names, X, Y

def get_schedule_data(dl, schedule):
    dataset = dl.dataset
    
    
    indices = [index for index in dl.sampler.indices 
                    if np.all(np.array(dataset.schedules[index].binary_repr) + np.array(schedule) != 1)]
    
    return indices
    
    
def get_data_with_prog_names(dl):
    dataset = dl.dataset
    names, X, Y = zip(*[(dataset.get_prog_name(index), *dataset[index]) for index in dl.sampler.indices])
    
    return names, X, Y

def joint_plot(df, title, val_range=list(range(-1, 15))):
    ax = sns.jointplot('target', 'prediction', df, ).ax_joint
    plt.suptitle(title)
    _ = ax.set_xticks(val_range)
    _ = ax.set_yticks(val_range)
    _ = ax.plot(val_range, val_range, ':k')
    
class NameGetter(object):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.get_prog_name(index)
    
def get_program_data(dl, prog_name):
    dataset = dl.dataset
    name_g = NameGetter(dataset)
    
    index1 = bisect_left(name_g, prog_name, 0)
    index2 = bisect_right(name_g, prog_name, index1)
    
    X, Y = zip(*[dataset[index] for index in range(index1, index2)])
   
    return torch.Tensor(X), torch.Tensor(Y)

def joint_plot_one_program(dl, prog_name, model):
    model.eval()
    
    X, Y = get_program_data(dl, prog_name)
    
    Y_hat = model(X.to(device))
    df = pd.DataFrame()
    df['prediction'] = np.array(Y_hat.view(-1,))
    df['target'] = np.array(Y)
    
    joint_plot(df, prog_name)
    
def joint_plot_one_schedule(dl, schedule, model, log=False):
    indices = get_schedule_data(dl, schedule)
    
    X, Y = dl.dataset[indices]
    X, Y = torch.Tensor(X), torch.Tensor(Y)
    
    Y_hat = model(X.to(device))

    
    df = pd.DataFrame()
    df['prediction'] = np.array(Y_hat.view(-1,))
    df['target'] = np.array(Y)
    
    joint_plot(df, schedule)
    



torch.cuda.device_count()


# In[ ]:


# print("loading data")
# train_dl, val_dl, test_dl = train_dev_split(dataset, batch_size, num_workers, log=log)

# db = fai.basic_data.DataBunch(train_dl, val_dl, test_dl, device=device)
# print("data loaded")


# In[2]:


print("loading data")
train_dl, val_dl, test_dl = train_dev_split2('/data/scratch/mmerouani/data/henni-datasets/speedup_dataset3.pkl',dataset, batch_size, num_workers, log=log)

db = fai.basic_data.DataBunch(train_dl, val_dl, test_dl, device=device)
print("data loaded")


# In[13]:


# # ds=np.concatenate(ds1, ds2)
# # ds=np.roll(ds, round(len(ds2)*val_test_size/len(ds)),axis=0)
# # tensor_x = torch.from_numpy(ds[:][0])
# # tensor_y = torch.from_numpy(ds[:][1])

# # ds = torch.utils.data.TensorDataset(tensor_x,tensor_y)
# val_test_size = test_size
# ds1ratio=round(len(ds1)*val_test_size/len(ds))
# ds2ratio=round(len(ds2)*val_test_size/len(ds))
# ds=torch.utils.data.ConcatDataset([ds1,ds2])
# indices = list(range(len(ds)) )
# test_indices= indices[:ds1ratio]+ indices[len(ds1):len(ds1)+ds2ratio] 
# val_indices= indices[ds1ratio:2*ds1ratio]+indices[len(ds1)+ds2ratio:len(ds1)+2*ds2ratio] 
# train_indices= indices[2*ds1ratio:len(ds1)]+indices[len(ds1)+2*ds2ratio:len(ds)] 
# # test_indices, val_indices, train_indices = indices[:test_size], \
# #                                             indices[test_size:test_size+validation_size], \
# #                                            indices[test_size+validation_size:]

# train_dl = DataLoader(ds, batch_size=batch_size,
#                     sampler=SubsetRandomSampler(train_indices),
#                      num_workers=num_workers)

# val_dl = DataLoader(ds, batch_size=batch_size, 
#                     sampler=SubsetRandomSampler(val_indices),
#                      num_workers=num_workers)

# test_dl = DataLoader(ds, batch_size=batch_size, 
#                      sampler=SubsetRandomSampler(test_indices),
#                      num_workers=num_workers)



# In[15]:





# In[ ]:





# In[3]:



print(len(train_dl.dataset))
#db.dl()
print(batch_size)
print(db.one_batch())


# In[4]:


input_size = train_dl.dataset.X.shape[1]
output_size = train_dl.dataset.Y.shape[1]
print(train_dl.dataset.X.shape)
print(train_dl.dataset.Y)

model = None 

if batch_norm:
    model = Model_BN(input_size, output_size, hidden_sizes=layers_sizes, drops=drops)
else:
    model = Model(input_size, output_size)

model = nn.DataParallel(model)
model.to(device)

if loss_func == 'MSE':
    criterion = nn.MSELoss()
else:
    criterion = mape_criterion

l = fai.basic_train.Learner(db, model, loss_func=criterion, metrics=[mape_criterion, rmse_criterion],
               callback_fns=[partial(EarlyStoppingCallback, mode='min', 
                                        monitor='mape_criterion', min_delta=0, patience=50)])

if optimizer == 'SGD':
    l.opt_func = optim.SGD 
    


# In[5]:
l = l.load(f"speedup_{optimizer}_batch_norm_{batch_norm}_{loss_func}_nlayers_{len(layers_sizes)}_log_{log}_merged")

# l.lr_find()


# In[6]:


# l.recorder.plot()


# In[7]:


lr = 1e-03


# In[ ]:


l.fit_one_cycle(1000, lr)


# In[7]:


l.recorder.plot_losses()


# In[10]:


l.save(f"speedup_{optimizer}_batch_norm_{batch_norm}_{loss_func}_nlayers_{len(layers_sizes)}_log_{log}_merged")


# In[93]:


val_df = get_results_df(val_dl, l.model)
train_df = get_results_df(train_dl, l.model)


# In[87]:


indices = train_dl.sampler.indices
inputs, targets = train_dl.dataset[indices]
#l.model.eval()
l.model(torch.Tensor(inputs))
#for name, param in l.model.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)


# In[88]:


print(train_dl.dataset[:])


# In[96]:


df = train_df


# In[80]:


indices = train_dl.sampler.indices
print(indices)


# In[99]:


val_df.describe()


# In[ ]:





# In[97]:


df[df['target']>0].describe()
    


# In[46]:


df[:][['prediction','target', 'abs_diff','APE']].describe()


# In[47]:


df[:][['prediction','target', 'abs_diff','APE']].mean()


# In[28]:


df[(df.interchange==1) & (df.unroll == 1) & (df.tile == 1)][['prediction','target', 'abs_diff','APE']].describe()


# In[16]:


df1 = df[(df.interchange==0) & (df.unroll == 0) & (df.tile == 0)]
#joint_plot(df1, f"Validation dataset, {loss_func} loss")
df2 = df
joint_plot(df2, f"Validation dataset, {loss_func} loss")


# In[17]:


df_ = df.sort_values(by=["APE"])

df_['x'] = range(len(df_))


# In[18]:


plt.plot('x', 'APE', 'bo', data=df_)


plt.xlabel('scheduled program')
plt.ylabel('APE')
plt.legend()


# In[19]:


plt.plot('x', 'APE', 'go', data=df_)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[85]:





# In[ ]:




