#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import environ
environ['train_device'] = 'cuda:2' # training device: 'cpu' or 'cuda:X'
environ['store_device'] = 'cuda:2' # Data storing device:  'cpu' or 'cuda:X'
environ['dataset_file'] = '/data/scratch/mmerouani/processed_datasets/dataset_batch4X_train_val_set.pkl' #training / validation set
environ['test_dataset_file'] = '/data/scratch/mmerouani/processed_datasets/dataset_batch4X_test_set.pkl' #test set
environ['benchmark_dataset_file']='/data/scratch/mmerouani/processed_datasets/dataset_Benchmark_batch10.pkl' #benchmarks set
#a copy of these datasets can be found in /data/commit/tiramisu/cost-model_datasets/processed_datasets/


from utils import *  # imports and defines cost-model utilities


# ### Data loading

# In[2]:


train_val_dataset, val_bl, val_indices, train_bl, train_indices = load_data(dataset_file, 0.2,max_batch_size=832)
test_dataset, test_bl, test_indices, _, _ = load_data(test_dataset_file, 1)


# ### Model definition

# In[3]:


input_size = 2534

model = None

model = Model_Recursive_LSTM_v2(input_size,drops=[0.112, 0.112, 0.112, 0.112])
model.to(train_device)

criterion = mape_criterion

optimizer = AdamW(model.parameters(),weight_decay=0.375e-2)



# ### Model training

# In[5]:


bl_dict={'train':train_bl, 'val':val_bl}
log_file = 'log_Recursive_LSTM_v2.txt'

losses, best_model = train_model(model, criterion, optimizer , max_lr=0.001, dataloader=bl_dict,
                                 num_epochs=800, logFile=log_file, log_every=1)


# ### Loading a pre-trained model

# In[6]:


model.load_state_dict(torch.load('Recursive_LSTM_v2_16.32.pkl',map_location=train_device))
model.to(train_device)
print()


# ### Basic results on the test and validation set

# In[8]:


val_df = get_results_df(train_val_dataset, val_bl, val_indices, model)
test_df = get_results_df(test_dataset, test_bl, test_indices, model)


# In[8]:


test_df.describe()


# In[9]:


val_df.describe()


# In[10]:


test_df[(test_df['exec_time']*test_df['target'])>5].describe()


# In[11]:


sample_size = 3200

fig = go.Figure()

fig.add_trace(go.Scatter(x=list(range(sample_size)), y=test_df.iloc[:sample_size].sort_values(by=['target'])['prediction'],
                    mode='markers', name='Predicted Speedup',marker=dict( size=2)))
fig.add_trace(go.Scatter(x=list(range(sample_size)), y=test_df.iloc[:sample_size].sort_values(by=['target'])['target'],
                    mode='lines', name='Measured Speedup' ))

fig.update_yaxes(type="log")
fig.show('png') # use fig.show() for interactive mode


# ### Basic results on the benchmark set

# In[11]:


benchmark_dataset, benchmark_bl, benchmark_indices, _, _ = load_data(benchmark_dataset_file, 1)


# In[12]:


benchmark_df = get_results_df(benchmark_dataset, benchmark_bl, benchmark_indices, model)


# In[13]:


benchmark_df.describe()


# In[ ]:
