#!/usr/bin/env python
# coding: utf-8

# Training the cost model with a ranking loss

# In[1]:


from os import environ
environ['train_device'] = 'cuda:1' # training device: 'cpu' or 'cuda:X'
environ['store_device'] = 'cuda:1' # Data storing device:  'cpu' or 'cuda:X'
environ['dataset_file'] = '/data/scratch/mmerouani/processed_datasets/dataset_batch4X_train_val_set.pkl' #training / validation set
environ['test_dataset_file'] = '/data/scratch/mmerouani/processed_datasets/dataset_batch4X_test_set.pkl' #test set
environ['benchmark_dataset_file']='/data/scratch/mmerouani/processed_datasets/dataset_Benchmark_batch10.pkl' #benchmarks set
#a copy of these datasets can be found in /data/commit/tiramisu/cost-model_datasets/processed_datasets/

from utils import *  # imports and defines cost-model utilities


# ### Data loading

# In[2]:


train_val_dataset, val_bl, val_indices, train_bl, train_indices = load_data_meta_batches(dataset_file, 0.2, max_batch_size=832)
test_dataset, test_bl, test_indices, _, _ = load_data_meta_batches(test_dataset_file, 1)


# ### Model definition

# In[3]:


input_size = 2534

criterion = ndcgLoss2PP_meta_batches # Using nDCG-Loss-2++

model = None
model = Model_Recursive_LSTM_v2_ranking(input_size, drops=[0.112, 0.112, 0.112, 0.112])
model.to(train_device)

optimizer = AdamW(model.parameters(),weight_decay=0.375e-2)



# ### Model training

# In[4]:


bl_dict={'train':train_bl, 'val':val_bl}
log_file = 'log_Recursive_LSTM_v2_ranking.txt'

losses, best_model = train_model_meta_batches(model, criterion, optimizer , max_lr=0.002, dataloader=bl_dict, num_epochs=800,
                                 logFile=log_file, log_every=1)


# ### Loading a pre-trained model

# In[5]:


model.load_state_dict(torch.load('Recursive_LSTM_v2_ndcgLoss2PP.pkl',map_location=train_device))
model.to(train_device)
print()


# ### Basic results on the test set
#

# In[6]:


test_df, test_df_rank_scores = get_results_df_meta_batches(test_dataset, test_bl, test_indices, model)


# In[8]:


test_df


# In[9]:


test_df_rank_scores.describe()


# In[12]:


cf_matrix = confusion_matrix(test_df['real_rank'].astype('int32'), test_df['predicted_rank'].astype('int32'))

fig = px.imshow(cf_matrix,
                labels=dict(x="Real rank", y="Predicted rank", color="Number of Schedules (out of 276k)" ),
                x=[str(i) for i in range(1,34)],
                y=[str(i) for i in range(1,34)]
               )
fig.update_xaxes(side="top")
fig.show('png') #use fig.show() for interactive mode


# ### Basic results on the benchmark set

# In[14]:


benchmark_dataset, benchmark_bl, benchmark_indices, _, _ = load_data_meta_batches(benchmark_dataset_file, 1)


# In[15]:


benchmark_df, benchmark_df_rank_scores = get_results_df_meta_batches(benchmark_dataset, benchmark_bl, benchmark_indices, model)


# In[16]:


benchmark_df_rank_scores.describe()


# In[11]:


pass


# In[ ]:
