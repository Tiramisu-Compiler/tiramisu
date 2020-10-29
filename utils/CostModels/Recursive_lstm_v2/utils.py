from os import environ
from pprint import pprint
import pickle
import numpy as np
import numpy as np
import torch 
import pandas as pd
import seaborn as sns
from torch import optim
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import random
import plotly.graph_objects as go
from src.torch141.lr_scheduler import *
from src.torch141.adamw import *
# from torch_lr_finder import LRFinder

train_device= torch.device(environ.get('train_device'))
store_device= torch.device(environ.get('store_device'))
dataset_file= environ.get('dataset_file')
test_dataset_file = environ.get('test_dataset_file')
benchmark_dataset_file=environ.get('benchmark_dataset_file')

def mape_criterion(inputs, targets):
    eps = 1e-5
    return 100*torch.mean(torch.abs(targets - inputs)/(targets+eps))
# def sum_ape_criterion(inputs, targets):
#     eps = 1e-5
#     return 100*(torch.abs(targets - inputs)/(targets+eps))

class LargeAccessMatices(Exception):
    pass
def get_representation(program_json, schedule_json):
    max_dims= 7
    max_accesses = 21 # TODO: check if 10 is enough
    program_representation = []
    indices_dict = dict()
    computations_dict = program_json['computations']
    ordered_comp_list = sorted(list(computations_dict.keys()), key = lambda x: computations_dict[x]['absolute_order'])
    
    for index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        comp_representation = []
        #         Is this computation a reduction 
        comp_representation.append(+comp_dict['comp_is_reduction'])


#         iterators representation + tiling and interchage
        iterators_repr = []
        for iterator_name in comp_dict['iterators']:
            
            iterator_dict = program_json['iterators'][iterator_name]
            iterators_repr.append(iterator_dict['upper_bound']) 
#             iterators_repr.append(iterator_dict['lower_bound'])
            # unfuse schedule replacing the lower bound for compability issue, this enables the use transfer learning from an older model 
            parent_iterator = program_json['iterators'][iterator_name]['parent_iterator']
            if parent_iterator in schedule_json['unfuse_iterators']:
                iterators_repr.append(1) #unfused true
            else:
                iterators_repr.append(0) #unfused false
            
            if iterator_name in schedule_json[comp_name]['interchange_dims']:
                iterators_repr.append(1) #interchanged true
            else:
                iterators_repr.append(0) #interchanged false
            
            if (schedule_json[comp_name]['tiling']!={}):
                if iterator_name in schedule_json[comp_name]['tiling']['tiling_dims']:
                    iterators_repr.append(1) #tiled: true
                    tile_factor_index = schedule_json[comp_name]['tiling']['tiling_dims'].index(iterator_name)
                    iterators_repr.append(int(schedule_json[comp_name]['tiling']['tiling_factors'][tile_factor_index])) #tile factor
                else:
                    iterators_repr.append(0) #tiled: false
                    iterators_repr.append(0) #tile factor 0
            else: #tiling = None
                iterators_repr.append(0) #tiled: false
                iterators_repr.append(0) #tile factor 0    
            # is this dimension saved (this dimension does not disapear aftre reduction)
            iterators_repr.append(+(iterator_name in comp_dict['real_dimensions']))
                    
        iterator_repr_size = int(len(iterators_repr)/len(comp_dict['iterators']))
        iterators_repr.extend([0]*iterator_repr_size*(max_dims-len(comp_dict['iterators']))) # adding iterators padding 

        comp_representation.extend(iterators_repr) #adding the iterators representation    

#         accesses representation
        accesses_repr=[]
        for access_dict in comp_dict['accesses']:
            access_matrix = access_dict['access_matrix']
            access_matrix = np.array(access_matrix)
            padded_access_matrix = np.zeros((max_dims, max_dims + 1))
            padded_access_matrix[:access_matrix.shape[0],:access_matrix.shape[1]-1] = access_matrix[:,:-1] #adding padding to the access matrix
            padded_access_matrix[:access_matrix.shape[0],-1] = access_matrix[:,-1] #adding padding to the access matrix
            access_repr = [access_dict['buffer_id']] + padded_access_matrix.flatten().tolist() # input_id + flattened access matrix 
            # is this access a reduction (the computation is accesing itself)
            access_repr.append(+access_dict['access_is_reduction'])
            accesses_repr.extend(access_repr)

        #access_repr_len = max_dims*(max_dims + 1)
        access_repr_len = max_dims*(max_dims + 1) + 1 +1 #+1 for input id, +1 for is_access_reduction
        accesses_repr.extend([0]*access_repr_len*(max_accesses-len(comp_dict['accesses']))) #adding accesses padding
    
        comp_representation.extend(accesses_repr) #adding access representation

#         operation histogram
        comp_representation.append(comp_dict['number_of_additions'])
        comp_representation.append(comp_dict['number_of_subtraction'])
        comp_representation.append(comp_dict['number_of_multiplication'])
        comp_representation.append(comp_dict['number_of_division'])

        
#         unrolling representation
        if (schedule_json[comp_name]['unrolling_factor']!=None):
            comp_representation.append(1) #unrolled True
            comp_representation.append(int(schedule_json[comp_name]['unrolling_factor'])) #unroll factor
        else:
            comp_representation.append(0) #unrolled false
            comp_representation.append(0) #unroll factor 0

        # adding log(x+1) of the representation
        log_rep = list(np.log1p(comp_representation))
        comp_representation.extend(log_rep)
        
        program_representation.append(comp_representation)
        indices_dict[comp_name] = index
    
    # transforming the schedule_json inorder to have loops as key instead of computations, this dict helps building the loop vectors
    loop_schedules_dict = dict()
    for loop_name in program_json['iterators']:
        loop_schedules_dict[loop_name]=dict()
        loop_schedules_dict[loop_name]['interchanged']=False
        loop_schedules_dict[loop_name]['interchanged_with']=None
        loop_schedules_dict[loop_name]['tiled']=False
        loop_schedules_dict[loop_name]['tile_depth']=None
        loop_schedules_dict[loop_name]['tiled_dims']=None
        loop_schedules_dict[loop_name]['tile_factor']=None
        loop_schedules_dict[loop_name]['unrolled']=False
        loop_schedules_dict[loop_name]['unroll_factor']=None
        loop_schedules_dict[loop_name]['unroll_comp']=None
        loop_schedules_dict[loop_name]['unfused']=False     
    for comp_name in schedule_json:
        if not comp_name.startswith('comp'): 
            continue # skip the non computation keys
        if schedule_json[comp_name]['interchange_dims']!=[]:
            interchanged_loop1=schedule_json[comp_name]['interchange_dims'][0]
            interchanged_loop2=schedule_json[comp_name]['interchange_dims'][1]
            loop_schedules_dict[interchanged_loop1]['interchanged']=True
            loop_schedules_dict[interchanged_loop1]['interchanged_with']=interchanged_loop2
            loop_schedules_dict[interchanged_loop2]['interchanged']=True
            loop_schedules_dict[interchanged_loop2]['interchanged_with']=interchanged_loop1
        if schedule_json[comp_name]['tiling']!={}:
            for tiled_loop_index,tiled_loop in enumerate(schedule_json[comp_name]['tiling']['tiling_dims']):
                loop_schedules_dict[tiled_loop]['tiled']=True
                loop_schedules_dict[tiled_loop]['tile_depth']=schedule_json[comp_name]['tiling']['tiling_depth']
                loop_schedules_dict[tiled_loop]['tiled_dims']=schedule_json[comp_name]['tiling']['tiling_dims']
                loop_schedules_dict[tiled_loop]['tile_factor']=int(schedule_json[comp_name]['tiling']['tiling_factors'][tiled_loop_index])
        if schedule_json[comp_name]['unrolling_factor']!=None:
            comp_innermost_loop=computations_dict[comp_name]['iterators'][-1] 
            tiling_dims = [] if schedule_json[comp_name]['tiling']=={} else schedule_json[comp_name]['tiling']['tiling_dims']
            interchange_dims =schedule_json[comp_name]['interchange_dims']
            if (not ((comp_innermost_loop in tiling_dims)or(comp_innermost_loop in interchange_dims))):#unrolling always applied to innermost loop, if tilling or interchange is applied to innermost, unroll is applied to the resulting loop instead of the orginal, hence we don't represent it
                loop_schedules_dict[comp_innermost_loop]['unrolled']=True
                loop_schedules_dict[comp_innermost_loop]['unroll_factor']=int(schedule_json[comp_name]['unrolling_factor'])
                loop_schedules_dict[comp_innermost_loop]['unroll_comp']=comp_name
    for unfuse_parent in schedule_json['unfuse_iterators'] :
        for unfused_loop in program_json['iterators'][unfuse_parent]['child_iterators']:
            loop_schedules_dict[unfused_loop]['unfused']=True
    
    # collect the set of iterators that are used for computation (to eleminate those that are only used for inputs)
    real_loops = set()
    for comp_name in computations_dict:
        real_loops.update(computations_dict[comp_name]['iterators'])
        
    #building loop tensor
    loops_representation_list = []
    loops_indices_dict = dict()
    loop_index=0
    for loop_name in program_json['iterators']:
        if not (loop_name in real_loops): # this removes the iterators that are only used for decraling inputs
            continue
        loop_representation=[]
        loop_dict = program_json['iterators'][loop_name]
        # upper and lower bound
        loop_representation.append(loop_dict['upper_bound'])
        loop_representation.append(loop_dict['lower_bound'])
        if loop_schedules_dict[loop_name]['unfused']:
            loop_representation.append(1) #unfused True
        else:
            loop_representation.append(0) #unfused False
        if loop_schedules_dict[loop_name]['interchanged']:
            loop_representation.append(1) #interchanged True
        else:
            loop_representation.append(0) #interchanged False            
        if loop_schedules_dict[loop_name]['tiled']:
            loop_representation.append(1) #tiled True
            loop_representation.append(loop_schedules_dict[loop_name]['tile_factor']) #tile factor
        else:
            loop_representation.append(0) #tiled False
            loop_representation.append(0) #tile factor 0
        # TODO: check if unroll representation should be moved to comp vector instead of loop vector
        if loop_schedules_dict[loop_name]['unrolled']:
            loop_representation.append(1) #unrolled True
            loop_representation.append(loop_schedules_dict[loop_name]['unroll_factor']) #unroll factor
        else:
            loop_representation.append(0) #unrolled False
            loop_representation.append(0) #unroll factor 0
        # adding log(x+1) of the loop representation
        loop_log_rep = list(np.log1p(loop_representation))
        loop_representation.extend(loop_log_rep)
        loops_representation_list.append(loop_representation)    
        loops_indices_dict[loop_name]=loop_index
        loop_index+=1
            
     
    def update_tree_atributes(node):     
        node['loop_index'] = torch.tensor(loops_indices_dict[node['loop_name'][:3]]).to(train_device)
        if node['computations_list']!=[]:
            node['computations_indices'] = torch.tensor([indices_dict[comp_name] for comp_name in node['computations_list']]).to(train_device)
            node['has_comps'] = True
        else:
            node['has_comps'] = False
        for child_node in node['child_list']:
            update_tree_atributes(child_node)
        return node
    
    tree_annotation = copy.deepcopy(schedule_json['tree_structure']) #to avoid altering the original tree from the json
    prog_tree = update_tree_atributes(tree_annotation) 
    
    loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_representation_list),0)#.to(device)
    computations_tensor = torch.unsqueeze(torch.FloatTensor(program_representation),0)#.to(device)     

    return prog_tree, computations_tensor, loops_tensor


#################################################

def get_tree_footprint(tree):
    footprint='<BL'+str(int(tree['loop_index']))
    if tree['has_comps']:
        footprint+='['
        for idx in tree['computations_indices']:
            footprint+='CI'+str(int(idx))
        footprint+=']'
    for child in tree['child_list']:
        footprint+= get_tree_footprint(child)
    footprint+='EL'+str(int(tree['loop_index']))+'>'
    return footprint

            

class Dataset():
    def __init__(self, dataset_filename, max_batch_size, filter_func=None, transform_func=None):
        super().__init__()
        
        self.dataset_name=dataset_filename
        f = open(dataset_filename, 'rb')
        self.programs_dict=pickle.load(f)
        f.close()
        
        self.X = []
        self.Y = []
        self.batched_program_names = []
        self.batched_schedule_names = []
        self.batched_exec_time = []
        self.nb_nan=0
        self.nb_long_access=0
        self.batches_dict=dict()
        
        if (filter_func==None):
            filter_func = lambda x : True
        if (transform_func==None):
            transform_func = lambda x : x
          
        for function_name in tqdm(self.programs_dict):
            program_json = self.programs_dict[function_name]['json']
            if (self.programs_dict[function_name]['schedules'][function_name+'_no_schedule']['exec_time']<0): #if less than x ms
                continue
            for schedule_name in self.programs_dict[function_name]['schedules']:
                if ((np.isnan(self.programs_dict[function_name]['schedules'][schedule_name]['speedup']))
                     or(self.programs_dict[function_name]['schedules'][schedule_name]['speedup']==0)): #Nan value means the schedule didn't run, zero values means exec time<1 micro-second, skip them
                    self.nb_nan+=1
                    continue
                if (not filter_func(self.programs_dict[function_name]['schedules'][schedule_name])):
                    continue
                schedule_json = self.programs_dict[function_name]['schedules'][schedule_name]['json']
                try:
                    tree, comps_tensor, loops_tensor = get_representation(program_json, schedule_json)
                except LargeAccessMatices:
                    self.nb_long_access +=1
                    continue
                tree_footprint=get_tree_footprint(tree)
                self.batches_dict[tree_footprint] = self.batches_dict.get(tree_footprint,{'tree':tree,'comps_tensor_list':[],'loops_tensor_list':[],'program_names_list':[],'sched_names_list':[],'speedups_list':[],'exec_time_list':[]})
#                 self.batches_dict[tree_footprint]['tree'].append(tree)
                self.batches_dict[tree_footprint]['comps_tensor_list'].append(comps_tensor)
                self.batches_dict[tree_footprint]['loops_tensor_list'].append(loops_tensor)
                self.batches_dict[tree_footprint]['sched_names_list'].append(schedule_name)
                self.batches_dict[tree_footprint]['program_names_list'].append(function_name)
                self.batches_dict[tree_footprint]['speedups_list'].append(self.programs_dict[function_name]['schedules'][schedule_name]['speedup'])
                self.batches_dict[tree_footprint]['exec_time_list'].append(self.programs_dict[function_name]['schedules'][schedule_name]['exec_time'])
        storing_device = store_device
        for tree_footprint in self.batches_dict:
            for chunk in range(0,len(self.batches_dict[tree_footprint]['sched_names_list']),max_batch_size):
                if storing_device.type=='cuda': # Check GPU memory in order to avoid Out of memory error
                    if ((torch.cuda.memory_allocated(storing_device.index)/torch.cuda.get_device_properties(storing_device.index).total_memory)>0.80):
                        print('GPU memory on '+str(storing_device)+' nearly full, switching to CPU memory')
                        storing_device = torch.device('cpu')
                self.batched_schedule_names.append(self.batches_dict[tree_footprint]['sched_names_list'][chunk:chunk+max_batch_size])
                self.batched_program_names.append(self.batches_dict[tree_footprint]['program_names_list'][chunk:chunk+max_batch_size])
                self.batched_exec_time.append(self.batches_dict[tree_footprint]['exec_time_list'][chunk:chunk+max_batch_size])
                self.X.append((self.batches_dict[tree_footprint]['tree'],
                               torch.cat(self.batches_dict[tree_footprint]['comps_tensor_list'][chunk:chunk+max_batch_size], 0).to(storing_device),
                               torch.cat(self.batches_dict[tree_footprint]['loops_tensor_list'][chunk:chunk+max_batch_size], 0).to(storing_device)))
                self.Y.append(torch.FloatTensor(self.batches_dict[tree_footprint]['speedups_list'][chunk:chunk+max_batch_size]).to(storing_device))
        print(f'Number of batches {len(self.Y)}')
        if self.nb_long_access>0:
            print('Number of batches dropped due to too much memory accesses:' +str(self.nb_long_access))
        
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            return self.X[index], self.Y[index] 

    def __len__(self):
        return len(self.Y)

def load_data(train_val_dataset_file,split_ratio=None, max_batch_size=2048, filter_func=None):
    print("loading batches from: "+train_val_dataset_file)
    dataset = Dataset(train_val_dataset_file, max_batch_size,filter_func=filter_func)
    if split_ratio == None:
        split_ratio=0.2
    if split_ratio > 1 : # not a ratio a number of batches
        validation_size = split_ratio
    else:
        validation_size = int(split_ratio * len(dataset))
    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    val_batches_indices, train_batches_indices = indices[:validation_size],\
                                               indices[validation_size:]
    val_batches_list=[]
    train_batches_list=[]
    for i in val_batches_indices:
        val_batches_list.append(dataset[i])
    for i in train_batches_indices:
        train_batches_list.append(dataset[i])
    print("Data loaded")
    print("Sizes: "+str((len(val_batches_list),len(train_batches_list)))+" batches")
    return dataset, val_batches_list, val_batches_indices, train_batches_list, train_batches_indices


def train_model(model, criterion, optimizer, max_lr, dataloader, num_epochs=100, log_every=5, logFile='log.txt'):
    since = time.time()    
    losses = []
    train_loss = 0
    best_loss = math.inf
    best_model = None
    dataloader_size = {'train':0,'val':0}
    for _,label in dataloader['train']:
        dataloader_size['train']+=label.shape[0]
    for _,label in dataloader['val']:
        dataloader_size['val']+=label.shape[0]

    model = model.to(train_device)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(dataloader['train']), epochs=num_epochs)
    for epoch in range(num_epochs):
        epoch_start=time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':                
                model.train()  
            else:
                model.eval()
            running_loss = 0.0         
            # Iterate over data. 
            for inputs, labels in dataloader[phase]:
                original_device = labels.device
                inputs=(inputs[0], inputs[1].to(train_device), inputs[2].to(train_device))
                labels=labels.to(train_device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  
                    assert outputs.shape == labels.shape
#                     print((outputs, labels))
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item()*labels.shape[0]
                inputs=(inputs[0], inputs[1].to(original_device), inputs[2].to(original_device))
                labels=labels.to(original_device)
                epoch_end=time.time()                
                #running_corrects += torch.sum((outputs.data - labels.data) < e)/inputs.shape[0]
            epoch_loss = running_loss / dataloader_size[phase]           
            if phase == 'val':
                losses.append((train_loss, epoch_loss))
                if (epoch_loss<=best_loss):
                    best_loss = epoch_loss
                    best_model= copy.deepcopy(model)
                print('Epoch {}/{}:  train Loss: {:.4f}   val Loss: {:.4f}   time: {:.2f}s   best: {:.4f}'
                      .format(epoch + 1, num_epochs, train_loss, epoch_loss, epoch_end - epoch_start, best_loss))     
                if (epoch%log_every==0):
                    with open(logFile, "a+") as f:
                        f.write('Epoch {}/{}:  train Loss: {:.4f}   val Loss: {:.4f}   time: {:.2f}s   best: {:.4f} \n'
                      .format(epoch + 1, num_epochs, train_loss, epoch_loss, epoch_end - epoch_start, best_loss))
            else:
                train_loss = epoch_loss
                scheduler.step()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s   best validation loss: {:.4f}'
          .format(time_elapsed // 60, time_elapsed % 60, best_loss)) 
    with open(logFile, "a+") as f:
        f.write('-----> Training complete in {:.0f}m {:.0f}s   best validation loss: {:.4f}\n '
          .format(time_elapsed // 60, time_elapsed % 60, best_loss))
        
    return losses, best_model

class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(self, input_size, comp_embed_layer_sizes=[600, 350, 200, 180], drops=[0.225, 0.225, 0.225, 0.225], output_size=1):
        super().__init__()
        embedding_size = comp_embed_layer_sizes[-1]
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [embedding_size*2+8*2] + comp_embed_layer_sizes[-2:]
        comp_embed_layer_sizes = [input_size] + comp_embed_layer_sizes
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts= nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts= nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts= nn.ModuleList()
        for i in range(len(comp_embed_layer_sizes)-1):
            self.comp_embedding_layers.append(nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i+1], bias=True))
            nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(regression_layer_sizes)-1):
            self.regression_layers.append(nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i+1], bias=True))
            nn.init.xavier_uniform_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(concat_layer_sizes)-1):
            self.concat_layers.append(nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i+1], bias=True))
#             nn.init.xavier_uniform_(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        nn.init.xavier_uniform_(self.predict.weight)
        self.ELU=nn.ELU()
        self.no_comps_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))
        self.no_nodes_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))
        self.comps_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.nodes_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        
    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        for n in node['child_list']:
            nodes_list.append(self.get_hidden_state(n, comps_embeddings,loops_tensor))
        if (nodes_list != []):
            nodes_tensor = torch.cat(nodes_list, 1) 
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1,0,2)
        else:       
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
        if (node['has_comps']):
            selected_comps_tensor = torch.index_select(comps_embeddings, 1, node['computations_indices'])
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(selected_comps_tensor) 
            comps_h_n = comps_h_n.permute(1,0,2)
        else:
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
        selected_loop_tensor = torch.index_select(loops_tensor,1,node['loop_index'])
        x = torch.cat((nodes_h_n, comps_h_n, selected_loop_tensor),2)
        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))
        return x  

    def forward(self, tree_tensors):
        tree, comps_tensor, loops_tensor = tree_tensors
        #computation embbedding layer
        x = comps_tensor
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))  
        comps_embeddings = x
        #recursive loop embbeding layer
        prog_embedding = self.get_hidden_state(tree, comps_embeddings, loops_tensor)
        #regression layer
        x = prog_embedding
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        out = self.predict(x)
            
        return self.ELU(out[:,0,0])

def get_results_df(dataset, batches_list, indices, model, log=False):   
    df = pd.DataFrame()
    model.eval()
    torch.set_grad_enabled(False)
    all_outputs=[]
    all_labels=[]
    prog_names=[]
    sched_names=[]
    exec_times=[]

    for k, (inputs, labels) in tqdm(list(enumerate(batches_list))):
        original_device = labels.device
        inputs=(inputs[0], inputs[1].to(train_device), inputs[2].to(train_device))
        labels=labels.to(train_device)
        outputs = model(inputs)
        assert outputs.shape == labels.shape
        all_outputs.append(outputs)
        all_labels.append(labels)
        assert len(outputs)==len(dataset.batched_schedule_names[indices[k]])
        assert len(outputs)==len(dataset.batched_program_names[indices[k]])
        for j, sched_name in enumerate(dataset.batched_schedule_names[indices[k]]):
            sched_names.append(sched_name)
            prog_names.append(dataset.batched_program_names[indices[k]][j])
            exec_times.append(dataset.batched_exec_time[indices[k]][j])
        inputs=(inputs[0], inputs[1].to(original_device), inputs[2].to(original_device))
        labels=labels.to(original_device)
    preds = torch.cat(all_outputs)
    targets = torch.cat(all_labels)
    preds = preds.cpu().detach().numpy().reshape((-1,))
    preds = np.around(np.abs(preds),decimals=6)
    targets = np.around(targets.reshape((-1,)),decimals=6)
                                            
    assert preds.shape == targets.shape 
    df['name'] = prog_names
    df['sched_name'] = sched_names
    df['exec_time'] = exec_times
    df['prediction'] = np.array(preds)
    df['target'] = np.array(targets)
    df['abs_diff'] = np.abs(preds - targets)
    df['APE'] = np.abs(df.target - df.prediction)/df.target * 100
        
    return df

