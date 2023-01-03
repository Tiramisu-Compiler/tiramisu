from os import environ
from pprint import pprint
import pickle
import numpy as np
import torch 
import pandas as pd
# import seaborn as sns
from torch import optim
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import random
#for plotting
# from IPython.display import clear_output
from matplotlib import pyplot as plt
# import plotly.graph_objects as go
import sys
from torch.optim import AdamW
from torch.optim.lr_scheduler import *
import json
import re

train_device= torch.device(environ.get('train_device'))
store_device= torch.device(environ.get('store_device'))
# dataset_file= environ.get('dataset_file')
# test_dataset_file = environ.get('test_dataset_file')
# benchmark_dataset_file=environ.get('benchmark_dataset_file')

#hyperparameter of the k-best model
k = 5
depth_max = 7

assert depth_max > 2
one_output = int(math.factorial(depth_max)/(math.factorial(depth_max-2)*2) + 1) # ex: 22 = C{2,7} + 1= 22 / 2! = 2

#################################### Filtering functions for Data loading ####################################

#Filtering Loop Interchange schedules for the multiple computations functions
def filter_schedule_MC(schedule_str): # needs to return True if we want the passed schedule to be dropped 
    
    regex1 = "^I\(\{[C0-9,]+\},L[0-9]+,L[0-9]+\)$"
    regex2 = "F\(\{[C0-9,]+\},L[0-9]+\)I\(\{[C0-9,]+\},L[0-9]+,L[0-9]+\)$"
    if re.search(regex1, schedule_str) or re.search(regex2, schedule_str) : # drops all the schedules except the loop interchanges
        return True
    if schedule_str=="":
        return True
    return False


#Filtering Loop Interchange schedules for the single computations functions
def filter_schedule_SC(schedule_str): # needs to return True if we want the passed schedule to be dropped 
    
    regex = "I\(L[0-9]+,L[0-9]+\)$"
    if re.search(regex, schedule_str): # drops all the schedules except the loop interchanges
        return True
    if schedule_str=="":
        return True
    return False

#################################### Creating dataset entries functions ####################################

#define exception to discard the 'too large' programs for our presentation
class LargeAccessMatices(Exception):
    pass


#format the json file into arrays (X representation)
#program_dict added to retrieve the tree structure
def get_representation(program_json, schedule_json, program_dict):
    max_dims= depth_max
    max_accesses = 15
    program_representation = []
    indices_dict = dict()
    computations_dict = program_json['computations'] #dict_keys(['memory_size', 'iterators', 'computations'])
    ordered_comp_list = sorted(list(computations_dict.keys()), key = lambda x: computations_dict[x]['absolute_order'])
    
    for index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        if len(comp_dict['accesses'])>max_accesses:
#             print('too much acc')
            raise LargeAccessMatices
        if len(comp_dict['accesses'])<1:
#             print('too little acc')
            raise LargeAccessMatices
        comp_representation = []
        #         Is this computation a reduction 
        comp_representation.append(+comp_dict['comp_is_reduction'])

        #fusion transformation -L
        fused_levels = []
        if 'fusions' in schedule_json and schedule_json['fusions']:
            for fusion in schedule_json['fusions']:#check if comp is involved in fusions 
                 # fusions format [compname1, compname2, loop depth]
                if comp_name in fusion:
                    fused_levels.append(fusion[2])
        
        iterators_repr = []
#         for iterator_name in comp_dict['iterators']: #-L
        for iter_i,iterator_name in enumerate(comp_dict['iterators']):
            
            iterator_dict = program_json['iterators'][iterator_name]
            iterators_repr.append(iterator_dict['upper_bound']) 
            iterators_repr.append(iterator_dict['lower_bound'])
            
            # Fusion representation -L
            if iter_i in fused_levels:
                iterators_repr.append(1) #fused true
            else:
                iterators_repr.append(0) #fused false
                            
        iterator_repr_size = int(len(iterators_repr)/len(comp_dict['iterators']))
        iterators_repr.extend([0]*iterator_repr_size*(max_dims-len(comp_dict['iterators']))) # adding iterators padding 

        comp_representation.extend(iterators_repr) #adding the iterators representation    

        #       write access represation 
        write_access_matrix = isl_to_write_matrix(comp_dict['write_access_relation'])
        write_access_matrix = np.array(write_access_matrix)
        write_access_matrix = np.c_[np.ones(write_access_matrix.shape[0]), write_access_matrix] # adding tags for marking the used rows
        write_access_matrix = np.r_[[np.ones(write_access_matrix.shape[1])], write_access_matrix] # adding tags for marking the used columns
        padded_write_matrix = np.zeros((max_dims + 1, max_dims + 2))
        padded_write_matrix[:write_access_matrix.shape[0],:write_access_matrix.shape[1]-1] = write_access_matrix[:,:-1] #adding padding to the access matrix
        padded_write_matrix[:write_access_matrix.shape[0],-1] = write_access_matrix[:,-1] #adding padding to the access matrix
        write_access_repr = [comp_dict['write_buffer_id']+1] + padded_write_matrix.flatten().tolist()
        comp_representation.extend(write_access_repr)
        
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
        
        #added in new reduced -L
        comp_representation.append(len(comp_dict['iterators'])) #the depth! 
   
        # adding log(x+1) of the representation
#         log_rep = list(np.log1p(comp_representation))
#         comp_representation.extend(log_rep)
        
        program_representation.append(comp_representation)
        indices_dict[comp_name] = index
    
    
    #added for fusion -L
#     # transforming the schedule_json in order to have loops as key instead of computations, this dict helps building the loop vectors
    loop_schedules_dict = dict()
    for loop_name in program_json['iterators']:
        loop_schedules_dict[loop_name]=dict()
        loop_schedules_dict[loop_name]['fused']=0
        
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_schedule_dict = schedule_json[comp_name]

    #update the fusions in loops dict 
    if 'fusions' in schedule_json and schedule_json['fusions']:
        for fusion in schedule_json['fusions']:
            fused_loop1 = computations_dict[fusion[0]]['iterators'][fusion[2]]
            fused_loop2 = computations_dict[fusion[1]]['iterators'][fusion[2]]
            loop_schedules_dict[fused_loop1]['fused']=1
            loop_schedules_dict[fused_loop2]['fused']=1
    #end of added for fusion

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
        
        #added for fusion
        loop_representation.append(loop_schedules_dict[loop_name]['fused'])
        
#      REMOVED IN NEW REDUCED REPRESENTATION        ----------------------------------------------
#         # adding log(x+1) of the loop representation
#         loop_log_rep = list(np.log1p(loop_representation)) #WHAT ?!! -L No need for log now! All numbers
#         loop_representation.extend(loop_log_rep)
#      REMOVED IN NEW REDUCED REPRESENTATION        ----------------------------------------------
        loops_representation_list.append(loop_representation)    
        loops_indices_dict[loop_name]=loop_index
        loop_index+=1
            
    def update_tree_atributes(node):     
        node['loop_index'] = torch.tensor(loops_indices_dict[node['loop_name'][:]]).to(train_device)
        if node['computations_list']!=[]:
            node['computations_indices'] = torch.tensor([indices_dict[comp_name] for comp_name in node['computations_list']]).to(train_device)
            node['has_comps'] = True
        else:
            node['has_comps'] = False
        for child_node in node['child_list']:
            update_tree_atributes(child_node)
        return node
    
    # getting the original tree structure 
    no_sched_json = program_dict['schedules_list'][0]
    assert 'fusions' not in no_sched_json or no_sched_json['fusions']==None
    orig_tree_structure = no_sched_json['tree_structure']
    tree_annotation = copy.deepcopy(orig_tree_structure) #to avoid altering the original tree from the json
    prog_tree = update_tree_atributes(tree_annotation) 
    
    
    #using resulting tree structure of the fused program
#     tree_annotation = copy.deepcopy(schedule_json['tree_structure']) #to avoid altering the original tree from the json
#     prog_tree = update_tree_atributes(tree_annotation) 
        
    
    loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_representation_list),0)#.to(device)
    computations_tensor = torch.unsqueeze(torch.FloatTensor(program_representation),0)#.to(device)     

    return prog_tree, computations_tensor, loops_tensor


#return a string describing the tree structure (loops and computations placement) for the current program.
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
    
    
#dataset creation
class Dataset():
    def __init__(self, dataset_MC_filename, dataset_SC_filename, max_batch_size, filter_func=None, filter_func_MC=None,filter_func_SC=None, transform_func=None):
        super().__init__()
        
        self.X = []
        self.Y = []
        self.batched_program_names = []
        self.batched_schedule_names = []
        self.batched_exec_time = []
        self.nb_nan=0
        self.nb_long_access=0
        self.batches_dict=dict()
            
        
        if(dataset_MC_filename!=None):
        
            #loading multi computation programs

            self.dataset_MC_name=dataset_MC_filename

            if dataset_MC_filename.endswith('json'):
                with open(dataset_MC_filename, 'r') as f:
                    dataset_MC_str = f.read()
                self.programs_dict_MC = json.loads(dataset_MC_str)
            elif dataset_MC_filename.endswith('pkl'):
                with open(dataset_MC_filename, 'rb') as f:
                    self.programs_dict_MC = pickle.load(f)

            if (filter_func_MC==None):
                filter_func_MC = lambda x : True
            if (transform_func==None):
                transform_func = lambda x : x


            for function_name in tqdm(self.programs_dict_MC):
#                 print("MC : ", function_name)
                if (np.min(self.programs_dict_MC[function_name]['schedules_list'][0]['execution_times'])<0): #if less than x ms
                    continue

                program_json = self.programs_dict_MC[function_name]['program_annotation']
                program_exec_time = self.programs_dict_MC[function_name]['initial_execution_time']
                loops = shared_loop_nest(program_json)
                
                schedules_dict = {}  
                explored_schedules = {}

                #checking all schedules
                for schedule_index in range(len(self.programs_dict_MC[function_name]['schedules_list'])):
                    sched_str = self.programs_dict_MC[function_name]['schedules_list'][schedule_index]["sched_str"]
                    schedule_json = self.programs_dict_MC[function_name]['schedules_list'][schedule_index]

                    #leave only LI functions, that are legal: happen in the perfectly nested part of the loop nest.
                    if (not filter_func_MC(sched_str)) or (not legal_LI(sched_str, schedule_index, loops)):
                        continue

                    sched_exec_time = np.min(schedule_json['execution_times'])
                    self.programs_dict_MC[function_name]['schedules_list'][schedule_index]['speedup'] = max(program_exec_time / sched_exec_time,0.01) #speedup clipping
                    if ((np.isnan(self.programs_dict_MC[function_name]['schedules_list'][schedule_index]['speedup']))
                         or(self.programs_dict_MC[function_name]['schedules_list'][schedule_index]['speedup']==0)): 
                        self.nb_nan+=1
                        continue


                    #look for the parent schedule
                    scheduleP_str = re.sub("I\(\{[C0-9,]+\},L[0-9]+,L[0-9]+\)", '', sched_str)
#                         print(self.programs_dict_MC[function_name]['schedules_list'])
                    if scheduleP_str in explored_schedules.keys():  #parent schedule exist, check if the current schedule is better
                        schedulePID = explored_schedules[scheduleP_str]
            
                        # logging all LIs in this vector, to extract the best k LI later
                        ind_LI = index_interchage(sched_str)
                        if sched_exec_time < schedules_dict[schedulePID]['all'][ind_LI]:
                            schedules_dict[schedulePID]['all'][ind_LI] = sched_exec_time
                    
                        if sched_exec_time >= schedules_dict[schedulePID]['best']: # not the best
                            continue
                        else :
                            schedules_dict[schedulePID]['best'] = sched_exec_time
                            schedules_dict[schedulePID]['sched'] = schedule_index
#                             sched_dict = {'best':sched_exec_time, 'sched':schedule_index}
#                             schedules_dict[schedulePID] = sched_dict
                        

                    else: # parent schedule is new 
                        #tracking fusions                
#                         if scheduleP_str != "":
#                             print(scheduleP_str, function_name)
                        
                        scheduleP, schedulePID = get_sched_by_string(scheduleP_str, self.programs_dict_MC[function_name]['schedules_list'])
#                         print(scheduleP, schedulePID )
                        explored_schedules[scheduleP_str]=schedulePID  # discoverd a new schedule

                        schedP_exec_time = np.min(scheduleP['execution_times'])

                        if sched_exec_time >= schedP_exec_time: # parent is best
                            schedP_dict = {'best':schedP_exec_time, 'sched':schedulePID}
                            schedules_dict[schedulePID] = schedP_dict
                        else : # schedule is best
                            sched_dict = {'best':sched_exec_time, 'sched':schedule_index}
                            schedules_dict[schedulePID] = sched_dict
                        
                        #create a vector, with 'one_output' size, to save execution times of all possible LIs, to be able to extract the k-best option later.
                        output = np.zeros(one_output,dtype=np.float64) + np.inf # because, a priori, all of them are very bad and does not execute well
                        output[0] = schedP_exec_time
                        ind_LI = index_interchage(sched_str)
                        output[ind_LI] = sched_exec_time
                        schedules_dict[schedulePID]['all'] = output

                #for each fused program or the original pgme
                for elem in schedules_dict.items():
                    sched_json = self.programs_dict_MC[function_name]['schedules_list'][elem[0]]
                    try:
                        tree, comps_tensor, loops_tensor = get_representation(program_json, sched_json, self.programs_dict_MC[function_name]) ######
                    except LargeAccessMatices:
                        self.nb_long_access +=1
                        continue   

                    # for each datapoint append its best LIs
                    tree_footprint=get_tree_footprint(tree) 
                    self.batches_dict[tree_footprint] = self.batches_dict.get(tree_footprint,{'tree':tree,'comps_tensor_list':[],'loops_tensor_list':[],'program_names_list':[],'sched_names_list':[],'speedups_list':[],'exec_time_list':[], 'output':[]}) 
                    self.batches_dict[tree_footprint]['comps_tensor_list'].append(comps_tensor)
                    self.batches_dict[tree_footprint]['loops_tensor_list'].append(loops_tensor)
#                     print(comps_tensor.shape, loops_tensor.shape)
                    self.batches_dict[tree_footprint]['sched_names_list'].append(elem[0])
                    self.batches_dict[tree_footprint]['program_names_list'].append(function_name)

                    speedup = max(program_exec_time / elem[1]['best'] , 0.01) #speedup clipping
                    self.batches_dict[tree_footprint]['speedups_list'].append(speedup)  
                    self.batches_dict[tree_footprint]['exec_time_list'].append(elem[1]['best'])

                    best_schedule_str = self.programs_dict_MC[function_name]['schedules_list'][elem[1]['sched']]['sched_str'] #does it contains only LI ?
                    
                    #produce the k-best concatinate output.
                    output= elem[1]['all']
                    output = output[0] / output 
                    order = np.flip(np.argsort(output, -1)) #highest to smallest
                    y = np.zeros(one_output * k,dtype=np.float64)
                    for i in range(k):
                        if output[order[i]] == 0:
                            y[one_output*i] = 1 # No LI
                            continue
                        y[one_output*i+order[i]] = 1 #define the output, using the best order we have in order % speedups. Because,it returns the indexes of the best options, and thus, the ones that should be put in 1.
                    self.batches_dict[tree_footprint]['output'].append(torch.tensor(y)) 

        #loading single computation programs
        if(dataset_SC_filename!=None):
            self.dataset_SC_name=dataset_SC_filename

            if dataset_SC_filename.endswith('json'):
                with open(dataset_SC_filename, 'r') as f:
                    dataset_SC_str = f.read()
                self.programs_dict_SC = json.loads(dataset_SC_str)
            elif dataset_SC_filename.endswith('pkl'):
                with open(dataset_SC_filename, 'rb') as f:
                    self.programs_dict_SC = pickle.load(f)

            if (filter_func_SC==None):
                filter_func_SC = lambda x : True
            if (transform_func==None):
                transform_func = lambda x : x


            for function_name in tqdm(self.programs_dict_SC):
                if (np.min(self.programs_dict_SC[function_name]['schedules_list'][0]['execution_times'])<0): #if less than x ms
                    continue
                program_json = self.programs_dict_SC[function_name]['program_annotation']
                program_exec_time = self.programs_dict_SC[function_name]['initial_execution_time']
                Parent_sched_json = self.programs_dict_SC[function_name]['schedules_list'][0]   # get the parent schedule


                try:
                    tree, comps_tensor, loops_tensor = get_representation(program_json, Parent_sched_json, self.programs_dict_SC[function_name]) ######
                except LargeAccessMatices:
                    self.nb_long_access +=1
                    continue        

                #######
                min_sch_time = np.inf   
                output = np.zeros(one_output,dtype=np.float64) + np.inf # because, a priori, all of them are very bad and does not execute well
                
                for schedule_index in range(len(self.programs_dict_SC[function_name]['schedules_list'])):
                    sched_str = sched_json_to_sched_str( self.programs_dict_SC[function_name]['schedules_list'][schedule_index] )
                    if (not filter_func_SC(sched_str)):
                        continue  

                    sched_exec_time = np.min(self.programs_dict_SC[function_name]['schedules_list'][schedule_index]['execution_times'])
                    self.programs_dict_SC[function_name]['schedules_list'][schedule_index]['speedup'] = max(program_exec_time / sched_exec_time,0.01) #speedup clipping
                    if ((np.isnan(self.programs_dict_SC[function_name]['schedules_list'][schedule_index]['speedup']))
                         or(self.programs_dict_SC[function_name]['schedules_list'][schedule_index]['speedup']==0)): #Nan value means the schedule didn't run, zero values means exec time<1 micro-second, skip them
                        self.nb_nan+=1
                        continue

                    #if another schedule used the same interchange before, and the current execution time is better
                    if output[index_interchage(sched_str)] > sched_exec_time:
                        output[index_interchage(sched_str)] = sched_exec_time 
                
                    if sched_exec_time >= min_sch_time:
                            continue   # we only keep the best loop interchange for a given program
                    min_sch_time = sched_exec_time
                    best_schedule_str = sched_json_to_sched_str( self.programs_dict_SC[function_name]['schedules_list'][schedule_index] )
               
                # for each function append its best LI

                tree_footprint=get_tree_footprint(tree) 
                self.batches_dict[tree_footprint] = self.batches_dict.get(tree_footprint,{'tree':tree,'comps_tensor_list':[],'loops_tensor_list':[],'program_names_list':[],'sched_names_list':[],'speedups_list':[],'exec_time_list':[], 'output':[]}) 
                self.batches_dict[tree_footprint]['comps_tensor_list'].append(comps_tensor)
                self.batches_dict[tree_footprint]['loops_tensor_list'].append(loops_tensor)
#                 print(comps_tensor.shape, loops_tensor.shape)
                #self.batches_dict[tree_footprint]['sched_names_list'].append(schedule_name)  ommit it bcs the schedule is always the 0th schedule ==> not significant
                self.batches_dict[tree_footprint]['program_names_list'].append(function_name)
                self.batches_dict[tree_footprint]['speedups_list'].append(self.programs_dict_SC[function_name]['schedules_list'][0]['speedup'])  ######
                self.batches_dict[tree_footprint]['exec_time_list'].append(program_exec_time)   ######
                
                #Y
                output = output[0] / output 
                order = np.flip(np.argsort(output, -1)) #highest to smallest
                y = np.zeros(one_output * k,dtype=np.float64)
                for i in range(k):
                    if output[order[i]] == 0:
                        y[one_output*i] = 1 # No LI
                        continue
                    y[one_output*i+order[i]] = 1 #define the output, using the best order we have in order % speedups. Because,it returns the indexes of the best options, and thus, the ones that should be put in 1.
                self.batches_dict[tree_footprint]['output'].append(torch.tensor(y))

# Batching the data. 
        storing_device = store_device
        for tree_footprint in self.batches_dict:
            for chunk in range(0,len(self.batches_dict[tree_footprint]['program_names_list']),max_batch_size):  #####
                if storing_device.type=='cuda': # Check GPU memory in order to avoid Out of memory error
                    if ((torch.cuda.memory_allocated(storing_device.index)/torch.cuda.get_device_properties(storing_device.index).total_memory)>0.80):
                        print('GPU memory on '+str(storing_device)+' nearly full, switching to CPU memory')
                        storing_device = torch.device('cpu')
                #self.batched_schedule_names.append(self.batches_dict[tree_footprint]['sched_names_list'][chunk:chunk+max_batch_size])   #####
                self.batched_program_names.append(self.batches_dict[tree_footprint]['program_names_list'][chunk:chunk+max_batch_size])
                self.batched_exec_time.append(self.batches_dict[tree_footprint]['exec_time_list'][chunk:chunk+max_batch_size])
                T = self.batches_dict[tree_footprint]['tree'] # tree of the program
                R = torch.cat(self.batches_dict[tree_footprint]['comps_tensor_list'][chunk:chunk+max_batch_size], 0).to(storing_device) #computation tensor
                S = torch.cat(self.batches_dict[tree_footprint]['loops_tensor_list'][chunk:chunk+max_batch_size], 0).to(storing_device) #loops tensor
                self.X.append( ( T, R , S ) )
                    
                temp_tens = torch.cat(self.batches_dict[tree_footprint]['output'][chunk:chunk+max_batch_size])
                dim_temp_tens = int(temp_tens.shape[0] / (one_output * k))
                self.Y.append(torch.reshape(temp_tens, (dim_temp_tens,one_output *k)).to(storing_device))
                
                #sanity check
                if len(self.X) != len(self.Y):
                    print(len(self.X[-1]), len(self.Y[-1]))   # to look here 
                    print(type(self.X),len(self.X), type(self.Y),len(self.Y),type(self.X[0][1]),len(self.X[0][1]))
                    print("stop")
                                                
        print(f'Number of batches {len(self.Y)}')
        if self.nb_long_access>0:
            print('Number of data points dropped due to too much memory accesses:' +str(self.nb_long_access))
            
            
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            return self.X[index], self.Y[index] 

    def __len__(self):
        return len(self.Y)

    
#loading data of single computation structure (precedently used, for compatibility and data reuse) and the multiple computation one.
def load_merge_data(train_val_MC,train_val_SC,split_ratio=None, max_batch_size=2048, filter_func_MC=None, filter_func_SC=None):
    full_dataset = Dataset(train_val_MC, train_val_SC, max_batch_size,filter_func_MC=filter_func_MC, filter_func_SC=filter_func_SC)
    if split_ratio == None:
        split_ratio=0.2
    if split_ratio > 1 : # not a ratio a number of batches
        validation_size = split_ratio
    else:
        validation_size = int(split_ratio * len(full_dataset))
    indices = list(range(len(full_dataset)))
    random.Random(42).shuffle(indices)
    val_batches_indices, train_batches_indices = indices[:validation_size],\
                                               indices[validation_size:]
    val_batches_list=[]
    train_batches_list=[]
    for i in val_batches_indices:
        val_batches_list.append(full_dataset[i])
    for i in train_batches_indices:
        train_batches_list.append(full_dataset[i])
    print("Data loaded")
    print("Sizes: "+str((len(val_batches_list),len(train_batches_list)))+" batches")
    return full_dataset, val_batches_list, val_batches_indices, train_batches_list, train_batches_indices


#################################### Training functions ####################################

def train_model(model, criterion, optimizer, max_lr, dataloader, num_epochs=100, log_every=5, logFile='log.txt'):
    since = time.time()    
    losses = []
    train_loss = 0
    best_loss = math.inf
    best_model = None
    soft = nn.Softmax(-1)
    
    dataloader_size = {'train':0,'val':0}
    for _,label in dataloader['train']: 
        dataloader_size['train']+=label.shape[0] ####
    for _,label in dataloader['val']:
        dataloader_size['val']+=label.shape[0] ####

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
                original_device = labels.device  ####
                inputs=(inputs[0], inputs[1].to(train_device), inputs[2].to(train_device))
                labels=labels.to(train_device)  ####
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
    
                    outputs = model(inputs)  
                    assert outputs.shape == labels.shape 
                    loss = 0
                    #apply cross entropy on each prediction in the sub-vectors.
                    weight = 100
                    for i in range(k):
                        loss += weight*criterion(outputs[...,one_output*i:one_output*(i+1)], torch.argmax(labels[...,one_output*i:one_output*(i+1)],-1)) 
                        weight /= 1.25
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                
                running_loss += loss.item()*labels.shape[0]
                #running_loss += loss.item()
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
                if(len(losses)>2 and (epoch%100 == 0 )) :
                    loss_plot(losses)
                if epoch == (num_epochs - 1):
                    loss_plot(losses, end = True)                    
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

#print the losses graph
def loss_plot(losses, figsize=(8,6), title='', end = False):   
    val_loss, train_loss = map(list, zip(*losses))  
    plt.figure(figsize=figsize)
    plt.plot(val_loss, label='Training Loss')
    plt.plot(train_loss, label='Validation Loss')
    plt.title(title)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='center left') # the plot evolves to the right
    if end == True:
        plt.savefig("output.png")
    plt.show();
    

#################################### Model definition functions ####################################

class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(self, input_size, comp_embed_layer_sizes=[600, 350, 200, 180], drops=[0.225, 0.225, 0.225, 0.225], output_size=1):
        super().__init__()
        embedding_size = comp_embed_layer_sizes[-1]
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
#         concat_layer_sizes = [embedding_size*2+24] + comp_embed_layer_sizes[-2:] # before reduced -L
        concat_layer_sizes = [embedding_size*2+3] + comp_embed_layer_sizes[-2:]
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
#         self.ELU = nn.Tanh()
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
#         print(out[:,0,:].shape)
        return out[:,0,:]  #(out[:,0])#self.ELU(out[:,0,0])       nb_elem/batch  nb_comp 15
    

#################################### Results retrieval functions ####################################
    
def get_results_df(dataset, batches_list, indices, model, log=False):   
    df = pd.DataFrame()
    model.eval()
    torch.set_grad_enabled(False)
    all_outputs=[]
    all_labels=[]
    prog_names=[]
    #sched_names=[]
    exec_times=[]
    soft = nn.Softmax(-1)

    for kk, (inputs, labels) in tqdm(list(enumerate(batches_list))):
        original_device = labels.device
        inputs=(inputs[0], inputs[1].to(train_device), inputs[2].to(train_device))
        labels=labels.to(train_device)
        outputs = model(inputs)
        assert outputs.shape == labels.shape
        
        outputs_soft = soft(outputs[...,0:one_output]) # special case
        for i in range(1,k):
            outputs_soft = torch.cat((outputs_soft,soft(outputs[...,one_output*i:one_output*(i+1)])),-1)
#         print(outputs_soft.shape, labels.shape)
        assert outputs_soft.shape == labels.shape

        #verified: outputs_softmax has the same shape
        
        all_outputs.append(outputs_soft) ##fixed
        all_labels.append(labels)

        for j, prog_name in enumerate(dataset.batched_program_names[indices[kk]]):   #####
            #sched_names.append(sched_name)
            prog_names.append(dataset.batched_program_names[indices[kk]][j])
            exec_times.append(dataset.batched_exec_time[indices[kk]][j])
        inputs=(inputs[0], inputs[1].to(original_device), inputs[2].to(original_device))
        labels=labels.to(original_device)
        
    preds = torch.cat(all_outputs).cpu().detach().numpy() #because it is modified above to be a tensor
    targets = torch.cat(all_labels).cpu().detach().numpy()
                                      
    assert preds.shape == targets.shape 
    df['name'] = prog_names
    df['exec_time'] = exec_times
    df['prediction'] = list(format_output(preds)[:,:]) #MISTAKES 
    df['target'] = list(targets[:,:])
    return df

#accuracy 1st-shot, 2-shots, 3-shots and 5-shots
def accuracy(df):
    #return np.sum(df["prediction"] == df["target"]) / len(df) * 100
    targets = np.array(df['target'].values.tolist()) # 1-dim array (#rows, ) = list
    #we went to lists and back to get the last dimension of the (#rows, #size_output) instead of (#rows,)
    predictions = np.array(df['prediction'].values.tolist())
    
    first = np.argmax(targets[:,0:one_output],-1) == np.argmax(predictions[:,0:one_output],-1) 
    second = np.argmax(targets[:,0:one_output],-1) ==  np.argmax(predictions[:,one_output:one_output+one_output],-1)
    third = np.argmax(targets[:,0:one_output],-1) ==  np.argmax(predictions[:,one_output*2:one_output*3],-1)
    forth = np.argmax(targets[:,0:one_output],-1) ==  np.argmax(predictions[:,one_output*3:one_output*4],-1 )  
    fifth = np.argmax(targets[:,0:one_output],-1) ==  np.argmax(predictions[:,one_output*4:one_output*5],-1)
    firstorsecond = np.logical_or(first,second)
    firstorsecondorsthird = firstorsecond | third # same as logical or
    anywhere = firstorsecondorsthird | forth | fifth
    return np.sum(first == True) / len(df) * 100, np.sum(firstorsecond == True) / len(df) * 100, np.sum(firstorsecondorsthird == True) / len(df) * 100, np.sum(anywhere == True) / len(df) * 100

#################################### Helper functions for data representation ####################################

def isl_to_write_matrix(isl_map): # for now this function only support reductions
    comp_iterators_str = re.findall(r'\[(.*)\]\s*->', isl_map)[0]
    buffer_iterators_str = re.findall(r'->\s*\w*\[(.*)\]', isl_map)[0]
    buffer_iterators_str=re.sub(r"\w+'\s=","",buffer_iterators_str)
    comp_iter_names = re.findall(r'(?:\s*(\w+))+', comp_iterators_str)
    buf_iter_names = re.findall(r'(?:\s*(\w+))+', buffer_iterators_str)
    matrix = np.zeros([len(buf_iter_names),len(comp_iter_names)+1])
    for i,buf_iter in enumerate(buf_iter_names):
        for j,comp_iter in enumerate(comp_iter_names):
            if buf_iter==comp_iter:
                matrix[i,j]=1
                break
    return matrix

def isl_to_write_dims(isl_map): # return the buffer iterator that defines the write buffer
    buffer_iterators_str = re.findall(r'->\s*\w*\[(.*)\]', isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=","",buffer_iterators_str)
    buf_iter_names = re.findall(r'(?:\s*(\w+))+', buffer_iterators_str)
    return buf_iter_names
                              

#################################### Helper functions for LI output ####################################

# return the position of a chosen LI  in the output vector
def pos(a,depth):  # calculate the position of the interchange in the output vector
    if(a==0):
        return 0
    elif (a==1):
        return depth-1
    else:
        return pos(a-1,depth)+depth-a

#encode the output vector to get '1' in the right LI, '0' elsewhere
def encode_interchage(str_sched):
    output = np.zeros(one_output,dtype=int) 
    str_interchange = re.findall('I\(.*\)', str_sched)  # get the LI string
    if str_interchange==[]:
        output[0] = 1
    else:
        m = re.findall(r'\d+', str_interchange[0])  # get the number of the loops
        a = min(int(m[-2]),int(m[-1]))
        b = max(int(m[-2]),int(m[-1]))
        i2 = b-a
        i1 = pos(a,depth_max)
        output[i1+i2]=1      
    return output

#same as precedent, but return the index
def index_interchage(str_sched): #encode the output vector to get '1' in the right LI, '0' elsewhere
    output = np.zeros(one_output,dtype=int)
    str_interchange = re.findall('I\(.*\)', str_sched)  # get the LI string
    if str_interchange==[]:
        return(0)
    else:
        m = re.findall(r'\d+', str_interchange[0])  # get the number of the loops
        a = min(int(m[-2]),int(m[-1]))
        b = max(int(m[-2]),int(m[-1]))
        i2 = b-a
        i1 = pos(a,depth_max)
        return(i1+i2)      

    
def get_sched_by_string(sched_str,schedules):
    for schedID in range(len(schedules)):
        sched = schedules[schedID]
        schedExp_str = sched['sched_str']
        if schedExp_str == sched_str: # found the right schedule
            return sched, schedID
        
def get_results_times_function(vector,schedules):
    
    sched_str = "I(L"
    np_vect= np.array(vector)
    indices = np.where(np_vect==1)[0]
    if len(indices) != 0: # check if a LI is applied
        sched_str = sched_str + str(indices[0]) + ",L" + str(indices[1]) + ")"
    else :
        sched_str = ""
        
    sched = get_sched_by_string(sched_str,schedules)
    if sched != None :
        sched_exec_time = np.min(sched['execution_times']) 
        return sched_exec_time
    else :
        return -1
    

def format_output(arr):  
    pred = np.zeros(arr.shape)
    for b in range(k):
        indices = np.argmax(arr[...,one_output*b:one_output*(b+1)], -1)
        for i in range(indices.shape[0]):
            pred[i,one_output*b + indices[i]] = 1
    return pred

#################################### Helper functions for Single computation functions ####################################

def speedup_clip(speedup):
    if speedup<0.01:
        speedup = 0.01
    return speedup

def sched_json_to_sched_str(sched_json): # Works only for 1 comp programs
    orig_loop_nest = []
    orig_loop_nest.append(sched_json['tree_structure']['loop_name'])
    child_list = sched_json['tree_structure']['child_list']
    while len(child_list)>0:
        child_loop = child_list[0]
        orig_loop_nest.append(child_loop['loop_name'])
        child_list = child_loop['child_list']
        
    comp_name = [n for n in sched_json.keys() if not n in ['unfuse_iterators','tree_structure','execution_times']][0]
    schedule = sched_json[comp_name]
    transf_loop_nest = orig_loop_nest
    sched_str = ''
    
    if schedule['interchange_dims']:
        first_dim_index = transf_loop_nest.index(schedule['interchange_dims'][0])
        second_dim_index = transf_loop_nest.index(schedule['interchange_dims'][1])
        sched_str+='I(L'+str(first_dim_index)+',L'+str(second_dim_index)+')'
        transf_loop_nest[first_dim_index], transf_loop_nest[second_dim_index] = transf_loop_nest[second_dim_index], transf_loop_nest[first_dim_index]
    if schedule['skewing']:
        first_dim_index = transf_loop_nest.index(schedule['skewing']['skewed_dims'][0])
        second_dim_index = transf_loop_nest.index(schedule['skewing']['skewed_dims'][1])
        first_factor = schedule['skewing']['skewing_factors'][0]
        second_factor = schedule['skewing']['skewing_factors'][1]
        sched_str+='S(L'+str(first_dim_index)+',L'+str(second_dim_index)+','+str(first_factor)+','+str(second_factor)+')'
    if schedule['parallelized_dim']:
        dim_index = transf_loop_nest.index(schedule['parallelized_dim'])
        sched_str+='P(L'+str(dim_index)+')'
    if schedule['tiling']:
        if schedule['tiling']['tiling_depth']==2:
            first_dim = schedule['tiling']['tiling_dims'][0]
            second_dim = schedule['tiling']['tiling_dims'][1]
            first_dim_index = transf_loop_nest.index(first_dim)
            second_dim_index = transf_loop_nest.index(second_dim)
            first_factor = schedule['tiling']['tiling_factors'][0]
            second_factor = schedule['tiling']['tiling_factors'][1]
            sched_str+='T2(L'+str(first_dim_index)+',L'+str(second_dim_index)+','+str(first_factor)+','+str(second_factor)+')'
            i = transf_loop_nest.index(first_dim)
            transf_loop_nest[i:i+1]=first_dim+'_outer', second_dim+'_outer'
            i = transf_loop_nest.index(second_dim)
            transf_loop_nest[i:i+1]=first_dim+'_inner', second_dim+'_inner'
        else: #tiling depth == 3
            first_dim = schedule['tiling']['tiling_dims'][0]
            second_dim = schedule['tiling']['tiling_dims'][1]
            third_dim = schedule['tiling']['tiling_dims'][2]
            first_dim_index = transf_loop_nest.index(first_dim)
            second_dim_index = transf_loop_nest.index(second_dim)
            third_dim_index = transf_loop_nest.index(third_dim)
            first_factor = schedule['tiling']['tiling_factors'][0]
            second_factor = schedule['tiling']['tiling_factors'][1]
            third_factor = schedule['tiling']['tiling_factors'][2]
            sched_str+='T3(L'+str(first_dim_index)+',L'+str(second_dim_index)+',L'+str(third_dim_index)+','+str(first_factor)+','+str(second_factor)+','+str(third_factor)+')'
            i = transf_loop_nest.index(first_dim)
            transf_loop_nest[i:i+1]=first_dim+'_outer', second_dim+'_outer', third_dim+'_outer'
            i = transf_loop_nest.index(second_dim)
            transf_loop_nest[i:i+1]=first_dim+'_inner', second_dim+'_inner', third_dim+'_inner'
            transf_loop_nest.remove(third_dim)
    if schedule['unrolling_factor']:
        dim_index = len(transf_loop_nest)-1
        dim_name =transf_loop_nest[-1]
        sched_str+='U(L'+str(dim_index)+','+schedule['unrolling_factor']+')'
        transf_loop_nest[dim_index:dim_index+1] = dim_name+'_Uouter', dim_name+'_Uinner'
    
    return sched_str

#################################### Helper functions for multiple computation functions ####################################

def shared_loop_nest(program_json):
    stop = False
    loop_nest = []
    iterators = program_json['iterators']
    j = 0
    for i in list(iterators.keys()) : 
        if not stop : 
            iterator = iterators[i]
            if iterator['child_iterators'] != [] :  #in the middle of the tree
                if iterator['computations_list'] != [] :  #has some computations
                    stop = True
                else:
                    if len( iterator['child_iterators'] ) >= 2 :
                        stop = True
            if stop == False :
                loop_nest.append( "i"+str(j) )
        else : 
            return loop_nest
        j = j + 1
    return loop_nest

def legal_LI(schedule_str, sched_ind, loops):
    
    if(schedule_str==""):
        return True
    else :
        regex1 = "I\(\{[C0-9,]+\},L[0-9]+,L[0-9]+\)$" 
        LI = re.findall(regex1, schedule_str)  # get the LI string
        
        m = re.findall(r'\d+', LI[0])  # get the loops
        L1 = 'i' + m[-2]
        L2 = 'i' + m[-1]
        if (L1 in loops) and (L2 in loops):
            return True
        else:
            return False


#################################### Search Performance Test helper functions ###############################
#restore the string of the LI from the index of the LI model output vector
#k: hyper parameter of the model; one_output, depth_max: representation hyperparameters
def get_LI_ordered(output):
    LIs = []
    taken = []
    for i in range(k):
        ind = np.argmax(output[...,one_output*i:one_output*(i+1)]) # return the indice of the chosen LI
        if ind in taken:
            continue
        taken.append(ind)
        if ind == 0:
            LIs.append("")
        else:
            for i in range(depth_max):
                if (ind < ( depth_max - i)):
                    str2 = "I(L" + str(i) + ",L" + str(ind+i)+ ")"
                    str2.strip()
                    if str2 not in LIs:
                        LIs.append(str2)
                    break
                else:
                    ind= ind - depth_max + i + 1
    return LIs

#################################### Beam Search Exploration Simulation Functions ###############################


def update_exploration_trace(node, func_predictions_df, func_enforced_df):
    if not node['schedule'] in list(func_predictions_df['sched_str']):
#         print(node['schedule'])
        return None
    sched = node['schedule']
    depth = node['depth']
    level_enforcement = func_enforced_df.query('depth==@depth') # it returns all the enforced nodes in the level

    if not level_enforcement.empty: #there is at least one enforced node in the level.
        node['level_has_enforcement'] = True
        res = level_enforcement.query('sched_str==@sched') 
        if not res.empty:
            node['enforced'] = True
            node['priority']= res.iloc[0]['priority']
        else:
            node['enforced'] = False
            node['priority']= 5 #because k = 5
        
    else:
        node['level_has_enforcement'] = False
        
    
#     node['id'] = int(func_predictions_df.query('sched_str == @sched')['sched_name'])
    new_children = []
    for child in node['children']:
        updated = update_exploration_trace(child, func_predictions_df, func_enforced_df)
        if updated != None:
            new_children.append(updated)
    node['children'] = new_children
    return node
  
def simulate_BeamSearch(root,beam_size,preds_dict=None,eval_mode='execution'): # given an exploration trace and a beam size, return the best candidate that can be found using that beam size
    children = root['children']
    if len(children)==0:
        return root
    
    if eval_mode != 'execution':
        if (children[0]['level_has_enforcement']): #that s our target level.
            if eval_mode == 'model':
                for child in children:
                    child['prediction'] = preds_dict[child['schedule']] #new column
                children = sorted(children, key = lambda x: x['prediction'], reverse=True) 
                if beam_size != 9:
                    children = children[:beam_size]
                #else: take them all.
            else: #if level1
                assert children[0]['level_has_enforcement'] == True # if some enforcement is set at this level = special model
                children = sorted(children, key = lambda x: x['priority']) #take all the children, from small priority to highest one.
                if beam_size != 9:
                    children = children[:beam_size]
            root['prediction'] = preds_dict[root['schedule']] #do we need this ?
        else:
            children = sorted(children, key = lambda x: x['evaluation']) 
            if beam_size != 9:
                children = children[:beam_size]
    else:
        children = sorted(children, key = lambda x: x['evaluation']) 
        if beam_size != 9:
            children = children[:beam_size]
#     print('----------------------------------------')
    bests = []
    for child in children:
        bests.append(simulate_BeamSearch(child,beam_size,preds_dict,eval_mode))
    bests.append(root) 
#     if eval_mode == 'model':
#         return max(bests, key = lambda x: x['prediction'])
#     elif eval_mode == 'execution':
#         return min(bests, key = lambda x: x['evaluation'])
    return min(bests, key = lambda x: x['evaluation'])
    
def simulate_TrueBeamSearch(root,beam_size,preds_dict=None,eval_mode='execution'): # given an exploration trace and a beam size, return the best candidate that can be found using that beam size
    if eval_mode == 'model':
        root['prediction'] = preds_dict[root['schedule']]
    candidates = [root]
    bests = [root]
    while len(candidates)!=0:
        new_candidates = []
        for candidate in candidates:
            new_candidates.extend(candidate['children'])
        if len(new_candidates)>0:    
            if eval_mode == 'model':
                for new_candidate in new_candidates: # sort candidates in both cases (with and without enforcement)
                    new_candidate['prediction'] = preds_dict[new_candidate['schedule']]
                new_candidates = sorted(new_candidates, key = lambda x: x['prediction'],reverse=True)
                if new_candidates[0]['level_has_enforcement']: # if some enforcement is set at this level 
                    candidates= [candidate for candidate in new_candidates if candidate['enforced']] # take the enforced childs only
                else: # if no enforcement, take BS best
                    candidates = new_candidates[:beam_size]
            elif eval_mode == 'execution':
                new_candidates = sorted(new_candidates, key = lambda x: x['evaluation'])
                candidates = new_candidates[:beam_size]

            bests.append(new_candidates[0])
        else:
            candidates= new_candidates #empty list
        
    if eval_mode == 'model':
        return max(bests, key = lambda x: x['prediction'])
    
    elif eval_mode == 'execution':
        return min(bests, key = lambda x: x['evaluation'])
        

def simulate_BeamSearch_on_Dataset(dataset, predictions_df, enforced_scheds_df,true_beam_search=False, get='speedups'):
    # I added the ground truth ^^
    if true_beam_search:
        bs_func = simulate_TrueBeamSearch
    else:
        bs_func = simulate_BeamSearch
    assert get in ['speedups', 'schedules']
        
    df = pd.DataFrame(columns = ['name','nb_scheds','base_time', 'eval_mode']+['bs='+str(i) for i in range(1,10)])
    for func_name in tqdm(sorted(list(predictions_df['name'].unique()),reverse=True)):
        
        func_dict = dataset.programs_dict[func_name]

        nb_scheds = len(func_dict['schedules_list'])
        init_exec_time = func_dict['initial_execution_time']
        root = update_exploration_trace(func_dict['exploration_trace'], predictions_df.query('name==@func_name'), enforced_scheds_df.query('name==@func_name'))
        root['depth'] = 0
#         best_candidate = simulate_BeamSearch(root,9999999,eval_mode='execution')
#         best_sp = round(root['evaluation']/best_candidate['evaluation'],2)
#         best_sched = best_candidate['schedule']
        sp_per_bs = dict()
        predictions_dict = predictions_df.query('name==@func_name')[['sched_str','prediction']].set_index('sched_str').to_dict()['prediction']    
#         print(func_name)
        for eval_mode in ['execution','model', 'level1']:
            sp_per_bs[eval_mode]=[]
            for i in range(1,10): #10 = max beam size
                if get=='schedules':
                    sp_per_bs[eval_mode].append(bs_func(root,i,preds_dict=predictions_dict,eval_mode=eval_mode)['schedule'])
                else:
                    sp_per_bs[eval_mode].append(round(root['evaluation']/bs_func(root,i,preds_dict=predictions_dict,eval_mode=eval_mode)['evaluation'],2)) ###  why is it evaluation ??

        df.loc[len(df)] = [func_name, str(nb_scheds), str(init_exec_time), 'execution'] + [i for i in sp_per_bs['execution']]
#         df.loc[len(df)] = [func_name, str(nb_scheds), str(init_exec_time), 'model'] + [i for i in sp_per_bs['model']]
        df.loc[len(df)] = [func_name, str(nb_scheds), str(init_exec_time), 'level1'] + [i for i in sp_per_bs['level1']]
    return df

def get_search_performance(dataset, predictions_df, enforced_scheds_df, true_beam_search=False, tira=True):
    if true_beam_search:
        bs_func = simulate_TrueBeamSearch
    else:
        bs_func = simulate_BeamSearch
        
    df = pd.DataFrame(columns = ['name','nb_scheds','base_time','tira']+['bs='+str(i) for i in range(1,10)])
    for func_name in tqdm(sorted(list(predictions_df['name'].unique()),reverse=True)):
        
        func_dict = dataset.programs_dict[func_name]

        nb_scheds = len(func_dict['schedules_list'])
        init_exec_time = func_dict['initial_execution_time']
        root = update_exploration_trace(func_dict['exploration_trace'], predictions_df.query('name==@func_name'), enforced_scheds_df.query('name==@func_name'))
        root['depth'] = 0
#         best_candidate = simulate_BeamSearch(root,9999999,eval_mode='execution')
#         best_sp = round(root['evaluation']/best_candidate['evaluation'],2)
#         best_sched = best_candidate['schedule']
        sp_per_bs = dict()
        predictions_dict = predictions_df.query('name==@func_name')[['sched_str','prediction']].set_index('sched_str').to_dict()['prediction']    
        for eval_mode in ['execution','model', 'level1']:
            sp_per_bs[eval_mode]=[]
            for i in range(1,10):
                sp_per_bs[eval_mode].append((round(root['evaluation']/bs_func(root,i,preds_dict=predictions_dict,eval_mode=eval_mode)['evaluation'],2)))
                
#         only difference with function with schedules
        if tira==True:
            df.loc[len(df)] = [func_name, str(nb_scheds), str(init_exec_time), 'True'] + [sp_per_bs['model'][i]/sp_per_bs['execution'][i]*100 for i in range(len(sp_per_bs['execution']))]
        else:
            df.loc[len(df)] = [func_name, str(nb_scheds), str(init_exec_time), 'False'] + [sp_per_bs['level1'][i]/sp_per_bs['execution'][i]*100 for i in range(len(sp_per_bs['execution']))]
    return df

#not_used_anymore
def encode_df_interchage(row, pred = True): #encode the output vector to get '1' in the right LI, '0' elsewhere
    if pred == True:
        str_sched = row["pred_str"]
    else:
        str_sched = row["target_str"]
    output = np.zeros(one_output,dtype=int) 
    str_interchange = re.findall('I\(.*\)', str_sched)  # get the LI string
    if str_interchange==[]:
        output[0] = 1
    else:
        m = re.findall(r'\d+', str_interchange[0])  # get the number of the loops
        a = min(int(m[-2]),int(m[-1]))
        b = max(int(m[-2]),int(m[-1]))
        i2 = b-a
        i1 = pos(a,depth_max)
        output[i1+i2]=1      
    return output

#not_used_anymore
def exist_merged(row, names_merged): #encode the output vector to get '1' in the right LI, '0' elsewhere
    return (row["name"] in names_merged)


#################################### Tiramisu Cost Model helper functions ###############################
#for cost model

class LoopsDepthException(Exception):
    pass

def drop_program(prog_dict):   
    if len(prog_dict['schedules_list'])<2:
        return True
    if has_skippable_loop_1comp(prog_dict):
        return True
    if 'node_name' in prog_dict and prog_dict['node_name']=='lanka24': # drop if we the program is run by lanka24 (because its measurements are inacurate)
        return True
    return False   

def drop_schedule(prog_dict, schedule_index):
    schedule_json =  prog_dict['schedules_list'][schedule_index]
    schedule_str = sched_json_to_sched_str(schedule_json)
    program_depth = len(prog_dict['program_annotation']['iterators'])
    if (not schedule_json['execution_times']) or min(schedule_json['execution_times'])<0: # exec time is set to -1 on datapoints that are deemed noisy, or if list empty
        return True
    if len(prog_dict['program_annotation']['computations'])==1: #this function works only on single comp programs
        if sched_is_prunable_1comp(schedule_str,program_depth):
            return True 
    
    if len(schedule_json['execution_times'])==1:
        total_def_eval+=1
        if 'function760518'<prog_dict['filename'][2:16]<'function761289': 
            def_eval_in_range+=1
        if total_def_eval%10==0:
            print(total_def_eval,def_eval_in_range)
        return True
    if wrongly_pruned_schedule(prog_dict, schedule_index):
        return True

    return False

def default_eval(prog_dict, schedule_index):
    schedule_json =  prog_dict['schedules_list'][schedule_index]
    schedule_str = sched_json_to_sched_str(schedule_json)
    program_depth = len(prog_dict['program_annotation']['iterators'])
    if len(prog_dict['program_annotation']['computations'])==1: #this function works only on single comp programs
        return can_set_default_eval_1comp(schedule_str,program_depth)
    else:
        return 0
    

def mape_criterion(inputs, targets):
    eps = 1e-5
    return 100*torch.mean(torch.abs(targets - inputs)/(targets+eps))

def get_tree_footprint_CM(tree):
    footprint='<L'+str(int(tree['loop_index']))+'>'
    if tree['has_comps']:
        footprint+='['
        for idx in tree['computations_indices']:
            footprint+='C'+str(int(idx))
        footprint+=']'
    for child in tree['child_list']:
        footprint+= get_tree_footprint_CM(child)
    footprint+='</L'+str(int(tree['loop_index']))+'>'
    return footprint

class Model_Recursive_LSTM_v2_CM(nn.Module):
    def __init__(self, input_size, comp_embed_layer_sizes=[600, 350, 200, 180], drops=[0.225, 0.225, 0.225, 0.225], output_size=1):
        super().__init__()
        embedding_size = comp_embed_layer_sizes[-1]
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [embedding_size*2+20] + comp_embed_layer_sizes[-2:]
        comp_embed_layer_sizes = [input_size] + comp_embed_layer_sizes
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts= nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts= nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts= nn.ModuleList()
        for i in range(len(comp_embed_layer_sizes)-1):
            self.comp_embedding_layers.append(nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i+1], bias=True))
#             nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            nn.init.zeros_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(regression_layer_sizes)-1):
            self.regression_layers.append(nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i+1], bias=True))
#             nn.init.xavier_uniform_(self.regression_layers[i].weight)
            nn.init.zeros_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(concat_layer_sizes)-1):
            self.concat_layers.append(nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i+1], bias=True))
#             nn.init.xavier_uniform_(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
#         nn.init.xavier_uniform_(self.predict.weight)
        nn.init.zeros_(self.predict.weight)
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


# In[3]:


def load_data_CM(train_val_dataset_file, split_ratio=None, max_batch_size=2048, drop_sched_func=None, drop_prog_func=None, default_eval=None, speedups_clip_func=None):
    print("loading batches from: "+train_val_dataset_file)
    dataset = Dataset_CM(train_val_dataset_file, max_batch_size, drop_sched_func, drop_prog_func, default_eval, speedups_clip_func)
    if split_ratio == None:
        split_ratio=0.2
    if split_ratio > 1 : # not a ratio a number of batches
        validation_size = split_ratio
    else:
        validation_size = int(split_ratio * len(dataset))
    indices = list(range(len(dataset)))
#     random.Random(42).shuffle(indices)
    val_batches_indices, train_batches_indices = indices[:validation_size],                                               indices[validation_size:]
    val_batches_list=[]
    train_batches_list=[]
    for i in val_batches_indices:
        val_batches_list.append(dataset[i])
    for i in train_batches_indices:
        train_batches_list.append(dataset[i])
    print("Data loaded")
    print("Sizes: "+str((len(val_batches_list),len(train_batches_list)))+" batches")
    return dataset, val_batches_list, val_batches_indices, train_batches_list, train_batches_indices


def get_representation_template_CM(program_dict, max_depth):
    max_accesses = 15
    min_accesses = 1
#     max_depth = 5 
    
    comps_repr_templates_list = []
    comps_indices_dict = dict()
    comps_placeholders_indices_dict = dict()
    
    program_json = program_dict['program_annotation']
    computations_dict = program_json['computations']
    ordered_comp_list = sorted(list(computations_dict.keys()), key = lambda x: computations_dict[x]['absolute_order'])
    
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        if len(comp_dict['accesses'])>max_accesses:
            raise NbAccessException
        if len(comp_dict['accesses'])<min_accesses:
            raise NbAccessException
        if len(comp_dict['iterators'])>max_depth:
            raise LoopsDepthException
            
        comp_repr_template = []
        # Is this computation a reduction 
        comp_repr_template.append(+comp_dict['comp_is_reduction'])


#         iterators representation + tiling and interchage
        iterators_repr = []
        for iter_i,iterator_name in enumerate(comp_dict['iterators']):
            iterator_dict = program_json['iterators'][iterator_name]
            iterators_repr.extend([iterator_dict['lower_bound'], iterator_dict['upper_bound']])
            
            # transformations placeholders
            c_code = 'C'+str(comp_index)
            l_code= c_code+'-L'+str(iter_i)
            iterators_repr.extend([l_code+'Parallelized',
                                   l_code+'Tiled', l_code+'TileFactor',
                                   l_code+'Fused']) #unrolling is skipped since it is only applied on innermost loop

        # Adding padding
        iterator_repr_size = int(len(iterators_repr)/len(comp_dict['iterators']))
        iterators_repr.extend([0]*iterator_repr_size*(max_depth-len(comp_dict['iterators']))) # adding iterators padding 

        # Adding unrolling placeholder since unrolling can only be applied to the innermost loop 
        iterators_repr.extend([c_code+'-Unrolled', c_code+'-UnrollFactor'])
        
        # Adding transformation matrix place holder
        iterators_repr.append(c_code+'-TransformationMatrixStart')
        iterators_repr.extend(['M']*((max_depth+1)**2-2))
        iterators_repr.append(c_code+'-TransformationMatrixEnd')
    
        # Adding the iterators representation to computation vector
        comp_repr_template.extend(iterators_repr)     

        #  Write access representation to computation vector
        padded_write_matrix = pad_access_matrix(isl_to_write_matrix(comp_dict['write_access_relation']), max_depth)
        write_access_repr = [comp_dict['write_buffer_id']+1] + padded_write_matrix.flatten().tolist() # buffer_id + flattened access matrix 
        
        # Adding write access representation to computation vector
        comp_repr_template.extend(write_access_repr)

        # Read Access representation 
        read_accesses_repr=[]
        for read_access_dict in comp_dict['accesses']:
            read_access_matrix = pad_access_matrix(read_access_dict['access_matrix'], max_depth)
            read_access_repr = [+read_access_dict['access_is_reduction']]+ [read_access_dict['buffer_id']+1] + read_access_matrix.flatten().tolist() # buffer_id + flattened access matrix 
            read_accesses_repr.extend(read_access_repr)

        access_repr_len = (max_depth+1)*(max_depth + 2) + 1 +1 # access matrix size +1 for buffer id +1 for is_access_reduction
        read_accesses_repr.extend([0]*access_repr_len*(max_accesses-len(comp_dict['accesses']))) #adding accesses padding

    
        comp_repr_template.extend(read_accesses_repr)

        # Adding Operations count to computation vector
        comp_repr_template.append(comp_dict['number_of_additions'])
        comp_repr_template.append(comp_dict['number_of_subtraction'])
        comp_repr_template.append(comp_dict['number_of_multiplication'])
        comp_repr_template.append(comp_dict['number_of_division'])
        

        # adding log(x+1) of the representation
#         log_rep = list(np.log1p(comp_representation))
#         comp_representation.extend(log_rep)
        
        comps_repr_templates_list.append(comp_repr_template)
        comps_indices_dict[comp_name] = comp_index
        for j, element in enumerate(comp_repr_template):
            if isinstance(element, str):
                comps_placeholders_indices_dict[element] = (comp_index,j)
    

        
    #building loop representation template
    
    loops_repr_templates_list = []
    loops_indices_dict = dict()
    loops_placeholders_indices_dict = dict()
#     assert len(program_json['iterators'])==len(set(program_json['iterators'])) #just to make sure that loop names are not duplicates, but this can't happen because it's a dict
    for loop_index, loop_name in enumerate(program_json['iterators']): # !! is the order in this list fix? can't we get new indices during schedule repr !!! should we use loop name in plchldrs instead of index ? !! #Edit: now it's using the name, so this issue shouldn't occure
        loop_repr_template=[]
        l_code = 'L'+loop_name
        # upper and lower bound
        loop_repr_template.extend([program_json['iterators'][loop_name]['lower_bound'],program_json['iterators'][loop_name]['upper_bound']])   
        loop_repr_template.extend([l_code+'Parallelized',
                                   l_code+'Tiled', l_code+'TileFactor',
                                   l_code+'Fused',
                                   l_code+'Unrolled', l_code+'UnrollFactor'])
        loop_repr_template.extend([l_code+'TransfMatRowStart']+['M']*(max_depth-2+1)+[l_code+'TransfMatRowEnd']) #+1 for the frame
        loop_repr_template.extend([l_code+'TransfMatColStart']+['M']*(max_depth-2+1)+[l_code+'TransfMatColEnd'])
        # adding log(x+1) of the loop representation
        loops_repr_templates_list.append(loop_repr_template)    
        loops_indices_dict[loop_name]=loop_index
        
        for j, element in enumerate(loop_repr_template):
            if isinstance(element, str):
                loops_placeholders_indices_dict[element] = (loop_index,j)
    
            
     
    def update_tree_atributes(node):     
        node['loop_index'] = torch.tensor(loops_indices_dict[node['loop_name']]).to(train_device)
        if node['computations_list']!=[]:
            node['computations_indices'] = torch.tensor([comps_indices_dict[comp_name] for comp_name in node['computations_list']]).to(train_device)
            node['has_comps'] = True
        else:
            node['has_comps'] = False
        for child_node in node['child_list']:
            update_tree_atributes(child_node)
        return node
    
    # getting the original tree structure 
    no_sched_json = program_dict['schedules_list'][0]
    assert 'fusions' not in no_sched_json or no_sched_json['fusions']==None
    orig_tree_structure = no_sched_json['tree_structure']
    tree_annotation = copy.deepcopy(orig_tree_structure) #to avoid altering the original tree from the json
    prog_tree = update_tree_atributes(tree_annotation) 
    
#     loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr_templates_list),0)#.to(device)
#     computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr_templates_list),0)#.to(device)     

    return prog_tree, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict


def get_schedule_representation_CM(program_json, schedule_json, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict, max_depth):

    comps_repr = copy.deepcopy(comps_repr_templates_list)
    loops_repr = copy.deepcopy(loops_repr_templates_list)
    
    computations_dict = program_json['computations']
    ordered_comp_list = sorted(list(computations_dict.keys()), key = lambda x: computations_dict[x]['absolute_order'])
    
    padded_tranf_mat_per_comp = dict()
    
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict =  program_json['computations'][comp_name]
        comp_schedule_dict=schedule_json[comp_name]
        c_code = 'C'+str(comp_index)
        
        
        #Fusion representation
        fused_levels = []
        if 'fusions' in schedule_json and schedule_json['fusions']:
            for fusion in schedule_json['fusions']:#check if comp is involved in fusions 
                 # fusions format [compname1, compname2, loop depth]
                if comp_name in fusion:
                    fused_levels.append(fusion[2])
                
            
        for iter_i,iterator_name in enumerate(comp_dict['iterators']):
            
            ### Updating the computations representation template 
            l_code= c_code+'-L'+str(iter_i)
            
             # Parallelization representation
            parallelized = 0
            if iterator_name == comp_schedule_dict['parallelized_dim']:
                parallelized = 1 # parallelized true
            p_index = comps_placeholders_indices_dict[l_code+'Parallelized']
            comps_repr[p_index[0]][p_index[1]]=parallelized
            
            # Tiling representation 
            tiled = 0
            tile_factor = 0
            if comp_schedule_dict['tiling'] and (iterator_name in comp_schedule_dict['tiling']['tiling_dims']):
                tiled = 1 #tiled: true
                tile_factor_index = comp_schedule_dict['tiling']['tiling_dims'].index(iterator_name)
                tile_factor = int(comp_schedule_dict['tiling']['tiling_factors'][tile_factor_index]) #tile factor
            p_index = comps_placeholders_indices_dict[l_code+'Tiled']
            comps_repr[p_index[0]][p_index[1]] = tiled
            p_index = comps_placeholders_indices_dict[l_code+'TileFactor']
            comps_repr[p_index[0]][p_index[1]] = tile_factor
            
            # Fusion representation
            fused = 0
            if iter_i in fused_levels:
                fused=1
            p_index = comps_placeholders_indices_dict[l_code+'Fused']
            comps_repr[p_index[0]][p_index[1]] = fused
            

         # Unrolling Representation 
        unrolled = 0
        unroll_factor = 0
        if comp_schedule_dict['unrolling_factor']: #Unrolling is always aplied to the innermost loop 
            unrolled=1 #unrolled True
            unroll_factor = int(comp_schedule_dict['unrolling_factor']) #unroll factor
        p_index = comps_placeholders_indices_dict[c_code+'-Unrolled']
        comps_repr[p_index[0]][p_index[1]] = unrolled
        p_index = comps_placeholders_indices_dict[c_code+'-UnrollFactor']
        comps_repr[p_index[0]][p_index[1]] = unroll_factor
        
        # Adding the transformation matrix
        # get the matrix start and end indices 
        mat_start = comps_placeholders_indices_dict[c_code+'-TransformationMatrixStart']
        mat_end = comps_placeholders_indices_dict[c_code+'-TransformationMatrixEnd']
        nb_mat_elements = mat_end[1] - mat_start[1] + 1
        max_depth = int(np.sqrt(nb_mat_elements))-1 # temporarily hack to get max_depth to use it in padding
        padded_matrix = get_padded_transformation_matrix(program_json, schedule_json, comp_name, max_depth)
    #     print(nb_mat_elements, padded_matrix, max_depth)
        assert len(padded_matrix.flatten().tolist()) == nb_mat_elements
    #     print(nb_mat_elements)
        comps_repr[mat_start[0]][mat_start[1]:mat_end[1]+1] = padded_matrix.flatten().tolist() 
        
        padded_tranf_mat_per_comp[comp_name] = padded_matrix #saving it for later to be used in loop repr
        
#     # transforming the schedule_json in order to have loops as key instead of computations, this dict helps building the loop vectors
    loop_schedules_dict = dict()
    for loop_name in program_json['iterators']:
        loop_schedules_dict[loop_name]=dict()
        loop_schedules_dict[loop_name]['TransformationMatrixCol']=[]
        loop_schedules_dict[loop_name]['TransformationMatrixRow']=[]
        loop_schedules_dict[loop_name]['tiled']=0
        loop_schedules_dict[loop_name]['tile_factor']=0
        loop_schedules_dict[loop_name]['unrolled']=0
        loop_schedules_dict[loop_name]['unroll_factor']=0
        loop_schedules_dict[loop_name]['parallelized']=0
        loop_schedules_dict[loop_name]['fused']=0
        
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_schedule_dict = schedule_json[comp_name]
        if comp_schedule_dict['tiling']:
            for tiled_loop_index,tiled_loop in enumerate(comp_schedule_dict['tiling']['tiling_dims']):
                loop_schedules_dict[tiled_loop]['tiled']=1
                assert loop_schedules_dict[tiled_loop]['tile_factor']==0 or loop_schedules_dict[tiled_loop]['tile_factor']==int(comp_schedule_dict['tiling']['tiling_factors'][tiled_loop_index]) #just checking that it hasn't been updated with a different value
                loop_schedules_dict[tiled_loop]['tile_factor']=int(comp_schedule_dict['tiling']['tiling_factors'][tiled_loop_index])
        if comp_schedule_dict['unrolling_factor']:
            comp_innermost_loop=computations_dict[comp_name]['iterators'][-1] 
            loop_schedules_dict[comp_innermost_loop]['unrolled']=1
            assert loop_schedules_dict[comp_innermost_loop]['unroll_factor']==0 or loop_schedules_dict[comp_innermost_loop]['unroll_factor']==int(comp_schedule_dict['unrolling_factor'])  #just checking that it hasn't been updated with a different value
            loop_schedules_dict[comp_innermost_loop]['unroll_factor']=int(comp_schedule_dict['unrolling_factor'])
        if comp_schedule_dict['parallelized_dim']:
            loop_schedules_dict[comp_schedule_dict['parallelized_dim']]['parallelized'] = 1
        
        # get the rows and cols transformation matrix for each iterator
        assert padded_tranf_mat_per_comp[comp_name].shape == (max_depth+1,max_depth+1) # make sure that the padding frame is applied, otherwise need to remove the +1 from iter_i+1 in the next few lines 
        for iter_i, loop_name in enumerate(computations_dict[comp_name]['iterators']):
            if len(loop_schedules_dict[loop_name]['TransformationMatrixCol'])>0:#if not empty
                assert (loop_schedules_dict[loop_name]['TransformationMatrixCol'] == padded_tranf_mat_per_comp[comp_name][:,iter_i+1]).all() #chck if the iterator what affected by a different matrix, that shouldn't happen
            else:
                loop_schedules_dict[loop_name]['TransformationMatrixCol'] = padded_tranf_mat_per_comp[comp_name][:,iter_i+1] #+1 for the padding frame
            if len(loop_schedules_dict[loop_name]['TransformationMatrixRow'])>0:#if not empty
                assert (loop_schedules_dict[loop_name]['TransformationMatrixRow'] == padded_tranf_mat_per_comp[comp_name][iter_i+1,:]).all() #chck if the iterator what affected by a different matrix, that shouldn't happen
            else:
                loop_schedules_dict[loop_name]['TransformationMatrixRow'] = padded_tranf_mat_per_comp[comp_name][iter_i+1,:]#+1 for the padding frame
    
    #update the fusions in loops dict 
    if 'fusions' in schedule_json and schedule_json['fusions']:
        for fusion in schedule_json['fusions']:
            fused_loop1 = computations_dict[fusion[0]]['iterators'][fusion[2]]
            fused_loop2 = computations_dict[fusion[1]]['iterators'][fusion[2]]
            loop_schedules_dict[fused_loop1]['fused']=1
            loop_schedules_dict[fused_loop2]['fused']=1
        
# Updating the loop representation templates
    for loop_name in program_json['iterators']:
        l_code = 'L'+loop_name
        
        p_index = loops_placeholders_indices_dict[l_code+'Parallelized']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['parallelized']
        
        p_index = loops_placeholders_indices_dict[l_code+'Tiled']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['tiled']
        p_index = loops_placeholders_indices_dict[l_code+'TileFactor']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['tile_factor']
        
        p_index = loops_placeholders_indices_dict[l_code+'Unrolled']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['unrolled']
        p_index = loops_placeholders_indices_dict[l_code+'UnrollFactor']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['unroll_factor']
        
        p_index = loops_placeholders_indices_dict[l_code+'Fused']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['fused']
        
        row_start = loops_placeholders_indices_dict[l_code+'TransfMatRowStart']
        row_end = loops_placeholders_indices_dict[l_code+'TransfMatRowEnd']
        nb_row_elements = row_end[1] - row_start[1] + 1
        assert len(loop_schedules_dict[loop_name]['TransformationMatrixRow']) == nb_row_elements
        loops_repr[row_start[0]][row_start[1]:row_end[1]+1] = loop_schedules_dict[loop_name]['TransformationMatrixRow']
        
        col_start = loops_placeholders_indices_dict[l_code+'TransfMatColStart']
        col_end = loops_placeholders_indices_dict[l_code+'TransfMatColEnd']
        nb_col_elements = col_end[1] - col_start[1] + 1
        assert len(loop_schedules_dict[loop_name]['TransformationMatrixCol']) == nb_col_elements
        loops_repr[col_start[0]][col_start[1]:col_end[1]+1] = loop_schedules_dict[loop_name]['TransformationMatrixCol']
    
    loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr),0)#.to(device)
    computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr),0)#.to(device)     

    return computations_tensor, loops_tensor


global_dioph_sols_dict = dict()
def get_padded_transformation_matrix(program_json, schedule_json, comp_name, max_depth=None):
    comp_name = list(program_json['computations'].keys())[0] # for single comp programs, there is only one computation
    comp_dict =  program_json['computations'][comp_name]
    comp_schedule_dict=schedule_json[comp_name]
    nb_iterators = len(comp_dict['iterators'])
    loop_nest = comp_dict['iterators'][:]
    
    if 'transformation_matrix' in comp_schedule_dict: # if the program is explored using matrices
        if comp_schedule_dict['transformation_matrix']!=[]: #if matrix applied, else set it to identity
            assert np.sqrt(len(comp_schedule_dict['transformation_matrix']))==nb_iterators
            final_mat = np.array(list(map(int,comp_schedule_dict['transformation_matrix']))).reshape(nb_iterators,nb_iterators)
        else:
            final_mat = np.zeros((nb_iterators,nb_iterators),int)
            np.fill_diagonal(final_mat,1)
        # just for checking
        comparison_matrix = np.zeros((nb_iterators,nb_iterators),int)
        np.fill_diagonal(comparison_matrix,1)
        for mat in comp_schedule_dict['transformation_matrices'][::-1]:
            comparison_matrix = comparison_matrix@np.array(list(map(int,mat))).reshape(nb_iterators,nb_iterators)
        assert (comparison_matrix==final_mat).all()
    else: # if the program is explored using tags
        interchange_matrix = np.zeros((nb_iterators,nb_iterators),int)
        np.fill_diagonal(interchange_matrix,1)
        if comp_schedule_dict['interchange_dims']:
            first_iter_index = loop_nest.index(comp_schedule_dict['interchange_dims'][0])
            second_iter_index = loop_nest.index(comp_schedule_dict['interchange_dims'][1])
            interchange_matrix[first_iter_index,first_iter_index]=0 #zeroing the diagonal elements
            interchange_matrix[second_iter_index,second_iter_index]=0 #zeroing the diagonal elements
            interchange_matrix[first_iter_index, second_iter_index]=1
            interchange_matrix[second_iter_index, first_iter_index]=1
            loop_nest[first_iter_index], loop_nest[second_iter_index] = loop_nest[second_iter_index], loop_nest[first_iter_index] # swapping iterators in loop nest

        skewing_matrix = np.zeros((nb_iterators,nb_iterators),int)
        np.fill_diagonal(skewing_matrix,1)
        if comp_schedule_dict['skewing']:
            first_iter_index = loop_nest.index(comp_schedule_dict['skewing']['skewed_dims'][0])
            second_iter_index = loop_nest.index(comp_schedule_dict['skewing']['skewed_dims'][1])
            first_factor = int(comp_schedule_dict['skewing']['skewing_factors'][0])
            second_factor = int(comp_schedule_dict['skewing']['skewing_factors'][1])
            # the skewing sub matrix should be in the form of 
            # [[fact1, fact2],
            #  [a,   , b    ]]
            # and we need to find a and b to make to matix det==1
    #         a, b = symbols('a b')
    #         sol = diophantine(first_factor*b - second_factor*a - 1) # solve the diophantine equation to keep a determinant of 1 in the matrix, 
    #         a, b = list(sol)[0] # since we know that there should at least (or only?) one solution 
    #         free_symbol = list(a.free_symbols)[0] # since we know that there should be only one free symbol
    #         a = int(a.subs({free_symbol:0})) #substitue the free symbol with 0 to get the initial solution
    #         b = int(b.subs({free_symbol:0}))
#             sol = simple_linear_diophantine_r(first_factor,second_factor)
            if (first_factor,second_factor) in global_dioph_sols_dict:
                a, b = global_dioph_sols_dict[(first_factor,second_factor)]
            else: 
                a, b = linear_diophantine_default(first_factor,second_factor)
            skewing_matrix[first_iter_index,first_iter_index] = first_factor # update the matrix
            skewing_matrix[first_iter_index,second_iter_index] = second_factor
            skewing_matrix[second_iter_index,first_iter_index] = a
            skewing_matrix[second_iter_index,second_iter_index] = b

        #multiply the mats 
        final_mat = skewing_matrix@interchange_matrix # Right order is skew_mat * interchange_mat
    
    padded_mat = final_mat
    
    
    #pad matrix if max_depth defined
    if max_depth!=None:
        padded_mat = np.c_[np.ones(padded_mat.shape[0]), padded_mat] # adding tags for marking the used rows
        padded_mat = np.r_[[np.ones(padded_mat.shape[1])], padded_mat] # adding tags for marking the used columns
        padded_mat = np.pad(padded_mat, [(0,max_depth-nb_iterators),(0,max_depth-nb_iterators)], mode='constant', constant_values=0)
    
    return padded_mat

    
def get_datapoint_attributes(func_name, program_dict, schedule_index, tree_footprint):
    schedule_json = program_dict['schedules_list'][schedule_index]
    sched_id = str(schedule_index).zfill(4)
    sched_str = sched_json_to_sched_str_CM(schedule_json)
    exec_time = np.min(schedule_json['execution_times'])
    memory_use = program_dict['program_annotation']['memory_size']
    node_name = program_dict['node_name'] if 'node_name' in program_dict else 'unknown'
    speedup = program_dict['initial_execution_time']/exec_time 

    return (func_name, sched_id, sched_str, exec_time, memory_use, node_name, tree_footprint, speedup)

def sched_json_to_sched_str_CM(sched_json): 
    
    if 'sched_str' in sched_json:
        return sched_json['sched_str']
    
    orig_loop_nest = []
    orig_loop_nest.append(sched_json['tree_structure']['loop_name'])
    child_list = sched_json['tree_structure']['child_list']
    while len(child_list)>0:
        child_loop = child_list[0]
        orig_loop_nest.append(child_loop['loop_name'])
        child_list = child_loop['child_list']
        
    comp_name = [n for n in sched_json.keys() if not n in ['unfuse_iterators','tree_structure','execution_times']][0]
    schedule = sched_json[comp_name]
    transf_loop_nest = orig_loop_nest
    sched_str = ''
    
    if 'Transformation Matrix' in schedule:
        if schedule['Transformation Matrix']:
            sched_str+='M('+','.join(schedule['Transformation Matrix'])+')'
    elif "transformation_matrix" in schedule:
        if schedule['transformation_matrix']:
            sched_str+='M('+','.join(schedule['transformation_matrix'])+')'
    if schedule['interchange_dims']:
        first_dim_index = transf_loop_nest.index(schedule['interchange_dims'][0])
        second_dim_index = transf_loop_nest.index(schedule['interchange_dims'][1])
        sched_str+='I(L'+str(first_dim_index)+',L'+str(second_dim_index)+')'
        transf_loop_nest[first_dim_index], transf_loop_nest[second_dim_index] = transf_loop_nest[second_dim_index], transf_loop_nest[first_dim_index]
    if schedule['skewing']:
        first_dim_index = transf_loop_nest.index(schedule['skewing']['skewed_dims'][0])
        second_dim_index = transf_loop_nest.index(schedule['skewing']['skewed_dims'][1])
        first_factor = schedule['skewing']['skewing_factors'][0]
        second_factor = schedule['skewing']['skewing_factors'][1]
        sched_str+='S(L'+str(first_dim_index)+',L'+str(second_dim_index)+','+str(first_factor)+','+str(second_factor)+')'
    if schedule['parallelized_dim']:
        dim_index = transf_loop_nest.index(schedule['parallelized_dim'])
        sched_str+='P(L'+str(dim_index)+')'
    if schedule['tiling']:
        if schedule['tiling']['tiling_depth']==2:
            first_dim = schedule['tiling']['tiling_dims'][0]
            second_dim = schedule['tiling']['tiling_dims'][1]
            first_dim_index = transf_loop_nest.index(first_dim)
            second_dim_index = transf_loop_nest.index(second_dim)
            first_factor = schedule['tiling']['tiling_factors'][0]
            second_factor = schedule['tiling']['tiling_factors'][1]
            sched_str+='T2(L'+str(first_dim_index)+',L'+str(second_dim_index)+','+str(first_factor)+','+str(second_factor)+')'
            i = transf_loop_nest.index(first_dim)
            transf_loop_nest[i:i+1]=first_dim+'_outer', second_dim+'_outer'
            i = transf_loop_nest.index(second_dim)
            transf_loop_nest[i:i+1]=first_dim+'_inner', second_dim+'_inner'
        else: #tiling depth == 3
            first_dim = schedule['tiling']['tiling_dims'][0]
            second_dim = schedule['tiling']['tiling_dims'][1]
            third_dim = schedule['tiling']['tiling_dims'][2]
            first_dim_index = transf_loop_nest.index(first_dim)
            second_dim_index = transf_loop_nest.index(second_dim)
            third_dim_index = transf_loop_nest.index(third_dim)
            first_factor = schedule['tiling']['tiling_factors'][0]
            second_factor = schedule['tiling']['tiling_factors'][1]
            third_factor = schedule['tiling']['tiling_factors'][2]
            sched_str+='T3(L'+str(first_dim_index)+',L'+str(second_dim_index)+',L'+str(third_dim_index)+','+str(first_factor)+','+str(second_factor)+','+str(third_factor)+')'
            i = transf_loop_nest.index(first_dim)
            transf_loop_nest[i:i+1]=first_dim+'_outer', second_dim+'_outer', third_dim+'_outer'
            i = transf_loop_nest.index(second_dim)
            transf_loop_nest[i:i+1]=first_dim+'_inner', second_dim+'_inner', third_dim+'_inner'
            transf_loop_nest.remove(third_dim)
    if schedule['unrolling_factor']:
        dim_index = len(transf_loop_nest)-1
        dim_name =transf_loop_nest[-1]
        sched_str+='U(L'+str(dim_index)+','+schedule['unrolling_factor']+')'
        transf_loop_nest[dim_index:dim_index+1] = dim_name+'_Uouter', dim_name+'_Uinner'
    
    
    return sched_str
    
def pad_access_matrix(access_matrix, max_depth):
    access_matrix = np.array(access_matrix)
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix] # adding tags for marking the used rows
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix] # adding tags for marking the used columns
    padded_access_matrix = np.zeros((max_depth + 1, max_depth + 2))
    padded_access_matrix[:access_matrix.shape[0],:access_matrix.shape[1]-1] = access_matrix[:,:-1] #adding padding to the access matrix before the last column
    padded_access_matrix[:access_matrix.shape[0],-1] = access_matrix[:,-1] #appending the last columns
    
    return padded_access_matrix

def get_results_df_CM(dataset, batches_list, indices, model, log=False):   
    df = pd.DataFrame()
    model.eval()
    torch.set_grad_enabled(False)
    all_outputs=[]
    all_labels=[]
    prog_names=[]
    sched_names=[]
    exec_times=[]
    sched_strs=[]
    memory_uses=[]
    node_names=[]
    tree_footprints = []

    for k, (inputs, labels) in tqdm(list(enumerate(batches_list))):
        original_device = labels.device
        inputs=(inputs[0], inputs[1].to(train_device), inputs[2].to(train_device))
        labels=labels.to(train_device)
        outputs = model(inputs)
        assert outputs.shape == labels.shape
        all_outputs.append(outputs)
        all_labels.append(labels)
#         assert len(outputs)==len(dataset.batched_schedule_names[indices[k]])
#         assert len(outputs)==len(dataset.batched_program_names[indices[k]])
#         for j, sched_name in enumerate(dataset.batched_schedule_names[indices[k]]):
#             sched_names.append(sched_name)
#             prog_names.append(dataset.batched_program_names[indices[k]][j])
#             exec_times.append(dataset.batched_exec_time[indices[k]][j])
        assert len(outputs)==len(dataset.batched_datapoint_attributes[indices[k]])
        zipped_attributes = list(zip(*dataset.batched_datapoint_attributes[indices[k]]))
        prog_names.extend(zipped_attributes[0])
        sched_names.extend(zipped_attributes[1])
        sched_strs.extend(zipped_attributes[2])
        exec_times.extend(zipped_attributes[3])
        memory_uses.extend(zipped_attributes[4])
        node_names.extend(zipped_attributes[5])
        tree_footprints.extend(zipped_attributes[6])
        inputs=(inputs[0], inputs[1].to(original_device), inputs[2].to(original_device))
        labels=labels.to(original_device)
    preds = torch.cat(all_outputs)
    targets = torch.cat(all_labels)
    preds = preds.cpu().detach().numpy().reshape((-1,))
    preds = np.around(preds,decimals=6)
    targets = np.around(targets.cpu().detach().numpy().reshape((-1,)),decimals=6)
                                            
    assert preds.shape == targets.shape 
    df['name'] = prog_names
    df['tree_struct'] = tree_footprints
    df['sched_name'] = sched_names
    df['sched_str'] = sched_strs
    df['exec_time'] = exec_times
    df['memory_use'] = list(map(float,memory_uses))
    df['node_name'] = node_names
    df['prediction'] = np.array(preds)
    df['target'] = np.array(targets)
#     df['abs_diff'] = np.abs(preds - targets)
    df['APE'] = np.abs(df.target - df.prediction)/df.target * 100
    df['sched_str'] = df['sched_str'].apply(lambda x: simplify_sched_str(x))
        
    return df

def simplify_sched_str(sched_str): #checks if the the same matrix is applied multiple computations, then merge the M() parts into a single 
#     print('before ')
    if sched_str.count('M')==1:
        return sched_str
    comps = re.findall('C\d+', sched_str)
    comps = set(comps)
    
    mats = set(re.findall(r'M\({[\dC\,]+},([\d\,\-]+)',sched_str))
    comps_per_mat = {mat:[] for mat in mats}
    new_mats_str = ''
    for mat in comps_per_mat:
        for mat_part in re.findall('M\({[C\d\,]+},'+mat,sched_str):
            comps_per_mat[mat].extend(re.findall('C\d+',mat_part))
        new_mats_str+='M({'+','.join(sorted(comps_per_mat[mat]))+'},'+mat+')'
    return re.sub('(M\({[\dC\,]+},[\d\,\-]+\))+',new_mats_str,sched_str)


class Dataset_CM():
    def __init__(self, dataset_filename, max_batch_size, drop_sched_func=None, drop_prog_func=None, can_set_default_eval=None , speedups_clip_func=None):
        
        if dataset_filename.endswith('json'):
            with open(dataset_filename, 'r') as f:
                dataset_str= f.read()
            self.programs_dict=json.loads(dataset_str)
        elif dataset_filename.endswith('pkl'):
            with open(dataset_filename, 'rb') as f:
                self.programs_dict = pickle.load(f)
        
        self.batched_X = []
        self.batched_Y = []
        self.batches_dict=dict()
        self.max_depth = 5 #WAS 5, LINA CHANGED IT FOR TESTING MISSING FUNCTIONS
        self.nb_dropped = 0
        self.pgme = 0
        self.nb_pruned = 0
        self.dropped_funcs = []
        self.batched_datapoint_attributes = []
        self.nb_datapoints = 0

        if (drop_sched_func==None):
            drop_sched_func = lambda x,y : False
        if (drop_prog_func==None):
            drop_prog_func = lambda x : False
        if (speedups_clip_func==None):
            speedups_clip_func = lambda x : x
        if (can_set_default_eval==None):
            can_set_default_eval = lambda x,y:0
                
        functions_list = list(self.programs_dict.keys())
        random.Random(42).shuffle(functions_list)
        for function_name in tqdm(functions_list):
            if drop_prog_func(self.programs_dict[function_name]):
                self.nb_dropped += len(self.programs_dict[function_name]['schedules_list'])
                self.pgme += 1
                self.dropped_funcs.append(function_name)
                continue
                
            program_json = self.programs_dict[function_name]['program_annotation']
            
            try:
                prog_tree, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict = get_representation_template_CM(self.programs_dict[function_name], max_depth = self.max_depth)
            except (NbAccessException, LoopsDepthException):
                self.nb_dropped += len(self.programs_dict[function_name]['schedules_list'])
                self.pgme += 1
                continue
            program_exec_time = self.programs_dict[function_name]['initial_execution_time']
            tree_footprint=get_tree_footprint_CM(prog_tree)
            self.batches_dict[tree_footprint] = self.batches_dict.get(tree_footprint,{'tree':prog_tree,'comps_tensor_list':[],'loops_tensor_list':[],'datapoint_attributes_list':[],'speedups_list':[],'exec_time_list':[]})
            
            for schedule_index in range(len(self.programs_dict[function_name]['schedules_list'])):
                schedule_json = self.programs_dict[function_name]['schedules_list'][schedule_index]
                sched_exec_time = np.min(schedule_json['execution_times'])
                if drop_sched_func(self.programs_dict[function_name], schedule_index) or (not sched_exec_time):
                    self.nb_dropped +=1
                    self.nb_pruned +=1
                    continue
                
                sched_speedup = program_exec_time / sched_exec_time
                
                def_sp = can_set_default_eval(self.programs_dict[function_name], schedule_index)
                if def_sp>0:
                    sched_speedup = def_sp
                    
                sched_speedup = speedups_clip_func(sched_speedup)

                comps_tensor, loops_tensor = get_schedule_representation_CM(program_json, schedule_json, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict, self.max_depth)
                
                datapoint_attributes = get_datapoint_attributes(function_name, self.programs_dict[function_name], schedule_index, tree_footprint)
                
                self.batches_dict[tree_footprint]['comps_tensor_list'].append(comps_tensor)
                self.batches_dict[tree_footprint]['loops_tensor_list'].append(loops_tensor)
                self.batches_dict[tree_footprint]['datapoint_attributes_list'].append(datapoint_attributes)
                self.batches_dict[tree_footprint]['speedups_list'].append(sched_speedup)
                self.nb_datapoints+=1

        storing_device = store_device
        print('Batching ...', self.pgme)
        for tree_footprint in tqdm(self.batches_dict):
            
            #shuffling the lists inside each footprint to avoid having batches with very low program diversity
            zipped = list(zip(self.batches_dict[tree_footprint]['datapoint_attributes_list'],
                              self.batches_dict[tree_footprint]['comps_tensor_list'],
                              self.batches_dict[tree_footprint]['loops_tensor_list'],
                              self.batches_dict[tree_footprint]['speedups_list']))
            random.shuffle(zipped)
            self.batches_dict[tree_footprint]['datapoint_attributes_list'],self.batches_dict[tree_footprint]['comps_tensor_list'],self.batches_dict[tree_footprint]['loops_tensor_list'],self.batches_dict[tree_footprint]['speedups_list']=zip(*zipped)
            
            for chunk in range(0,len(self.batches_dict[tree_footprint]['speedups_list']),max_batch_size):
                if (storing_device.type=='cuda' and (torch.cuda.memory_allocated(storing_device.index)/torch.cuda.get_device_properties(storing_device.index).total_memory)>0.80):  # Check GPU memory in order to avoid Out of memory error
                    print('GPU memory on '+str(storing_device)+' nearly full, switching to CPU memory')
                    storing_device = torch.device('cpu')
                self.batched_datapoint_attributes.append(self.batches_dict[tree_footprint]['datapoint_attributes_list'][chunk:chunk+max_batch_size])
                self.batched_X.append((self.batches_dict[tree_footprint]['tree'],
                               torch.cat(self.batches_dict[tree_footprint]['comps_tensor_list'][chunk:chunk+max_batch_size], 0).to(storing_device),
                               torch.cat(self.batches_dict[tree_footprint]['loops_tensor_list'][chunk:chunk+max_batch_size], 0).to(storing_device)))
                self.batched_Y.append(torch.FloatTensor(self.batches_dict[tree_footprint]['speedups_list'][chunk:chunk+max_batch_size]).to(storing_device))
        
        #shuffling batches to avoid having the same footprint in consecutive batches
        zipped = list(zip(self.batched_X, self.batched_Y, self.batched_datapoint_attributes))
        random.shuffle(zipped)
        self.batched_X, self.batched_Y, self.batched_datapoint_attributes = zip(*zipped)
        
        print(f'Number of datapoints {self.nb_datapoints} Number of batches {len(self.batched_Y)}')
#         del self.programs_dict
        del self.batches_dict
        
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            return self.batched_X[index], self.batched_Y[index] 

    def __len__(self):
        return len(self.batched_Y)


def has_skippable_loop_1comp(prog_dict): # check if the program has a non time-step free iterator 
                                   # (has an iterator that is not used in accesses and the expression doesn't have reduction stentcils)
    
    program_json =  prog_dict['program_annotation']
    if not len(program_json['computations'])==1: #this function works only on single comp programs
        return False
    comp_name = list(program_json['computations'].keys())[0]
    comp_dict = program_json['computations'][comp_name]
    write_buffer_id = comp_dict['write_buffer_id']
    iterators = comp_dict['iterators']
    write_dims =  isl_to_write_dims(comp_dict['write_access_relation'])
    read_buffer_ids = [e['buffer_id'] for e in comp_dict['accesses']]
    
    
    if len(write_dims)==len(iterators): # if all loops used in write, no free loops
        # one special case of empty program
        if len(read_buffer_ids) == 1 and read_buffer_ids[0]==write_buffer_id and comp_dict['number_of_additions'] ==0 and comp_dict['number_of_subtraction'] ==0 and comp_dict['number_of_multiplication'] ==0 and comp_dict['number_of_division'] ==0: 
            return True
        return False
    
    if not write_buffer_id in read_buffer_ids: # if the calculation is clearly overwritten
        return True
    
    # find the simle reduction access
    found = False
    for access in comp_dict['accesses']:
        if access['buffer_id']==write_buffer_id and not access_is_stencil(access):
            found = True
            break
    if not found: # no simple reduction access is found, but we know that there is a reduction access in expression, so there is a skippable loop if the reduction is performed on last iterator, otherwise it's hardly skippable
        if write_dims[-1]!=iterators[-1]: # reduction is performed on the last iterator
            return True
    
    # find the non simple reduction accesses
    for access in comp_dict['accesses']:
        if access['buffer_id']==write_buffer_id and access_is_stencil(access): # a stencil access pattern is used
            return False
    
    # checking if there is a free loop (not used in write nor in read)
    read_dims_bools = []
    for access in comp_dict['accesses']: 
        read_dims_bools.append(np.any(access['access_matrix'], axis=0))
    read_dims_bools = np.any(read_dims_bools,axis=0)
    read_iterators = [iterators[i] for i, is_used in enumerate(read_dims_bools[:-1]) if is_used==True]
    used_iterators = set(write_dims+read_iterators)
    if len(used_iterators)==len(iterators): # all iterators are used in the computation
        return False
    
    if iterators[-1] in used_iterators: # the last iterator is not the dropped one, so the dropped loop shouldn't be skippable (knowing that there is a reduction access)
        if len(comp_dict['accesses'])>2:# has to have more than 2 accesses to make sure the loop isn't skippable, adding this condition for strictness
            return False
        
    return True

def sched_is_prunable_1comp(schedule_str, prog_depth):
    if re.search('P\(L2\)U\(L3,\d+\)', schedule_str):
        return True
    if prog_depth==2:
        if re.search('P\(L1\)(?:[^T]|$)', schedule_str):
            return True
    if prog_depth==3:
        if re.search('P\(L2\)(?:[^T]|$|T2\(L0,L1)', schedule_str):
            return True
    return False

def can_set_default_eval_1comp(schedule_str, prog_depth):
    def_sp = 0
#     print(schedule_str, type(schedule_str))
    if prog_depth==2:
        if re.search('P\(L1\)$', schedule_str):
            def_sp = 0.001
    if prog_depth==3:
        if re.search('P\(L2\)$', schedule_str):
            def_sp = 0.001
    return def_sp

def access_is_stencil(access):
    return np.any(access['access_matrix'], axis=0)[-1]

def linear_diophantine_default(f_i,f_j):
    found = False
    gamma = 0
    sigma = 1
    if ((f_j == 1) or (f_i == 1)):
        gamma = f_i - 1
        sigma = 1
    else:
        if((f_j == -1) and (f_i > 1)):
            gamma = 1
            sigma = 0       
        else:     
            i =0
            while((i < 100) and (not found)):     
                if (((sigma * f_i ) % abs(f_j)) ==  1):
                            found = True
                else:
                    sigma+=1
                    i+=1
            if(not found):
                print('Error cannof find solution to diophantine equation')
                return
            gamma = ((sigma * f_i) - 1 ) / f_j
    
    return gamma, sigma




def wrongly_pruned_schedule(prog_dict, schedule_index):
    schedule_dict = prog_dict['schedules_list'][schedule_index]
    if not "sched_str" in schedule_dict: # this function concerns multicomp progs only, if sched str not in annot it means that the prog is single comp
        return False 
    sched_str = schedule_dict["sched_str"]
#     if(schedule_dict["execution_times"] == None):
#         return True
    target = prog_dict['initial_execution_time'] / np.min(schedule_dict["execution_times"])
    depths = []
    for depth in prog_dict['program_annotation']['computations']:
        depths.append(len(prog_dict['program_annotation']['computations'][depth]['iterators']))
    if(not (target>0.0113 or target<0.0087)):
        reg_str = ""
        for j in reversed(range(len(depths))):
            for i in range(depths[j]-1):
                reg_str += ".*P\(\{(C[0-9],)*C" + str(j) + "(,C[0-9])*\},L"+ str(i) +"\)$|"
        reg_str= reg_str[:-1]
        if(re.search(reg_str, sched_str)):
#             if len(schedule_dict["execution_times"])==1:
            print(prog_dict['filename'][2:16], schedule_index, sched_str, len(schedule_dict["execution_times"]),'yes')
            
#                             if not 'function760518'<prog_dict['filename'][2:16]<'function761289':
#                 print(prog_dict['filename'][2:16], schedule_index, sched_str, len(schedule_dict["execution_times"]),'are you sure?')
#             else:
#                 print(prog_dict['filename'][2:16], schedule_index, sched_str, len(schedule_dict["execution_times"]),'yes')
#             assert 'function760518'<prog_dict['filename'][2:16]<'function761289' #since we know that the ranges of programs affected by this bug, we can make this assertion
            return True
        else:
            return False
    else:
        return False
   
