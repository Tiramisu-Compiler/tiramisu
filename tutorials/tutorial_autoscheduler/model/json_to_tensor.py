import numpy as np
import torch
import math
import copy

device = "cpu"

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
            # unfuse schedule replacing the low bound for testing transfer learning  
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
#             padded_access_matrix[:access_matrix.shape[0],:access_matrix.shape[1]] = access_matrix #adding padding to the access matrix
            padded_access_matrix[:access_matrix.shape[0],:access_matrix.shape[1]-1] = access_matrix[:,:-1] #adding padding to the access matrix
            padded_access_matrix[:access_matrix.shape[0],-1] = access_matrix[:,-1] #adding padding to the access matrix
            #access_repr = access_dict['comp_id'] +1 + padded_access_matrix.flatten() # input_id + flattened access matrix
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
    def update_tree_atributes(node):
        if node['computations_list']!=[]:
            node['computations_indices'] = torch.tensor([indices_dict[comp_name] for comp_name in node['computations_list']]).to(device)
            node['has_comps'] = 1
        else:
            node['has_comps'] = 0
        for child_node in node['child_list']:
            update_tree_atributes(child_node)
        return node
    
    tree_annotation = copy.deepcopy(schedule_json['tree_structure']) #to avoid altering the original tree from the json
    prog_tree = update_tree_atributes(tree_annotation) 
              
    program_tensor = torch.unsqueeze(torch.FloatTensor(program_representation),0).to(device)     

    return prog_tree, program_tensor
