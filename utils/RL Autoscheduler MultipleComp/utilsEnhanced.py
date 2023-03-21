import numpy as np
import re
import sys, os, subprocess
from pathlib import Path
from datetime import datetime
import re
import torch

train_device_name = 'cpu' # choose training/storing device, either 'cuda:X' or 'cpu'
store_device_name = 'cpu'

store_device = torch.device(store_device_name)
train_device = torch.device(train_device_name)

class LargeAccessMatices(Exception):
    pass
class NbAccessException(Exception):
    pass
class LoopsDepthException(Exception):
    pass
class TimeOutException(Exception):
    pass
class LoopExtentException(Exception):
    pass

def get_dataset(path):
    os.getcwd()
    print("***************************",os.getcwd())
    prog_list=os.listdir(path)
    return prog_list


def pad_access_matrix(access_matrix, max_depth):
    access_matrix = np.array(access_matrix)
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix] # adding tags for marking the used rows
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix] # adding tags for marking the used columns
    padded_access_matrix = np.zeros((max_depth + 1, max_depth + 2))
    padded_access_matrix[:access_matrix.shape[0],:access_matrix.shape[1]-1] = access_matrix[:,:-1] #adding padding to the access matrix before the last column
    padded_access_matrix[:access_matrix.shape[0],-1] = access_matrix[:,-1] #appending the last columns
    
    return padded_access_matrix


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

def sched_json_to_sched_str(sched_json, prog_it): # Works only for 1 comp programs

    orig_loop_nest = []
    orig_loop_nest.append(list(prog_it.keys())[0])
    child_list = prog_it[list(prog_it.keys())[0]]['child_iterators']
    while len(child_list)>0:
        child_loop = prog_it[child_list[0]]
        orig_loop_nest.append(child_list[0])
        child_list = child_loop['child_iterators']
        
    comp_name = [n for n in sched_json.keys() if not n in ['unfuse_iterators','tree_structure','execution_times']][0]
    schedule = sched_json[comp_name]
    transf_loop_nest = orig_loop_nest
    sched_str = ''
    
    if schedule['interchange_dims']:
        first_dim_index = transf_loop_nest.index(schedule['interchange_dims'][0])
        second_dim_index = transf_loop_nest.index(schedule['interchange_dims'][1])
        sched_str+='I(L'+str(first_dim_index)+',L'+str(second_dim_index)+')'
        transf_loop_nest[first_dim_index], transf_loop_nest[second_dim_index] = transf_loop_nest[second_dim_index], transf_loop_nest[first_dim_index]
    if schedule['skewing']['skewed_dims']:
        first_dim_index = transf_loop_nest.index(schedule['skewing']['skewed_dims'][0])
        second_dim_index = transf_loop_nest.index(schedule['skewing']['skewed_dims'][1])
        first_factor = schedule['skewing']['skewing_factors'][0]
        second_factor = schedule['skewing']['skewing_factors'][1]
        sched_str+='S(L'+str(first_dim_index)+',L'+str(second_dim_index)+','+str(first_factor)+','+str(second_factor)+')'
    if schedule['parallelized_dim']:
        dim_index = transf_loop_nest.index(schedule['parallelized_dim'])
        sched_str+='P(L'+str(dim_index)+')'
    if schedule['tiling']['tiling_dims']:
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
        sched_str+='U(L'+str(dim_index)+','+str(schedule['unrolling_factor'])+')'
        transf_loop_nest[dim_index:dim_index+1] = dim_name+'_Uouter', dim_name+'_Uinner'
    if schedule["reversed_dim"]:
        dim_index = transf_loop_nest.index(schedule["reversed_dim"])
        sched_str+='R(L'+str(dim_index)+')'
    
    return sched_str



def get_representation(program_annot):
    max_dims= 7
    max_depth=5
    max_accesses = 21 # TODO: check if 10 is enough
    program_representation = []
    indices_dict = dict()
    computations_dict = program_annot['computations']
    ordered_comp_list = sorted(list(computations_dict.keys()), key = lambda x: computations_dict[x]['absolute_order'])

    placeholders_comp = {}
    
    for index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        comp_representation = []
  
        iterators_repr = [] 
        for iter_i,iterator_name in enumerate(comp_dict['iterators']):
            iterator_dict = program_annot['iterators'][iterator_name]
            iterators_repr.extend([iterator_dict['lower_bound'], iterator_dict['upper_bound']])
        
            # transformations placeholders
            l_code='L'+iterator_name
            iterators_repr.extend([l_code+'Interchanged', 
                                l_code+'Skewed', l_code+'SkewFactor', 
                                l_code+'Parallelized',
                                l_code+'Tiled', l_code+'TileFactor',
                                l_code+'Reversed',
                                l_code+'Fused',
                                0, 
                                0,
                                l_code+"_1"+'Interchanged',
                                l_code+"_1"+'Skewed', l_code+"_1"+'SkewFactor',
                                l_code+"_1"+'Parallelized',
                                l_code+"_1"+'Tiled', l_code+"_1"+'TileFactor',
                                l_code+"_1"+'Reversed',
                                l_code+"_1"+'Fused']) #unrolling is skipped since it is added only once
        
        # Adding padding
        iterator_repr_size = int(len(iterators_repr)/(2*len(comp_dict['iterators'])))
        iterators_repr.extend([0]*iterator_repr_size*2*(max_depth-len(comp_dict['iterators']))) # adding iterators padding 

        # Adding unrolling placeholder since unrolling can only be applied to the innermost loop 
        iterators_repr.extend(['Unrolled', 'UnrollFactor'])
        
        # Adding the iterators representation to computation vector       
        comp_representation.extend(iterators_repr)
        
        #  Write access representation to computation vector
        padded_write_matrix = pad_access_matrix(isl_to_write_matrix(comp_dict['write_access_relation']), max_depth)
        write_access_repr = [comp_dict['write_buffer_id']+1] + padded_write_matrix.flatten().tolist() # buffer_id + flattened access matrix 

    #     print('write ', comp_dict['write_buffer_id']+1,'\n',padded_write_matrix)
        
        # Adding write access representation to computation vector
        comp_representation.extend(write_access_repr)
        
        # Read Access representation 
        read_accesses_repr=[]
        for read_access_dict in comp_dict['accesses']:
            read_access_matrix = pad_access_matrix(read_access_dict['access_matrix'], max_depth)
            read_access_repr = [read_access_dict['buffer_id']+1] + read_access_matrix.flatten().tolist() # buffer_id + flattened access matrix 
            read_accesses_repr.extend(read_access_repr)
    #         print('read ', read_access_dict['buffer_id']+1,'\n',read_access_matrix)

            
        access_repr_len = (max_depth+1)*(max_depth + 2) + 1 # access matrix size +1 for buffer id
        read_accesses_repr.extend([0]*access_repr_len*(max_accesses-len(comp_dict['accesses']))) #adding accesses padding


        # Adding read Accesses to the representation to computation vector
        comp_representation.extend(read_accesses_repr)
        
        # Adding Operations count to computation vector
        comp_representation.append(comp_dict['number_of_additions'])
        comp_representation.append(comp_dict['number_of_subtraction'])
        comp_representation.append(comp_dict['number_of_multiplication'])
        comp_representation.append(comp_dict['number_of_division'])

        #print("comp rep before placeholders", comp_representation)



        placeholders_indices_dict = {}
        for i, element in enumerate(comp_representation):
            if isinstance(element, str):
                placeholders_indices_dict[element] = i
                comp_representation[i]=0
        placeholders_comp[comp_name]= placeholders_indices_dict
        


        # adding log(x+1) of the representation
        # log_rep = list(np.log1p(comp_representation))
        # comp_representation.extend(log_rep)
        
        program_representation.append(comp_representation)
        indices_dict[comp_name] = index
    
    return program_representation, placeholders_comp, indices_dict

def get_representation_template(program_annot):
    print("in repr template")
    max_accesses = 15
    min_accesses = 1
    max_depth = 5 

    comp_name = list(program_annot['computations'].keys())[0] # for single comp programs, there is only one computation
    comp_dict = program_annot['computations'][comp_name] 
    
    if len(comp_dict['accesses'])>max_accesses:
        raise NbAccessException
    if len(comp_dict['accesses'])<min_accesses:
        raise NbAccessException
    if len(comp_dict['iterators'])>max_depth:
        raise LoopsDepthException

    
    comp_repr_template = []
#         iterators representation + transformations placeholders
    iterators_repr = []    
    for iter_i,iterator_name in enumerate(comp_dict['iterators']):
        iterator_dict = program_annot['iterators'][iterator_name]
        iterators_repr.extend([iterator_dict['lower_bound'], iterator_dict['upper_bound']])
      
        # transformations placeholders
        l_code='L'+iterator_name
        iterators_repr.extend([l_code+'Interchanged', 
                               l_code+'Skewed', l_code+'SkewFactor', 
                               l_code+'Parallelized',
                               l_code+'Tiled', l_code+'TileFactor',
                               l_code+'Reversed',
                               0, 
                               0,
                               l_code+"_1"+'Interchanged',
                               l_code+"_1"+'Skewed', l_code+"_1"+'SkewFactor',
                               l_code+"_1"+'Parallelized',
                               l_code+"_1"+'Tiled', l_code+'TileFactor',
                               l_code+"_1"+'Reversed']) #unrolling is skipped since it is added only once
    
    # Adding padding
    iterator_repr_size = int(len(iterators_repr)/(2*len(comp_dict['iterators'])))
    iterators_repr.extend([0]*iterator_repr_size*2*(max_depth-len(comp_dict['iterators']))) # adding iterators padding 

    # Adding unrolling placeholder since unrolling can only be applied to the innermost loop 
    iterators_repr.extend(['Unrolled', 'UnrollFactor'])
    
    # Adding the iterators representation to computation vector       
    comp_repr_template.extend(iterators_repr)
    
    #  Write access representation to computation vector
    padded_write_matrix = pad_access_matrix(isl_to_write_matrix(comp_dict['write_access_relation']), max_depth)
    write_access_repr = [comp_dict['write_buffer_id']+1] + padded_write_matrix.flatten().tolist() # buffer_id + flattened access matrix 

#     print('write ', comp_dict['write_buffer_id']+1,'\n',padded_write_matrix)
    
    # Adding write access representation to computation vector
    comp_repr_template.extend(write_access_repr)
    
    # Read Access representation 
    read_accesses_repr=[]
    for read_access_dict in comp_dict['accesses']:
        read_access_matrix = pad_access_matrix(read_access_dict['access_matrix'], max_depth)
        read_access_repr = [read_access_dict['buffer_id']+1] + read_access_matrix.flatten().tolist() # buffer_id + flattened access matrix 
        read_accesses_repr.extend(read_access_repr)
#         print('read ', read_access_dict['buffer_id']+1,'\n',read_access_matrix)

        
    access_repr_len = (max_depth+1)*(max_depth + 2) + 1 # access matrix size +1 for buffer id
    read_accesses_repr.extend([0]*access_repr_len*(max_accesses-len(comp_dict['accesses']))) #adding accesses padding


    # Adding read Accesses to the representation to computation vector
    comp_repr_template.extend(read_accesses_repr)
    
    # Adding Operations count to computation vector
    comp_repr_template.append(comp_dict['number_of_additions'])
    comp_repr_template.append(comp_dict['number_of_subtraction'])
    comp_repr_template.append(comp_dict['number_of_multiplication'])
    comp_repr_template.append(comp_dict['number_of_division'])
    
    # Track the indices to the placeholders in a a dict
    placeholders_indices_dict = {}
    for i, element in enumerate(comp_repr_template):
        if isinstance(element, str):
            placeholders_indices_dict[element] = i
            comp_repr_template[i]=0

    
    return comp_repr_template, placeholders_indices_dict

def get_orig_tree_struct(program_json,root_iterator):
    tree_struct = {'loop_name':root_iterator,'computations_list':program_json['iterators'][root_iterator]['computations_list'][:],'child_list':[]}
    for child_iterator in program_json['iterators'][root_iterator]['child_iterators']:
        tree_struct['child_list'].append(get_orig_tree_struct(program_json,child_iterator))
    return tree_struct

#c++ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 -o ${FILE_PATH}.o -c ${FILE_PATH};\
#c++ -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 ${FILE_PATH}.o -o ./${FILE_PATH}.out   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl' 


#${CXX} -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 -o ${FILE_PATH}.o -c ${FILE_PATH};\
#${CXX} -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 ${FILE_PATH}.o -o ./${FILE_PATH}.out   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl' 

def compile_and_run_tiramisu_code(file_path, log_message='No message'):
    print("inside compile and run")
    os.environ['FUNC_DIR'] = ('/'.join(Path(file_path).parts[:-1]) if len(Path(file_path).parts)>1 else '.') +'/'
    os.environ['FILE_PATH'] = file_path
    log_message_cmd = 'printf "'+log_message+'\n">> ${FUNC_DIR}log.txt'
    compile_tiramisu_cmd = 'printf "Compiling ${FILE_PATH}\n" >> ${FUNC_DIR}log.txt;\
    ${CXX} -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 -o ${FILE_PATH}.o -c ${FILE_PATH};\
    ${CXX} -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 ${FILE_PATH}.o -o ./${FILE_PATH}.out   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl' 
    run_tiramisu_cmd = 'printf "Running ${FILE_PATH}.out\n">> ${FUNC_DIR}log.txt;\
    ./${FILE_PATH}.out>> ${FUNC_DIR}log.txt;'
    launch_cmd(log_message_cmd,'')
    
    failed = launch_cmd(compile_tiramisu_cmd, file_path)
    if failed:
        print(f'Error occured while compiling {file_path}')
        return False
    else:
        failed = launch_cmd(run_tiramisu_cmd, file_path)
        if failed:
            print(f'Error occured while running {file_path}')
            return False
    return True


def launch_cmd(step_cmd, file_path, cmd_type=None,nb_executions=None, initial_exec_time=None):
    failed = False
    try:
        if cmd_type == 'initial_exec':
            out = subprocess.run(step_cmd, check=True ,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15*nb_executions)
            print("after running initial exec")
        elif cmd_type == 'sched_eval':
            out = subprocess.run(step_cmd, check=True ,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15+10*nb_executions*initial_exec_time/1000)
            print("after running sched eval")

        else:
            out = subprocess.run(step_cmd, check=True ,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    except subprocess.TimeoutExpired:
        raise TimeOutException

    except Exception as e:
        print(f'\n# {str(datetime.now())} ---> Error running {step_cmd} \n'+e.stderr.decode('UTF-8'), file=sys.stderr, flush=True)
        out = e
        failed = True
    else: # no exception rised
        if 'error' in out.stderr.decode('UTF-8'):
            print(f'\n# {str(datetime.now())} ---> Error running {step_cmd} \n'+out.stderr.decode('UTF-8'), file=sys.stderr, flush=True)
            failed = True
    if failed:
        func_folder = ('/'.join(Path(file_path).parts[:-1]) if len(Path(file_path).parts)>1 else '.') +'/'
        with open(func_folder+'error.txt', 'a') as f:
            f.write('\nError running '+step_cmd+'\n---------------------------\n'+out.stderr.decode('UTF-8')+'\n')
    return failed




def update_iterators(id, it_list, action_params, added_iterators, comp_indic_dict):
    for comp in it_list:
        if id in range(28):
            tmp=it_list[comp][action_params["first_dim_index"]]
            it_list[comp][action_params["first_dim_index"]]=it_list[comp].pop(action_params["second_dim_index"])
            it_list[comp][action_params["second_dim_index"]]=tmp

        if id in range(28,41):
            depth_1=action_params["first_dim_index"]
            depth_2=action_params["second_dim_index"]

            keys=list(it_list[comp].keys())
            print("keys: ", keys)

            i=len(keys)-1

            
            if action_params["tiling_depth"]==2:
                while i>depth_2:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"]:
                            it_list[comp][i+2]=it_list[comp][i]
                    elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"]:   
                            it_list[comp][i+1]=it_list[comp][i]
                    i-=1

            else:
                if action_params["tiling_depth"]==3:
                    depth_3=action_params["third_dim_index"]
                    print("third depth is", depth_3)
                    while i>depth_3:
                        if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:   
                            it_list[comp][i+3]=it_list[comp][i]
                        else:
                            booleans=[action_params["tiling_loop_1"], action_params["tiling_loop_2"], action_params["tiling_loop_3"]]
                            if booleans.count(True)==2:
                                it_list[comp][i+2]=it_list[comp][i]
                            elif booleans.count(True)==1:
                                it_list[comp][i+1]=it_list[comp][i]
                        i-=1
            
                            

            if action_params["tiling_depth"]==2:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"]:
                    print("in action params == 7 and tiling_loop_1 and tiling_loop_2")


                    #update the loop bounds if tiling is applied on loop 1
                    it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                    it_list[comp][depth_1+2]={}
                    it_list[comp][depth_1+2]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])
                    it_list[comp][depth_1+2]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                    it_list[comp][depth_1+2]['upper_bound']=action_params["first_factor"]
                    #Add the new iterator to added_iterators
                    added_iterators.append(it_list[comp][depth_1+2]['iterator'])
                    #update the loop bounds if tiling is applied on loop 2
                    it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                    it_list[comp][depth_2+2]={}
                    it_list[comp][depth_2+2]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                    it_list[comp][depth_2+2]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                    it_list[comp][depth_2+2]['upper_bound']=action_params["second_factor"]
                    #Add the new iterator to added_iterators
                    added_iterators.append(it_list[comp][depth_2+2]['iterator'])

                else:
                    if action_params["tiling_loop_1"]:
                        print("in action params == 7 and tiling_loop_1")
                        #update the loop bounds if tiling is applied on loop 1
                        it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                        it_list[comp][depth_1+2]={}
                        it_list[comp][depth_1+2]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])
                        it_list[comp][depth_1+2]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                        it_list[comp][depth_1+2]['upper_bound']=action_params["first_factor"]

                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_1+2]['iterator'])

                    elif action_params["tiling_loop_2"]:
                        print("in action params == 7 and tiling_loop_2")
                        #update the loop bounds if tiling is applied on loop 2
                        it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                        it_list[comp][depth_2+1]={}
                        it_list[comp][depth_2+1]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                        it_list[comp][depth_2+1]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                        it_list[comp][depth_2+1]['upper_bound']=action_params["second_factor"]

                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_2+1]['iterator'])

            elif action_params["tiling_depth"]==3:

                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    print("in action params == 10 and tiling_loop_1 and tiling_loop_2 and tiling_loop_3")

                    #update the loop bounds if tiling is applied on loop 1
                    it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                    it_list[comp][depth_1+3]={}
                    it_list[comp][depth_1+3]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])   
                    it_list[comp][depth_1+3]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                    it_list[comp][depth_1+3]['upper_bound']=action_params["first_factor"]

                    #Add the new iterator to added_iterators
                    added_iterators.append(it_list[comp][depth_1+3]['iterator'])

                    #update the loop bounds if tiling is applied on loop 2
                    it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                    it_list[comp][depth_2+3]={}
                    it_list[comp][depth_2+3]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                    it_list[comp][depth_2+3]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                    it_list[comp][depth_2+3]['upper_bound']=action_params["second_factor"]

                    #Add the new iterator to added_iterators
                    added_iterators.append(it_list[comp][depth_2+3]['iterator'])

                    #update the loop bounds if tiling is applied on loop 1=3
                    it_list[comp][depth_3]['upper_bound']=it_list[comp][depth_3]['upper_bound']/action_params["third_factor"]             
                    it_list[comp][depth_3+3]={}
                    it_list[comp][depth_3+3]['iterator']="{}_1".format(it_list[comp][depth_3]['iterator'])
                    it_list[comp][depth_3+3]['lower_bound']=it_list[comp][depth_3]['lower_bound']
                    it_list[comp][depth_3+3]['upper_bound']=action_params["third_factor"]

                    #Add the new iterator to added_iterators
                    added_iterators.append(it_list[comp][depth_3+3]['iterator'])
                
                elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"]:
                    print("in action params == 10 and tiling_loop_1 and tiling_loop_2")

                    #update the loop bounds if tiling is applied on loop 1
                    it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                    it_list[comp][depth_1+3]={}
                    it_list[comp][depth_1+3]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])   
                    it_list[comp][depth_1+3]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                    it_list[comp][depth_1+3]['upper_bound']=action_params["first_factor"]
                    #Add the new iterator to added_iterators
                    added_iterators.append(it_list[comp][depth_1+3]['iterator'])

                    #update the loop bounds if tiling is applied on loop 2
                    it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                    it_list[comp][depth_2+3]={}
                    it_list[comp][depth_2+3]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                    it_list[comp][depth_2+3]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                    it_list[comp][depth_2+3]['upper_bound']=action_params["second_factor"]
                    #Add the new iterator to added_iterators
                    added_iterators.append(it_list[comp][depth_2+3]['iterator'])

                elif action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:
                    print("in action params == 10 and tiling_loop_2 and tiling_loop_3")

                    #update the loop bounds if tiling is applied on loop 2
                    it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                    it_list[comp][depth_2+2]={}
                    it_list[comp][depth_2+2]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                    it_list[comp][depth_2+2]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                    it_list[comp][depth_2+2]['upper_bound']=action_params["second_factor"]

                    #Add the new iterator to added_iterators
                    added_iterators.append(it_list[comp][depth_2+2]['iterator'])

                    #update the loop bounds if tiling is applied on loop 1
                    it_list[comp][depth_3]['upper_bound']=it_list[comp][depth_3]['upper_bound']/action_params["third_factor"]
                    it_list[comp][depth_3+2]={}
                    it_list[comp][depth_3+2]['iterator']="{}_1".format(it_list[comp][depth_3]['iterator'])   
                    it_list[comp][depth_3+2]['lower_bound']=it_list[comp][depth_3]['lower_bound']
                    it_list[comp][depth_3+2]['upper_bound']=action_params["third_factor"]
                    #Add the new iterator to added_iterators
                    added_iterators.append(it_list[comp][depth_3+2]['iterator'])

                elif action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:
                    print("in action params == 10 and tiling_loop_1 and tiling_loop_3")

                    #update the loop bounds if tiling is applied on loop 2
                    it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                    it_list[comp][depth_1+3]={}
                    it_list[comp][depth_1+3]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])
                    it_list[comp][depth_1+3]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                    it_list[comp][depth_1+3]['upper_bound']=action_params["first_factor"]
                    #Add the new iterator to added_iterators
                    added_iterators.append(it_list[comp][depth_1+3]['iterator'])

                    #update the loop bounds if tiling is applied on loop 3
                    it_list[comp][depth_3]['upper_bound']=it_list[comp][depth_3]['upper_bound']/action_params["third_factor"]
                    it_list[comp][depth_3+2]={}
                    it_list[comp][depth_3+2]['iterator']="{}_1".format(it_list[comp][depth_3]['iterator'])   
                    it_list[comp][depth_3+2]['lower_bound']=it_list[comp][depth_3]['lower_bound']
                    it_list[comp][depth_3+2]['upper_bound']=action_params["third_factor"]
                    #Add the new iterator to added_iterators
                    added_iterators.append(it_list[comp][depth_3+2]['iterator'])
                else:
                    if action_params["tiling_loop_1"]:
                        print("in action params == 10 and tiling_loop_1")

                        it_list[comp][depth_1]['upper_bound']=it_list[comp][depth_1]['upper_bound']/action_params["first_factor"]
                        it_list[comp][depth_1+3]={}
                        it_list[comp][depth_1+3]['iterator']="{}_1".format(it_list[comp][depth_1]['iterator'])   
                        it_list[comp][depth_1+3]['lower_bound']=it_list[comp][depth_1]['lower_bound']
                        it_list[comp][depth_1+3]['upper_bound']=action_params["first_factor"] 
                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_1+3]['iterator']) 

                    elif action_params["tiling_loop_2"]:
                        print("in action params == 10 and tiling_loop_2")

                        it_list[comp][depth_2]['upper_bound']=it_list[comp][depth_2]['upper_bound']/action_params["second_factor"]
                        it_list[comp][depth_2+2]={}
                        it_list[comp][depth_2+2]['iterator']="{}_1".format(it_list[comp][depth_2]['iterator'])
                        it_list[comp][depth_2+2]['lower_bound']=it_list[comp][depth_2]['lower_bound']
                        it_list[comp][depth_2+2]['upper_bound']=action_params["second_factor"]
                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_2+2]['iterator'])

                    elif action_params["tiling_loop_3"]:
                        print("in action params == 10 and tiling_loop_3")

                        #update the loop bounds if tiling is applied on loop 1
                        it_list[comp][depth_3]['upper_bound']=it_list[comp][depth_3]['upper_bound']/action_params["third_factor"]
                        it_list[comp][depth_3+1]={}
                        it_list[comp][depth_3+1]['iterator']="{}_1".format(it_list[comp][depth_3]['iterator'])   
                        it_list[comp][depth_3+1]['lower_bound']=it_list[comp][depth_3]['lower_bound']
                        it_list[comp][depth_3+1]['upper_bound']=action_params["third_factor"]
                        #Add the new iterator to added_iterators
                        added_iterators.append(it_list[comp][depth_3+1]['iterator'])

        elif id in range(41,44): #Unrolling
            it_list[comp][action_params["dim_index"]]['upper_bound']=it_list[comp][action_params["dim_index"]]['upper_bound']/action_params['unrolling_factor']
        
        elif id in range(44,46):#Skewing
            depth_1=action_params["first_dim_index"]
            depth_2=action_params["second_dim_index"]

            l1_lower_bound=it_list[comp][depth_1]["lower_bound"]
            l1_upper_bound=it_list[comp][depth_1]["upper_bound"]
            l2_lower_bound=it_list[comp][depth_2]["lower_bound"]
            l2_upper_bound=it_list[comp][depth_2]["upper_bound"]

            l1_extent = abs(l1_upper_bound - l1_lower_bound)
            l2_extent = abs(l2_upper_bound - l2_lower_bound)

            l2_lower_bound = 0
            l1_lower_bound = abs(action_params["first_factor"]) * l1_lower_bound
            l1_upper_bound = l1_lower_bound + abs(action_params["first_factor"]) * l1_extent + abs(action_params["second_factor"]) * l2_extent
            l2_upper_bound = ((l1_extent * l2_extent) / (l1_upper_bound - l1_lower_bound)) + 1

            it_list[comp][depth_1]["lower_bound"]=l1_lower_bound
            it_list[comp][depth_1]["upper_bound"]=l1_upper_bound
            it_list[comp][depth_2]["lower_bound"]=l2_lower_bound
            it_list[comp][depth_2]["upper_bound"]=l2_upper_bound  
            
        elif id in range(48,56):#Reversal
            tmp=it_list[comp][action_params["dim_index"]]['lower_bound']
            it_list[comp][action_params["dim_index"]]['lower_bound']=it_list[comp][action_params["dim_index"]]['upper_bound']
            it_list[comp][action_params["dim_index"]]['upper_bound']=tmp 

    
    it_list=dict(sorted(it_list.items()))

    return it_list


def sched_str(sched_str, id, params, comp_indic):
    if id in range(28):
        sched_str+='I(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+')'
    else:
        if id in range(28, 41):
            if params["tiling_depth"]==2:
                sched_str+='T2(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+','+str(params["first_factor"])+','+str(params["second_factor"])+')'
            else:
                sched_str+='T3(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+',L'+str(params["third_dim_index"])+','+str(params["first_factor"])+','+str(params["second_factor"])+','+str(params["third_factor"])+')'
        else:
            if id in range(41,44):
                for comp in params:
                    sched_str+='U(L'+str(params[comp]["dim_index"])+','+str(params[comp]['unrolling_factor'])+ ",C"+ str(comp_indic[comp])+')'
            else:
                if id in range(44,46):
                    sched_str+='S(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+','+str(params["first_factor"])+','+str(params["second_factor"])+')'
                else:
                    if id in range(46,48):
                        sched_str+='P(L'+str(params["dim_index"])+')'
                    else:
                        if id in range(48,56):
                            sched_str+='R(L'+str(params["dim_index"])+')'
                        else:
                            if id in range(56,61):
                                sched_str+='F(L'+str(params["dim_index"])+')'

    return sched_str


def get_schedules_str(programs_list,programs_dict):

    if programs_dict != {}:
    
        functions_set={}#a dict containing all existed programs in the dataset with their evaluated schedules

        for fun in programs_list: 
            #print(programs_dict[fun]['schedules_list'])
            if 'schedules_list' in programs_dict[fun].keys():
                schedules=programs_dict[fun]['schedules_list']#[:2]
                #print(schedules)
                schedules_set={}#schedules_program_x 

                
                for schedule in schedules:
                    #schedule_str = sched_json_to_sched_str(schedule, prog_it) 
                    comp=list(schedule.keys())[0] #we have only one computation
                    schedule_str = schedule[comp]["schedule_str"]    
                    schedules_set[schedule_str]=schedule[comp]["execution_times"]

                functions_set[fun]=schedules_set
            #schedules_set.append(schedules_subset)#appending schedules_program_x to schedules_set
            
        return functions_set
    else:
        return {}





def get_comp_name(file):
    f = open(file, "r+")
    comp_name=''
    for l in f.readlines():
        if not "//" in l:
            l=l.split()
            if "computation" in l or "tiramisu::computation" in l :
                l=l[1].split("(")
                comp_name=l[0]
                break
    f.close()
    return comp_name


def get_cpp_file(Dataset_path,func_name):
    
    file_name=func_name+'_generator.cpp'
    original_path = Dataset_path+'/'+func_name + '/' + file_name
    dc_path=Path(Dataset_path).parts[:-1]
    print('dc path',dc_path)
    if len(dc_path)==0:
        
        target_path = "{}/Dataset_copies/{}".format(os.path.join(*dc_path),func_name)

    else:

        target_path = "{}/Dataset_copies/{}".format(os.path.join(*dc_path),func_name)
        
    if os.path.isdir(target_path):
        os.system("rm -r {}".format(target_path))
        print("directory removed")
    
    os.mkdir(target_path)
    os.system("cp -r {} {}".format(original_path, target_path))
    return target_path+'/'+file_name
