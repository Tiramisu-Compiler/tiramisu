import numpy as np
import re
import sys, os, subprocess, shutil
from pathlib import Path
from datetime import datetime
import re


class NbAccessException(Exception):
    pass
class LoopsDepthException(Exception):
    pass
class TimeOutException(Exception):
    pass
class LoopExtentException(Exception):
    pass

def get_dataset(path):
    print("**********************************",os.getcwd())
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


def get_representation_template(program_annot):
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
    
    # Adding padding so that all the vectors have the same size
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

#c++ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 -o ${FILE_PATH}.o -c ${FILE_PATH};\
#c++ -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 ${FILE_PATH}.o -o ./${FILE_PATH}.out   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl' 



#${CXX} -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 -o ${FILE_PATH}.o -c ${FILE_PATH};\
#${CXX} -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 ${FILE_PATH}.o -o ./${FILE_PATH}.out   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl' 

def compile_and_run_tiramisu_code(file_path, log_message='No message'):
    print("in compile and run")
    os.environ['FUNC_DIR'] = ('/'.join(Path(file_path).parts[:-1]) if len(Path(file_path).parts)>1 else '.') +'/'
    os.environ['FILE_PATH'] = file_path
    log_message_cmd = 'printf "'+log_message+'\n">> ${FUNC_DIR}log.txt'
    compile_tiramisu_cmd = 'printf "Compiling ${FILE_PATH}\n" >> ${FUNC_DIR}log.txt;\
c++ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 -o ${FILE_PATH}.o -c ${FILE_PATH};\
c++ -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 ${FILE_PATH}.o -o ./${FILE_PATH}.out   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl'     
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
    
    print("done with compile and run")
    return True


def launch_cmd(step_cmd, file_path, cmd_type=None,nb_executions=None, initial_exec_time=None):
    failed = False
    try:
        if cmd_type == 'initial_exec':
            out = subprocess.run(step_cmd, check=True ,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15*nb_executions)
            
        elif cmd_type == 'sched_eval':
            out = subprocess.run(step_cmd, check=True ,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15+10*nb_executions*initial_exec_time/1000)
           

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



# Updates the iterators according to the applied the transformation, if 
# - Interchange: We interchange the two loop levels
# - Tiling: If it generates one or more iterators we added them and change the loop bounds
# - Skewing: Change the loop bounds
# - Unrolling: Change the loop bounds
# - Reversal: Change the loop bounds
def update_iterators(id, it_list, action_params):
    if id in range(28):
        tmp=it_list[action_params["first_dim_index"]]
        it_list[action_params["first_dim_index"]]=it_list.pop(action_params["second_dim_index"])
        it_list[action_params["second_dim_index"]]=tmp

    if id in range(28,41):
        depth_1=action_params["first_dim_index"]
        depth_2=action_params["second_dim_index"]

        keys=list(it_list.keys())

        i=len(keys)-1

        
        if action_params["tiling_depth"]==2:
            while i>depth_2:
                if action_params["tiling_loop_1"] and action_params["tiling_loop_2"]:
                        it_list[i+2]=it_list[i]
                elif action_params["tiling_loop_1"] or action_params["tiling_loop_2"]:   
                        it_list[i+1]=it_list[i]
                i-=1

        else:
            if action_params["tiling_depth"]==3:
                depth_3=action_params["third_dim_index"]
                
                while i>depth_3:
                    if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:   
                        it_list[i+3]=it_list[i]
                    else:
                        booleans=[action_params["tiling_loop_1"], action_params["tiling_loop_2"], action_params["tiling_loop_3"]]
                        if booleans.count(True)==2:
                            it_list[i+2]=it_list[i]
                        elif booleans.count(True)==1:
                            it_list[i+1]=it_list[i]
                    i-=1
        
                        

        if action_params["tiling_depth"]==2:
            if action_params["tiling_loop_1"] and action_params["tiling_loop_2"]:

                #update the loop bounds if tiling is applied on loop 1
                it_list[depth_1]['upper_bound']=it_list[depth_1]['upper_bound']/action_params["first_factor"]
                it_list[depth_1+2]={}
                it_list[depth_1+2]['iterator']="{}_1".format(it_list[depth_1]['iterator'])
                it_list[depth_1+2]['lower_bound']=it_list[depth_1]['lower_bound']
                it_list[depth_1+2]['upper_bound']=action_params["first_factor"]

                #update the loop bounds if tiling is applied on loop 2
                it_list[depth_2]['upper_bound']=it_list[depth_2]['upper_bound']/action_params["second_factor"]
                it_list[depth_2+2]={}
                it_list[depth_2+2]['iterator']="{}_1".format(it_list[depth_2]['iterator'])
                it_list[depth_2+2]['lower_bound']=it_list[depth_2]['lower_bound']
                it_list[depth_2+2]['upper_bound']=action_params["second_factor"]

            else:
                if action_params["tiling_loop_1"]:
                    #update the loop bounds if tiling is applied on loop 1
                    it_list[depth_1]['upper_bound']=it_list[depth_1]['upper_bound']/action_params["first_factor"]
                    it_list[depth_1+2]={}
                    it_list[depth_1+2]['iterator']="{}_1".format(it_list[depth_1]['iterator'])
                    it_list[depth_1+2]['lower_bound']=it_list[depth_1]['lower_bound']
                    it_list[depth_1+2]['upper_bound']=action_params["first_factor"]

                elif action_params["tiling_loop_2"]:
                    #update the loop bounds if tiling is applied on loop 2
                    it_list[depth_2]['upper_bound']=it_list[depth_2]['upper_bound']/action_params["second_factor"]
                    it_list[depth_2+1]={}
                    it_list[depth_2+1]['iterator']="{}_1".format(it_list[depth_2]['iterator'])
                    it_list[depth_2+1]['lower_bound']=it_list[depth_2]['lower_bound']
                    it_list[depth_2+1]['upper_bound']=action_params["second_factor"]

        elif action_params["tiling_depth"]==3:

            if action_params["tiling_loop_1"] and action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:

                #update the loop bounds if tiling is applied on loop 1
                it_list[depth_1]['upper_bound']=it_list[depth_1]['upper_bound']/action_params["first_factor"]
                it_list[depth_1+3]={}
                it_list[depth_1+3]['iterator']="{}_1".format(it_list[depth_1]['iterator'])   
                it_list[depth_1+3]['lower_bound']=it_list[depth_1]['lower_bound']
                it_list[depth_1+3]['upper_bound']=action_params["first_factor"]

                #update the loop bounds if tiling is applied on loop 2
                it_list[depth_2]['upper_bound']=it_list[depth_2]['upper_bound']/action_params["second_factor"]
                it_list[depth_2+3]={}
                it_list[depth_2+3]['iterator']="{}_1".format(it_list[depth_2]['iterator'])
                it_list[depth_2+3]['lower_bound']=it_list[depth_2]['lower_bound']
                it_list[depth_2+3]['upper_bound']=action_params["second_factor"]

                #update the loop bounds if tiling is applied on loop 1=3
                it_list[depth_3]['upper_bound']=it_list[depth_3]['upper_bound']/action_params["third_factor"]             
                it_list[depth_3+3]={}
                it_list[depth_3+3]['iterator']="{}_1".format(it_list[depth_3]['iterator'])
                it_list[depth_3+3]['lower_bound']=it_list[depth_3]['lower_bound']
                it_list[depth_3+3]['upper_bound']=action_params["third_factor"]
            
            elif action_params["tiling_loop_1"] and action_params["tiling_loop_2"]:

                #update the loop bounds if tiling is applied on loop 1
                it_list[depth_1]['upper_bound']=it_list[depth_1]['upper_bound']/action_params["first_factor"]
                it_list[depth_1+3]={}
                it_list[depth_1+3]['iterator']="{}_1".format(it_list[depth_1]['iterator'])   
                it_list[depth_1+3]['lower_bound']=it_list[depth_1]['lower_bound']
                it_list[depth_1+3]['upper_bound']=action_params["first_factor"]

                #update the loop bounds if tiling is applied on loop 2
                it_list[depth_2]['upper_bound']=it_list[depth_2]['upper_bound']/action_params["second_factor"]
                it_list[depth_2+3]={}
                it_list[depth_2+3]['iterator']="{}_1".format(it_list[depth_2]['iterator'])
                it_list[depth_2+3]['lower_bound']=it_list[depth_2]['lower_bound']
                it_list[depth_2+3]['upper_bound']=action_params["second_factor"]

            elif action_params["tiling_loop_2"] and action_params["tiling_loop_3"]:

                #update the loop bounds if tiling is applied on loop 2
                it_list[depth_2]['upper_bound']=it_list[depth_2]['upper_bound']/action_params["second_factor"]
                it_list[depth_2+2]={}
                it_list[depth_2+2]['iterator']="{}_1".format(it_list[depth_2]['iterator'])
                it_list[depth_2+2]['lower_bound']=it_list[depth_2]['lower_bound']
                it_list[depth_2+2]['upper_bound']=action_params["second_factor"]

                #update the loop bounds if tiling is applied on loop 1
                it_list[depth_3]['upper_bound']=it_list[depth_3]['upper_bound']/action_params["third_factor"]
                it_list[depth_3+2]={}
                it_list[depth_3+2]['iterator']="{}_1".format(it_list[depth_3]['iterator'])   
                it_list[depth_3+2]['lower_bound']=it_list[depth_3]['lower_bound']
                it_list[depth_3+2]['upper_bound']=action_params["third_factor"]

            elif action_params["tiling_loop_1"] and action_params["tiling_loop_3"]:

                #update the loop bounds if tiling is applied on loop 2
                it_list[depth_1]['upper_bound']=it_list[depth_1]['upper_bound']/action_params["first_factor"]
                it_list[depth_1+3]={}
                it_list[depth_1+3]['iterator']="{}_1".format(it_list[depth_1]['iterator'])
                it_list[depth_1+3]['lower_bound']=it_list[depth_1]['lower_bound']
                it_list[depth_1+3]['upper_bound']=action_params["first_factor"]

                #update the loop bounds if tiling is applied on loop 3
                it_list[depth_3]['upper_bound']=it_list[depth_3]['upper_bound']/action_params["third_factor"]
                it_list[depth_3+2]={}
                it_list[depth_3+2]['iterator']="{}_1".format(it_list[depth_3]['iterator'])   
                it_list[depth_3+2]['lower_bound']=it_list[depth_3]['lower_bound']
                it_list[depth_3+2]['upper_bound']=action_params["third_factor"]
            else:
                if action_params["tiling_loop_1"]:

                    it_list[depth_1]['upper_bound']=it_list[depth_1]['upper_bound']/action_params["first_factor"]
                    it_list[depth_1+3]={}
                    it_list[depth_1+3]['iterator']="{}_1".format(it_list[depth_1]['iterator'])   
                    it_list[depth_1+3]['lower_bound']=it_list[depth_1]['lower_bound']
                    it_list[depth_1+3]['upper_bound']=action_params["first_factor"] 

                elif action_params["tiling_loop_2"]:

                    it_list[depth_2]['upper_bound']=it_list[depth_2]['upper_bound']/action_params["second_factor"]
                    it_list[depth_2+2]={}
                    it_list[depth_2+2]['iterator']="{}_1".format(it_list[depth_2]['iterator'])
                    it_list[depth_2+2]['lower_bound']=it_list[depth_2]['lower_bound']
                    it_list[depth_2+2]['upper_bound']=action_params["second_factor"]

                elif action_params["tiling_loop_3"]:

                    #update the loop bounds if tiling is applied on loop 1
                    it_list[depth_3]['upper_bound']=it_list[depth_3]['upper_bound']/action_params["third_factor"]
                    it_list[depth_3+1]={}
                    it_list[depth_3+1]['iterator']="{}_1".format(it_list[depth_3]['iterator'])   
                    it_list[depth_3+1]['lower_bound']=it_list[depth_3]['lower_bound']
                    it_list[depth_3+1]['upper_bound']=action_params["third_factor"]

    elif id==41: #Unrolling
        it_list[action_params["dim_index"]]['upper_bound']=it_list[action_params["dim_index"]]['upper_bound']/action_params['unrolling_factor']
    
    elif id==42:#Skewing
        depth_1=action_params["first_dim_index"]
        depth_2=action_params["second_dim_index"]

        l1_lower_bound=it_list[depth_1]["lower_bound"]
        l1_upper_bound=it_list[depth_1]["upper_bound"]
        l2_lower_bound=it_list[depth_2]["lower_bound"]
        l2_upper_bound=it_list[depth_2]["upper_bound"]

        l1_extent = l1_upper_bound - l1_lower_bound
        l2_extent = l2_upper_bound - l2_lower_bound

        l2_lower_bound = 0
        l1_lower_bound = abs(action_params["first_factor"]) * l1_lower_bound
        l1_upper_bound = l1_lower_bound + abs(action_params["first_factor"]) * l1_extent + abs(action_params["second_factor"]) * l2_extent
        l2_upper_bound = ((l1_extent * l2_extent) / (l1_upper_bound - l1_lower_bound)) + 1

        it_list[depth_1]["lower_bound"]=l1_lower_bound
        it_list[depth_1]["upper_bound"]=l1_upper_bound
        it_list[depth_2]["lower_bound"]=l2_lower_bound
        it_list[depth_2]["upper_bound"]=l2_upper_bound  
        
    elif id in range(44,52):#Reversal
        tmp=it_list[action_params["dim_index"]]['lower_bound']
        it_list[action_params["dim_index"]]['lower_bound']=it_list[action_params["dim_index"]]['upper_bound']
        it_list[action_params["dim_index"]]['upper_bound']=tmp 

    
    it_list=dict(sorted(it_list.items()))

    return it_list


def sched_str(sched_str, id, params):
    if id in range(28):
        sched_str+='I(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+')'
    else:
        if id in range(28, 41):
            if params["tiling_depth"]==2:
                sched_str+='T2(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+','+str(params["first_factor"])+','+str(params["second_factor"])+')'
            else:
                sched_str+='T3(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+',L'+str(params["third_dim_index"])+','+str(params["first_factor"])+','+str(params["second_factor"])+','+str(params["third_factor"])+')'
        else:
            if id==41:
                sched_str+='U(L'+str(params["dim_index"])+','+str(params['unrolling_factor'])+')'
            else:
                if id==42:
                    sched_str+='S(L'+str(params["first_dim_index"])+',L'+str(params['second_dim_index'])+','+str(params["first_factor"])+','+str(params["second_factor"])+')'
                else:
                    if id==43:
                        sched_str+='P(L'+str(params["dim_index"])+')'
                    else:
                        if id in range(44,52):
                            sched_str+='R(L'+str(params["dim_index"])+')'

    return sched_str


# Get the schedules saved by previous iterations
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
    if len(dc_path)==0:
        
        target_path = "{}/Dataset_copies/{}".format(os.path.join(*dc_path),func_name)

    else:

        target_path = "{}/Dataset_copies/{}".format(os.path.join(*dc_path),func_name)
        
    if os.path.isdir(target_path):
        os.system("rm -r {}".format(target_path))
    
    os.mkdir(target_path)
    os.system("cp -r {} {}".format(original_path, target_path))
    return target_path+'/'+file_name
