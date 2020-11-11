import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import dill
from os import listdir
import json 
from tqdm import tqdm
import re
import subprocess
import random 
import time 
from multiprocessing import Pool
from shutil import rmtree
from os import path
import sys
sys.path.append("TiramisuCodeGenerator")
from TiramisuCodeGenerator import code_generator


class cluster_utilities():
    def __init__(self, data_path, generator_script, wrappers_script, execute_script, log_path, batchName, nb_nodes, tmp_files_dir):
        self.batchName = batchName
        self.data_path = Path(data_path)
        self.generator_script = Path(generator_script)
        self.wrappers_script = Path(wrappers_script)
        self.execute_script = Path(execute_script)
        self.log_path = Path(log_path)
        self.nb_nodes=nb_nodes    
        self.tmp_files_dir = tmp_files_dir
        self.compile_jobs_ids = self.wrap_jobs_ids = self.exec_jobs_ids = '1'
        if path.exists(self.log_path):
            rmtree(self.log_path)
        if path.exists(self.tmp_files_dir+ "job_files_"+self.batchName):
            rmtree(self.tmp_files_dir+ "job_files_"+self.batchName)
            
            
    def generate_prog_list(self):
        """
        Create the list of programs that will be executed.
        The result is a list of tuples of the following format :

        (function_id, schedule_id)

        Example:
        (function524, function524_schedule_125)
        """

        # Path to where to store the list of programs
        dst_path = Path(self.tmp_files_dir + "progs_list_"+self.batchName+".pickle")

        self.progs_list = []

        for func_path in self.data_path.iterdir():


            if (str(func_path.parts[-1]).startswith('.')):
                continue

            # We discard programs that have no schedule.
            # We don't need to execute those programs as they just have a speedup of 1,
            # and they have no programs with schedules.
            # If you want them in the dataset, just include them with speedup = 1.
            if len(list(func_path.iterdir())) <= 2:
                rmtree(str(func_path))
        #         func_path = func_path.rename(str(self.data_path)+'/_'+str(func_path.parts[-1]))
                continue

            for sched_path in func_path.iterdir():
                if not sched_path.is_dir():
                    continue

                if (str(sched_path.parts[-1]).startswith('.')):
                    continue

                func_id = func_path.parts[-1]
                sched_id = sched_path.parts[-1]

                self.progs_list.append((func_id, sched_id))

        random.Random(42).shuffle(self.progs_list) # shuffling the prog list for having a similar exec time per node  

        with open(dst_path, "wb") as f:
            pickle.dump(self.progs_list, f)
        print("Total number of schedules generated " + str(len(self.progs_list)) )
        
        
        
    def generate_compile_jobs(self):

        """
        Generate the job files needed by the sbatch command.

        Here's an example of a job file :

        #!/bin/bash
        #SBATCH --job-name=comp2
        #SBATCH --output=log/log_comp_2_6842_10263
        #SBATCH -N 1
        #SBATCH --exclusive
        #SBATCH -p research
        srun python3 compile_tiramisu_code.py 6842 10263 2 
        """

        # Path to the list of programs
        self.progs_list_path = Path(self.tmp_files_dir + "progs_list_"+self.batchName+".pickle")

        # Path where to store the job files
        dst_path = Path(self.tmp_files_dir+ "job_files_"+self.batchName)
        dst_path.mkdir(parents=True, exist_ok=True)

        # Path to the script that will be distributed
        # self.generator_script = Path("/data/scratch/mmerouani/data_scripts/compile_tiramisu_code.py")

        # Path to where to store the logs of the jobs
        self.log_path = Path(self.tmp_files_dir + "log_"+self.batchName+"/")
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Content of the job files
        job_file_content = "\
#!/bin/bash\n\
#SBATCH --job-name=comp{2}_{3}\n\
#SBATCH --output=%s/log_comp_{2}_{0}_{1}\n\
#SBATCH -N 1\n\
#SBATCH --exclusive\n\
#SBATCH -p lanka-v3\n\
#SBATCH --exclude=lanka21,lanka33\n\
srun python3 %s {0} {1} {3}" % (str(self.log_path), str(self.generator_script)) # This replaces the %s
# SBATCH --exclude=lanka21,lanka04\n\


        # Content of the submit script
        submit_script_content = '\
#!/bin/bash\n\
\n\
for file in %sjob_files_%s/compile_job*\n\
do\n\
  sbatch "$file"\n\
done' % (self.tmp_files_dir, self.batchName) # This replaces the %s

        with open(self.tmp_files_dir + 'submit_compile_jobs_' + self.batchName + '.sh', "w") as f:
            f.write(submit_script_content)

        with open(self.progs_list_path, "rb") as f:
            self.progs_list = pickle.load(f)

        nb_progs = len(self.progs_list)
        progs_per_node = nb_progs // self.nb_nodes

        for i in range(self.nb_nodes):
            # Each node will process the programs in the range progs_list[start, end)
            start = i * progs_per_node

            if i < self.nb_nodes - 1:
                end = (i + 1) * progs_per_node
            else:
                end = nb_progs

            with open(dst_path / ("compile_job_%s_%s.batch" % (start, end)), "w") as f:
                f.write(job_file_content.format(start, end, i, self.batchName))
    
    def submit_compile_jobs(self):
        exec_output = subprocess.check_output(['sh', self.tmp_files_dir + 'submit_compile_jobs_' + self.batchName + '.sh'])
        print(exec_output.decode("utf-8"))
        self.compile_jobs_ids = ','.join(re.findall(r'\d+',exec_output.decode("utf-8")))
        
    def check_compile_progress(self):
        print(subprocess.check_output(["squeue --format='%.18i %.9P %.15j %.8u %.2t %.10M %.6D %R'| grep 'comp\|JOBID' "], shell=True).decode("utf-8"))
        print(subprocess.check_output(['tail -n 3 '+ str(self.log_path) + '/log_comp*'], shell=True).decode("utf-8"))    
                
        
    def generate_wrapper_jobs(self):
        """
        Generate the job files needed by the sbatch command.
        Two type of job files are generated :

        - One for editing and compiling the wrappers.
        - The other for executing the compiled wrappers and measuring execution time.

        Here's an example of a job file of type execute :

        #!/bin/bash
        #SBATCH --job-name=exec17
        #SBATCH --output=/data/scratch/k_abdous/autoscheduling_tiramisu/execute_all/log/log_exec_17_1096959_1161486
        #SBATCH -N 1
        #SBATCH --exclusive
        srun python3 /data/scratch/k_abdous/autoscheduling_tiramisu/execute_all/execute_programs.py 1096959 1161486 17
        """

        # Path to the list of programs
        self.progs_list_path = Path(self.tmp_files_dir + "progs_list_"+self.batchName+".pickle")

        # Path where to store the job files
        # This script will use two subdirectories (don't forget to create them first) : wrappers and execute
        dst_path = Path(self.tmp_files_dir + "job_files_"+self.batchName)
        Path(dst_path / "wrappers").mkdir(parents=True, exist_ok=True)
        Path(dst_path / "execute").mkdir(parents=True, exist_ok=True)
        # Path to the scrip "wrappers"t that edits and compiles the wrappers
        # If your wrappers are already in the good format, point this script to compile_tiramisu_wrappers.py
        # wrappers_script = Path("/data/scratch/mmerouani/data_scripts/compile_tiramisu_wrappers.py")

        # Path to the script that execute the compiled wrappers
        # execute_script = Path("/data/scratch/mmerouani/data_scripts/execute_programs.py")

        # Path to where to store the logs of the jobs
        log_path = Path(self.tmp_files_dir + "log_"+self.batchName+"/")
        log_path.mkdir(parents=True, exist_ok=True)

        # Content of the job files of type wrappers
        wrappers_job = "\
#!/bin/bash\n\
#SBATCH --job-name=wrap{2}_{3}\n\
#SBATCH --output=%s/log_wrap_{2}_{0}_{1}\n\
#SBATCH -N 1\n\
#SBATCH --exclusive\n\
#SBATCH -p lanka-v3\n\
#SBATCH --exclude=lanka21,lanka33\n\
srun python3 %s {0} {1} {3}" % (str(self.log_path), str(self.wrappers_script)) # This replaces the %s
#SBATCH --exclude=lanka21,lanka04\n\

        # Content of the job files of type execute
        execute_job = "\
#!/bin/bash\n\
#SBATCH --job-name=exec{2}_{3}\n\
#SBATCH --output=%s/log_exec_{2}_{0}_{1}\n\
#SBATCH -N 1\n\
#SBATCH --exclusive\n\
#SBATCH -p lanka-v3\n\
#SBATCH --exclude=lanka21,lanka33\n\
srun python3 %s {0} {1} {2} {3}" % (str(self.log_path), str(self.execute_script)) # This replaces the %s
#SBATCH --exclude=lanka21,lanka04\n\
        submit_wrap_script_content='\
#!/bin/bash \n\
\n\
for file in %sjob_files_%s/wrappers/wrappers_job*\n\
do\n\
  sbatch --dependency=afterok:%s "$file"\n\
done' % (self.tmp_files_dir, self.batchName, self.compile_jobs_ids) # This replaces the %s

        with open(self.tmp_files_dir + 'submit_wrapper_jobs_' + self.batchName + '.sh', "w") as f:
            f.write(submit_wrap_script_content)

        with open(self.progs_list_path, "rb") as f:
            self.progs_list = pickle.load(f)

        nb_progs = len(self.progs_list)
        progs_per_node = nb_progs // self.nb_nodes

        for i in range(self.nb_nodes):
            # Each node will process the programs in the range progs_list[start, end)
            start = i * progs_per_node

            if i < self.nb_nodes - 1:
                end = (i + 1) * progs_per_node
            else:
                end = nb_progs

            with open(dst_path / "wrappers" / ("wrappers_job_%s_%s.batch" % (start, end)), "w") as f:
                f.write(wrappers_job.format(start, end, i, self.batchName))

            with open(dst_path / "execute" / ("execute_job_%s_%s.batch" % (start, end)), "w") as f:
                f.write(execute_job.format(start, end, i, self.batchName))
        
    
    def submit_wrapper_compilation_jobs(self):
        wrap_cmd_output = subprocess.check_output(['sh', self.tmp_files_dir + 'submit_wrapper_jobs_' + self.batchName + '.sh'])
        print(wrap_cmd_output.decode("utf-8"))
        self.wrap_jobs_ids = ','.join(re.findall(r'\d+',wrap_cmd_output.decode("utf-8")))
       
    
    def check_wrapper_compilation_progress(self):
        print(subprocess.check_output(["squeue --format='%.18i %.9P %.15j %.8u %.2t %.10M %.6D %R' | grep 'wrap\|JOBID' "], shell=True).decode("utf-8"))
        print(subprocess.check_output(['tail -n 3 '+ str(self.log_path) + '/log_wrap*'], shell=True).decode("utf-8"))
    
    def generate_execution_slurm_script(self):
        submit_exec_script_content='\
        #!/bin/bash \n\
        \n\
        for file in %sjob_files_%s/execute/execute_job*\n\
        do\n\
          sbatch --dependency=afterok:%s "$file"\n\
        done' % (self.tmp_files_dir, self.batchName, self.wrap_jobs_ids) # This replaces the %s

        with open(self.tmp_files_dir + 'submit_execute_jobs_' + self.batchName + '.sh', "w") as f:
            f.write(submit_exec_script_content)
            
    def submit_execution_jobs(self):
        exec_cmd_output = subprocess.check_output(['sh', self.tmp_files_dir + 'submit_execute_jobs_' + self.batchName + '.sh'])
        print(exec_cmd_output.decode("utf-8"))
        self.exec_jobs_ids = ','.join(re.findall(r'\d+',exec_cmd_output.decode("utf-8")))
        
    def check_execution_progress(self):
        print(subprocess.check_output(["squeue --format='%.18i %.9P %.15j %.8u %.2t %.10M %.6D %R' | grep 'exec\|JOBID' "], shell=True).decode("utf-8"))
        print(subprocess.check_output(['tail -n 3 '+ str(self.log_path) + '/log_exec*'], shell=True).decode("utf-8"))
        
        
        
class annotation_utilities():
    def __init__(self):
        pass
    
    def get_function_annotation(self, filename):
        with open(filename,'r') as file:
            code = file.read()

        #Constantas        
        cst_lines=re.findall(r'\n\s*constant\s+.*;',code) # list of lines [constant c0("c0", 64), c1("c1", 128);]
        cst_def=[]
        for line in cst_lines:
            cst_def.extend(re.findall(r'\w+\s*\(\s*\"\w+\s*\"\s*,\s*\d+\s*\)',line)) # list ['c0("c0", 64)','c1("c1", 128)','c20("c20", 64)'...
        constants_dict=dict()    
        for cst in cst_def:
            name= re.findall(r'(\w+)\s*\(', cst)[0] #the name before parenthesis 
        #     name2= re.findall(r'\"\w+\"', cst)[0][1:-1] #the name between "" 
            value= re.findall(r'(\d+)\s*\)', cst)[0] #the value before )
            constants_dict[name]=int(value)


        #Inputs    
        input_lines = re.findall(r'\n\s*input\s+.*;',code) # gets the lines where inputs are difined 
        input_defs = []
        buffer_id=1
        for line in input_lines:
            input_defs.extend(re.findall(r'\w*\s*\(\s*\"\w*\"\s*,\s*{\s*\w*\s*(?:,\s*\w*\s*)*}\s*,\s*\w*\s*\)',line)) # get the input buffers definition
        inputs_dict = dict()
        for inp in input_defs:
            name = re.findall(r'(\w*)\s*\(', inp)[0]
            inp_interators = re.findall(r'\w+',re.findall(r'{([\w\s,]*)}',inp)[0]) # gets a list of iterators
            inp_type = re.findall(r',\s*(\w+)\s*\)',inp)[0] # gets data type of the input
            inputs_dict[name]=dict()
            inputs_dict[name]['id'] = buffer_id
            buffer_id += 1
            inputs_dict[name]['iterators_list']=inp_interators
            inputs_dict[name]['data_type']=inp_type

        # Computations    
        comp_lines = re.findall(r'\n\s*computation\s+.*;',code)
        computation_dict=dict()
        for comp_def in comp_lines: #assuming that each computation is declared in a separate line
            computation_is_reduction = False #by default the computation is not a reduction
            name = re.findall(r'(\w*)\s*\(\s*\"',comp_def)[0]
            comp_interators = re.findall(r'\w+',re.findall(r'{([\w\s,]*)}',comp_def)[0]) # gets a list of iterators
            comp_assingment = re.findall(r'}\s*,\s*(.*)\);', comp_def)[0] # gets the assignment expression
            comp_accesses = re.findall(r'\w*\((?:\s*[^()]\s*,?)+\)',comp_assingment) # gets the list of buffer accesses form the assingment
            if (comp_accesses == []): #if assignment is not specified in computation definition, search for assignment with the .set_expression expression
                comp_assingment = re.findall(r''+name+'\.set_expression\(\s*(.*)\);',code)# gets the assignment expression
                if (comp_assingment!=[]): # if the .set_expression found
                    comp_assingment = comp_assingment[0]
                    comp_accesses = re.findall(r'\w*\((?:\s*[^()]\s*,?)+\)',comp_assingment)# gets the list of buffer accesses form the assingment
            if (comp_assingment != []):    
                nb_addition = len(re.findall(r'\)\s*\+|\+\s*\w*[^)]\(',comp_assingment)) #gets the number of additions in assignment, support input + input , cst + input, input + cst
                nb_multiplication = len(re.findall(r'\)\s*\*|\*\s*\w*[^)]\(',comp_assingment)) #gets the number of multiplications in assignment, support input * input , cst * input, input * cst
                nb_division = len(re.findall(r'\)\s*\/|\/\s*\w*[^)]\(',comp_assingment)) #gets the number of multiplications in assignment, support input * input , cst * input, input * cst
                nb_subtraction = len(re.findall(r'\)\s*\-|\-\s*\w*[^)]\(',comp_assingment)) #gets the number of subtracrions in assignment, support input - input , cst - input, input - cst
            else:
                nb_addition=0
                nb_multiplication=0
                nb_division=0
                nb_subtraction=0
            accesses_list=[]
            for access in comp_accesses:
                access_is_reduction = False #by default the acces is not a reduction 
                accessed_buf_name = re.findall(r'(\w*)\s*\(',access)[0] # gets the accessed input buffer name
                if (accessed_buf_name == name): # if the the computation itself is used in assignment, we assume that it is a reduction
                    access_is_reduction = True 
                    computation_is_reduction = True
                    accessed_buf_id = buffer_id
                    access_matrix = np.zeros([len(comp_interators), len(comp_interators)+1], dtype = int) # initialises the access matrix
                else: #the accessed is an input buffer      
                    accessed_buf_id = inputs_dict[accessed_buf_name]['id'] # gets the accessed input buffer id
                    access_matrix = np.zeros([len(inputs_dict[accessed_buf_name]['iterators_list']), len(comp_interators)+1], dtype = int) # initialises the access matrix
                dim_accesses = re.findall(r'[\w\s\+\-\*\/]+', re.findall(r'\(([\w\s,+\-*/]*)\)',access)[0]) # returns a list where n'th element is the access to the n'th dimension of the buffer
                for i, dim_acc in enumerate(dim_accesses):
                    left_coef_iter1, used_iterator_iter1, right_coef_iter1, left_coef_iter2, used_iterator_iter2, right_coef_iter2 =  re.findall(r'(?:(\d+)\s*\*\s*)?([A-Za-z]\w*)(?:\s*\*\s*(\d+))?(?:\s*[\+\*\-]\s*(?:(\d+)\s*\*\s*)?([A-Za-z]\w*)(?:\s*\*\s*(\d+))?)?', dim_acc)[0] # gets iterator and coeficients, only supports patterns like 4*i1*2 + 6*i2*5  + 1
                    cst_shift = re.findall(r'(?:\s*[\-\+]\s*\d+)+',dim_acc) # gets the '+ cst' of the access
                    cst_shift= 0 if ( not(cst_shift) ) else eval(cst_shift[0])
                    left_coef_iter1 = 1 if (left_coef_iter1=='') else int(left_coef_iter1)
                    right_coef_iter1 = 1 if (right_coef_iter1=='') else int(right_coef_iter1)
                    coef_iter1 =left_coef_iter1*right_coef_iter1
                    j = comp_interators.index(used_iterator_iter1)
                    access_matrix[i,j]=coef_iter1
                    access_matrix[i,-1]=cst_shift
                    if (used_iterator_iter2 != ''):
                        left_coef_iter2 = 1 if (left_coef_iter2=='') else int(left_coef_iter2)
                        right_coef_iter2 = 1 if (right_coef_iter2=='') else int(right_coef_iter2)
                        coef_iter2 =left_coef_iter2*right_coef_iter2
                        j = comp_interators.index(used_iterator_iter2)
                        access_matrix[i,j]=coef_iter2

                access_dict=dict()
                access_dict['access_is_reduction']=access_is_reduction
                access_dict['buffer_id']=accessed_buf_id
                access_dict['buffer_name']=accessed_buf_name   
                access_dict['access_matrix']=access_matrix.tolist() 
                accesses_list.append(access_dict)
            real_dimensions = re.findall(r'\w+',re.findall(r''+name+'\.store_in\s*\(\s*&\w*\s*,?\s*{?([\w\s,]*)}?',code)[0]) #get the list of this computation's dimensions that are saved to a buffer
            computation_dict[name]=dict()    
            computation_dict[name]['id']=buffer_id
            buffer_id +=1
            computation_dict[name]['absolute_order'] = None # None means not explicitly ordered, this value is updated in computation oredering is specified
            computation_dict[name]['iterators']=comp_interators
            if (real_dimensions!=[]): # if dimensions are explicitly specified in the store_in instruction
                computation_dict[name]['real_dimensions'] = real_dimensions
            else: # else by default the real dimensions are the iterators
                computation_dict[name]['real_dimensions'] = comp_interators
            computation_dict[name]['comp_is_reduction']=computation_is_reduction
            computation_dict[name]['number_of_additions']=nb_addition
            computation_dict[name]['number_of_subtraction']=nb_subtraction
            computation_dict[name]['number_of_multiplication']=nb_multiplication
            computation_dict[name]['number_of_division']=nb_division
            computation_dict[name]['accesses']=accesses_list



        #Iterators
        iterator_lines=re.findall(r'\n\s*var\s+.*;',code) #list of lines var [var i0("i0", 0, c0), i1("i1", 0, c1),......]
        iterator_def = []
        iterator_def_no_bounds = []
        for line in iterator_lines:
            iterator_def.extend(re.findall(r'\w+\s*\(\s*\"\w+\s*\"\s*,\s*\d+\s*,(?:\s*\w*\s*[+\-\*/]?\s*)*\w+\s*\)',line)) # find iteratros with lower and upper bound deifined [i0("i0", 0, c0), i1("i1", 0, c1 - 2)...]
            iterator_def_no_bounds.extend(re.findall(r'\w+\s*\(\s*\"\w+\s*\"\s*\)', line)) #finds iteratros with bounds not defined [i01("i01"), i02("i02"), i03("i03") ....]
        iterators_dict=dict()
        for iterator in iterator_def:
            name=re.findall(r'(\w+)\s*\(',iterator)[0]
            lower_bound = re.findall(r',((?:\s*\w*\s*[+\-\*/]?\s*)*\w+\s*),',iterator)[0] # gets the lower bound of the iterator
            upper_bound = re.findall(r',((?:\s*\w*\s*[+\-\*/]?\s*)*\w+\s*)\)',iterator)[0] #gets the uppper bound of the iterator
            iterators_dict[name]=dict()
            iterators_dict[name]['lower_bound'] = eval(lower_bound,{},constants_dict) #evaluates the bounds in case it's and expression 
            iterators_dict[name]['upper_bound'] = eval(upper_bound,{},constants_dict)
            iterators_dict[name]['parent_iterator'] = None # None mean no parent, hierarchy attributes are set later
            iterators_dict[name]['child_iterators'] = []
            iterators_dict[name]['iterator_order'] = None # None means not explicitly ordered, this value is updated in computation oredering is specified
            iterators_dict[name]['computations_list'] = []       
        for iterator in iterator_def_no_bounds:
            name=re.findall(r'(\w+)\s*\(',iterator)[0]
            iterators_dict[name]=dict()
            iterators_dict[name]['lower_bound']=None
            iterators_dict[name]['upper_bound']=None
            iterators_dict[name]['parent_iterator'] = None
            iterators_dict[name]['child_iterators'] = []
            iterators_dict[name]['iterator_order'] = None
            iterators_dict[name]['computations_list'] = []

        #Setting hierarchy attributes
        for comp in computation_dict:
            for i in range(1, len(computation_dict[comp]['iterators'])):
                iterators_dict[computation_dict[comp]['iterators'][i]]['parent_iterator'] = computation_dict[comp]['iterators'][i-1]

            for i in range(0, len(computation_dict[comp]['iterators'])-1):
                iterators_dict[computation_dict[comp]['iterators'][i]]['child_iterators'].append(computation_dict[comp]['iterators'][i+1])

            iterators_dict[computation_dict[comp]['iterators'][-1]]['computations_list'].append(comp)


        #Removing duplicates in lists
        for iterator in iterators_dict:
            iterators_dict[iterator]['child_iterators'] = list(set(iterators_dict[iterator]['child_iterators']))
            iterators_dict[iterator]['computations_list'] = list(set(iterators_dict[iterator]['computations_list']))


        #Ordering
        orderings_list = self.get_ordering_sequence(code)        


    #     for ordering in orderings_list:
        ordering = orderings_list
        for rank, ord_tuple in enumerate(ordering):
            computation_dict[ord_tuple[0]]['absolute_order'] = rank + 1
            # update iterator order
            if (rank != 0): # if not first comp
                interator_index = computation_dict[ord_tuple[0]]['iterators'].index(ord_tuple[1])
                if (interator_index<len(computation_dict[ord_tuple[0]]['iterators']) - 1):
                    iterators_dict[computation_dict[ord_tuple[0]]['iterators'][interator_index+1]]['iterator_order'] = rank+1
                if (iterators_dict[ord_tuple[1]]['iterator_order'] == None):
                    iterators_dict[ord_tuple[1]]['iterator_order'] = rank+1
            else: # if first comp
                interator_index = computation_dict[ord_tuple[0]]['iterators'].index(ordering[rank+1][1])
                if (interator_index<len(computation_dict[ord_tuple[0]]['iterators']) - 1):
                    iterators_dict[computation_dict[ord_tuple[0]]['iterators'][interator_index+1]]['iterator_order'] = rank+1

        #Creating Json                
        function_name = re.findall(r'tiramisu::init\s*\(\s*\"(\w*)\"\s*\)', code)[0]
        function_json = {
                            'function_name': function_name,
                            'constants': constants_dict,
                            'iterators': iterators_dict,
                            'inputs': inputs_dict,
                            'computations': computation_dict                    
                        }

        json_dump = json.dumps(function_json, indent = 4)

        def format_array(array_str):
            array_str = array_str.group()    
            array_str = array_str.replace('\n','')
            array_str = array_str.replace('\t','')
            array_str = array_str.replace('  ','')
            array_str = array_str.replace(',',', ')
            return array_str

        json_dump = re.sub(r'\[[\s*\w*,\"]*\]',format_array, json_dump)

        return json_dump

    def get_ordering_sequence(self, code): #returns the ordering instructions as one sequence    
        #Ordering
        comps_order_instructions = re.findall(r'\w+(?:\s*\n*\s*\.\s*then\(\s*\w+\s*,\s*\w+\s*\))+;',code) # gets the lines where the computations are orderd eg: comp3.then(comp4, i1).then(comp0, i1020).then(comp2, i1).then(comp1, i1);
        orderings_list=[]
        for comps_order_instr in comps_order_instructions:
            first_comp_name = re.findall(r'(\w+)\s*\n*\s*.then', comps_order_instr)[0] # gets the name of the first computation {comp3}   .then(...).then..
            ordered_comps_tuples = re.findall(r'.\s*then\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)', comps_order_instr) # gets an orderd list (of tuples) of rest of the computations ordering eg: comp3.then(  {comp4, i1}  ).then(  {comp0, i1020}  )
            ordered_comps_tuples.insert(0, (first_comp_name, ''))
            if (len(orderings_list)==0): #first oreding line
                orderings_list.extend(ordered_comps_tuples)
            elif (ordered_comps_tuples[0][0]==orderings_list[-1][0]): #the new ordering line starts at the end of the previous
                ordered_comps_tuples.remove(ordered_comps_tuples[0])
                orderings_list.extend(ordered_comps_tuples)
            elif (ordered_comps_tuples[-1][0]==orderings_list[0][0]): #the new ordering line ends at the begining of the previous
                orderings_list.remove(orderings_list[0])
                orderings_list= ordered_comps_tuples + orderings_list           
            else: # WARNING: this can cause an infinite loop if the input code is incoherent 
                comps_order_instructions.append(comps_order_instr) # if we can't yet chain it with the other instructions, postpone the instruction till later
    #             print(orderings_list)
    #             print(ordered_comps_tuples)
    #             raise Exception('This case of ordering is not yet supported by the parser ', comps_order_instructions,re.findall(r'tiramisu::init\s*\(\s*\"(\w*)\"\s*\)', code))
        return orderings_list

    def get_schedule_annotation(self, filename):

        with open(filename,'r') as file:
            code = file.read()
        tile_2d_list = re.findall(r'(\w*)\s*.\s*tile\s*\(\s*(\w*)\s*,\s*(\w*)\s*,\s*(\d*)\s*,\s*(\d*)\s*,\s*(\w*)\s*,\s*(\w*)\s*,\s*(\w*)\s*,\s*(\w*)\s*\);',code) # gets tuples comp0.tile(i0, i1, 32, 64, i01, i02, i03, i04); --> (comp0,i0, i1, 32, 64, i01, i02, i03, i04)
        tile_3d_list = re.findall(r'(\w*)\s*.\s*tile\s*\(\s*(\w*)\s*,\s*(\w*)\s*,\s*(\w*)\s*,\s*(\d*)\s*,\s*(\d*)\s*,\s*(\d*)\s*,\s*(\w*)\s*,\s*(\w*)\s*,\s*(\w*)\s*,\s*(\w*)\s*,\s*(\w*)\s*,\s*(\w*)\s*\);',code) # same as previous with 3d tiling
        interchange_list = re.findall(r'(\w*)\s*.\s*interchange\s*\(\s*(\w*)\s*,\s*(\w*)\s*\);', code) #gets tuples comp0.interchange(i1, i2); --> (comp0, i1, i2);
        unroll_list = re.findall(r'(\w*)\s*.\s*unroll\s*\(\s*(\w*)\s*,\s*(\d*)\s*\);', code) # gets tuples comp3.unroll(i1100, 8); -->  (comp3, i1100, 8)

        schedules_dict = dict()
        schedule_name = re.findall(r'tiramisu::init\s*\(\s*\"(\w*)\"\s*\)', code)[0]
        schedules_dict['schedule_name'] = schedule_name

        #get all the computation names
        computations_list = []
        comp_lines = re.findall(r'\n\s*computation\s+.*;',code)
        for comp_def in comp_lines: #assuming that each computation is declared in a separate line
            computations_list.append(re.findall(r'(\w*)\s*\(\s*\"',comp_def)[0])
        for comp in computations_list:
            schedules_dict[comp] = {'interchange_dims':[], 'tiling' : {}, 'unrolling_factor' : None }

        for tile_2d_tuple in tile_2d_list:
            schedules_dict[tile_2d_tuple[0]]['tiling']['tiling_depth'] = 2
            schedules_dict[tile_2d_tuple[0]]['tiling']['tiling_dims'] = [tile_2d_tuple[1], tile_2d_tuple[2]]
            schedules_dict[tile_2d_tuple[0]]['tiling']['tiling_factors'] = [tile_2d_tuple[3], tile_2d_tuple[4]]

        for tile_3d_tuple in tile_3d_list:
            schedules_dict[tile_3d_tuple[0]]['tiling']['tiling_depth'] = 3
            schedules_dict[tile_3d_tuple[0]]['tiling']['tiling_dims'] = [tile_3d_tuple[1], tile_3d_tuple[2], tile_3d_tuple[3]]
            schedules_dict[tile_3d_tuple[0]]['tiling']['tiling_factors'] = [tile_3d_tuple[4], tile_3d_tuple[5], tile_3d_tuple[6]]

        for interchange_tuple in interchange_list :
            schedules_dict[interchange_tuple[0]]['interchange_dims']= [interchange_tuple[1], interchange_tuple[2]]

        for unroll_tuple in unroll_list:
            schedules_dict[unroll_tuple[0]]['unrolling_factor'] = unroll_tuple[2]

        #Computation ordering 
          #get the computations iterators first
        comp_lines = re.findall(r'\n\s*computation\s+.*;',code)
        computation_dict=dict()
        for comp_def in comp_lines: #assuming that each computation is declared in a separate line
            name = re.findall(r'(\w*)\s*\(\s*\"',comp_def)[0]
            comp_iterators = re.findall(r'\w+',re.findall(r'{([\w\s,]*)}',comp_def)[0]) # gets a list of iterators
            computation_dict[name]=dict()
            computation_dict[name]['iterators']=comp_iterators
          #get the ordering seq
        orderings_list = self.get_ordering_sequence(code)
        unfuse_iterators=[]
        if len(orderings_list)>0:# Computation ordering is defined
            for i in range(1,len(orderings_list)):
                fused_at = orderings_list[i][1]
                for depth in range(min(len(computation_dict[orderings_list[i][0]]['iterators']),len(computation_dict[orderings_list[i-1][0]]['iterators']))):
                    if computation_dict[orderings_list[i][0]]['iterators'][depth]==computation_dict[orderings_list[i-1][0]]['iterators'][depth]:
                        deepest_shared_loop = computation_dict[orderings_list[i][0]]['iterators'][depth]
                    else:
                        break
                if deepest_shared_loop!=fused_at: #the fuse iterator is not the deepest shared loop between the two computations
                    unfuse_iterators.append(fused_at)
        schedules_dict['unfuse_iterators']=unfuse_iterators

    # custructing the tree structure from ordering instruction
        tree = dict()
        last_path = []
        if orderings_list==[] : #no ordering is defined, we assume that there is only one computation
            dummy_ord_tuple = (list(computation_dict.keys())[0], '')
            orderings_list.append(dummy_ord_tuple)
        for comp_name,fuse_at in orderings_list:
            if fuse_at == '': # first computation
                fuse_at = computation_dict[comp_name]['iterators'][0]
                tree['loop_name']=fuse_at+'_c'+comp_name[-1] # the tree root is the first loop, adding suffix for readability
                tree['computations_list']=[]
    #             index_lastpath = 0
                last_path=[fuse_at+'_c'+comp_name[-1]]

            for index,i in enumerate(last_path):
                if i.startswith(fuse_at):
                    index_lastpath=index
                    break

            tree_browser = tree
            index_comp_iterators = computation_dict[comp_name]['iterators'].index(fuse_at)
            del last_path[index_lastpath+1:]
            for itr in computation_dict[comp_name]['iterators'][index_comp_iterators+1:]: # update last_path to with the new computation iterators
                last_path.append(itr+'_c'+comp_name[-1])
    #         print(comp_name,fuse_at,last_path,last_path[:index_lastpath+1])
            for itr in last_path[:index_lastpath+1]: #browse untill the node of fuse_at 
                if tree_browser['loop_name'] == itr: #for the first computation, TODO: check if this intruction in necessary
                    continue    
    #             print(filename, itr, tree_browser['loop_name'])
                for child in tree_browser['child_list']:
                    if child['loop_name']==itr:
                        tree_browser=child
                        break
            for itr in computation_dict[comp_name]['iterators'][index_comp_iterators+1:]: #builld the rest of the branch
                new_tree = dict()
                new_tree['loop_name'] = itr+'_c'+comp_name[-1]
                new_tree['computations_list']=[]
    #             new_tree['has_comps'] = False
                tree_browser['child_list'] = tree_browser.get('child_list',[])
                tree_browser['child_list'].append(new_tree)
                tree_browser = new_tree
    #         tree_browser['has_comps'] = True
            tree_browser['computations_list'] = tree_browser.get('computations_list', [])
            tree_browser['computations_list'].append(comp_name)
            tree_browser['child_list'] = tree_browser.get('child_list',[])

        schedules_dict['tree_structure'] = tree




        # custructing the tree structure from ordering instruction
        tree = dict()
        last_path = []
        for comp_name,fuse_at in orderings_list:
            if fuse_at == '': # first computation
                fuse_at = computation_dict[comp_name]['iterators'][0]
                tree['loop_name']=fuse_at+'_c'+comp_name[-1] # the tree root is the first loop, adding suffix for readability
                tree['computations_list']=[]
    #             index_lastpath = 0
                last_path=[fuse_at+'_c'+comp_name[-1]]

            for index,i in enumerate(last_path):
                if i.startswith(fuse_at):
                    index_lastpath=index
                    break

            tree_browser = tree
            index_comp_iterators = computation_dict[comp_name]['iterators'].index(fuse_at)
            del last_path[index_lastpath+1:]
            for itr in computation_dict[comp_name]['iterators'][index_comp_iterators+1:]: # update last_path to with the new computation iterators
                last_path.append(itr+'_c'+comp_name[-1])
    #         print(comp_name,fuse_at,last_path,last_path[:index_lastpath+1])
            for itr in last_path[:index_lastpath+1]: #browse untill the node of fuse_at 
                if tree_browser['loop_name'] == itr: #for the first computation, TODO: check if this intruction in necessary
                    continue    
    #             print(filename, itr, tree_browser['loop_name'])
                for child in tree_browser['child_list']:
                    if child['loop_name']==itr:
                        tree_browser=child
                        break
            for itr in computation_dict[comp_name]['iterators'][index_comp_iterators+1:]: #builld the rest of the branch
                new_tree = dict()
                new_tree['loop_name'] = itr+'_c'+comp_name[-1]
                new_tree['computations_list']=[]
    #             new_tree['has_comps'] = False
                tree_browser['child_list'] = tree_browser.get('child_list',[])
                tree_browser['child_list'].append(new_tree)
                tree_browser = new_tree
    #         tree_browser['has_comps'] = True
            tree_browser['computations_list'] = tree_browser.get('computations_list', [])
            tree_browser['computations_list'].append(comp_name)
            tree_browser['child_list'] = tree_browser.get('child_list',[])

        schedules_dict['tree_structure'] = tree



        schedules_json = schedules_dict
        sched_json_dump = json.dumps(schedules_json, indent = 4)

        def format_array(array_str):
            array_str = array_str.group()    
            array_str = array_str.replace('\n','')
            array_str = array_str.replace('\t','')
            array_str = array_str.replace('  ','')
            array_str = array_str.replace(',',', ')
            return array_str
        sched_json_dump = re.sub(r'\[[\s*\w*,\"]*\]',format_array, sched_json_dump)



        return sched_json_dump


    def generate_json_annotations(self, programs_folder):    
        program_names = listdir(programs_folder)
        program_names = sorted(filter(lambda x:not (x.endswith('.json') or x.startswith('.') or x.startswith('-') or x.startswith('_')), program_names))

        for program in tqdm(program_names):
            function_annotation = self.get_function_annotation(programs_folder + '/' + program + '/' + program + '_no_schedule' + '/' + program + '_no_schedule.cpp')
            with open(programs_folder + '/' + program + '/' + program + '_fusion_v3.json', 'w') as file:
                file.write(function_annotation)
            schedule_names = sorted(filter(lambda x:not (x.endswith('.json') or x.startswith('.') or x.startswith('-') or x.startswith('_')), listdir(programs_folder + '/' + program)))
            for schedule in schedule_names: 
                schedule_annotation = self.get_schedule_annotation(programs_folder + '/' + program + '/' + schedule + '/' + schedule +'.cpp')
                with open(programs_folder + '/' + program + '/' + schedule + '/' + schedule + '_fusion_v3.json', 'w') as file:
                    file.write(schedule_annotation)

    def generate_json_annotations_parallel(self, programs_folder,nb_threads=24): # same as previous but parallelized
        program_names = listdir(programs_folder)
        program_names = sorted(filter(lambda x:not (x.endswith('.json') or x.startswith('.') or x.startswith('-') or x.startswith('_')), program_names))
        programs_folder_list = [programs_folder]*len(program_names)
        args_tuples = list(zip(programs_folder_list,program_names))
        with Pool(nb_threads) as pool:
            pool.map(self.write_annots_parallel, args_tuples)
            
    def write_annots_parallel(self, args_tuple):
        programs_folder, program = args_tuple
        function_annotation = self.get_function_annotation(programs_folder + '/' + program + '/' + program + '_no_schedule' + '/' + program + '_no_schedule.cpp')
        with open(programs_folder + '/' + program + '/' + program + '_fusion_v3.json', 'w') as file:
            file.write(function_annotation)
        schedule_names = sorted(filter(lambda x:not (x.endswith('.json') or x.startswith('.') or x.startswith('-') or x.startswith('_')), listdir(programs_folder + '/' + program)))
        for schedule in schedule_names: 
            schedule_annotation = self.get_schedule_annotation(programs_folder + '/' + program + '/' + schedule + '/' + schedule +'.cpp')
            with open(programs_folder + '/' + program + '/' + schedule + '/' + schedule + '_fusion_v3.json', 'w') as file:
                file.write(schedule_annotation)

                
class dataset_utilities():
    def __init__(self):
        pass
    
    # Path to the files generated by the execution jobs
    # example src_path = "/data/scratch/mmerouani/time_measurement/results_batch11001-11500/parts/"
    def post_process_exec_times(self, partial_exec_times_folder):
        """
        Fuse the files generated by the execution jobs to a single file.

        The final file will be a dictionary with this format:
        { 
            'func_name1':{
                'sched_name1': {
                    'exec_time':<> 
                    'speedup':<>
                }
                'sched_name2': {
                    'exec_time':<> 
                    'speedup':<>
                }
                ...
            }
            'func_name2':{
                'sched_name1': {
                    'exec_time':<> 
                    'speedup':<>
                }
                'sched_name2': {
                    'exec_time':<> 
                    'speedup':<>
                }
                ...
            }
            ...
        }
        """
        src_path = Path(partial_exec_times_folder)    
        final_exec_times = []

        # Fuse all execution times to a single list
        for file_path in src_path.iterdir():
            if file_path.name.startswith("final_exec_times"):
                with open(file_path, "rb") as f:
                    final_exec_times.extend(pickle.load(f))

        print(f'nb schedules {len(final_exec_times)}')

        # Compute the medians
        final_exec_times_median = []

        for x in final_exec_times:
            func_id, sched_id, e = x
            final_exec_times_median.append((func_id, sched_id, e, np.median(e)))

        # Compute the speedups
        ref_progs = dict()

        for x in final_exec_times_median:
            func_id, sched_id, _, median = x
            if sched_id.endswith("no_schedule"):
                if (func_id in ref_progs):
                    print("duplicate found, taking non zero "+ str((func_id, ref_progs[func_id],median )))
                    if (median==0):
                        continue # if zero keep the old value
                ref_progs[func_id] = median

        final_exec_times_median_speedup = []
        for x in final_exec_times_median:
            func_id, sched_id, e, median = x
            if (median == 0):
                speedup = np.NaN
            else:
                if not (func_id in ref_progs):
                    print("error ref_prog", func_id," not found")
                    continue
                speedup = float(ref_progs[func_id] / median)

            final_exec_times_median_speedup.append((func_id, sched_id, e, median, speedup))


    #     # Save results
    #     with open(dst_path, "wb") as f:
    #         pickle.dump(final_exec_times_median_speedup, f)

        # Transform to dict
        final_exec_times_median_speedup = sorted(final_exec_times_median_speedup, key=lambda x: x[1])
        programs = dict()
        for l in final_exec_times_median_speedup:
            programs[l[0]]=programs.get(l[0], dict())
            if (l[1] in programs[l[0]]): #duplicate found
                if (l[3]==0): # if the new is zero, dont change the value
                    print("duplicate found, taking the oldest"+str((l[1], programs[l[0]][l[1]]['exec_time'],l[3])))
                    continue 
                else: 
                    print("duplicate found, taking the latest"+str((l[1], programs[l[0]][l[1]]['exec_time'],l[3])))

            programs[l[0]][l[1]]=dict()

            programs[l[0]][l[1]]['exec_time']=l[3]/1000
            programs[l[0]][l[1]]['speedup']=l[4]

        return programs


    def merge_datasets(self, datasets_file_list, result_dataset_file):
        '''
        merges multiple dataset file into a single one
        eg:
        merge_datasets(['/data/scratch/mmerouani/processed_datasets/dataset_batch250000-254999.pkl',
                        '/data/scratch/mmerouani/processed_datasets/dataset_batch255000-259999.pkl',
                        '/data/scratch/mmerouani/processed_datasets/dataset_batch260000-264999.pkl'],
                        result_dataset_file='/data/scratch/mmerouani/processed_datasets/dataset_batch250000-264999.pkl')
        '''
        merged_programs_dict = dict()
        for dataset_filename in tqdm(datasets_file_list):
            f = open(dataset_filename, 'rb')
            programs_dict=pickle.load(f)
            f.close()
            merged_programs_dict.update(programs_dict)

        f = open(result_dataset_file, 'wb')
        pickle.dump(merged_programs_dict, f)
        f.close()


    def save_pkl_dataset(self, programs_folder, partial_exec_times_folder, output_filename):
        '''
        Creates and dumps a dataset as dictionary with this format

        programs_dict={
            'func_name1':{
                'json':<function annotation file content>
                'schedules':{
                    'schedule_name1':{ 
                        'json':<schedule annotation file content>
                        'exec_time': <value>
                        'speedup': <value>
                    }
                    'schedule_name2':{ 
                        'json':<schedule annotation file content>
                        'exec_time': <value>
                        'speedup': <value>
                    }
                    ...
                }
            }
            'func_name2':{
                'json':<function annotation file content>
                'schedules':{
                    'schedule_name1':{ 
                        'json':<schedule annotation file content>
                        'exec_time': <value>
                        'speedup': <value>
                    }
                    'schedule_name2':{ 
                        'json':<schedule annotation file content>
                        'exec_time': <value>
                        'speedup': <value>
                    }
                    ...
                }
            }
            ...
        }
        '''
        exec_dict= self.post_process_exec_times(partial_exec_times_folder)

        program_names = listdir(programs_folder)
        program_names = sorted(filter(lambda x:not (x.startswith('.') or x.startswith('-') or x.startswith('_')),program_names))
        programs_dict = dict()

        for program in tqdm(program_names):
            programs_dict[program] = dict()
            programs_dict[program]['json'] = json.load(open(programs_folder + '/' + program + '/' + program + '_fusion_v3.json', 'r'))
            schedule_names = sorted(filter(lambda x:not (x.endswith('.json') or x.startswith('.') or x.startswith('-') or x.startswith('_')), listdir(programs_folder + '/' + program)))
            programs_dict[program]['schedules'] = dict()
            if not (program in exec_dict):
                print('error ',program,' not found in exec_dict')
                continue
            for schedule in schedule_names:
                if not (schedule in exec_dict[program]):
                    print ("error schedules ",schedule," not found in exec_dict")
                    continue
                programs_dict[program]['schedules'][schedule] = dict()
    #             print(schedule)
                programs_dict[program]['schedules'][schedule]['json'] = json.load(open(programs_folder + '/' + program + 
                                                                            '/' + schedule + '/' + schedule + '_fusion_v3.json', 'r'))
                programs_dict[program]['schedules'][schedule]['exec_time'] = exec_dict[program][schedule]['exec_time']
                programs_dict[program]['schedules'][schedule]['speedup'] = exec_dict[program][schedule]['speedup']

        f = open(output_filename, 'wb')
        pickle.dump(programs_dict, f)
        f.close()
        
    def get_dataset_df(self, dataset_filename):

        dataset_name=dataset_filename

        f = open(dataset_filename, 'rb')
        programs_dict=pickle.load(f)
        f.close()


        Y = []
        program_names = []
        schedule_names = []
        exec_time = []
        nb_nan=0

        for function_name in tqdm(programs_dict):
            program_json = programs_dict[function_name]['json']
            for schedule_name in programs_dict[function_name]['schedules']:
                if ((np.isnan(programs_dict[function_name]['schedules'][schedule_name]['speedup']))
                     or(programs_dict[function_name]['schedules'][schedule_name]['speedup']==0)): #Nan value means the schedule didn't run, zero values means exec time<1 micro-second, skip them
                    nb_nan+=1
                    continue

                schedule_json = programs_dict[function_name]['schedules'][schedule_name]['json']
                Y.append(programs_dict[function_name]['schedules'][schedule_name]['speedup'])
                program_names.append(function_name)
                schedule_names.append(schedule_name)
                exec_time.append(programs_dict[function_name]['schedules'][schedule_name]['exec_time'])

        print('Dataset location: ',dataset_filename)
        print(f'Number of schedules {len(Y)}')

        df = pd.DataFrame(data=list(zip(program_names,schedule_names,exec_time,Y)),
                          columns=['function', 'schedule', 'exec_time',  'speedup'])
        print("Schedules that didn't run: {:.2f}%".format(nb_nan/len(Y)*100) )
        print('Speedups >1 :{:.2f}%'.format(len(df['speedup'][df['speedup']>1])/len(df['speedup'])*100))
        print('Speedups >2 :{:.2f}%'.format(len(df['speedup'][df['speedup']>2])/len(df['speedup'])*100))
        print('Speedups <0.1 :{:.2f}%'.format(len(df['speedup'][df['speedup']<0.1])/len(df['speedup'])*100))
        print('Speedups 0.9<s<1.1 :{:.2f}%'.format((len(df['speedup'][df['speedup']<1.1])-len(df['speedup'][df['speedup']<0.9]))/len(df['speedup'])*100))
        print('Mean speedup: {:.2f}'.format( df['speedup'].mean()))
        print('Median speedup: {:.2f}'.format( np.median(df['speedup'])))
        print('Max speedup: {:.2f}'.format(df['speedup'].max()))
        print('Min speedup: {:.3f}'.format(df['speedup'].min()))
        print('Speedup variance : {:.3f}'.format((df['speedup']).var()))
        print('Mean execution time: {:.3f}s'.format(df.exec_time.mean()/1000))
        print('Max execution time: {:.3f}s'.format(df.exec_time.max()/1000))
        return df