import re
import os
import json
from pathlib import Path
import random, time
from utilsLocal import TimeOutException, compile_and_run_tiramisu_code, launch_cmd

class InternalExecException(Exception):
    pass

class Tiramisu_Program():
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.original_str = f.read()
        self.func_folder = ('/'.join(Path(file_path).parts[:-1]) if len(Path(file_path).parts)>1 else '.') +'/'
        self.body = re.findall(r'(tiramisu::init(?s:.)+)tiramisu::codegen', self.original_str)[0]
        self.name = re.findall(r'tiramisu::init\(\"(\w+)\"\);', self.original_str)[0]
        self.comp_name = re.findall(r'computation (\w+)\(', self.original_str)[0]
        self.code_gen_line = re.findall(r'tiramisu::codegen\({.+;', self.original_str)[0]
        buffers_vect = re.findall(r'{(.+)}', self.code_gen_line)[0]
        self.IO_buffer_names = re.findall(r'\w+', buffers_vect)
        self.buffer_sizes = []
        for buf_name in self.IO_buffer_names:
            sizes_vect = re.findall(r'buffer '+buf_name+'.*{(.*)}', self.original_str)[0]
            self.buffer_sizes.append(re.findall(r'\d+',sizes_vect))
        self.program_annotations = ''
        self.wrapper_is_compiled = False
        
    
    def get_program_annotations(self):
        if not self.program_annotations=='':
            return self.program_annotations
        get_json_lines = '''
    auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function());
    std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_program_json(ast);
    std::ofstream out("'''+self.func_folder+self.name+'''_program_annotations.json");
    out << program_json;
    out.close();
    '''
        get_json_prog = self.original_str.replace(self.code_gen_line, get_json_lines)
        output_file = self.func_folder+self.name+'_get_prog_annot.cpp'
        with open(output_file, 'w') as f:
            f.write(get_json_prog)
        compile_and_run_tiramisu_code(output_file, 'Generating program annotations')
        with open(self.func_folder+self.name+'_program_annotations.json','r') as f:
            self.program_annotations = json.loads(f.read())
        return self.program_annotations
    
    def check_legality_of_schedule(self,optims_list): #Optims_list should be the order of which they should be applied # works only for single comp
        legality_check_lines = '''
    prepare_schedules_for_legality_checks();
    perform_full_dependency_analysis();
    
    bool is_legal=true;
'''
#         if_not_legal_quit = ""
#         check_legality_for_func_line = 
        for optim in optims_list:
            if optim.type == 'Interchange':
                legality_check_lines += optim.tiramisu_optim_str+'\n'
            elif optim.type == 'Reversal':
                legality_check_lines += optim.tiramisu_optim_str+'\n'
            elif optim.type == 'Skewing':
                legality_check_lines += optim.tiramisu_optim_str+'\n'
            elif optim.type == 'Parallelization':
                legality_check_lines += '''
    is_legal &= loop_parallelization_is_legal('''+str(optim.params_list[0])+''', {&'''+self.comp_name+'''});
'''
                legality_check_lines += optim.tiramisu_optim_str+'\n' #not sure if this line is necessary
            elif optim.type == 'Tiling':
                legality_check_lines += optim.tiramisu_optim_str+'\n'
            elif optim.type == 'Unrolling':
                legality_check_lines += '''
    is_legal &= loop_unrolling_is_legal('''+str(optim.params_list[0])+''', {&'''+self.comp_name+'''});
'''
                legality_check_lines += optim.tiramisu_optim_str+'\n' #not sure if this line is necessary
                
        legality_check_lines+='''
    is_legal &= check_legality_of_function();
    
    std::ofstream out("'''+self.func_folder+'''legality_check_result.txt");
    out << is_legal;
    out.close();
        '''
        
        LC_code = self.original_str.replace(self.code_gen_line, legality_check_lines)
        # print("\n *-*-*- Legality Check File -*-*-* \n")
        # print(LC_code)
        output_file = self.func_folder+self.name+'_legality_check.cpp'
        with open(output_file, 'w') as f:
            f.write(LC_code)
        self.reset_legality_check_result_file()
        log_message = 'Checking legality for: ' + ' '.join([o.tiramisu_optim_str for o in optims_list])
        compile_and_run_tiramisu_code(output_file, log_message)
        lc_result = self.read_legality_check_result_file()
        
        return lc_result



    def call_solver(self, comp, params):#single comp
        lc_file=self.func_folder+self.name+'_legality_check.cpp'
        if os.path.isfile(lc_file):
            with open(lc_file, 'r') as f:
                original_str = f.read()

            to_replace = re.findall(r'(std::ofstream out(?s:.)+)return', original_str)[0]
            header="function * fct = tiramisu::global::get_implicit_function();\n"
        else:
            
            original_str=self.original_str
            to_replace=self.code_gen_line
            header='''
    
    perform_full_dependency_analysis();
    prepare_schedules_for_legality_checks();
    function * fct = tiramisu::global::get_implicit_function();
    '''
        
        
        solver_lines=header + "auto auto_skewing_result = fct->skewing_local_solver({&" + comp + "}},{},{},1);\n".format(params["first_dim_index"],params["second_dim_index"])

        solver_lines+='''
    std::ofstream out("'''+self.func_folder+'''solver_result.txt");
    std::vector<std::pair<int,int>> outer1, outer2,outer3;
    tie( outer1,  outer2,  outer3 )= auto_skewing_result;
    if (outer1.size()>0){
            out << outer1.front().first << std::endl;
            out << outer1.front().second << std::endl;
        }
    if (outer2.size()>0){
            out << outer2.front().first << std::endl;
            out << outer2.front().second << std::endl;
        }
    if (outer3.size()>0){
        out << outer3.front().first << std::endl;
        out << outer3.front().second << std::endl;
    }
    
    '''
    #for (const auto &e : outer) out << e ;
    #std::tuple<int, int>
        solver_code = original_str.replace(to_replace, solver_lines)
        
        output_file = self.func_folder+self.name+'_solver.cpp'
        
        with open(output_file, 'w') as f:
            f.write(solver_code)
        self.reset_solver_result_file()
        
        log_message = 'Solver results for: computation {}'.format(comp) + ' '.join([p for p in params])
        if compile_and_run_tiramisu_code(output_file, log_message):
            solver_result = self.read_solver_result_file()
            if len(solver_result) == 0:
                return None
            else:
                return solver_result
        else:
            raise InternalExecException

    
    def evaluate_schedule(self, optims_list, cmd_type, nb_executions, initial_exec_time=None):
        optim_lines = ''
        for optim in optims_list:
            if optim.type == 'Interchange':
                optim_lines += optim.tiramisu_optim_str+'\n'
            elif optim.type == 'Skewing':
                optim_lines += optim.tiramisu_optim_str+'\n'
            elif optim.type == 'Parallelization':
                optim_lines += optim.tiramisu_optim_str+'\n'
            elif optim.type == 'Tiling':
                optim_lines += optim.tiramisu_optim_str+'\n'
            elif optim.type == 'Unrolling':
                optim_lines += optim.tiramisu_optim_str+'\n'
            elif optim.type == 'Reversal':
                optim_lines += optim.tiramisu_optim_str+'\n'
               
        codegen_code = self.original_str.replace(self.code_gen_line, optim_lines + '\n' + self.code_gen_line.replace(self.name,self.func_folder+self.name))
        output_file = self.func_folder+self.name+'_schedule_codegen.cpp'
        # print("\n *-*-*- Le fichier du code -*-*-* \n")
        # print(codegen_code)
        with open(output_file, 'w') as f:
            f.write(codegen_code)
        log_message = 'Applying schedule: ' + ' '.join([o.tiramisu_optim_str for o in optims_list])
        start_time=time.time()
        if(compile_and_run_tiramisu_code(output_file, log_message)): 
            #print("COMPILE/RUN SCHEDULE CODEGEN :\n",time.time()- start_time) 
            try:
                execution_times = self.get_measurements(cmd_type, nb_executions, initial_exec_time)
                if len(execution_times)!=0:
                    #print("execution times are: ",execution_times)
                    return min(execution_times)
                else:
                    return 0
            except TimeOutException: 
                return 10*nb_executions*initial_exec_time
        else:
            raise InternalExecException
        
    def get_measurements(self, cmd_type, nb_executions, initial_exec_time):
        os.environ['FUNC_DIR'] = ('/'.join(Path(self.file_path).parts[:-1]) if len(Path(self.file_path).parts)>1 else '.') +'/'
        os.environ['FILE_PATH'] = self.file_path
        os.environ['FUNC_NAME'] = self.name
        if not self.wrapper_is_compiled:
            self.write_wrapper_code()
            log_message_cmd = 'printf "Compiling wrapper\n">> ${FUNC_DIR}log.txt'
            compile_wrapper_cmd = 'cd ${FUNC_DIR};\
            g++ -shared -o ${FUNC_NAME}.o.so ${FUNC_NAME}.o;\
            g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/benchmarks -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -o ${FUNC_NAME}_wrapper -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ./${FUNC_NAME}_wrapper.cpp ./${FUNC_NAME}.o.so -ltiramisu -lHalide -ldl -lpthread -lz -lm'
            launch_cmd(log_message_cmd,'')
            failed = launch_cmd(compile_wrapper_cmd, self.file_path)
            if failed:
                print('Failed compiling wrapper')
                return
            self.wrapper_is_compiled = True
        self.reset_measurements_file()
        log_message_cmd = 'printf "Running wrapper nb_exec = '+str(nb_executions)+'\n">> ${FUNC_DIR}log.txt'
        run_wrapper_cmd = 'cd ${FUNC_DIR};\
        g++ -shared -o ${FUNC_NAME}.o.so ${FUNC_NAME}.o;\
        ./${FUNC_NAME}_wrapper '+str(nb_executions)
        launch_cmd(log_message_cmd, '')
        s_time=time.time()
        failed = launch_cmd(run_wrapper_cmd, self.file_path, cmd_type,nb_executions,initial_exec_time)
        #print("WRAPPER RUN in : ", time.time()-s_time)
        
        if failed:
            print('Failed running wrapper')
            return
        return self.read_measurements_file()

    def write_wrapper_code(self): #construct the wrapper.cpp and wrapper.h from the program
        
        buffers_init_lines = ''
        for i,buffer_name in enumerate(self.IO_buffer_names):
            buffers_init_lines+=f'''
    double *c_{buffer_name} = (double*)malloc({'*'.join(self.buffer_sizes[i][::-1])}* sizeof(double));
    parallel_init_buffer(c_{buffer_name}, {'*'.join(self.buffer_sizes[i][::-1])}, (double){str(random.randint(1,10))});
    Halide::Buffer<double> {buffer_name}(c_{buffer_name}, {','.join(self.buffer_sizes[i][::-1])});
    '''
        wrapper_cpp_code = wrapper_cpp_template.replace('$func_name$',self.name)
        wrapper_cpp_code = wrapper_cpp_code.replace('$buffers_init$',buffers_init_lines)
        wrapper_cpp_code = wrapper_cpp_code.replace('$func_folder_path$',self.func_folder)
        wrapper_cpp_code = wrapper_cpp_code.replace('$func_params$',','.join([name+'.raw_buffer()' for name in self.IO_buffer_names]))
        output_file = self.func_folder+self.name+'_wrapper.cpp'
        with open(output_file, 'w') as f:
            f.write(wrapper_cpp_code)
        
        wrapper_h_code = wrapper_h_template.replace('$func_name$',self.name)
        wrapper_h_code = wrapper_h_code.replace('$func_params$',','.join(['halide_buffer_t *'+name for name in self.IO_buffer_names]))
        output_file = self.func_folder+self.name+'_wrapper.h'
        with open(output_file, 'w') as f:
            f.write(wrapper_h_code)
                                                

    def read_legality_check_result_file(self):
        with open(self.func_folder+"legality_check_result.txt",'r') as f:
            res = int(f.read())
        return res
    def reset_legality_check_result_file(self):
        with open(self.func_folder+"legality_check_result.txt",'w') as f:
            f.write('-1')
    def read_measurements_file(self):
        with open(self.func_folder+"measurements_file.txt",'r') as f:
            res = [float(i) for i in f.read().split()]
        return res
    def reset_measurements_file(self):
        with open(self.func_folder+"measurements_file.txt",'w') as f:
            f.write('-1')
    def read_solver_result_file(self):
        with open(self.func_folder+"solver_result.txt",'r') as f:
            res = f.readlines()
        return res
    def reset_solver_result_file(self):
        with open(self.func_folder+"solver_result.txt",'w') as f:
            f.write('-1')
    def clean_up_genrated_files(): # TODO: clean up all the temporary files generated in the function folder
        pass
                



wrapper_h_template = '''#include <tiramisu/utils.h>
#include <sys/time.h>
#include <cstdlib>
#include <algorithm>
#include <vector>

#define NB_THREAD_INIT 48
struct args {
    double *buf;
    unsigned long long int part_start;
    unsigned long long int part_end;
    double value;
};

void *init_part(void *params)
{
   double *buffer = ((struct args*) params)->buf;
   unsigned long long int start = ((struct args*) params)->part_start;
   unsigned long long int end = ((struct args*) params)->part_end;
   double val = ((struct args*) params)->value;
   for (unsigned long long int k = start; k < end; k++){
       buffer[k]=val;
   }
   pthread_exit(NULL);
}

void parallel_init_buffer(double* buf, unsigned long long int size, double value){
    pthread_t threads[NB_THREAD_INIT]; 
    struct args params[NB_THREAD_INIT];
    for (int i = 0; i < NB_THREAD_INIT; i++) {
        unsigned long long int start = i*size/NB_THREAD_INIT;
        unsigned long long int end = std::min((i+1)*size/NB_THREAD_INIT, size);
        params[i] = (struct args){buf, start, end, value};
        pthread_create(&threads[i], NULL, init_part, (void*)&(params[i])); 
    }
    for (int i = 0; i < NB_THREAD_INIT; i++) 
        pthread_join(threads[i], NULL); 
    return;
}
#ifdef __cplusplus
extern "C" {
#endif
int $func_name$($func_params$);
#ifdef __cplusplus
}  // extern "C"
#endif'''

wrapper_cpp_template = '''#include "Halide.h"
#include "$func_name$_wrapper.h"
#include "tiramisu/utils.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

using namespace std::chrono;
using namespace std;

int main(int, char **argv){
        
$buffers_init$

    int nb_execs = atoi(argv[1]);

    std::ofstream out("measurements_file.txt");
    double duration;
    
    for (int i = 0; i < nb_execs; ++i) {
        auto begin = std::chrono::high_resolution_clock::now(); 
        $func_name$($func_params$);
        auto end = std::chrono::high_resolution_clock::now(); 

        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000;
        out << duration << " "; 

    }
    
    out.close();
    return 0;
}'''
