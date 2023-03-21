from configuration import *
import subprocess
from pathlib import Path
from tqdm import tqdm


# general notes
# - TODO each class should have an init method that initializes the class with the given parameters and a create method that initializes it randomly, currently the init method initializes randomly
# - TODO define __str__ for each class that will be used for writing in the text files and printing

class Program:
    def __init__(self, program_id):
        self.id = program_id
        self.nb_computation = 0
        self.nb_nests = 0
        self.depth = 0
        self.name = 'function' + str(program_id).zfill(6)  # find a naming convention, probably 'function'+'id in fixed position' or function+'batch code automatically generated from config'+id
        self.ordered_computations_list = []
        self.iterators_list = []
        self.buffer_list = []
        self.input_list = []
        self.data_type = 'p_float64'  # TODO implement datatype randomization
        self.problem_size = None
        self.root_loop_list = None
        self.nb_branches = 1  # ie the number of leaf loops

    def create(self):
        # randomly initializes the program object
        local_random.seed(self.id)
        self.problem_size = choose_problem_size()
        nb_root_loop = choose_nb_root_loops()
        self.root_loop_list = [Loop(parent_loop=None, depth_level=0, siblings_order=i, program=self) for i in range(nb_root_loop)]  # create loop/compuations tree
        for loop in self.root_loop_list:
            loop.create()
        self.correct_loop_bounds()
        self.correct_buffer_sizes()
        # get_skippable_loop(self)
        if has_overwritten_comp(self) or has_skippable_loop_multi_comp(self):
            # if has_overwritten_comp(self) or has_skippable_loop_multi_comp(self):
            #     if has_overwritten_comp(self) or has_skippable_loop_multi_comp(self):
            # print(self.name)
        # print(self.write_tiramisu_program())
        #     print(self.print())
        #     raise Exception('fdsafdaf')
            pass  # we can assume that we don't need to do verification at this stage since they are done at the computation stage
            # print('----------------------------------------')

    def correct_loop_bounds(self):  # make corrections to loop bounds that are used for accesses with a negative shift
        # for now the only needed correction is by incrementing the upper and lower bounds by max(abs([negative shifts used with the iterator]) ( both bounds are incremented inorder to not change the extent)
        for comp in self.ordered_computations_list:
            for read_access in comp.expression.read_access_list:
                last_col = read_access.access_pattern[:, -1]
                if np.any(last_col < 0):
                    for i in range(len(last_col)):
                        if last_col[i] < 0:
                            shift = abs(min(last_col))
                            loop = read_access.buffer.defining_iterators[i]
                            if loop.lower_bound < shift:  # if not already shifted by the same shift value
                                loop.upper_bound -= loop.lower_bound  # in case a smaller shift has already been applied
                                loop.lower_bound = shift
                                loop.upper_bound += shift

        pass

    def correct_buffer_sizes(self):  # correct the buffer extents to avoid out of bound accesses that can be caused by stencils
        # this involves correcting the defining iterators of the wrapping inputs
        # currently correction only supports stencils, convolution like patterns need to be added
        for buffer in self.buffer_list:
            if buffer.read_list != []:  # if there are read_accesses
                last_cols_list = []
                for read_access in buffer.read_list:
                    last_col = read_access.access_pattern[:, -1]
                    last_cols_list.append(last_col)
                    # if np.any(last_col):
                    #     max_positiv = max(last_col)
                last_cols_matrix = np.array(last_cols_list).T
                for i in range(len(buffer.defining_iterators)):
                    # max_neg = 0
                    max_pos = 0
                    # if np.any(last_cols_matrix[i, :] < 0):
                    #     max_neg = max(abs(last_cols_matrix[i, :]))
                    if np.any(last_cols_matrix[i, :] > 0):
                        max_pos = max(abs(last_cols_matrix[i, :]))
                    # total_shift = max_pos+max_neg
                    if buffer.sizes[i] < (buffer.defining_iterators[i].upper_bound + max_pos):
                        buffer.sizes[i] = buffer.defining_iterators[i].upper_bound + max_pos
                        if '_p' in buffer.wrapping_input.defining_iterators[i].name:  # if the iterator has already been extended
                            if '_p' + str(max_pos) in buffer.wrapping_input.defining_iterators[i].name:  # the extension was made with the same shift
                                continue  # nothing to do
                            else:  # the extension was made with a different shift, remove the iterator and declare a new one
                                self.iterators_list.remove(buffer.wrapping_input.defining_iterators[i])
                                buffer.wrapping_input.defining_iterators[i] = Loop.input_iterator(lower_bound=0, upper_bound=buffer.sizes[i], name=buffer.defining_iterators[i].name + '_p' + str(max_pos), program=self)
                        else:  # the iterator hasn't been extended yet, create a new extended version
                            buffer.wrapping_input.defining_iterators[i] = Loop.input_iterator(lower_bound=0, upper_bound=buffer.sizes[i], name=buffer.defining_iterators[i].name + '_p' + str(max_pos), program=self)

            if buffer.write_list != []:  # check if the buffer sizes are still compatible with the iterators' corrected bound
                for i in range(len(buffer.defining_iterators)):
                    if buffer.sizes[i] < buffer.defining_iterators[i].upper_bound:
                        shift = buffer.defining_iterators[i].upper_bound - buffer.sizes[i]
                        buffer.sizes[i] = buffer.defining_iterators[i].upper_bound
                        if buffer.wrapping_input != None:  # if the computation has a wrapping input declared, update it's defining iterators
                            if '_p' in buffer.wrapping_input.defining_iterators[i].name:  # if the iterator has already been extended
                                if '_p' + str(shift) in buffer.wrapping_input.defining_iterators[i].name:  # the extension was made with the same shift
                                    continue  # nothing to do
                                else:  # the extension was made with a different shift, remove the iterator and declare a new one
                                    self.iterators_list.remove(buffer.wrapping_input.defining_iterators[i])
                                    buffer.wrapping_input.defining_iterators[i] = Loop.input_iterator(lower_bound=0, upper_bound=buffer.sizes[i], name=buffer.defining_iterators[i].name + '_p' + str(shift), program=self)
                            else:  # the iterator hasn't been extended yet, create a new extended version
                                buffer.wrapping_input.defining_iterators[i] = Loop.input_iterator(lower_bound=0, upper_bound=buffer.sizes[i], name=buffer.defining_iterators[i].name + '_p' + str(shift), program=self)

    def write_tiramisu_program(self):  # write the program object into a text file
        tiramisu_source = ''
        tiramisu_source += self.write_tiramisu_header()
        tiramisu_source += self.write_iterators()
        tiramisu_source += self.write_inputs()
        tiramisu_source += self.write_computations()
        tiramisu_source += self.write_computation_ordering()
        tiramisu_source += self.write_buffers()
        tiramisu_source += self.write_input_store()
        tiramisu_source += self.write_computation_store()
        tiramisu_source += self.write_footer()
        return tiramisu_source

    def write_tiramisu_header(self):
        return '#include <tiramisu/tiramisu.h> ' \
               '\n#include <tiramisu/auto_scheduler/evaluator.h>' \
               '\n#include <tiramisu/auto_scheduler/search_method.h>' \
                                            '\n\nusing namespace tiramisu;' \
                                            '\n\nint main(int argc, char **argv){ \
               \n\ttiramisu::init("' + self.name + '");\n'

    def write_iterators(self):
        var_line = "\tvar"
        for loop in self.iterators_list:
            var_line += " " + loop.write() + ","
        var_line = var_line[:-1]  # remove last comma
        var_line += ";\n"
        return var_line

    def write_inputs(self):
        inputs_text = ''
        for inp in self.input_list:
            inputs_text += inp.write()
        return inputs_text

    def write_computations(self):
        computations_text = ''
        for computation in self.ordered_computations_list:
            computations_text += computation.write()
        return computations_text

    def write_computation_ordering(self):  # assuming everything is max fused
        if len(self.ordered_computations_list) < 2:
            return ''  # no computation to order
        prev_comp = self.ordered_computations_list[0]
        comp_ordering = '\t' + prev_comp.name + '.then('
        for comp in self.ordered_computations_list[1:]:
            deepest_shared_loop = prev_comp.get_shared_parent_loop(comp)
            if deepest_shared_loop is None:  # no shared loop
                order_at = 'computation::root'
            else:
                order_at = deepest_shared_loop.name
            comp_ordering += comp.name + ', ' + order_at + ')\n\t\t.then('
            prev_comp = comp
        comp_ordering = comp_ordering[:-9]  # remove last \n\t.then(
        comp_ordering += ';\n'
        return comp_ordering

    def write_buffers(self):
        buffers_declaration = ''
        for buffer in self.buffer_list:
            buffers_declaration += buffer.write()
        return buffers_declaration

    def write_input_store(self):
        input_store_text = ''
        for inp in self.input_list:
            input_store_text += inp.write_store()
        return input_store_text

    def write_computation_store(self):
        computation_store_text = ''
        for comp in self.ordered_computations_list:
            computation_store_text += comp.write_store()
        return computation_store_text

    def write_footer(self):
        io_buffer_names = []
        for buf in self.buffer_list:
            if buf.type != 'a_temporary':
                io_buffer_names.append(buf.name)

        footer_text = '\ttiramisu::codegen({' + ','.join(['&' + name for name in io_buffer_names]) \
                      + '}, "' + self.name + '.o"); \n\treturn 0; \n}'

        return footer_text

    def print(self):
        print(self.name + '\n')
        for loop in self.root_loop_list:
            loop.print()

    def write_autoscheduler(self, py_cmd_path='/data/scratch/mmerouani/anaconda/bin/python', py_interface_path='/data/scratch/mmerouani/k_autosched/demo_tiramisu_autoscheduler/model/main.py'):
        autoscheduler_source = ''
        # autoscheduler_source += '#include <string>\n\n'
        # autoscheduler_source += 'const std::string py_cmd_path = "' + py_cmd_path + '";\n'
        # autoscheduler_source += 'const std::string py_interface_path = "' + py_interface_path + '";\n\n'
        autoscheduler_source += self.write_tiramisu_header()
        autoscheduler_source += self.write_iterators()
        autoscheduler_source += self.write_inputs()
        autoscheduler_source += self.write_computations()
        autoscheduler_source += self.write_computation_ordering()
        autoscheduler_source += self.write_buffers()
        autoscheduler_source += self.write_input_store()
        autoscheduler_source += self.write_computation_store()
        # autoscheduler_source += '\n\n\t'
        autoscheduler_source += '\n\tprepare_schedules_for_legality_checks();'
        autoscheduler_source += '\n\tperforme_full_dependency_analysis();'

        autoscheduler_source += '\n\n\t'
        autoscheduler_source += 'const int beam_size = get_beam_size();\n\t'
        autoscheduler_source += 'const int max_depth = get_max_depth();\n\t'
        autoscheduler_source += 'declare_memory_usage();\n\n\t'

        autoscheduler_source += 'auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();\n\t'
        autoscheduler_source += 'auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution('
        io_buffer_names = []
        for buf in self.buffer_list:
            if buf.type != 'a_temporary':
                io_buffer_names.append(buf.name)
        io_buffers = '{' + ','.join(['&' + name for name in io_buffer_names]) + '}'
        autoscheduler_source += io_buffers + ', "' + self.name + '.o", "./' + self.name + '_wrapper");\n\t'
        # currently we use beam search guided by evaluation for data generation
        autoscheduler_source += 'auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);\n\t'
        autoscheduler_source += 'auto_scheduler::auto_scheduler as(bs, exec_eval);\n\t'
        autoscheduler_source += 'as.set_exec_evaluator(exec_eval);\n\t'
        autoscheduler_source += 'as.sample_search_space("./' + self.name + '_explored_schedules.json", true);\n\t'
        autoscheduler_source += 'delete scheds_gen;\n\t'
        autoscheduler_source += 'delete exec_eval;\n\t'
        autoscheduler_source += 'delete bs;\n\t'
        autoscheduler_source += 'return 0;\n}'
        return autoscheduler_source

    def test_compilation(self):  # tests whether the tiramisu program compiles and whether the (obj file) generator runs #TODO this currently doesn't work proprely, doesn't detect/print errors
        with open('test_compil.cpp', 'w') as f:
            f.write(self.write_tiramisu_program())
        compile_cmd = 'g++ -std=c++11 -O3 -fno-rtti -mavx2 -I/include/ -I/home/masci/tiramisu/include/ ' \
                      '-I/home/masci/tiramisu/3rdParty/Halide/include/ -I/home/masci/tiramisu/3rdParty/isl/build/include/ ' \
                      'test_compil.cpp ' \
                      '-L/lib/ -L/home/masci/tiramisu/3rdParty/Halide/lib/ ' \
                      '-L/home/masci/tiramisu/3rdParty/isl/build/lib// -L/home/masci/tiramisu/build/ ' \
                      '-ltiramisu -lHalide -lisl -lz -lpthread -ldl -o generator'
        run_generator_cmd = 'LD_LIBRARY_PATH=:/home/masci/tiramisu/3rdParty/Halide/lib:/home/masci/tiramisu/3rdParty/isl/build/lib/:/home/masci/tiramisu/build/:/lib/' \
                            'DYLD_LIBRARY_PATH=:/home/masci/tiramisu/3rdParty/Halide/lib:/home/masci/tiramisu/build/:/lib/' \
                            './generator'

        compile_output = subprocess.run(compile_cmd.split(), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(compile_output.stderr, compile_output.stdout)
        run_gen_output = subprocess.run(run_generator_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(run_gen_output.stdout, run_gen_output.stdout)

    def write_wrapper(self):  # writes the program's wrapper into a text file
        return self.write_wrapper_header() + self.write_wrapper_buffers() + self.write_wrapper_footer()

    def write_wrapper_header(self):
        return '#include "Halide.h"\n#include "' + self.name + '_wrapper.h"\n#include "tiramisu/utils.h"\n#include <iostream>\n#include <time.h>\n#include <fstream>\n#include <chrono>\nusing namespace std::chrono;\nusing namespace std;\
                \nint main(int, char **argv)\n{\n'

    def write_wrapper_buffers(self):
        buffer_declaration = ''
        buffer_initialization = ''
        for buffer in self.buffer_list:
            if buffer.type == 'a_output':
                buffer_declaration += buffer.write_wrapper()
            elif buffer.type == 'a_input':
                buffer_initialization += buffer.write_wrapper() + '\n'
        return buffer_declaration + '\n' + buffer_initialization

    def write_wrapper_footer(self):
        func_call = self.name + '(' + ','.join([buffer.name + '.raw_buffer()' for buffer in self.buffer_list if buffer.type != 'a_temporary']) + ');'
        # text = '\tstd::vector<double> duration_vector;\n\tdouble start, end;\n\n\tfor (int i = 0; i < 1; ++i) '
        # text = '\tint nb_exec = get_nb_exec();\n'
        # text += '\n\t' + func_call
        # text += '\n\tfor (int i = 0; i < 5; i++) \n\t{  \n\t\tstart = rtclock(); \n\t'+func_call
        # text += '\t\tend = rtclock(); \n\n\t\tduration_vector.push_back((end - start) * 1000); \n\t}\n\n\tstd::cout << median(duration_vector) << std::endl;\n\n\treturn 0; \n}\n'
        # text += '\n\tfor (int i = 0; i < nb_exec; i++) \n\t{  \n\t\tauto begin = std::chrono::high_resolution_clock::now(); \n\t' + func_call
        # text += '\t\tauto end = std::chrono::high_resolution_clock::now(); \n\n' \
        #         '\t\tstd::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; \n' \
        #         '\t}\n' \
        #         '\t\tstd::cout << std::endl;\n\n'\
        #         '\treturn 0; \n' \
        #         '}\n'

        text = '''\
    bool nb_runs_dynamic = is_nb_runs_dynamic();
    
    if (!nb_runs_dynamic){ 
        
        int nb_exec = get_max_nb_runs();    
        for (int i = 0; i < nb_exec; i++) 
        {  
            auto begin = std::chrono::high_resolution_clock::now(); 
            '''+func_call+'''
            auto end = std::chrono::high_resolution_clock::now(); 

            std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
        }
    }
    
    else{ // Adjust the number of runs depending on the measured time on the firs runs
    
        std::vector<double> duration_vector;
        double duration;
        int nb_exec = get_min_nb_runs();    
        
        for (int i = 0; i < nb_exec; i++) 
        {  
            auto begin = std::chrono::high_resolution_clock::now(); 
            '''+func_call+'''
            auto end = std::chrono::high_resolution_clock::now(); 

            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000;
            std::cout << duration << " "<< std::flush; 
            duration_vector.push_back(duration);
        }

        int nb_exec_remaining = choose_nb_runs(duration_vector);

        for (int i = 0; i < nb_exec_remaining; i++) 
        {  
            auto begin = std::chrono::high_resolution_clock::now(); 
            '''+func_call+'''
            auto end = std::chrono::high_resolution_clock::now(); 

            std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() / (double)1000000 << " " << std::flush; 
        }
    }
    std::cout << std::endl;

	return 0; 
}

        '''
        return text

    # v.erase(std::remove(v.begin(), v.end(), 0.0), v.end());
    def write_wrapper_h(self):
        dtype = ''
        if self.data_type == 'p_float64':
            dtype = 'double'
        else:
            raise Exception('unrecognized data type')

        text = '''#ifndef HALIDE__generated_function_blur_no_schedule_h
#define HALIDE__generated_function_blur_no_schedule_h
#include <tiramisu/utils.h>
#include <sys/time.h>
#include <cstdlib>
#include <algorithm>
#include <vector>

#define NB_THREAD_INIT 48
struct args {
    ''' + dtype + ''' *buf;
    unsigned long long int part_start;
    unsigned long long int part_end;
    ''' + dtype + ''' value;
};

void *init_part(void *params)
{
   ''' + dtype + ''' *buffer = ((struct args*) params)->buf;
   unsigned long long int start = ((struct args*) params)->part_start;
   unsigned long long int end = ((struct args*) params)->part_end;
   ''' + dtype + ''' val = ((struct args*) params)->value;
   for (unsigned long long int k = start; k < end; k++){
       buffer[k]=val;
   }
   pthread_exit(NULL);
}

void parallel_init_buffer(''' + dtype + '''* buf, unsigned long long int size, ''' + dtype + ''' value){
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
#endif\n'''
        text += '''int ''' + self.name + '''(''' + ''', '''.join(['''halide_buffer_t *''' + buf.name for buf in self.buffer_list]) + ''');\n'''
        text += '''#ifdef __cplusplus
}  // extern "C"
#endif

int get_beam_size(){
    if (std::getenv("BEAM_SIZE")!=NULL)
        return std::stoi(std::getenv("BEAM_SIZE"));
    else{
        std::cerr<<"error: Environment Variable BEAM_SIZE not declared"<<std::endl;
        exit(1);
    }
}

int get_max_depth(){
    if (std::getenv("MAX_DEPTH")!=NULL)
        return std::stoi(std::getenv("MAX_DEPTH"));
    else{
        std::cerr<<"error: Environment Variable MAX_DEPTH not declared"<<std::endl;
        exit(1);
    }
}

void declare_memory_usage(){
    setenv("MEM_SIZE", "'''+str(self.get_memory_usage())+'''", true); // This value was set by the Code Generator
}


int get_max_nb_runs(){
    if (std::getenv("MAX_RUNS")!=NULL)
        return std::stoi(std::getenv("MAX_RUNS"));
    else{
        std::cerr<<"error: Environment Variable MAX_RUNS not declared"<<std::endl;
        exit(1);
    }
}

int get_min_nb_runs(){
    if (std::getenv("MIN_RUNS")!=NULL)
        return std::stoi(std::getenv("MIN_RUNS"));
    else{
        std::cerr<<"error: Environment Variable MIN_RUNS not declared"<<std::endl;
        exit(1);
    }
}

double get_init_exec_time(){
    if (std::getenv("INIT_EXEC_TIME")!=NULL)
        return std::stod(std::getenv("INIT_EXEC_TIME"));
    else{
        std::cerr<<"error: Environment Variable INIT_EXEC_TIME not declared"<<std::endl;
        exit(1);
    }
}

bool is_nb_runs_dynamic(){
    if (std::getenv("DYNAMIC_RUNS")!=NULL){
        if (std::stoi(std::getenv("DYNAMIC_RUNS"))==1){
            // just for checking if that the needed env vars are defined
            get_max_nb_runs();
            get_min_nb_runs();
            get_init_exec_time();
            return true;
        }
        else 
            return false;
    }
    else{
        std::cerr<<"error: Environment Variable DYNAMIC_RUNS not declared"<<std::endl;
        exit(1);
    }
}

int choose_nb_runs(std::vector<double> duration_vect){

    int max_nb_runs=get_max_nb_runs();

    if (max_nb_runs>30){
        std::cerr<<"error: max_nb_runs>30 not supported by choose_nb_runs()"<<std::endl;
        exit(1);
    }
           
    double init_exec = get_init_exec_time();
    if (init_exec==0)  // init_exec==0 means this is the no schedule version of the program
        return max_nb_runs-duration_vect.size();
    
    double eps[] = {0.3326, 0.1166, 0.0896, 0.0747, 0.0642, 0.0568, 0.0506, 0.0455, 0.0410, 0.0371, 0.0335, 0.0305, 0.0278, 0.0252, 0.0227, 0.0203, 0.0182, 0.0163, 0.0146, 0.0130, 0.0115, 0.0100, 0.0086, 0.0072, 0.0059, 0.0047, 0.0034 ,0.0022, 0.0011, 0.0000};
    
    double sched_approx_exec = *min_element(duration_vect.begin(), duration_vect.end());
    
    if ((sched_approx_exec*max_nb_runs)<=15*1000) // if can do all the runs in less than 15s
        return max_nb_runs-duration_vect.size();
    
    double estimate = 1/(init_exec/(0.1*sched_approx_exec)+1); // Please check the documentation for more details about this formula

    int i;
    for (i=duration_vect.size()-1; i<max_nb_runs; i++)
        if (eps[i]<estimate)
            break;

    return i+1-duration_vect.size(); 
}

#endif'''
        return text

    def test_execution(self):
        with open('test_exec.cpp', 'w') as f:
            f.write(self.write_wrapper())
        with open('wrapper.h', 'w') as f:
            f.write(self.write_wrapper_h())
        compile_cmd = 'g++ -std=c++11 -O3 -fno-rtti -mavx2 -I/include/ -I/home/masci/tiramisu/include/ -I/home/masci/tiramisu/3rdParty/Halide/include/ ' \
                      '-I/home/masci/tiramisu/3rdParty/isl/build/include/ -I./ -DTIRAMISU_MEDIUM test_exec.cpp ' \
                      '-L/lib/ -L/home/masci/tiramisu/3rdParty/Halide/lib/ -L/home/masci/tiramisu/3rdParty/isl/build/lib// ' \
                      '-L/home/masci/tiramisu/build/ -ltiramisu -lHalide -lisl -lz -lpthread -ldl ' + self.name + '.o ' \
                                                                                                                  '-ltiramisu -lHalide -lisl -lz -lpthread -ldl -o wrapper'
        exec_cmd = 'LD_LIBRARY_PATH=:/home/masci/tiramisu/3rdParty/Halide/lib:/home/masci/tiramisu/3rdParty/isl/build/lib/:/home/masci/tiramisu/build/:/lib/' \
                   'DYLD_LIBRARY_PATH=:/home/masci/tiramisu/3rdParty/Halide/lib:/home/masci/tiramisu/build/:/lib/' \
                   '.\wrapper'
        compile_output = subprocess.run(compile_cmd.split(), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(compile_output.stderr, compile_output.stdout)
        run_gen_output = subprocess.run(exec_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(run_gen_output.stdout, run_gen_output.stdout)

    def get_memory_usage(self):  # Returns the total memory allocated for the programs buffers in MegaBytes
        total = 0
        for buffer in self.buffer_list:
            nb_points = 1
            for dim_size in buffer.sizes:
                nb_points *= dim_size
            if buffer.data_type == 'p_float64':
                buffer_size = nb_points*8
            else:
                raise Exception('unimplemented data type')
            total += buffer_size
        return total/1024/1024 #in MBytes


class Loop:
    def __init__(self, parent_loop, depth_level, siblings_order, program):
        self.depth = depth_level
        program.depth = max(program.depth, self.depth)  # update max program depth, for stats
        self.parent_loop = parent_loop
        self.name = 'i' + str(len(program.iterators_list))  # TODO find a convention for naming loops, e.g. transform depth into letter, sibling order into number
        self.siblings_order = siblings_order
        self.program = program
        self.program.iterators_list.append(self)
        self.lower_bound = None
        self.upper_bound = None

    @classmethod
    def input_iterator(cls, name, lower_bound, upper_bound, program):  # iterator used only for defining input size
        loop = cls(parent_loop=None, depth_level=0, siblings_order=0, program=program)
        loop.lower_bound = lower_bound
        loop.upper_bound = upper_bound
        loop.name = name
        # look for potential redeclaration of the iterator
        for iterator in loop.program.iterators_list[:-1]:  # iterate over all loops except itself (most recently added)
            if iterator.name == loop.name:
                if loop.lower_bound == iterator.lower_bound and loop.upper_bound == iterator.upper_bound:
                    # print(loop.name, iterator.name, [it.name for it in loop.program.iterators_list])
                    # print(loop, iterator, [it for it in loop.program.iterators_list])
                    loop.program.iterators_list.remove(iterator)
                else:
                    loop.name = loop.name+'_'+str(lower_bound)+'_'+str(upper_bound)
                    # loop.program.iterators_list.remove(loop)
                    # loop = iterator
                    # print(loop.program.name)
                    # raise Exception('Iterator re-declared with a different extent')
        return loop

    def create(self):
        self.lower_bound = choose_lower_bound(self.program.problem_size)
        self.upper_bound = choose_upper_bound(self.program.problem_size, self.get_parents_upper_bounds())
        if self.siblings_order > 0:
            if (local_random.random()<0.8): # 80% chances of just copying sibling loop bounds
                self.upper_bound = self.parent_loop.ordered_child_list[self.siblings_order - 1].upper_bound
                self.lower_bound = self.parent_loop.ordered_child_list[self.siblings_order - 1].lower_bound
        self.nb_child_loops = choose_nb_child_loops(self)
        self.program.nb_branches += 1 if self.nb_child_loops == 2 else 0
        self.nb_child_comps = choose_nb_child_comps(self)
        self.ordered_child_list = []
        # getting child order between loops and computations
        child_positions_list = choose_child_order(self.nb_child_loops, self.nb_child_comps)
        for position, type_ in enumerate(child_positions_list):
            if type_ == 'L':
                # create loop and append it to ordered child list
                loop = Loop(parent_loop=self, depth_level=self.depth + 1, siblings_order=position, program=self.program)
                self.ordered_child_list.append(loop)
                loop.create()
            else:  # type == 'C'
                # create computation and append it to ordered child list
                # init comp absolute order = program.nb_comptation; program.nb_comp +=1
                # init siblings order in both comps and loops
                # add to program's ordered computation list
                computation = Computation(self, self.program.nb_computation, position, self.program)
                self.program.nb_computation += 1
                self.ordered_child_list.append(computation)
                self.program.ordered_computations_list.append(computation)
                computation.create()

    def write(self):
        return self.name + '("' + self.name + '", ' + str(self.lower_bound) + ', ' + str(self.upper_bound) + ')'

    def print(self):
        text = self.depth * '    '
        if self.depth > 0:
            text += 'L__'
        text += self.name + '(' + str(self.lower_bound) + ', ' + str(self.upper_bound) + '): '
        print(text)
        for child in self.ordered_child_list:
            child.print()

    def get_parents_upper_bounds(self):  # returns the upper bounds of the parent loops
        loop = self.parent_loop
        upper_bounds = []
        while loop is not None:
            upper_bounds.append(loop.upper_bound)
            loop = loop.parent_loop
        upper_bounds.reverse()
        return upper_bounds


class Computation:
    def __init__(self, parent_loop, absolute_order, siblings_order, program):
        self.parent_loop = parent_loop
        self.absolute_order = absolute_order
        self.siblings_order = siblings_order
        self.program = program
        self.name = "comp" + str(self.absolute_order).zfill(2)  # TODO find a naming convention for computations, can include depth, dim, sibling order, abs order, comp type
        self.data_type = 'p_float64'  # by default, temporarily
        self.expression = ComputationExpression(self, self.program)
        self.write_access = WriteAccess(computation=self, program=self.program)
        self.parent_iterators_list = self.get_parent_iterators_list()

    def create(self):
        self.write_access.create()
        self.expression.create()
        nb_tries = 1
        while has_skippable_loop_multi_comp(self.program) or has_overwritten_comp(self.program):
            nb_tries+=1
            if nb_tries>1000:
                self.expression.expression = '#could not create valid program#'
                # print('***************************************************************************************************')
                return
            # print('regen')
            self.expression.reset()
            self.expression.create()

        # choose read bufs
        # choose read accesses for each buf
        # choose coefs for read accesses
        # choose operators between read accesses

    def get_parent_iterators_list(self):
        parent_iterators_list = []
        loop = self.parent_loop
        while loop is not None:
            parent_iterators_list.append(loop)
            loop = loop.parent_loop
        parent_iterators_list.reverse()
        return parent_iterators_list

    def get_shared_parent_loop(self, other_computation):  # return the deepest shared loop between this computation and an other
        deepest_shared_loop = None
        for i in range(min(len(self.parent_iterators_list), len(other_computation.parent_iterators_list))):
            if self.parent_iterators_list[i] is other_computation.parent_iterators_list[i]:
                deepest_shared_loop = self.parent_iterators_list[i]
        return deepest_shared_loop

    def write(self):
        computation_def = '\tcomputation ' + self.name + '("' + self.name + '", {'
        for iterator in self.parent_iterators_list:
            computation_def += iterator.name + ','
        computation_def = computation_def[:-1]  # remove last comma
        computation_def += '}, '
        expression_text = self.expression.write()
        if self.name in expression_text:  # temporary solution for checking if a computation uses itself
            computation_def += ' ' + self.data_type + ');\n'
            computation_def += '\t' + self.name + '.set_expression('
        computation_def += expression_text
        computation_def += ');\n'
        return computation_def

    def write_store(self):
        store_line = '\t' + self.name + '.store_in(&' + self.write_access.buffer.name
        if not self.write_access.is_default_mapping():
            store_line += ', {'
            store_line += ','.join([i.name for i in self.write_access.buffer.defining_iterators])
            store_line += '}'
        store_line += ');\n'
        return store_line

    def print(self):
        text = (self.parent_loop.depth + 1) * '    ' + '|__'
        text += self.name + ': {' + self.write_access.write_buffer_access() + '<--' + ','.join([b.write_buffer_access() for b in self.expression.read_access_list]) + '}'
        print(text)


class WriteAccess:
    def __init__(self, computation, program):
        self.computation = computation
        self.program = program
        self.access_pattern = None
        self.buffer = None

    def create(self):
        self.access_pattern, defining_iterators = choose_write_access_pattern(self.computation)
        self.buffer = choose_write_buf(defining_iterators, self.program)
        self.buffer.write_list.append(self)

    def is_default_mapping(self):  # checks if the write access uses the default mapping between the computation and the buffer, this used to decide whether to specify to iterators as .store_in()
        if len(self.buffer.defining_iterators) != len(self.computation.parent_iterators_list):
            return False
        for i in range(len(self.buffer.defining_iterators)):
            if self.buffer.defining_iterators[i].name != self.computation.parent_iterators_list[i].name:
                return False
        return True

    def write_buffer_access(self):  # This is not actually used for writing the program, just to get how the real mem access looks like
        text = self.buffer.name + '('
        for i in range(len(self.buffer.defining_iterators)):
            if not np.any(self.access_pattern[i]):  # all the row is zeros, then the access is 0
                text += '0'
            else:  # if there is at least one non-zero value in the row
                for j in range(len(self.computation.parent_iterators_list)):
                    if self.access_pattern[i, j] == 1:  # no need for a coefficient
                        text += self.computation.parent_iterators_list[j].name + '+'
                    elif self.access_pattern[i, j] > 1:  # a coefficient is used
                        text += str(self.access_pattern[i, j]) + '*' + self.computation.parent_iterators_list[
                            j].name + '+'
                if self.access_pattern[i, -1] > 0:  # a positive constant is used
                    text += str(self.access_pattern[i, -1])
                elif self.access_pattern[i, -1] < 0:  # a negative constant is used
                    text = text[:-1] + str(self.access_pattern[i, -1])  # remove the last + and put a minus
                else:
                    text = text[:-1]  # remove the last '+'
            text += ','
        text = text[:-1] + ')'  # remove last comma and add a parenthesis

        return text


class ComputationExpression:
    def __init__(self, computation, program):
        self.read_access_list = []  # list of ReadAccess object
        self.expression = ''  # the expression is formatted as a string
        self.computation = computation
        self.program = program

    def create(self):
        self.read_access_list = choose_read_accesses(self.computation, self.program)  # chooses buffers to read and access patter to each
        self.expression = choose_expression(self.computation, self.read_access_list)

    def write(self):
        return self.expression
        pass

    def reset(self):
        self.expression = ''
        for read_access in self.read_access_list:
            read_access.buffer.read_list.remove(read_access)
            if read_access.buffer.read_list == [] and read_access.buffer.write_list == []:  # if buffer no longer used in read nor write, delete it from program
                self.program.buffer_list.remove(read_access.buffer)
                self.program.input_list.remove(read_access.buffer.wrapping_input)
        self.read_access_list = []


class Buffer:
    def __init__(self, defining_iterators, program):
        # optimally, only buffers that are read before being written on need an input wrapper (accesses as input) but to avoid complications, we create an input for each buffer with at least one read access
        self.name = 'buf' + str(len(program.buffer_list)).zfill(2)  # TODO find a naming convention for buffers
        self.type = ''  # either a_output, a_input, a_temporary, to define when writing
        self.read_list = []  # a list of ReadAccess that are used to read from the buffer
        self.write_list = []  # a list of WriteAccess that are used to write on the buffer
        self.defining_iterators = defining_iterators  # the iterator who's sizes are used to define the buffer
        self.sizes = [i.upper_bound - i.lower_bound for i in defining_iterators]
        self.wrapping_input = None  # the input wrapper that will be used to read from the buffer. will be defined a write
        self.program = program
        self.program.buffer_list.append(self)
        self.data_type = 'p_float64'  # temporarily fixed

    # def fit_size_to_accesses(self):  # will be called when a stencil access (optimaly, but it can be called for every access) is used on the buf, this will correct the extent
    #     # must also create new defining iterators for the wrapping input, and add them to the programs iterators list
    #     for read_access in self.read_list:
    #         if np.any(read_access.access_pattern[:, -1]):
    #             for i in range(len(read_access.access_pattern[:, -1])):
    #                 if self.sizes[i] < (self.wrapping_input.defining_iterators[i].upper_bound - self.wrapping_input.defining_iterators[i].lower_bound) + read_access.access_pattern[i, -1]:
    #                     self.sizes[i] = (self.defining_iterators[i].upper_bound - self.defining_iterators[i].lower_bound) + read_access.access_pattern[i, -1]
    #                     # print(self.wrapping_input.defining_iterators[i].name)
    #                     # if '_p' in self.wrapping_input.defining_iterators[i].name:  # if the iterator has already been extended
    #                     #     self.program.iterators_list.remove(self.wrapping_input.defining_iterators[i])
    #                     self.wrapping_input.defining_iterators[i] = \
    #                         Loop.input_iterator(lower_bound=self.defining_iterators[i].lower_bound, upper_bound=self.defining_iterators[i].upper_bound + read_access.access_pattern[i, -1], name=self.defining_iterators[i].name+'_p'+str(read_access.access_pattern[i, -1]), program=self.program)
    #     pass

    def update_wrapping_input(self):  # will be called for every read access created
        if self.wrapping_input is None:
            self.wrapping_input = Input(self, self.program)
        # self.fit_size_to_accesses()

    def write(self):
        if self.write_list != []:  # updating the buffer type
            self.type = 'a_output'
        else:
            self.type = 'a_input'
        buffer_declaration = '\tbuffer ' + self.name + '("' + self.name + '", {' + \
                             ','.join([str(size) for size in self.sizes]) + '}, ' + self.data_type + \
                             ', ' + self.type + ');\n'
        return buffer_declaration

    def write_wrapper(self):
        dtype = ''
        if self.data_type == 'p_float64':
            dtype = 'double'
        else:
            raise Exception('unrecognized data type')

        text = ''
        if self.type == 'a_input' or self.read_list != []:  # if input or read
            text += '\t' + dtype + ' *c_' + self.name + ' = (' + dtype + '*)malloc(' + '*'.join([str(size) for size in self.sizes[::-1]]) + '* sizeof(' + dtype + '));\n'
            text += '\tparallel_init_buffer(c_' + self.name + ', ' + '*'.join([str(size) for size in self.sizes[::-1]]) + ', (' + dtype + ')' + str(random.randint(1, 100)) + ');\n'
            text += '\tHalide::Buffer<' + dtype + '> ' + self.name + '(c_' + self.name + ', ' + ','.join([str(size) for size in self.sizes[::-1]]) + ');\n'
        elif self.type == 'a_output':
            text += '\tHalide::Buffer<' + dtype + '> ' + self.name + '(' + ','.join([str(size) for size in self.sizes[::-1]]) + ');\n'

        return text


class Input:
    def __init__(self, buffer, program):
        self.buffer = buffer
        self.defining_iterators = self.buffer.defining_iterators.copy()
        if self.buffer.write_list != []:
            self.name = 'i' + self.buffer.write_list[-1].computation.name
        else:
            self.name = 'input' + self.buffer.name[-2:]
        self.program = program
        self.program.input_list.append(self)
        self.data_type = 'p_float64'  # temporarily fixed data type

    def write(self):
        input_line = '\tinput ' + self.name + '("' + self.name + '", {'
        for iterator in self.defining_iterators:
            input_line += iterator.name + ','
        input_line = input_line[:-1]  # remove last comma
        input_line += '}, ' + self.data_type + ');\n'
        return input_line

    def write_store(self):
        return '\t' + self.name + '.store_in(&' + self.buffer.name + ');\n'


class ReadAccess:
    def __init__(self, buffer, access_pattern, computation):
        # self.buffer = Buffer(defining_iterators=computation.get_parent_iterators_list(), buffer_type='a_input') # temporarily just create a new input buffer
        # self.access_pattern = computation.get_parent_iterators_list() # temporarily
        self.buffer = buffer
        self.buffer.read_list.append(self)
        self.access_pattern = access_pattern
        self.computation = computation
        self.buffer.update_wrapping_input()
        pass

    def write(self):
        text = self.buffer.wrapping_input.name + '('
        for i in range(len(self.buffer.defining_iterators)):
            if not np.any(self.access_pattern[i]):  # all the row is zeros, then the access is 0
                text += '0'
            else:  # if there is at least one non-zero value in the row
                for j in range(len(self.computation.parent_iterators_list)):
                    if self.access_pattern[i, j] == 1:  # no need for a coefficient
                        text += self.computation.parent_iterators_list[j].name + '+'
                    elif self.access_pattern[i, j] > 1:  # a coefficient is used
                        text += str(self.access_pattern[i, j]) + '*' + self.computation.parent_iterators_list[j].name + '+'
                if self.access_pattern[i, -1] > 0:  # a positive constant is used
                    text += str(self.access_pattern[i, -1])
                elif self.access_pattern[i, -1] < 0:  # a negative constant is used
                    text = text[:-1] + str(self.access_pattern[i, -1])  # remove the last + and put a minus
                else:
                    text = text[:-1]  # remove the last '+'
            text += ','
        text = text[:-1] + ')'  # remove last comma and add a parenthesis

        return text

    def write_buffer_access(self):  # This is not actually used for writing the program, just to get how the real mem access looks like
        text = self.buffer.name + '('
        for i in range(len(self.buffer.defining_iterators)):
            if not np.any(self.access_pattern[i]):  # all the row is zeros, then the access is 0
                text += '0'
            else:  # if there is at least one non-zero value in the row
                for j in range(len(self.computation.parent_iterators_list)):
                    if self.access_pattern[i, j] == 1:  # no need for a coefficient
                        text += self.computation.parent_iterators_list[j].name + '+'
                    elif self.access_pattern[i, j] > 1:  # a coefficient is used
                        text += str(self.access_pattern[i, j]) + '*' + self.computation.parent_iterators_list[
                            j].name + '+'
                if self.access_pattern[i, -1] > 0:  # a positive constant is used
                    text += str(self.access_pattern[i, -1])
                elif self.access_pattern[i, -1] < 0:  # a negative constant is used
                    text = text[:-1] + str(self.access_pattern[i, -1])  # remove the last + and put a minus
                else:
                    text = text[:-1]  # remove the last '+'
            text += ','
        text = text[:-1] + ')'  # remove last comma and add a parenthesis

        return text

    def access_pattern_is_simple(self):  # checks if all elements of the buffer are read and in order, i.e. nb_col = nb_row +1 and last_col =0 and the rest is a identity matrix
        if self.access_pattern.shape[0] + 1 != self.access_pattern.shape[1]:
            return False
        if np.any(self.access_pattern[:, -1]):
            return False
        x = self.access_pattern[:, :-1]
        if not (x == np.eye(x.shape[0])).all():
            return False
        return True


def generate_programs(output_path, nb_programs, first_seed='auto'):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if first_seed == 'auto':
        with open('RandomTiramisu/last_seed.txt', 'r') as f:
            first_seed = int(f.read())
        with open('RandomTiramisu/last_seed.txt', 'w') as f:
            f.write(str(first_seed + nb_programs))
    for i in tqdm(range(first_seed, first_seed + nb_programs)):
        program = Program(i)
        program.create()
        program_folder = output_path + '/' + program.name
        Path(program_folder).mkdir(parents=True, exist_ok=True)
        with open(program_folder + '/' + program.name + '_generator.cpp', 'w') as f:
            f.write(program.write_tiramisu_program())
        with open(program_folder + '/' + program.name + '_autoscheduler.cpp', 'w') as f:
            f.write(program.write_autoscheduler())
        with open(program_folder + '/' + program.name + '_wrapper.cpp', 'w') as f:
            f.write(program.write_wrapper())
        with open(program_folder + '/' + program.name + '_wrapper.h', 'w') as f:
            f.write(program.write_wrapper_h())


if __name__ == '__main__':
    generate_programs(output_path='Dataset_multi', first_seed=105100, nb_programs=20000)

