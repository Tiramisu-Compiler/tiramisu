#ifndef HALIDE__generated_function_blur_no_schedule_h
#define HALIDE__generated_function_blur_no_schedule_h
#include <tiramisu/utils.h>
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
int function1050000(halide_buffer_t *buf00, halide_buffer_t *buf01);
#ifdef __cplusplus
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
    setenv("MEM_SIZE", "144.19337463378906", true); // This value was set by the Code Generator
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

#endif