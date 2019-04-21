#ifndef RESTRUCTURED_TIRAMISU_CODE_GENERATOR_H
#define RESTRUCTURED_TIRAMISU_CODE_GENERATOR_H


//========================define tiramisu_code_generator
#define ASSIGNMENT 0
#define ASSIGNMENT_INPUTS 1
#define STENCIL 2
#define TILE_2_PROB 0.5
#define MAX_STENCIL_SIZE 3
#define SHUFFLE_SAME_SIZE_PROB 0.1



//======================================define stats
#define MAX_NB_OPERATIONS_ASSIGNMENT 3
#define MAX_ASSIGNMENT_VAL 10

#define MAX_COMP_DIM 4
#define MAX_MEMORY_SIZE 27 //2²⁷
#define MIN_LOOP_DIM 6      //2⁶
#define MAX_UNROLL_FACTOR 128 // used to limit unroll factors in  exaustive exploration

#define INF (-100)


//====================================================

#define MAX_CONST_VALUE 100

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <sys/stat.h>
#include <cmath>
#include <map>
#include "classes.h"
#include "external_writers.h"


using namespace std;

bool inf(int i, int j);



//=====================================================================random_generator==========================================================================================================
computation *generate_computation(string name, vector<variable*> computation_variables, int computation_type, vector<computation_abstract*> inputs, vector<int> var_num, int offset, string data_type, int id);
string assignment_expression_generator(int **op_stats);
string assignment_expression_generator_inputs(vector<computation_abstract*> inputs, int **op_stats);
string stencil_expression_generator(string input_name, computation_abstract* in, vector<int> *var_nums, int offset, int **op_stats, vector<vector<vector<int>>> *accesses);
vector <int> computation_types(int nb_comps, double *probs);
vector <int> generate_padding_types(int nb_layers, double *padding_probs);




//=====================================================================tiramisu_code_generator==========================================================================================================
vector <variable*> generate_variables(int nb_variables, int from, int *inf_values, vector <constant*> constants);

void generate_tiramisu_code_single_computation(int code_id, int computation_type,
                                               vector<int> *var_nums, int nb_inputs, string *default_type_tiramisu,
                                               string *default_type_wrapper, int offset);
void generate_tiramisu_code_conv(int code_id, int nb_layers, double *padding_probs, string *default_type_tiramisu, string *default_type_wrapper);
void generate_tiramisu_code_multiple_computations(int code_id, int nb_stages, double *probs,
                                                  int max_nb_dims, vector<int> scheduling_commands, bool all_schedules,
                                                  int nb_inputs, string *default_type_tiramisu,
                                                  string *default_type_wrapper, int offset,
                                                  vector<int> *tile_sizes, vector<int> *unrolling_factors,
                                                  int nb_rand_schedules, vector<dim_stats *> *stats);

node_class *comp_to_node(computation *comp, int seed);
void generate_all_schedules(vector<schedule_params> schedules, computation *comp, vector<vector <schedule*>> *generated_schedules, vector<vector<variable*>> *generated_variables, vector<schedules_class*> *schedule_classes);
void generate_random_schedules(int nb_schedules, vector<schedule_params> schedules, computation *comp, vector<vector <schedule*>> *generated_schedules, vector<vector<variable*>> *generated_variables, vector<schedules_class*> *schedule_classes);
schedules_class *confs_to_sc(vector<configuration> schedules);
string to_base_2(int num, int nb_pos);


#endif //RESTRUCTURED_TIRAMISU_CODE_GENERATOR_H
