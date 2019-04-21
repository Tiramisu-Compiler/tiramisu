
#ifndef RESTRUCTURED_CLASSES_H
#define RESTRUCTURED_CLASSES_H



#include <string>
#include <vector>
#include <deque>
#include <map>
#include <iostream>
#include <fstream>


//========================define tiramisu_code_generator
#define ASSIGNMENT 0
#define ASSIGNMENT_INPUTS 1
#define STENCIL 2
#define TILE_2_PROB 0.5
#define MAX_STENCIL_SIZE 3
#define SHUFFLE_SAME_SIZE_PROB 0.1


//=======================================define random_generator
#define REDUCTION 3
#define CONV 4
#define SAME 10
#define VALID 11

//=======================================define schedule
#define  NONE 0
#define NONE_UNROLL 210
#define INTERCHANGE 20
#define UNROLL 21
#define TILE_2 22
#define TILE_3 23
#define TILING 220
#define THEN 24
#define VECTORIZE 25
#define PARALLELIZE 26
#define AFTER 27

#define VECTOR_SIZE 8

//=======================================define buffer
#define INPUT_BUFFER 0
#define OUTPUT_BUFFER 1
#define INPUT_INIT_0 2

#define NB_TYPES 7

#define INF (-100)


using namespace std;

class stats_per_type{
public:
    int nb_assignments;
    int *nb_each_schedule;
    stats_per_type();
};


class dim_stats{
public:
    int nb_dims;
    int nb_progs;
    int *data_sizes;
    vector <stats_per_type*> types;
    dim_stats(int i, vector <stats_per_type*> types);

};


//========================================================================================constant class==========================================================================================================
class constant {
public:
    string name;
    int value;
    constant(string name, int value);
};


//=====================================================================variable class==========================================================================================================
class variable {
public:
    int id;
    string name;
    int inf_value;
    constant *sup_value;

    variable(string name, int id, int inf_value, constant *sup_value);
    //variable(string name, int inf_value, int sup_value);
    variable(string name, int id);
};



//========================================================================================computation_abstract class==========================================================================================================
class computation_abstract {
public:
    int id;
    string name, data_type;
    vector<variable*> variables;
    string variables_to_string();
    string vars_to_string();
    vector <string> for_stencils(vector<int> var_nums, int offset, vector<vector<vector<int>>> *accesses);


private:
    string to_base_3(int num);
};



//========================================================================================computation class==========================================================================================================
class computation : public computation_abstract{
public:
    int type, **op_stats;
    string expression;
    vector <computation_abstract*> used_comps;
    vector<vector<vector<int>>> used_comps_accesses = {{{}}};
    vector <variable*> updated_vars;
    computation(string name, int type, vector<variable*> *variables, string data_type, int id);
};


//=====================================================================input class==========================================================================================================
class input : public computation_abstract{
public:
    input(string name, vector<variable*> *variables, string data_type, int id);
};


//========================================================================================buffer class==========================================================================================================
class buffer {
public:
    string name;
    int type;
    vector<int> dimensions;
    vector <computation_abstract*> computations;
    buffer(string name, vector<int> dimensions, int type, vector <computation_abstract*> *computations);
    string dimensions_to_string(bool inverted);

};


//========================================================================================schedule class==========================================================================================================
class schedule {
public:
    vector <computation*> comps;
    int type;
    vector <int> factors;
    vector <variable*> vars;
    schedule(vector <computation*> comps, int type, vector <int> factors, vector <variable*> vars);
    void write(string *code_buffer);

private:
    string vars_to_string(int from, int to);

};

//=====================================================================tiramisu_code class==========================================================================================================
class tiramisu_code{
private:
    void write_variables();
    void write_constants();
    void write_computations();
    void write_inputs();
    void write_buffers();
    void write_schedules();
    computation_abstract *generate_tiramisu_conv_layer(vector<int> *input_dimensions, computation_abstract *input_comp, vector<int> *filter_dimensions, int padding_type, int layer_number);
public:
    string code_buffer, function_name;
    string default_type;
    vector<constant*> constants;
    vector<variable*> variables;
    vector<computation*> computations;
    vector<input*> inputs;
    vector<buffer*> buffers;
    vector<schedule*> schedules;
    int indentation_level, id;
    ofstream output_file;
    tiramisu_code(int code_id, string function_name, int schedule_n, vector<computation*> *computations, vector <variable*> *variables, vector<constant*> *constants, vector<input*> *inputs, vector<buffer*> *buffers, string *default_type, vector <schedule*> *schedules);
    tiramisu_code(string function_name, vector<int> *padding_types, string *default_type);

    void generate_code();
    void write_footer();


};


class tiling_class{
public:
    int tiling_depth;         //2D or 3D
    vector<int> tiling_dims;
    vector<int> tiling_factors;
    tiling_class(int tiling_depth, vector<int> tiling_dims, vector<int> tiling_factors);
};


class schedules_class{
public:
    vector<int> interchange_dims;
    int unrolling_factor;
    tiling_class *tiling;
    schedules_class(vector <int> interchange_dims, int unrolling_factor, tiling_class *tiling);
};

class iterator_class{
public:
    int id;
    int lowe_bound;
    int upper_bound;
    iterator_class(int id, int lower_bound, int upper_bound);
};

class iterators_class{
public:
    int n;
    vector<iterator_class*> it_array;
    iterators_class(int n, vector<iterator_class*> it_array);
};

class input_class{
public:
    int input_id;
    string data_type;
    vector <int> iterators;
    input_class(int id, string data_type, vector <int> iterators);
    string get_iterators_string();
};



class inputs_class{
public:
    int n;
    vector <input_class*> inputs_array;
    inputs_class(int n, vector<input_class*> inputs_array);

};


class mem_access_class{
public:
    int comp_id;
    vector <vector <int>> accesses;
    mem_access_class(int comp_id, vector<vector<int>> accesses);
    vector <string> accesses_to_string();
};

class mem_accesses_class{
public:
    int n;
    vector <mem_access_class*> accesses_array;
    mem_accesses_class(int n, vector<mem_access_class*> accesses_array);
};

class computation_class{
public:
    int comp_id;
    string lhs_data_type;
    vector<int> iterators;
    int **operations_histogram;
    mem_accesses_class *rhs_accesses;
    computation_class(int comp_id, string lhs_data_type, vector<int> iterators, int **operations_histogram, mem_accesses_class *rhs_accesses);
    string get_iterators_string();
    string get_op_stats(int n);

};

class computations_class{
public:
    int n;
    vector <computation_class*> computations_array;
    computations_class(int n, vector <computation_class*> computations_array);
};

class assignment_class{
public:
    int assignment_id;
    int position;
    assignment_class(int id, int position);
};

class assignments_class{
public:
    int n;
    vector<assignment_class*> assignments;
    assignments_class(int n, vector<assignment_class*> assignments);
};

class loop_class{
public:
    int loop_id;
    int parent;
    int position;
    int loop_it;
    assignments_class *assignments;
    loop_class(int loop_id, int parent, int position, int loop_it, assignments_class *assignments);
};

class loops_class{
public:
    int n;
    vector <loop_class*> loops_array;
    loops_class(int n, vector<loop_class*> loops_array);
};

class node_class{
public:
    int seed;
    int code_type;
    loops_class *loops;
    computations_class *computations;
    iterators_class *iterators;
    inputs_class *inputs;
    node_class(int seed, loops_class *loops, computations_class *computations, iterators_class *iterators, inputs_class *inputs);
};

struct configuration{
    int schedule, computation_type;
    vector<int> factors = {};
    vector<variable*> in_variables = {}, out_variables = {};
};



struct schedule_params{
    int schedule;
    vector<int> factors;
    double prob;
};


configuration random_conf(schedule_params schedule_parameters, int computation_type, vector<variable*> in_vars);

class state{
public:
    vector<configuration> schedules;
    int level;
    bool is_extendable(int nb_schedules);
    bool is_appliable(int nb_schedules);
    state(vector<configuration> schedules, int level);
    vector<schedule*> apply(computation *comp);
};
bool is_valid(configuration conf);



//---------------------------------------------------------Helper functions--------------------------------------------------------------------------
int find(vector<int> ints, int e);
int find_schedule(vector<configuration> schedules, int schedule);
int find_schedule(vector<schedule*> schedules, int schedule);

bool contains(vector<variable*> v, variable *e);
map <int, vector<int>> indexes_by_size(vector<variable*> vars);

vector<int> new_indexes(vector<vector<int>> indexes_by_size, int size);

#endif //RESTRUCTURED_CLASSES_H
