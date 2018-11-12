#ifndef CODE_GENERATOR_TIRAMISU_CODE_GENERATOR_H
#define CODE_GENERATOR_TIRAMISU_CODE_GENERATOR_H


//========================define tiramisu_code_generator
#define ASSIGNMENT 0
#define ASSIGNMENT_INPUTS 1
#define STENCIL 2

//=======================================define buffer
#define INPUT_BUFFER 0
#define OUTPUT_BUFFER 1
#define INPUT_INIT_0 2


//=======================================define random_generator
#define ASSIGNMENT 0
#define ASSIGNMENT_INPUTS 1
#define STENCIL 2
#define REDUCTION 3
#define CONV 4
#define SAME 10
#define VALID 11

#define MAX_NB_OPERATIONS_ASSIGNMENT 3
#define MAX_ASSIGNMENT_VAL 10


//====================================================

#define MAX_BUFFER_SIZE 100
#define MAX_CONST_VALUE 100

#include <string>
#include <vector>
#include <iostream>
#include <time.h>
#include <sys/stat.h>
#include <fstream>
#include <cmath>


using namespace std;


//=====================================================================variable class==========================================================================================================
class variable {
public:
    string name;
    int inf_value, sup_value;
    //constant *sup_value;

    //variable(string name, int inf_value, constant *sup_value);
    variable(string name, int inf_value, int sup_value);
};

//========================================================================================constant class==========================================================================================================
class constant {
public:
    string name;
    int value;
    constant(string name, int value);
};


//========================================================================================computation_abstract class==========================================================================================================
class computation_abstract {
public:
    string name;
    vector<variable*> variables;
    string variables_to_string();
    string vars_to_string();
    vector <string> for_stencils(vector<int> var_nums, int offset);


private:
    string to_base_3(int num);
};



//========================================================================================computation class==========================================================================================================
class computation : public computation_abstract{
public:
    int type;
    string expression;
    computation(string name, int type, vector<variable*> *variables);
};


//=====================================================================input class==========================================================================================================
class input : public computation_abstract{
public:
    input(string name, vector<variable*> *variables);
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

//=====================================================================layout==========================================================================================================
void new_line(int nb_lines, int indentation_level, string *code_buffer);
void check_buffer_size(string *code_buffer, ostream *output_file);

//=====================================================================random_generator==========================================================================================================
computation *generate_computation(string name, vector<variable*> computation_variables, int computation_type, vector<computation_abstract*> inputs, vector<int> var_num, int offset);
string assignment_expression_generator();
string assignment_expression_generator_inputs(vector<computation_abstract*> inputs);
string stencil_expression_generator(string input_name, computation_abstract* in, vector<int> *var_nums, int offset);
vector <int> computation_types(int nb_comps, double *probs);
vector <int> generate_padding_types(int nb_layers, double *padding_probs);



//=====================================================================tiramisu_code_generator==========================================================================================================
vector <variable*> generate_variables(int nb_variables, int from, int *inf_values, int *sup_values);

void generate_tiramisu_code_single_computation(int code_id, int computation_type, vector<int> *computation_dims,
                                               vector<int> *var_nums, int nb_inputs, string *default_type_tiramisu,
                                               string *default_type_wrapper, int offset);
void generate_tiramisu_code_conv(int code_id, int nb_layers, double *padding_probs, string *default_type_tiramisu, string *default_type_wrapper);
void generate_tiramisu_layer();
void generate_tiramisu_code_multiple_computations(int code_id, vector<int> *computation_dims, int nb_stages, double *probs,
                                                  vector<int> *var_nums, int nb_inputs, string *default_type_tiramisu,
                                                  string *default_type_wrapper, int offset);

//=====================================================================tiramisu_code class==========================================================================================================
class tiramisu_code{
private:
    void write_variables();
    void write_constants();
    void write_computations();
    void write_inputs();
    void write_buffers();
    string default_type;
    computation_abstract *generate_tiramisu_conv_layer(vector<int> *input_dimensions, computation_abstract *input_comp, vector<int> *filter_dimensions, int padding_type, int layer_number);
public:
    string code_buffer, function_name;
    vector<constant*> constants;
    vector<variable*> variables;
    vector<computation*> computations;
    vector<input*> inputs;
    vector<buffer*> buffers;
    int indentation_level;
    ofstream output_file;
    tiramisu_code(string function_name, vector<computation*> *computations, vector <variable*> *variables, /*vector<constant*> *constants,*/ vector<input*> *inputs, vector<buffer*> *buffers, string *default_type);
    tiramisu_code(string function_name, vector<int> *padding_types, string *default_type);

    void generate_code();
    void write_footer();


};

//=====================================================================wrapper==========================================================================================================
void generate_cpp_wrapper(string function_name, vector <buffer*> buffers, string *default_type_wrapper);
void generate_h_wrapper(string function_name, vector <buffer*> buffers);
string random_array_initialization(buffer *buffer, int *indentation_level, string *default_type_wrapper);
string random_array_initialization_same_dimensions(vector<buffer*> same_size_input_buffers, int *indentation_level, string *default_type_wrapper);


#endif //CODE_GENERATOR_TIRAMISU_CODE_GENERATOR_H
