#ifndef RESTRUCTURED_EXTERNAL_WRITERS_H
#define RESTRUCTURED_EXTERNAL_WRITERS_H

#include <string>
#include <iostream>
#include <fstream>
#include "classes.h"


//=======================================define buffer
#define INPUT_BUFFER 0
#define OUTPUT_BUFFER 1
#define INPUT_INIT_0 2

#define MAX_BUFFER_SIZE 100

using namespace std;

void generate_cpp_wrapper(string function_name, vector <buffer*> buffers, string *default_type_wrapper, int code_id, int schedule_n);
void generate_h_wrapper(string function_name, vector <buffer*> buffers, int code_id, int schedule_n);
string random_array_initialization(buffer *buffer, int *indentation_level, string *default_type_wrapper);
string random_array_initialization_same_dimensions(vector<buffer*> same_size_input_buffers, int *indentation_level, string *default_type_wrapper);
void generate_json_one_node(node_class *represented_node, int code_id);
void generate_json_schedules(schedules_class *schedules, int code_id, int schedule_n, string function_name);
void write_stats(vector <dim_stats*> stats);

//=====================================================================layout==========================================================================================================
void new_line(int nb_lines, int indentation_level, string *code_buffer);
void check_buffer_size(string *code_buffer, ostream *output_file);

#endif //RESTRUCTURED_EXTERNAL_WRITERS_H
