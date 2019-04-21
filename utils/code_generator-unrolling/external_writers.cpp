#include "external_writers.h"
#include "classes.h"
#include "tiramisu_code_generator.h"


void generate_json_one_node(node_class *represented_node, int code_id){
    int indentation_level = 0;
    string json = "{";
    new_line(1, ++indentation_level, &json);
    json += "\"seed\" : " + to_string(represented_node->seed) + ",";

    new_line(1, ++indentation_level, &json);
    json += "\"type\" : " + to_string(represented_node->code_type) + ",";

    new_line(1, indentation_level, &json);
    json += "\"loops\" : {";

    new_line(1, ++indentation_level, &json);
    json += "\"n\" : " + to_string(represented_node->loops->n) + ",";
    new_line(1, indentation_level, &json);

    json += "\"loops_array\" : [";
    indentation_level++;
    for (int i = 0; i < represented_node->loops->n; ++i) {
        new_line(1, indentation_level, &json);
        json += "{";
        new_line(1, ++indentation_level, &json);
        json += "\"loop_id\" : " + to_string(represented_node->loops->loops_array[i]->loop_id) + ",";
        new_line(1, indentation_level, &json);
        json += "\"parent\" : " + to_string(represented_node->loops->loops_array[i]->parent) + ",";
        new_line(1, indentation_level, &json);
        json += "\"position\" : " + to_string(represented_node->loops->loops_array[i]->position) + ",";
        new_line(1, indentation_level, &json);
        json += "\"loop_it\" : " + to_string(represented_node->loops->loops_array[i]->loop_it) + ",";

        new_line(1, indentation_level, &json);
        json += "\"assignments\" : {";

        new_line(1, ++indentation_level, &json);
        json += "\"n\" : " + to_string(represented_node->loops->loops_array[i]->assignments->n) + ",";
        new_line(1, indentation_level, &json);

        json += "\"assignments_array\" : [";
        for (int j = 0; j < represented_node->loops->loops_array[i]->assignments->n; ++j) {
            new_line(1, ++indentation_level, &json);
            json += "{";
            new_line(1, ++indentation_level, &json);
            json += "\"id\" : " +
                    to_string(represented_node->loops->loops_array[i]->assignments->assignments[j]->assignment_id) +
                    ",";
            new_line(1, indentation_level, &json);
            json += "\"position\" : " +
                    to_string(represented_node->loops->loops_array[i]->assignments->assignments[j]->position);
            new_line(1, --indentation_level, &json);
            json += "},";
        }
        if (represented_node->loops->loops_array[i]->assignments->n > 0) {
            json = json.substr(0, json.size() - 1);
            indentation_level--;
        }
        new_line(1, indentation_level, &json);
        json += "]";
        new_line(1, --indentation_level, &json);
        json += "}";  //end assignments
        new_line(1, --indentation_level, &json);
        json += "},";
    }
    if (represented_node->loops->n > 0) {
        json = json.substr(0, json.size() - 1);
    }
    new_line(1, --indentation_level, &json);
    json += "]";
    new_line(1, --indentation_level, &json);
    json += "}, ";  //end loops


    new_line(1, indentation_level, &json);
    json += "\"computations\" : {";

    new_line(1, ++indentation_level, &json);
    json += "\"n\" : " + to_string(represented_node->computations->n) + ",";
    new_line(1, indentation_level, &json);

    json += "\"computations_array\" : [";
    indentation_level++;
    for (int i = 0; i < represented_node->computations->n; ++i) {
        new_line(1, indentation_level, &json);
        json += "{";
        new_line(1, ++indentation_level, &json);
        json += "\"comp_id\" : " + to_string(represented_node->computations->computations_array[i]->comp_id) + ",";
        new_line(1, indentation_level, &json);
        json += "\"lhs_data_type\" : \"" + represented_node->computations->computations_array[i]->lhs_data_type +
                "\" ,";
        new_line(1, indentation_level, &json);
        json += "\"loop_iterators_ids\" : " +
                represented_node->computations->computations_array[i]->get_iterators_string() + ",";
        new_line(1, indentation_level, &json);
        json += "\"operations_histogram\" : [";
        indentation_level++;
        for (int j = 0; j < NB_TYPES; ++j) {
            new_line(1, indentation_level, &json);
            json += represented_node->computations->computations_array[i]->get_op_stats(j) + ",";
        }
        json = json.substr(0, json.size() - 1);
        new_line(1, --indentation_level, &json);
        json += "],";


        new_line(1, indentation_level, &json);
        json += "\"rhs_accesses\" : {";

        new_line(1, ++indentation_level, &json);
        json += "\"n\" : " + to_string(represented_node->computations->computations_array[i]->rhs_accesses->n) + ",";
        new_line(1, indentation_level, &json);

        json += "\"accesses\" : [";
        indentation_level++;
        for (int j = 0; j < represented_node->computations->computations_array[i]->rhs_accesses->n; ++j) {
            new_line(1, indentation_level, &json);
            json += "{";
            new_line(1, ++indentation_level, &json);
            json += "\"comp_id\" : " + to_string(
                    represented_node->computations->computations_array[i]->rhs_accesses->accesses_array[j]->comp_id) +
                    ",";
            new_line(1, indentation_level, &json);
            json += "\"access\" : [";
            indentation_level++;
            for (int k = 0; k <
                            represented_node->computations->computations_array[i]->rhs_accesses->accesses_array[j]->accesses_to_string().size(); ++k) {
                new_line(1, indentation_level, &json);
                json += represented_node->computations->computations_array[i]->rhs_accesses->accesses_array[j]->accesses_to_string()[k] +
                        ",";
            }
            if (represented_node->computations->computations_array[i]->rhs_accesses->accesses_array[j]->accesses_to_string().size() >
                0) {
                json = json.substr(0, json.size() - 1);
            }
            new_line(1, --indentation_level, &json);
            json += "]";
            new_line(1, --indentation_level, &json);
            json += "},";
        }
        if (represented_node->computations->computations_array[i]->rhs_accesses->n > 0) {
            json = json.substr(0, json.size() - 1);
        }
        new_line(1, --indentation_level, &json);
        json += "]";
        new_line(1, --indentation_level, &json);
        json += "}";       //end rhs_accesses
        new_line(1, --indentation_level, &json);
        json += "},";
    }
    if (represented_node->computations->n > 0) {
        json = json.substr(0, json.size() - 1);
    }
    new_line(1, --indentation_level, &json);
    json += "]";  //end computations_array
    new_line(1, --indentation_level, &json);
    json += "}, ";  //end computations


    new_line(1, indentation_level, &json);
    json += "\"inputs\" : {";

    new_line(1, ++indentation_level, &json);
    json += "\"n\" : " + to_string(represented_node->inputs->n) + ",";
    new_line(1, indentation_level, &json);

    json += "\"inputs_array\" : [";
    indentation_level++;
    for (int i = 0; i < represented_node->inputs->n; ++i) {
        new_line(1, indentation_level, &json);
        json += "{";
        new_line(1, ++indentation_level, &json);
        json += "\"input_id\" : " + to_string(represented_node->inputs->inputs_array[i]->input_id) + ",";
        new_line(1, indentation_level, &json);
        json += "\"data_type\" : \"" + represented_node->inputs->inputs_array[i]->data_type + "\" ,";
        new_line(1, indentation_level, &json);
        json += "\"loop_iterators_ids\" : " + represented_node->inputs->inputs_array[i]->get_iterators_string();
        new_line(1, indentation_level, &json);
        new_line(1, --indentation_level, &json);
        json += "},";
    }
    if (represented_node->inputs->n > 0) {
        json = json.substr(0, json.size() - 1);
    }
    new_line(1, --indentation_level, &json);
    json += "]";  //end inputs_array
    new_line(1, --indentation_level, &json);
    json += "}, ";  //end inputs


    new_line(1, indentation_level, &json);
    json += "\"iterators\" : {";

    new_line(1, ++indentation_level, &json);
    json += "\"n\" : " + to_string(represented_node->iterators->n) + ",";
    new_line(1, indentation_level, &json);

    json += "\"iterators_array\" : [";
    indentation_level++;
    for (int i = 0; i < represented_node->iterators->n; ++i) {
        new_line(1, indentation_level, &json);
        json += "{";
        new_line(1, ++indentation_level, &json);
        json += "\"it_id\" : " + to_string(represented_node->iterators->it_array[i]->id) + ",";
        new_line(1, indentation_level, &json);
        json += "\"lower_bound\" : " + to_string(represented_node->iterators->it_array[i]->lowe_bound) + ",";
        new_line(1, indentation_level, &json);
        json += "\"upper_bound\" : " + to_string(represented_node->iterators->it_array[i]->upper_bound);
        new_line(1, --indentation_level, &json);
        json += "},";
    }

    if (represented_node->iterators->n > 0) {
        json = json.substr(0, json.size() - 1);
    }
    new_line(1, --indentation_level, &json);
    json += "]";
    new_line(1, --indentation_level, &json);
    json += "}";
    new_line(1, --indentation_level, &json);
    json += "}";

    ofstream output_file;
    output_file.open("samples/function" + to_string(code_id) + "/function" + to_string(code_id) + ".json");
    output_file << json;
    output_file.close();

}

//=====================================================================wrapper==========================================================================================================
void generate_h_wrapper(string function_name, vector<buffer *> buffers, int code_id, int schedule_n) {
    ofstream output;

    string fpath = "samples/function" + to_string(code_id) + "/" + "function" + to_string(code_id) + "_schedule_" + to_string(schedule_n) + "/" + function_name;
    output.open(fpath + "/" + function_name + "_wrapper.h");
    string code_buffer = "#ifndef HALIDE__generated_" + function_name + "_h\n"
                                                                        "#define HALIDE__generated_" + function_name +
                         "_h\n"
                         "\n"
                         "#include <tiramisu/utils.h>\n"
                         "\n"
                         "#ifdef __cplusplus\n"
                         "extern \"C\" {\n"
                         "#endif\n\nint ";
    code_buffer += function_name + "(";
    for (int i = 0; i < buffers.size(); ++i) {
        code_buffer += "halide_buffer_t *" + buffers[i]->name + ", ";
    }
    code_buffer = code_buffer.substr(0, code_buffer.size() - 2) + ");";
    code_buffer += "\n"
                   "\n"
                   "#ifdef __cplusplus\n"
                   "}  // extern \"C\"\n"
                   "#endif\n"
                   "#endif";
    output << code_buffer;
    output.close();
}


/*void generate_cpp_wrapper(string function_name, vector <buffer*> buffers, string *default_type_wrapper, int code_id){
    ofstream output;
    output.open("samples/function" + to_string(code_id)+ "/" + function_name + "/" + function_name +"_wrapper.cpp");
    string code_buffer = "#include \"Halide.h\"\n"
                         "#include \"" + function_name + "_wrapper.h\"\n"
                                                         "#include \"tiramisu/utils.h\"\n"
                                                         "#include <cstdlib>\n"
                                                         "#include <iostream>\n"
                                                         "#include <time.h>\n"
                                                         "#include <fstream>\n"
                                                         "\n"
                                                         "#define MAX_RAND 200\n"
                                                         "\n"
                                                         "int main(int, char **){";

    int indentation_level = 1;


    for (int i = 0; i < buffers.size(); ++i) {
        if(buffers[i]->type == INPUT_BUFFER) {
            code_buffer += random_array_initialization(buffers[i], &indentation_level, default_type_wrapper);
            new_line(2, indentation_level, &code_buffer);
            check_buffer_size(&code_buffer, &output);
        }
        else{
            code_buffer += "Halide::Buffer<" + *default_type_wrapper + "> " + buffers[i]->name + buffers[i]->dimensions_to_string(true) + ";";
            new_line(1, indentation_level, &code_buffer);
            code_buffer += "init_buffer(" + buffers[i]->name + ", (" + *default_type_wrapper + ")0);";
            new_line(2, indentation_level, &code_buffer);
            check_buffer_size(&code_buffer, &output);

        }
    }

    //compute time
    new_line(1, indentation_level, &code_buffer);
    code_buffer += "clock_t t = clock();";

    new_line(2, indentation_level, &code_buffer);
    code_buffer += function_name + "(";
    for (int i = 0; i < buffers.size(); ++i) {
        code_buffer += buffers[i]->name + ".raw_buffer(), ";
    }


    code_buffer = code_buffer.substr(0, code_buffer.size() - 2);

    code_buffer += ");";

    new_line(2, indentation_level, &code_buffer);
    code_buffer += "t = clock() - t;";

    new_line(2, indentation_level, &code_buffer);

    code_buffer += "std::ofstream exec_times_file;";
    new_line(1, indentation_level, &code_buffer);
    code_buffer += "exec_times_file.open(\"../data/programs/function" + to_string(code_id) + "/" + function_name + "/exec_times.txt\", std::ios_base::app);";
    new_line(1, indentation_level, &code_buffer);
    code_buffer += "if (exec_times_file.is_open()){";
    new_line(1, ++indentation_level, &code_buffer);
    code_buffer += "exec_times_file << \"" + function_name + "\" << \" : \" << ((float)t) / CLOCKS_PER_SEC * 1000 << \"ms\" <<std::endl;";
    new_line(1, indentation_level, &code_buffer);
    code_buffer += "exec_times_file.close();";
    new_line(1, --indentation_level, &code_buffer);
    code_buffer += "}";

    code_buffer += "\n\n    return 0;"
                   "\n}";
    output << code_buffer;
    output.close();
}*/




void generate_cpp_wrapper(string function_name, vector<buffer *> buffers, string *default_type_wrapper, int code_id, int schedule_n) {
    ofstream output;
    string fpath = "samples/function" + to_string(code_id) + "/" + "function" + to_string(code_id) + "_schedule_" + to_string(schedule_n) + "/" + function_name;
    output.open(fpath + "/" + function_name + "_wrapper.cpp");
    string code_buffer = "#include \"Halide.h\"\n"
                         "#include \"" + function_name + "_wrapper.h\"\n"
                                                         "#include \"tiramisu/utils.h\"\n"
                                                         "#include <cstdlib>\n"
                                                         "#include <iostream>\n"
                                                         "#include <time.h>\n"
                                                         "#include <fstream>\n"
                                                         "#include <chrono>\n"
                                                         "\n"
                                                         "#define MAX_RAND 200\n"
                                                         "\n"
                                                         "int main(int, char **){";

    int indentation_level = 1;


    for (int i = 0; i < buffers.size(); ++i) {
        if (buffers[i]->type == INPUT_BUFFER) {
            code_buffer += random_array_initialization(buffers[i], &indentation_level, default_type_wrapper);
            new_line(2, indentation_level, &code_buffer);
            check_buffer_size(&code_buffer, &output);
        } else {
            code_buffer += "Halide::Buffer<" + *default_type_wrapper + "> " + buffers[i]->name +
                           buffers[i]->dimensions_to_string(true) + ";";
            new_line(1, indentation_level, &code_buffer);
            code_buffer += "init_buffer(" + buffers[i]->name + ", (" + *default_type_wrapper + ")0);";
            new_line(2, indentation_level, &code_buffer);
            check_buffer_size(&code_buffer, &output);

        }
    }

    //compute time
    new_line(1, indentation_level, &code_buffer);
    code_buffer += "auto t1 = std::chrono::high_resolution_clock::now();";

    new_line(2, indentation_level, &code_buffer);
    code_buffer += function_name + "(";
    for (int i = 0; i < buffers.size(); ++i) {
        code_buffer += buffers[i]->name + ".raw_buffer(), ";
    }


    code_buffer = code_buffer.substr(0, code_buffer.size() - 2);

    code_buffer += ");";

    new_line(2, indentation_level, &code_buffer);
    code_buffer += "auto t2 = std::chrono::high_resolution_clock::now();";


    new_line(2, indentation_level, &code_buffer);
    code_buffer += "std::chrono::duration<double> diff = t2 - t1;";


    new_line(2, indentation_level, &code_buffer);

    code_buffer += "std::ofstream exec_times_file;";
    new_line(1, indentation_level, &code_buffer);
    std::size_t pos = function_name.find("_unroll"); 
    std::string str2 = function_name.substr (0,pos);  
   // code_buffer += "exec_times_file.open(\"function" + to_string(code_id) + "/" + str2 + "/" + function_name +
                 //  "/exec_times.txt\", std::ios_base::app);";

         code_buffer += "exec_times_file.open(\"../exec_times.txt\", std::ios_base::app);";
    new_line(1, indentation_level, &code_buffer);
    code_buffer += "if (exec_times_file.is_open()){";
    new_line(1, ++indentation_level, &code_buffer);
    code_buffer += "exec_times_file << diff.count() * 1000000 <<std::endl;";
    new_line(1, indentation_level, &code_buffer);
    code_buffer += "exec_times_file.close();";
    new_line(1, --indentation_level, &code_buffer);
    code_buffer += "}";

    code_buffer += "\n\n    return 0;"
                   "\n}";
    output << code_buffer;
    output.close();
}


string random_array_initialization_same_dimensions(vector<buffer *> same_size_input_buffers, int *indentation_level,
                                                   string *default_type_wrapper) {
    string initialize_array;
    for (int i = 0; i < same_size_input_buffers.size(); ++i) {
        new_line(1, *indentation_level, &initialize_array);
        initialize_array += "Halide::Buffer<" + *default_type_wrapper + "> " + same_size_input_buffers[i]->name +
                            same_size_input_buffers[i]->dimensions_to_string(
                                    true) + ";";
    }

    new_line(1, *indentation_level, &initialize_array);
    string indexes = ")";
    for (int i = 0; i < same_size_input_buffers[0]->dimensions.size(); ++i) {
        new_line(1, *indentation_level++, &initialize_array);
        initialize_array += "for (int " + string(1, 105 + i);
        initialize_array += " = 0; " + string(1, 105 + i);
        initialize_array += " < " + to_string(same_size_input_buffers[0]->dimensions[i]);
        initialize_array += "; ++" + string(1, 105 + i);
        initialize_array += "){";
        indexes = ", " + string(1, 105 + i) + indexes;
    }
    indexes = "(" + indexes.substr(2);
    indexes += " = (rand() % MAX_RAND) + 1;";

    for (int i = 0; i < same_size_input_buffers.size(); ++i) {
        new_line(1, *indentation_level, &initialize_array);
        initialize_array += same_size_input_buffers[i]->name + indexes;
    }

    for (int i = 0; i < same_size_input_buffers[0]->dimensions.size(); ++i) {
        new_line(1, --*indentation_level, &initialize_array);
        initialize_array += "}";
    }
    return initialize_array;
}

string random_array_initialization(buffer *buffer, int *indentation_level, string *default_type_wrapper) {
    string initialize_array;
    new_line(1, *indentation_level, &initialize_array);
    initialize_array +=
            "Halide::Buffer<" + *default_type_wrapper + "> " + buffer->name + buffer->dimensions_to_string(true) + ";";

    /*new_line(1, *indentation_level, &initialize_array);
    string indexes = ")";
    for (int i = 0; i < buffer->dimensions.size(); ++i) {
        new_line(1, (*indentation_level)++, &initialize_array);
        initialize_array += "for (int " + string(1, 105 + i);
        initialize_array += " = 0; " + string(1, 105 + i);
        initialize_array += " < " + to_string(buffer->dimensions[i]);
        initialize_array += "; ++" + string(1, 105 + i);
        initialize_array += "){";
        indexes = ", " + string(1, 105 + i) + indexes;
    }
    indexes = "(" + indexes.substr(2);
    indexes += " = (rand() % MAX_RAND) + 1;";

    new_line(1, *indentation_level, &initialize_array);
    initialize_array += buffer->name + indexes;


    for (int i = 0; i < buffer->dimensions.size(); ++i) {
        new_line(1, --(*indentation_level), &initialize_array);
        initialize_array += "}";
    }*/
    return initialize_array;
}


void generate_json_schedules(schedules_class *schedules, int code_id, int schedule_n, string function_name) {
    int indentattion_level = 0;
    string json = "{";
    new_line(1, ++indentattion_level, &json);
    json += "\"interchange_dims\" : ";
    if (!(schedules->interchange_dims.empty())) {
        json += "[" + to_string(schedules->interchange_dims[0]) + ", " +
                to_string(schedules->interchange_dims[1]) + "],";
    } else {
        json += "[],";
    }
    new_line(1, indentattion_level, &json);
    json += "\"tiling\" : ";
    if (schedules->tiling != nullptr) {
        json += "{";
        new_line(1, ++indentattion_level, &json);
        json += "\"tiling_depth\" : " + to_string(schedules->tiling->tiling_depth) + ",";
        new_line(1, indentattion_level, &json);
        json += "\"tiling_dims\" : [" + to_string(schedules->tiling->tiling_dims[0]) + ", " +
                to_string(schedules->tiling->tiling_dims[1]);
        if (schedules->tiling->tiling_depth == 3) {
            json += ", " + to_string(schedules->tiling->tiling_dims[2]);
        }
        json += "],";
        new_line(1, indentattion_level, &json);
        json += "\"tiling_factors\" : [" + to_string(schedules->tiling->tiling_factors[0]) + ", " +
                to_string(schedules->tiling->tiling_factors[1]);
        if (schedules->tiling->tiling_depth == 3) {
            json += ", " + to_string(schedules->tiling->tiling_factors[2]);
        }
        json += "]";
        new_line(1, --indentattion_level, &json);
        json += "},";
    } else json += "null,";
    new_line(1, indentattion_level, &json);
    json += "\"unrolling_factor\" : ";
    if (schedules->unrolling_factor == 0) {
        json += "null";
    } else {
        json += to_string(schedules->unrolling_factor);
    }
    new_line(1, --indentattion_level, &json);
    json += "}";


    ofstream output_file;
    string fpath = "samples/function" + to_string(code_id) + "/" + "function" + to_string(code_id) + "_schedule_" + to_string(schedule_n) + "/" + function_name;

    output_file.open(fpath + "/" + function_name + ".json");
    output_file << json;
    output_file.close();

}

//=====================================================================layout==========================================================================================================
void new_line(int nb_lines, int indentation_level, string *code_buffer) {
    for (int i = 0; i < nb_lines; ++i) {
        *code_buffer += "\n";
    }

    for (int i = 0; i < indentation_level; ++i) {
        *code_buffer += "    ";
    }
}

void check_buffer_size(string *code_buffer, ostream *output_file) {
    if ((*code_buffer).size() >= MAX_BUFFER_SIZE) {
        (*output_file) << (*code_buffer);
        (*code_buffer) = "";
    }
}


void write_stats(vector <dim_stats*> stats){
    ofstream output_file;
    output_file.open("stats.json");
    int indentation_level = 0;
    string json = "{";
    indentation_level++;
    for (int i = 0; i < stats.size(); ++i) {
        new_line(1, indentation_level, &json);
        json += "\"" + to_string(stats[i]->nb_dims) + "D_comps\" : {";
        new_line(1, indentation_level, &json);
        json += "\"nb\" : " + to_string(stats[i]->nb_progs) + ",";
        new_line(1, indentation_level, &json);
        json += "\"data_sizes\" : {";


        indentation_level++;
        for (int k = (i + 2) * MIN_LOOP_DIM; k < 28; ++k) {
            new_line(1, indentation_level, &json);
            json += "\"size_" + to_string((int)pow(2.0, k)) + "\" : " + to_string(stats[i]->data_sizes[k - MIN_LOOP_DIM * (i + 2)]) + ",";
        }
        json = json.substr(0, json.size()-1);
        new_line(1, --indentation_level, &json);
        json += "},";
        for (int j = 0; j < stats[i]->types.size(); ++j) {
            new_line(1, indentation_level, &json);
            switch (j){
                case 0:
                    json += "\"assignments\" : {";
                    break;
                case 1:
                    json += "\"assignments_inputs\" : {";
                    break;
                case 2:
                    json += "\"stencils\" : {";
                    break;
            }
            indentation_level++;
            new_line(1, indentation_level, &json);
            json += "\"nb\" : " + to_string(stats[i]->types[j]->nb_assignments) + ",";
            new_line(1, indentation_level, &json);
            /*json += "\"schedules\" : {";
            indentation_level++;
            for (int k = 0; k < 12; ++k) {
                new_line(1, indentation_level, &json);
                json += "\"schedule_" + to_string(k) + "\" : " + to_string(stats[i]->types[j]->nb_each_schedule[k]) + ",";
            }
            json = json.substr(0, json.size()-1);
            new_line(1, --indentation_level, &json);
            json += "},";*/
            new_line(1, --indentation_level, &json);
            json += "},";

            new_line(1, --indentation_level, &json);
            json += "},";
        }
        json = json.substr(0, json.size());
    }
    new_line(1, --indentation_level, &json);
    json += "}";
    output_file << json;
    output_file.close();

}