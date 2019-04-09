#include <cmath>
#include <sys/stat.h>
#include "classes.h"
#include "external_writers.h"

stats_per_type::stats_per_type() {
    nb_assignments = 0;
    nb_each_schedule = new int[12];
    for (int i = 0; i < 12; ++i) {
        nb_each_schedule[i] = 0;
    }
}

dim_stats::dim_stats(int i, vector<stats_per_type *> types) {
    nb_dims = i;
    this->types = types;
    nb_progs = 0;
    data_sizes = new int[28];
    for (int j = 0; j < 28; ++j) {
        data_sizes[j] = 0;
    }
}

iterator_class::iterator_class(int id, int lower_bound, int upper_bound) {
    this->id = id;
    this->lowe_bound = lower_bound;
    this->upper_bound = upper_bound;
}

mem_access_class::mem_access_class(int comp_id, vector<vector<int>> accesses) {
    this->comp_id = comp_id;
    this->accesses = accesses;
}

vector<string> mem_access_class::accesses_to_string() {
    vector<string> vec;
    string s;
    for (int i = 0; i < accesses.size(); ++i) {
        s = "[" + to_string(accesses[i][0]);
        for (int j = 1; j < accesses[i].size(); ++j) {
            s += ", " + to_string(accesses[i][j]);
        }
        s += "]";
        vec.push_back(s);
    }
    return vec;
}

mem_accesses_class::mem_accesses_class(int n, vector<mem_access_class *> accesses_array) {
    this->n = n;
    this->accesses_array = accesses_array;

}

computation_class::computation_class(int comp_id, string lhs_data_type, vector<int> iterators,
                                     int **operations_histogram, mem_accesses_class *rhs_accesses) {
    this->comp_id = comp_id;
    this->lhs_data_type = lhs_data_type;
    this->iterators = iterators;
    this->operations_histogram = operations_histogram;
    this->rhs_accesses = rhs_accesses;

}

string computation_class::get_iterators_string() {
    string s = "[" + to_string(iterators[0]);
    for (int i = 1; i < iterators.size(); ++i) {
        s += ", " + to_string(iterators[i]);
    }
    s += "]";
    return s;
}

string computation_class::get_op_stats(int n) {
    string s = "[" + to_string(operations_histogram[n][0]);
    for (int i = 1; i < 4; ++i) {
        s += ", " + to_string(operations_histogram[n][i]);
    }
    s += "]";
    return s;
}

computations_class::computations_class(int n, vector<computation_class *> computations_array) {
    this->n = n;
    this->computations_array = computations_array;

}

assignment_class::assignment_class(int id, int position) {
    this->assignment_id = id;
    this->position = position;

}

assignments_class::assignments_class(int n, vector<assignment_class *> assignments) {
    this->n = n;
    this->assignments = assignments;

}

loop_class::loop_class(int loop_id, int parent, int position, int loop_it, assignments_class *assignments) {
    this->loop_id = loop_id;
    this->parent = parent;
    this->position = position;
    this->loop_it = loop_it;
    this->assignments = assignments;

}

loops_class::loops_class(int n, vector<loop_class *> loops_array) {
    this->n = n;
    this->loops_array = loops_array;

}

node_class::node_class(int seed, loops_class *loops, computations_class *computations, iterators_class *iterators,
                       inputs_class *inputs) {
    this->seed = seed;
    this->loops = loops;
    this->computations = computations;
    this->iterators = iterators;
    this->inputs = inputs;

}

iterators_class::iterators_class(int n, vector<iterator_class *> it_array) {
    this->n = n;
    this->it_array = it_array;

}


//=====================================================================buffer class==========================================================================================================
buffer::buffer(string name, vector<int> dimensions, int type, vector<computation_abstract *> *computations) {
    this->name = name;
    this->dimensions = dimensions;
    this->type = type;
    this->computations = *computations;
}

string buffer::dimensions_to_string(bool inverted) {
    string s = "(";
    if (!inverted) {
        s += to_string(dimensions[0]);
        for (int i = 1; i < dimensions.size(); ++i) {
            s += ", " + to_string(dimensions[i]);
        }
    } else {
        s += to_string(dimensions[dimensions.size() - 1]);
        for (int i = dimensions.size() - 2; i >= 0; i--) {
            s += ", " + to_string(dimensions[i]);
        }
    }
    s += ")";
    return s;
}

//========================================================================================schedule class==========================================================================================================
schedule::schedule(vector<computation *> comps, int type, vector<int> factors, vector<variable *> vars) {
    this->comps = comps;
    this->type = type;
    this->vars = vars;
    this->factors = factors;

}

void schedule::write(string *code_buffer) {
    switch (type) {
        case INTERCHANGE:
            *code_buffer += comps[0]->name + ".interchange(" + this->vars_to_string(0, this->vars.size()) + ");";
            break;
        case UNROLL:
            *code_buffer += comps[0]->name + ".unroll(" + vars[0]->name + ", " + to_string(factors[0]) + ");";
            break;
        case TILE_2:
            *code_buffer +=
                    comps[0]->name + ".tile(" + this->vars_to_string(0, 2) + ", " + to_string(factors[0]) + ", " +
                    to_string(factors[1]) + ", " + this->vars_to_string(2, this->vars.size()) + ");";
            break;
        case TILE_3:
            *code_buffer +=
                    comps[0]->name + ".tile(" + this->vars_to_string(0, 3) + ", " + to_string(factors[0]) + ", " +
                    to_string(factors[1]) + ", " + to_string(factors[2]) + ", " +
                    this->vars_to_string(3, this->vars.size()) + ");";
            break;
        case THEN:
            if (comps.size() > 1) {
                *code_buffer += comps[0]->name;
                for (int i = 1; i < comps.size(); ++i) {
                    *code_buffer += ".then(" + comps[i]->name + ", computation::rootroot)";
                }
                *code_buffer += ";";
            }
            break;
        case VECTORIZE:
            *code_buffer +=
                    comps[0]->name + ".vectorize(" + this->vars_to_string(0, 1) + ", " + to_string(factors[0]) + ", " +
                    this->vars_to_string(1, this->vars.size()) + ");";
            break;
        case PARALLELIZE:
            *code_buffer += comps[0]->name + ".parallelize(" + this->vars_to_string(0, 1) + ");";
            break;

        case AFTER:
            if (comps.size() > 1) {
                *code_buffer += comps[0]->name;
                for (int i = 1; i < comps.size(); ++i) {
                    *code_buffer += ".after(" + comps[i]->name + ", " + to_string(this->factors[i - 1]) + ")";
                }
                *code_buffer += ";";
            }

    }

}

string schedule::vars_to_string(int from, int to) {
    string s = vars[from]->name;
    for (int i = from + 1; i < to; ++i) {
        s += ", " + vars[i]->name;
    }
    return s;

}


//========================================================================================computation_abstract class==========================================================================================================

string computation_abstract::variables_to_string() {
    string vars;
    if (!variables.empty()) {
        vars += "{" + variables[0]->name;
        for (int i = 1; i < variables.size(); ++i) {
            vars += ", " + variables[i]->name;
        }
        vars += "}";
    }
    return vars;
}

string computation_abstract::vars_to_string() {
    string vars;
    if (!variables.empty()) {
        vars += "(" + variables[0]->name;
        for (int i = 1; i < variables.size(); ++i) {
            vars += ", " + variables[i]->name;
        }
        vars += ")";
    }
    return vars;
}

//returns all combinations used in stencils
//ex: {(i0, i1, i2), (i0, i1 - 1, i2), (i0, i1 + 1, i2), (i0 - 1, i1, i2), (i0 - 1, i1 - 1, i2), (i0 - 1, i1 + 1, i2), (i0 + 1, i1, i2), (i0 + 1, i1 - 1, i2), (i0 + 1, i1 + 1, i2)}
//with offset = 1 and var_nums = {0, 1}
vector<string>
computation_abstract::for_stencils(vector<int> var_nums, int offset, vector<vector<vector<int>>> *accesses) {
    vector<string> strings;
    strings.push_back(vars_to_string());
    string s;
    string zeros;
    string vars;
    int pos;
    for (int i = 1; i < pow(3.0, var_nums.size()); ++i) {
        s = to_base_3(i);
        zeros = "";
        for (int j = 0; j < var_nums.size() - s.size(); ++j) {
            zeros += "0";
        }
        s = zeros + s;

        for (int k = 1; k < offset + 1; ++k) {
            pos = 0;
            vars = "(" + variables[0]->name;
            if (var_nums[0] == 0) {
                if (s[0] == '1') {     //+ offset
                    vars += " + " + to_string(k);
                    (*accesses)[(i - 1) * offset + k - 1][0][variables.size()] = k;
                } else if (s[0] == '2') {        //- offset
                    vars += " - " + to_string(k);
                    (*accesses)[(i - 1) * offset + k - 1][0][variables.size()] = k * (-1);
                }
                pos++;
            }
            for (int j = 1; j < variables.size(); ++j) {
                vars += ", " + variables[j]->name;
                if (var_nums[pos] == j) {
                    if (s[pos] == '1') {     //+ offset
                        vars += " + " + to_string(k);
                        (*accesses)[(i - 1) * offset + k - 1][j][variables.size()] = k;
                    } else if (s[pos] == '2') {        //- offset
                        vars += " - " + to_string(k);
                        (*accesses)[(i - 1) * offset + k - 1][j][variables.size()] = k * (-1);
                    }
                    pos++;
                }
            }
            vars += ")";
            strings.push_back(vars);
            vars = "";
        }
    }
    return strings;
}

//used in "for_stecils" function
string computation_abstract::to_base_3(int num) {
    string vals = "012";
    string num_to_base_3;
    while (num > 0) {
        num_to_base_3 = vals[num % 3] + num_to_base_3;
        num /= 3;
    }
    return num_to_base_3;
}


//=====================================================================computation class==========================================================================================================
computation::computation(string name, int type, vector<variable *> *variables, string data_type, int id) {
    this->id = id;
    this->data_type = data_type;
    this->type = type;
    this->variables = *variables;
    this->updated_vars = *variables;
    this->name = name;
    this->op_stats = new int *[NB_TYPES];
    for (int i = 0; i < NB_TYPES; ++i) {
        this->op_stats[i] = new int[4];
        op_stats[i][0] = 0;
        op_stats[i][1] = 0;
        op_stats[i][2] = 0;
        op_stats[i][3] = 0;

    }

}

//=====================================================================constant class==========================================================================================================
constant::constant(string name, int value) {
    this->name = name;
    this->value = value;
}

//=====================================================================input class==========================================================================================================
input::input(string name, vector<variable *> *variables, string data_type, int id) {
    this->id = id;
    this->data_type = data_type;
    this->name = name;
    this->variables = *variables;
}


//=====================================================================tiramisu_code class==========================================================================================================

tiramisu_code::tiramisu_code(int code_id, string function_name, int schedule_n, vector<computation *> *computations,
                             vector<variable *> *variables, vector<constant *> *constants, vector<input *> *inputs,
                             vector<buffer *> *buffers, string *default_type, vector<schedule *> *schedules) {

    this->code_buffer = "#include <tiramisu/tiramisu.h>\n"
                        "\n"
                        "using namespace tiramisu;\n"
                        "\n"
                        "int main(int argc, char **argv){"
                        "\n"
                        "    tiramisu::init(\"" + function_name + "\");";

    this->function_name = function_name;
    this->default_type = *default_type;
    this->computations = *computations;
    this->variables = *variables;
    this->constants = *constants;
    this->inputs = *inputs;
    this->buffers = *buffers;
    this->schedules = *schedules;
    this->indentation_level = 1;
    this->id = code_id;

    string fpath = "samples/function" + to_string(code_id) + "/" + "function" + to_string(code_id) + "_schedule_" + to_string(schedule_n) + "/" + function_name;
    mkdir(fpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    this->output_file.open(fpath + "/" + function_name + ".cpp");

    write_constants();
    write_variables();
    write_inputs();
    write_computations();
    write_schedules();
    write_buffers();
    generate_code();
    write_footer();


}

void tiramisu_code::write_variables() {
    if (!variables.empty()) {
        new_line(2, this->indentation_level, &this->code_buffer);
        if (variables[0]->inf_value != INF) {
            this->code_buffer += "var " + variables[0]->name + "(\"" + variables[0]->name + "\", " +
                                 to_string(variables[0]->inf_value) + ", " + variables[0]->sup_value->name + ")";
            //this->code_buffer += "var " + variables[0]->name + "(\"" + variables[0]->name + "\", " + to_string(variables[0]->inf_value) + ", " + to_string(variables[0]->sup_value) + ")";
        } else code_buffer += "var " + variables[0]->name + "(\"" + variables[0]->name + "\")";
        for (int i = 1; i < variables.size(); ++i) {
            if (variables[i]->inf_value != INF) {
                this->code_buffer += ", " + variables[i]->name + "(\"" + variables[i]->name + "\", " +
                                     to_string(variables[i]->inf_value) + ", " + variables[i]->sup_value->name + ")";
                //this->code_buffer += ", " + variables[i]->name + "(\"" + variables[i]->name + "\", " + to_string(variables[i]->inf_value) + ", " + to_string(variables[i]->sup_value) + ")";
            } else code_buffer += ", " + variables[i]->name + "(\"" + variables[i]->name + "\")";

            check_buffer_size(&code_buffer, &output_file);
        }
        this->code_buffer += ";";
    }
}


void tiramisu_code::write_constants() {
    if (!constants.empty()) {
        new_line(2, this->indentation_level, &this->code_buffer);

        this->code_buffer += "constant " + constants[0]->name + "(\"" + constants[0]->name + "\", " +
                             to_string(constants[0]->value) + ")";
        for (int i = 1; i < constants.size(); ++i) {
            this->code_buffer +=
                    ", " + constants[i]->name + "(\"" + constants[i]->name + "\", " + to_string(constants[i]->value) +
                    ")";
            check_buffer_size(&code_buffer, &output_file);
        }
        this->code_buffer += ";";
    }
}


void tiramisu_code::write_footer() {
    this->code_buffer += "\n\n    return 0;"
                         "\n}";
    this->output_file << code_buffer;
    this->output_file.close();
}


void tiramisu_code::write_computations() {
    for (int i = 0; i < computations.size(); ++i) {
        if (computations[i]->type <= 3) {
            new_line(2, indentation_level, &code_buffer);
            code_buffer += "computation " + computations[i]->name + "(\"" + computations[i]->name
                           + "\", " + computations[i]->variables_to_string() + ", " + computations[i]->expression +
                           ");";
            check_buffer_size(&code_buffer, &output_file);
        } else {
            if (!computations[i]->expression.empty()) {
                new_line(2, indentation_level, &code_buffer);
                code_buffer += "computation " + computations[i]->name + "(\"" + computations[i]->name
                               + "\", " + computations[i]->variables_to_string() + ", " + default_type + ");";
                check_buffer_size(&code_buffer, &output_file);

                new_line(1, indentation_level, &code_buffer);
                code_buffer += computations[i]->name + ".set_expression(" + computations[i]->expression + ");";
                check_buffer_size(&code_buffer, &output_file);
            }
        }
    }
}

void tiramisu_code::write_inputs() {
    for (int i = 0; i < inputs.size(); ++i) {
        new_line(2, indentation_level, &code_buffer);
        code_buffer += "input " + inputs[i]->name + "(\"" + inputs[i]->name
                       + "\", " + inputs[i]->variables_to_string() + ", " + default_type + ");";
        check_buffer_size(&code_buffer, &output_file);
    }

}

void tiramisu_code::generate_code() {
    new_line(2, indentation_level, &code_buffer);
    code_buffer += "tiramisu::codegen({&" + buffers[0]->name;
    for (int i = 1; i < buffers.size(); ++i) {
        code_buffer += ", &" + buffers[i]->name;
    }
  std::size_t pos = function_name.find("_unroll"); 
  std::string str2 = function_name.substr (0,pos);  
    //code_buffer +=
            //"}, \"function" + to_string(id) + "/" + str2 + "/" +function_name+ "/" + function_name + ".o\");";
            code_buffer +=
            "}, \""+ function_name + ".o\");";
}

void tiramisu_code::write_buffers() {
    new_line(1, indentation_level, &code_buffer);
    for (int i = 0; i < buffers.size(); ++i) {
        new_line(1, indentation_level, &code_buffer);
        code_buffer += "buffer " + buffers[i]->name + "(\"" + buffers[i]->name + "\", {"
                       + buffers[i]->dimensions_to_string(false).substr(1,
                                                                        buffers[i]->dimensions_to_string(false).size() -
                                                                        2)
                       + "}, " + default_type + ", ";
        if (buffers[i]->type == INPUT_BUFFER) {
            code_buffer += "a_input";
        } else {
            code_buffer += "a_output";
        }
        code_buffer += ");";

        check_buffer_size(&code_buffer, &output_file);
    }

    new_line(1, indentation_level, &code_buffer);

    for (int i = 0; i < buffers.size(); ++i) {
        for (int j = 0; j < buffers[i]->computations.size(); ++j) {
            new_line(1, indentation_level, &code_buffer);
            code_buffer += buffers[i]->computations[j]->name + ".store_in(&" + buffers[i]->name;
            if (buffers[i]->dimensions.size() != buffers[i]->computations[j]->variables.size()) {
                code_buffer += ", {" + buffers[i]->computations[j]->variables[0]->name;
                for (int k = 1; k < buffers[i]->dimensions.size(); ++k) {
                    code_buffer += ", " + buffers[i]->computations[j]->variables[k]->name;
                }
                code_buffer += "}";
            }
            check_buffer_size(&code_buffer, &output_file);
            code_buffer += ");";
        }
    }
}

//TODO: add bias
computation_abstract *
tiramisu_code::generate_tiramisu_conv_layer(vector<int> *input_dimensions, computation_abstract *input_comp,
                                            vector<int> *filter_dimensions, int padding_type, int layer_number) {
    //input_dimensions : #channels, height, width
    //filter_dimensions : #filters, #channels, height, width
    int padding, id = 0;

    constant *output_height, *output_width;

    //constants
    constant *nb_channels = new constant("nb_channels" + to_string(layer_number), (*input_dimensions)[0]);
    constant *input_height = new constant("input_height" + to_string(layer_number), (*input_dimensions)[1]);
    constant *input_width = new constant("input_width" + to_string(layer_number), (*input_dimensions)[2]);
    constant *nb_filters = new constant("nb_filters" + to_string(layer_number), (*filter_dimensions)[0]);
    constant *filter_height = new constant("filter_height" + to_string(layer_number), (*filter_dimensions)[2]);
    constant *filter_width = new constant("filter_width" + to_string(layer_number), (*filter_dimensions)[3]);

    constants.push_back(nb_channels);
    constants.push_back(input_height);
    constants.push_back(input_width);
    constants.push_back(nb_filters);
    constants.push_back(filter_height);
    constants.push_back(filter_width);


    switch (padding_type) {
        case SAME:
            padding = ((*filter_dimensions)[2] - 1) / 2;
            output_height = input_height;
            output_width = input_width;

            break;
        case VALID:
            padding = 0;
            output_height = new constant("output_height" + to_string(layer_number),
                                         (*input_dimensions)[1] + 2 * padding - (*filter_dimensions)[2] + 1);
            output_width = new constant("output_width" + to_string(layer_number),
                                        (*input_dimensions)[2] + 2 * padding - (*filter_dimensions)[3] + 1);



            constants.push_back(output_height);
            constants.push_back(output_width);
            break;
    }

    int id_var = 0;

    //variables
    vector<variable *> filter_vars, output_vars;
    variable *h_var = new variable("h" + to_string(layer_number), id_var++, 0, output_height);
    variable *w_var = new variable("w" + to_string(layer_number), id_var++, 0, output_width);
    variable *c_var = new variable("c" + to_string(layer_number), id_var++, 0, nb_channels);
    variable *f_h_var = new variable("f_h" + to_string(layer_number), id_var++, 0, filter_height);
    variable *f_w_var = new variable("f_w" + to_string(layer_number), id_var++, 0, filter_width);
    variable *f_c_var = new variable("f_c" + to_string(layer_number), id_var++, 0, nb_filters);

    variables.push_back(h_var);
    variables.push_back(w_var);
    variables.push_back(c_var);
    variables.push_back(f_h_var);
    variables.push_back(f_w_var);
    variables.push_back(f_c_var);

    filter_vars.push_back(f_c_var);
    filter_vars.push_back(c_var);
    filter_vars.push_back(f_h_var);
    filter_vars.push_back(f_w_var);

    output_vars.push_back(f_c_var);
    output_vars.push_back(h_var);
    output_vars.push_back(w_var);
    output_vars.push_back(c_var);
    output_vars.push_back(f_h_var);
    output_vars.push_back(f_w_var);

    //computations
    input *filter_comp = new input("filter" + to_string(layer_number), &filter_vars, default_type, id++);
    inputs.push_back(filter_comp);

    computation_abstract *init_input;

    if (padding_type == SAME) {     //computation for adding the padding
        vector<variable *> init_same_variables;
        init_same_variables.push_back(new variable("chan" + to_string(layer_number), id_var++, 0, nb_channels));
        init_same_variables.push_back(new variable("i" + to_string(layer_number), id_var++, padding,
                                                   new constant(input_height->name + " + " + to_string(padding),
                                                                padding + (*input_dimensions)[1])));
        init_same_variables.push_back(new variable("j" + to_string(layer_number), id_var++, padding,
                                                   new constant(input_width->name + " + " + to_string(padding),
                                                                padding + (*input_dimensions)[2])));
        // init_same_variables.push_back(new variable("i" + to_string(layer_number), padding, padding + (*input_dimensions)[1]));
        //init_same_variables.push_back(new variable("j" + to_string(layer_number), padding, padding + (*input_dimensions)[2]));


        variables.push_back(new variable("chan" + to_string(layer_number), id_var++, 0, nb_channels));
        variables.push_back(new variable("i" + to_string(layer_number), id_var++, padding,
                                         new constant(input_height->name + " + " + to_string(padding),
                                                      padding + (*input_dimensions)[1])));
        variables.push_back(new variable("j" + to_string(layer_number), id_var++, padding,
                                         new constant(input_width->name + " + " + to_string(padding),
                                                      padding + (*input_dimensions)[2])));
        //variables.push_back(new variable("i" + to_string(layer_number), padding, padding + (*input_dimensions)[1]));
        //variables.push_back(new variable("j" + to_string(layer_number), padding, padding + (*input_dimensions)[2]));


        computation *init_input_same = new computation("init_same" + to_string(layer_number), ASSIGNMENT_INPUTS,
                                                       &init_same_variables, default_type, id++);
        init_input_same->expression =
                input_comp->name + "(chan" + to_string(layer_number) + ", i" + to_string(layer_number) + " - " +
                to_string(padding) + ", j" + to_string(layer_number) + "- " + to_string(padding);
        if (input_comp->variables.size() > 3) {
            init_input_same->expression +=
                    ", " + nb_channels->name + ", " + filter_height->name + ", " + filter_width->name;
            // init_input_same->expression += ", " + to_string(nb_channels) + " - 1, " + to_string(filter_height) + " - 1, " + to_string(filter_width) + " - 1";
        }
        init_input_same->expression += ")";
        init_input = init_input_same;
        computations.push_back(init_input_same);
    } else {
        init_input = input_comp;
    }

    computation *output_comp = new computation("output" + to_string(layer_number), CONV, &output_vars, default_type,
                                               id++);
    string output_comp_vars = output_comp->vars_to_string();
    output_comp->expression = output_comp->name + output_comp_vars.substr(0, output_comp_vars.size() - 1) + " - 1) + " +
                              init_input->name + "(c" + to_string(layer_number) +
                              ", h" + to_string(layer_number) + " + f_h" + to_string(layer_number) +
                              ", w" + to_string(layer_number) + " + f_w" + to_string(layer_number);
    if (init_input->variables.size() > 3) {
        output_comp->expression += ", " + nb_channels->name + ", " + filter_height->name + ", " + filter_width->name;
    }
    output_comp->expression += ") * " + filter_comp->name + filter_comp->vars_to_string();

    computations.push_back(output_comp);

    //buffers
    vector<computation_abstract *> comps;
    comps.push_back(init_input);

    if (padding_type == SAME) {
        buffer *b_input = new buffer("buf_input" + to_string(layer_number),
                                     {(*input_dimensions)[0], (*input_dimensions)[1] + 2 * padding,
                                      (*input_dimensions)[2] + 2 * padding}, INPUT_INIT_0, &comps);
        buffers.push_back(b_input);
    }

    comps.clear();
    comps.push_back(filter_comp);
    buffer *b_filter = new buffer("buf_filter" + to_string(layer_number), *filter_dimensions, INPUT_BUFFER, &comps);

    comps.clear();
    comps.push_back(output_comp);
    buffer *b_output = new buffer("buf_output" + to_string(layer_number),
                                  {(*filter_dimensions)[0], output_height->value, output_width->value}, INPUT_INIT_0,
                                  &comps);

    buffers.push_back(b_filter);
    buffers.push_back(b_output);

    return output_comp;

}

tiramisu_code::tiramisu_code(string function_name, vector<int> *padding_types, string *default_type) {
    this->code_buffer = "#include <tiramisu/tiramisu.h>\n"
                        "\n"
                        "using namespace tiramisu;\n"
                        "\n"
                        "int main(int argc, char **argv){"
                        "\n"
                        "    tiramisu::init(\"" + function_name + "\");";

    this->function_name = function_name;
    this->default_type = *default_type;
    this->indentation_level = 1;
    string fpath = "samples/" + function_name;
    mkdir(fpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    this->output_file.open("samples/" + function_name + "/" + function_name + "_file.cpp");

    int nb_layers = (*padding_types).size();

    //creating the first input "layer"
    vector<int> inits = {3, 1024, 1024};
    vector<vector<int>> inits_filter;
    for (int j = 0; j < nb_layers; ++j) {
        inits_filter.push_back({3, 3, 3, 3});
    }

    vector<variable *> variables_test;
    vector<computation_abstract *> comps;

    constant *init_nb_channels = new constant("nb_chans0", inits[0]);
    constant *init_input_height = new constant("in_height0", inits[1]);
    constant *init_input_width = new constant("in_width0", inits[2]);

    constants.push_back(init_nb_channels);
    constants.push_back(init_input_height);
    constants.push_back(init_input_width);


    int id_var = 0;

    variables_test.push_back(new variable("chan0", id_var++, 0, init_nb_channels));
    variables_test.push_back(new variable("i0", id_var++, 0, init_input_height));
    variables_test.push_back(new variable("j0", id_var++, 0, init_input_width));

    variables.push_back(new variable("chan0", id_var++, 0, init_nb_channels));
    variables.push_back(new variable("i0", 0, id_var++, init_input_height));
    variables.push_back(new variable("j0", 0, id_var++, init_input_width));

    inputs.push_back(new input("input0", &variables_test, *default_type, 0));
    comps.push_back(inputs[0]);

    buffers.push_back(new buffer("buf0", inits, INPUT_BUFFER, &comps));

    computation_abstract *new_input = inputs[0];


    for (int i = 1; i < nb_layers; ++i) {
        new_input = generate_tiramisu_conv_layer(&inits, new_input, &inits_filter[i], (*padding_types)[i - 1], i);
        //f_c1, h1, w1, c1, f_h1, f_w1
        inits = {inits_filter[i][0], new_input->variables[1]->sup_value->value,
                 new_input->variables[2]->sup_value->value};

    }

    write_constants();
    write_variables();
    write_inputs();
    write_computations();
    write_buffers();
    generate_code();
    write_footer();
}


void tiramisu_code::write_schedules() {
    new_line(1, indentation_level, &code_buffer);
    for (int i = 0; i < schedules.size(); ++i) {
        new_line(1, indentation_level, &code_buffer);
        schedules[i]->write(&code_buffer);
    }

}


//=====================================================================variable class==========================================================================================================
variable::variable(string name, int id, int inf_value, constant *sup_value) {
    this->id = id;
    this->name = name;
    this->inf_value = inf_value;
    this->sup_value = sup_value;
}

variable::variable(string name, int id) {
    this->id = id;
    this->name = name;
    this->inf_value = INF;
}


tiling_class::tiling_class(int tiling_depth, vector<int> tiling_dims, vector<int> tiling_factors) {
    this->tiling_depth = tiling_depth;
    this->tiling_factors = tiling_factors;
    this->tiling_dims = tiling_dims;
}

schedules_class::schedules_class(vector<int> interchange_dims, int unrolling_factor, tiling_class *tiling) {
    this->interchange_dims = interchange_dims;
    this->unrolling_factor = unrolling_factor;
    this->tiling = tiling;
}

inputs_class::inputs_class(int n, vector<input_class *> inputs_array) {
    this->n = n;
    this->inputs_array = inputs_array;
}


input_class::input_class(int id, string data_type, vector<int> iterators) {
    this->input_id = id;
    this->data_type = data_type;
    this->iterators = iterators;
}

string input_class::get_iterators_string() {
    string s = "[" + to_string(iterators[0]);
    for (int i = 1; i < iterators.size(); ++i) {
        s += ", " + to_string(iterators[i]);
    }
    s += "]";
    return s;
}

bool is_valid(configuration conf) {
    if ((conf.schedule == TILE_2) || (conf.schedule == TILE_3)) {
        for (int i = 0; i < conf.factors.size(); ++i) {
            if (conf.factors[i] > conf.in_variables[i]->sup_value->value) return false;
        }
    }
    if ((conf.schedule == UNROLL) && (conf.in_variables.back()->inf_value != INF)){
         if (conf.factors[0] > conf.in_variables.back()->sup_value->value / 2) return false;
    }
    return true;
}

bool state::is_extendable(int nb_schedules) {
    return (is_valid(schedules.back()) && (schedules.size() < nb_schedules));
}


bool state::is_appliable(int nb_schedules) {
    int tiling  = find_schedule(this->schedules, TILE_2), unrolling  = find_schedule(this->schedules, UNROLL);
    if (tiling == -1) tiling = find_schedule(this->schedules, TILE_3);
    if ((tiling != -1) && (unrolling != -1)){
        return !((this->schedules[tiling].out_variables.back() == this->schedules[unrolling].in_variables[0]) && (this->schedules[unrolling].factors[0] > this->schedules[tiling].factors.back()));
    }
    return (is_valid(schedules.back()) && (schedules.size() == nb_schedules));
}

state::state(vector<configuration> schedules, int level) {
    this->schedules = schedules;
    this->level = level;

}

vector<schedule *> state::apply(computation *comp) {
    vector <variable*> vect_vars;
    //variable *v1 = new variable("i_vec", 21), *v2 = new variable("i_vec1", 22);
    vector<schedule *> scheds;
    for (int i = 0; i < this->schedules.size(); ++i) {
        vect_vars.clear();
        if ((this->schedules[i].schedule != NONE) && (this->schedules[i].schedule != NONE_UNROLL)) {
            if (this->schedules[i].schedule == UNROLL){
                vect_vars.push_back(this->schedules[i].in_variables[0]);
                //vect_vars.push_back(v1);
               // vect_vars.push_back(v2);
               // scheds.push_back(new schedule({comp}, VECTORIZE, {VECTOR_SIZE}, vect_vars));
                scheds.push_back(new schedule({comp}, this->schedules[i].schedule, this->schedules[i].factors, this->schedules[i].in_variables));
            }
            else {
                scheds.push_back(new schedule({comp}, this->schedules[i].schedule, this->schedules[i].factors,
                                              this->schedules[i].in_variables));
            }
        }
        if (this->schedules[i].schedule == NONE_UNROLL) {
            vect_vars.push_back(this->schedules[i - 1].out_variables.back());
           // vect_vars.push_back(v1);
           // vect_vars.push_back(v2);
          //  scheds.push_back(new schedule({comp}, VECTORIZE, {VECTOR_SIZE}, vect_vars));
        }
        if (comp->type == STENCIL){
         //   comp->variables.back()->inf_value = VECTOR_SIZE;
        }
    }
    scheds.push_back(new schedule({comp}, PARALLELIZE, {}, {this->schedules.back().out_variables[0]}));
    return scheds;

}

//------------------------------------------------------------Helper functions----------------------------------------------------------------------------

bool contains(vector<variable *> v, variable *e) {
    for (int i = 0; i < v.size(); i++) {
        if (v[i]->id == e->id) return true;
    }
    return false;
}

int find_schedule(vector<configuration> schedules, int schedule){
    for (int i = 0; i < schedules.size(); ++i) {
        if (schedules[i].schedule == schedule) return i;
    }
    return -1;
}

int find_schedule(vector<schedule*> schedules, int schedule){
    for (int i = 0; i < schedules.size(); ++i) {
        if (schedules[i]->type == schedule) return i;
    }
    return -1;
}


map<int, vector<int>> indexes_by_size(
        vector<variable *> vars) {           //returns a list of indexes, each representing variables having the same size
    vector<int> sizes;
    vector<vector<int>> indexes;
    int pos;
    for (int i = 0; i < vars.size(); ++i) {
        pos = find(sizes, vars[i]->sup_value->value);
        if (pos != -1) {
            indexes[pos].push_back(i);
        } else {
            sizes.push_back(vars[i]->sup_value->value);
            vector<int> indexes_same_size = {i};
            indexes.push_back(indexes_same_size);
        }
    }

    map<int, vector<int>> indexes_map;
    for (int i = 0; i < sizes.size(); ++i) {
        indexes_map.insert(pair<int, vector<int>>(sizes[i], indexes[i]));
    }


    return indexes_map;
}

int find(vector<int> ints, int e) {
    for (int i = 0; i < ints.size(); ++i) {
        if (ints[i] == e) return i;
    }
    return -1;
}


configuration random_conf(schedule_params schedule_parameters, int computation_type, vector<variable*> in_vars){
    variable *v1, *v2, *v3, *v4, *v5, *v6;
    configuration conf;
    conf.computation_type = computation_type;
    conf.out_variables = in_vars;
    switch (schedule_parameters.schedule){
        case INTERCHANGE:{
            conf.schedule = INTERCHANGE;
            int rand1 = rand() % in_vars.size();
            int rand2 = rand() % in_vars.size();
            while (rand2 == rand1){
                rand2 = rand() % in_vars.size();
            }
            conf.in_variables = {in_vars[rand1], in_vars[rand2]};
            iter_swap(conf.out_variables.begin() + rand1, conf.out_variables.begin() + rand2);
            break;
        }
        case TILING:{
            v1 = new variable("i01", 11);
            v2 = new variable("i02", 12);
            v3 = new variable("i03", 13);
            v4 = new variable("i04", 14);
            double num = (double) rand() / (RAND_MAX);
            int rand1 = rand() % schedule_parameters.factors.size();
            int rand2 = rand() % schedule_parameters.factors.size();
            if ((num < TILE_2_PROB) || (in_vars.size() < 3)){
                conf.schedule = TILE_2;
                int random = rand() % (in_vars.size() - 1);
                conf.factors = {schedule_parameters.factors[rand1], schedule_parameters.factors[rand2]};
                conf.in_variables = {in_vars[random], in_vars[random + 1], v1, v2, v3, v4};
                conf.out_variables.insert(conf.out_variables.begin() + random, v1);
                conf.out_variables.insert(conf.out_variables.begin() + random + 1, v2);
                conf.out_variables.insert(conf.out_variables.begin() + random + 2, v3);
                conf.out_variables.insert(conf.out_variables.begin() + random + 3, v4);
                conf.out_variables.erase(conf.out_variables.begin() + random + 4, conf.out_variables.begin() + random + 6);
            }
            else{
                conf.schedule = TILE_3;
                v5 = new variable("i05", 15);
                v6 = new variable("i06", 16);
                int random = rand() % (in_vars.size() - 2);
                int rand3 = rand() % schedule_parameters.factors.size();
                conf.factors = {schedule_parameters.factors[rand1], schedule_parameters.factors[rand2], schedule_parameters.factors[rand3]};
                conf.in_variables = {in_vars[random], in_vars[random + 1], in_vars[random + 2], v1, v2, v3, v4, v5, v6};
                conf.out_variables.insert(conf.out_variables.begin() + random, v1);
                conf.out_variables.insert(conf.out_variables.begin() + random + 1, v2);
                conf.out_variables.insert(conf.out_variables.begin() + random + 2, v3);
                conf.out_variables.insert(conf.out_variables.begin() + random + 3, v4);
                conf.out_variables.insert(conf.out_variables.begin() + random + 4, v5);
                conf.out_variables.insert(conf.out_variables.begin() + random + 5, v6);
                conf.out_variables.erase(conf.out_variables.begin() + random + 6, conf.out_variables.begin() + random + 9);
            }
            break;
        }
        case UNROLL:{
            conf.schedule = UNROLL;
            conf.in_variables = {in_vars.back()};
            int rand1 = rand() % schedule_parameters.factors.size();
            conf.factors = {schedule_parameters.factors[rand1]};
            break;
        }
    }
    return conf;
}
