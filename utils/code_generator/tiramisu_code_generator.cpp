#include "tiramisu_code_generator.h"

//=====================================================================tiramisu_code_generator==========================================================================================================

vector<variable*> generate_variables(int nb_variables, int from, int *inf_values, int *constants){
    vector<variable*> variables;
    for (int i = from; i <nb_variables + from; ++i) {
        variables.push_back(new variable("i" + to_string(i), inf_values[i - from], constants[i - from]));
    }
    return variables;
}

//automatically generating a single computation tiramisu code with the associated wrapper
void generate_tiramisu_code_single_computation(int code_id, int computation_type, vector<int> *computation_dims,
                                               vector<int> *var_nums, int nb_inputs, string *default_type_tiramisu,
                                               string *default_type_wrapper, int offset){
    //initializations
    string function_name = "function" + to_string(code_id);
    vector <variable*> variables;
    vector <computation*> computations;
    vector <buffer*> buffers;
    vector <computation_abstract*> inputs;
    vector <input*> input_vec;
    vector <computation_abstract*> abs;

    int *variables_min_values = new int[(*computation_dims).size()];
    int *variable_max_values = new int[(*computation_dims).size()];
    for (int i = 0; i < (*computation_dims).size(); ++i) {
        variables_min_values[i] = 0;
        variable_max_values[i] = (*computation_dims)[i];
    }
    variables = generate_variables((*computation_dims).size(), 0, variables_min_values, variable_max_values);

    switch (computation_type){
        case ASSIGNMENT:{
            computations.push_back(generate_computation("comp0", variables, ASSIGNMENT, {}, {}, 0));
            for (int i = 0; i < computations.size(); ++i) {
                abs.push_back(computations[i]);
            }
            buffers.push_back(new buffer("buf" + to_string(0), *computation_dims, OUTPUT_BUFFER, &abs));
            break;
        }
        case ASSIGNMENT_INPUTS:{
            for (int i = 0; i < nb_inputs; ++i) {
                inputs.push_back(new input("input" + to_string(i), &variables));
                input_vec.push_back(new input("input" + to_string(i), &variables));
                abs = {inputs[i]};
                buffers.push_back(new buffer("buf" + to_string(i), *computation_dims, INPUT_BUFFER, &abs));
            }
            computations.push_back(generate_computation("comp0", variables, ASSIGNMENT_INPUTS, inputs, {}, 0));
            abs = {computations[0]};
            buffers.push_back(new buffer("buf" + to_string(nb_inputs + 1), *computation_dims, OUTPUT_BUFFER, &abs));
            break;
        }
        case STENCIL:{
            for (int i = 0; i < (*computation_dims).size(); ++i) {
                variables_min_values[i] = offset;
                variable_max_values[i] = (*computation_dims)[i] - offset;
            }
            variables = generate_variables((*computation_dims).size(), 0, variables_min_values, variable_max_values);
            inputs.push_back(new input("input" + to_string(0), &variables));
            input_vec.push_back(new input("input" + to_string(0), &variables));
            abs = {inputs[0]};
            buffers.push_back(new buffer("buf" + to_string(0), *computation_dims, INPUT_BUFFER, &abs));
            computations.push_back(generate_computation("comp0", variables, STENCIL, inputs, *var_nums, offset));
            abs = {computations[0]};
            buffers.push_back(new buffer("buf" + to_string(1), *computation_dims, OUTPUT_BUFFER, &abs));

            break;
        }
    }

    tiramisu_code *code = new tiramisu_code(function_name, &computations, &variables, &input_vec, &buffers, default_type_tiramisu);

    generate_cpp_wrapper(code->function_name, buffers, default_type_wrapper);
    generate_h_wrapper(code->function_name, buffers);



    //destruction TODO


}


//automatically generating a tiramisu code with convolutional layers
void generate_tiramisu_code_conv(int code_id, int nb_layers, double *padding_probs, string *default_type_tiramisu, string *default_type_wrapper){
    //initializations
    vector <int> padding_types = generate_padding_types(nb_layers, padding_probs);
    tiramisu_code *code = new tiramisu_code("function" + to_string(code_id), &padding_types, default_type_tiramisu);

    generate_cpp_wrapper(code->function_name, code->buffers, default_type_wrapper);
    generate_h_wrapper(code->function_name, code->buffers);
}



//automatically generating a multiple computations tiramisu code with the associated wrapper
void generate_tiramisu_code_multiple_computations(int code_id, vector<int> *computation_dims, int nb_stages, double *probs,
                                                  vector<int> *var_nums, int nb_inputs, string *default_type_tiramisu,
                                                  string *default_type_wrapper, int offset){
    //initializations
    string function_name = "function" + to_string(code_id);
    vector <variable*> variables;
    vector <variable*> variables_stencils;
    vector <computation*> computations;
    vector <buffer*> buffers;
    vector <input*> inputs;
    vector <computation_abstract*> abs;
    vector <computation_abstract*> abs1;

    int *variables_min_values = new int[(*computation_dims).size()];
    int *variable_max_values = new int[(*computation_dims).size()];
    int *variables_min_values_stencils = new int[(*computation_dims).size()];
    int *variable_max_values_stencils = new int[(*computation_dims).size()];
    for (int i = 0; i < (*computation_dims).size(); ++i) {
        variables_min_values[i] = 0;
        variable_max_values[i] = (*computation_dims)[i];
        variables_min_values_stencils[i] = offset;
        variable_max_values_stencils[i] = (*computation_dims)[i] - offset;
    }
    variables = generate_variables((*computation_dims).size(), 0, variables_min_values, variable_max_values);
    variables_stencils = generate_variables((*computation_dims).size(), (*computation_dims).size(), variables_min_values_stencils, variable_max_values_stencils);


    vector <int> types = computation_types(nb_stages, probs);
    computation *stage_computation;

    for (int i = 0; i < nb_stages; ++i) {
        switch (types[i]){
            case ASSIGNMENT:
                stage_computation = generate_computation("comp" + to_string(i), variables, ASSIGNMENT, {}, {}, 0);
                computations.push_back(stage_computation);
                abs = {stage_computation};
                buffers.push_back(new buffer("buf" + to_string(i), *computation_dims, OUTPUT_BUFFER, &abs));
                break;
            case ASSIGNMENT_INPUTS:
                abs1.clear();
                for (int j = 0; j < nb_inputs; ++j) {
                    input *in = new input("input" + to_string(i) + to_string(j), &variables);
                    inputs.push_back(in);
                    abs = {in};
                    buffers.push_back(new buffer("buf" + to_string(i) + to_string(j), *computation_dims, INPUT_BUFFER, &abs));
                    abs1.push_back(in);
                }
                //TODO: use previous stages as inputs in abs1
                stage_computation = generate_computation("comp" + to_string(i), variables, ASSIGNMENT_INPUTS, abs1, {}, 0);
                computations.push_back(stage_computation);
                abs = {stage_computation};
                buffers.push_back(new buffer("buf" + to_string(i), *computation_dims, OUTPUT_BUFFER, &abs));
                break;
            case STENCIL:
                stage_computation = generate_computation("comp" + to_string(i), variables_stencils, STENCIL, abs, *var_nums, offset);
                computations.push_back(stage_computation);
                abs = {stage_computation};
                buffers.push_back(new buffer("buf" + to_string(i), *computation_dims, OUTPUT_BUFFER, &abs));
                break;
        }
    }

    for (int k = 0; k < variables_stencils.size(); ++k) {
        variables.push_back(variables_stencils[k]);
    }


    tiramisu_code *code = new tiramisu_code(function_name, &computations, &variables, &inputs, &buffers, default_type_tiramisu);

    generate_cpp_wrapper(code->function_name, buffers, default_type_wrapper);
    generate_h_wrapper(code->function_name, buffers);

}



//=====================================================================buffer class==========================================================================================================
buffer::buffer(string name, vector<int> dimensions, int type, vector <computation_abstract*> *computations) {
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
    }
    else{
        s += to_string(dimensions[dimensions.size() - 1]);
        for (int i = dimensions.size() - 2; i >= 0; i--) {
            s += ", " + to_string(dimensions[i]);
        }
    }
    s += ")";
    return s;
}

//========================================================================================computation_abstract class==========================================================================================================

string computation_abstract::variables_to_string() {
    string vars;
    if (!variables.empty()){
        vars += "{" + variables[0]->name;
        for (int i = 1; i < variables.size(); ++i) {
            vars += ", " + variables[i]->name;
        }
        vars += "}";
    }
    return vars;
}

string computation_abstract::vars_to_string(){
    string vars;
    if(!variables.empty()){
        vars += "(" + variables[0]->name;
        for(int i = 1; i < variables.size(); ++i){
            vars += ", " + variables[i]->name;
        }
        vars += ")";
    }
    return vars;
}

//returns all combinations used in stencils
//ex: {(i0, i1, i2), (i0, i1 - 1, i2), (i0, i1 + 1, i2), (i0 - 1, i1, i2), (i0 - 1, i1 - 1, i2), (i0 - 1, i1 + 1, i2), (i0 + 1, i1, i2), (i0 + 1, i1 - 1, i2), (i0 + 1, i1 + 1, i2)}
//with offset = 1 and var_nums = {0, 1}
vector<string> computation_abstract::for_stencils(vector<int> var_nums, int offset) {
    vector<string> strings;
    strings.push_back(vars_to_string());
    string s;
    string zeros;
    string vars;
    int pos;
    for (int i = 1; i < pow(3.0,var_nums.size()); ++i) {
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
                } else if (s[0] == '2') {        //- offset
                    vars += " - " + to_string(k);
                }
                pos++;
            }
            for (int j = 1; j < variables.size(); ++j) {
                vars += ", " + variables[j]->name;
                if (var_nums[pos] == j) {
                    if (s[pos] == '1') {     //+ offset
                        vars += " + " + to_string(k);
                    } else if (s[pos] == '2') {        //- offset
                        vars += " - " + to_string(k);
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
    while (num > 0){
        num_to_base_3 = vals[num % 3] + num_to_base_3;
        num /= 3;
    }
    return num_to_base_3;
}


//=====================================================================computation class==========================================================================================================
computation::computation(string name, int type, vector<variable *> *variables) {
    this->type = type;
    this->variables = *variables;
    this->name = name;

}

//=====================================================================constant class==========================================================================================================
constant::constant(string name, int value) {
    this->name = name;
    this->value = value;
}

//=====================================================================input class==========================================================================================================
input::input(string name,vector<variable*> *variables){
    this->name = name;
    this->variables = *variables;
}


//=====================================================================layout==========================================================================================================
void new_line(int nb_lines, int indentation_level, string *code_buffer){
    for (int i = 0; i < nb_lines; ++i) {
        *code_buffer += "\n";
    }

    for (int i = 0; i < indentation_level; ++i) {
        *code_buffer += "    ";
    }
}

void check_buffer_size(string *code_buffer, ostream *output_file ) {
    if ((*code_buffer).size() >= MAX_BUFFER_SIZE){
        (*output_file) << (*code_buffer);
        (*code_buffer) = "";
    }
}

//=====================================================================random_generator==========================================================================================================
//automatically generating computation
computation *generate_computation(string name, vector<variable*> computation_variables, int computation_type, vector<computation_abstract*> inputs, vector<int> var_nums, int offset){
    computation *c = new computation(name, computation_type, &computation_variables);
    if (computation_type == ASSIGNMENT){
        c->expression = assignment_expression_generator();
    }

    if (computation_type == ASSIGNMENT_INPUTS){
        c->expression = assignment_expression_generator_inputs(inputs);
    }

    if (computation_type == STENCIL){
        c->expression = stencil_expression_generator(inputs[0]->name, c, &var_nums, offset);
    }
    return c;
}


//automatically generating computation expression in case of a simple assignment
string assignment_expression_generator(){
    string expr = to_string(rand() % MAX_ASSIGNMENT_VAL);
    for (int i = 0; i < rand() % (MAX_NB_OPERATIONS_ASSIGNMENT - 1) + 1 ; ++i) {
        int op = rand() % 3;
        switch (op) {
            case 0:
                expr += " + " + to_string(rand() % MAX_ASSIGNMENT_VAL);
                break;
            case 1:
                expr += " - " + to_string(rand() % MAX_ASSIGNMENT_VAL);
                break;
            case 2:
                expr += " * " + to_string(rand() % MAX_ASSIGNMENT_VAL);
                break;
                //       case 3:
                //         expr += " / " + to_string(rand() % (MAX_ASSIGNMENT_VAL - 1) + 1);
                //       break;

        }
    }
    return expr;
}


//automatically generating computation expression in case of an assignment using other computations
string assignment_expression_generator_inputs(vector<computation_abstract *> inputs){
    string vars = inputs[0]->vars_to_string();
    string expr = inputs[0]->name + vars;
    for (int i = 1; i <inputs.size() ; ++i) {
        int op = rand() % 3;
        switch (op) {
            case 0:
                expr += " + " + inputs[i]->name + vars;
                break;
            case 1:
                expr += " - " + inputs[i]->name + vars;
                break;
            case 2:
                expr += " * " + inputs[i]->name + vars;
                break;

        }
    }
    return expr;
}


//automatically generating computation expression in case of stencils
string stencil_expression_generator(string input_name, computation_abstract* in, vector<int> *var_nums, int offset){
    vector <string> vars = in->for_stencils(*var_nums, offset);
    string expr = "(";
    for (int i = 0; i < vars.size() - 1; ++i) {
        expr += input_name + vars[i];
        if (rand() % 2){
            expr += " + ";
        } else expr += " - ";
    }
    expr += input_name + vars[vars.size() - 1] + ")";

    return expr;
}


//returns a vector of computation types according to the probability of their occurence (used in multiple computations codes)
vector <int> computation_types(int nb_comps, double *probs){
    vector <int> types;
    double num = (double) rand() / (RAND_MAX);
    if (num < 1){
        types.push_back(ASSIGNMENT);
    }
    else{
        types.push_back(ASSIGNMENT_INPUTS);
    }
    for (int i = 1; i < nb_comps; ++i) {
        num = (double) rand() / (RAND_MAX);
        if (num < probs[0]){
            types.push_back(ASSIGNMENT);
        }
        else if (num < probs[0] + probs[1]){
            types.push_back(ASSIGNMENT_INPUTS);
        }
        else{
            types.push_back(STENCIL);
        }
    }
    return types;

}


//returns a vector of padding types according to the probability of their occurence (used in convolutions)
vector <int> generate_padding_types(int nb_layers, double *padding_probs){
    vector <int> types;
    double num;
    for (int i = 0; i < nb_layers; ++i) {
        num = (double) rand() / (RAND_MAX);
        if (num < padding_probs[0]){
            types.push_back(SAME);
        }
        else {
            types.push_back(VALID);
        }
    }
    return types;
}

//=====================================================================tiramisu_code class==========================================================================================================

tiramisu_code::tiramisu_code(string function_name, vector<computation*> *computations, vector <variable*> *variables, /*vector<constant*> *constants,*/ vector<input*> *inputs, vector<buffer*> *buffers, string *default_type){
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
    this->inputs = *inputs;
    this->buffers = *buffers;
    this->indentation_level = 1;

    string fpath = "samples/" + function_name;
    mkdir(fpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    this->output_file.open("samples/" + function_name + "/" + function_name +"_file.cpp");

    write_constants();
    write_variables();
    write_inputs();
    write_computations();
    write_buffers();
    generate_code();
    write_footer();



}

void tiramisu_code::write_variables(){
    if (!variables.empty()){
        new_line(2, this->indentation_level, &this->code_buffer);

        this->code_buffer += "var " + variables[0]->name + "(\"" + variables[0]->name + "\", " + to_string(variables[0]->inf_value) + ", " + to_string(variables[0]->sup_value) + ")";
        for (int i = 1; i < variables.size(); ++i) {
             this->code_buffer += ", " + variables[i]->name + "(\"" + variables[i]->name + "\", " + to_string(variables[i]->inf_value) + ", " + to_string(variables[i]->sup_value) + ")";
            check_buffer_size(&code_buffer, &output_file);
        }
        this->code_buffer += ";";
    }
}


void tiramisu_code::write_constants(){
    if (!constants.empty()){
        new_line(2, this->indentation_level, &this->code_buffer);

        this->code_buffer += "constant " + constants[0]->name + "(\"" + constants[0]->name + "\", " + to_string(constants[0]->value) + ")";
        for (int i = 1; i < constants.size(); ++i) {
            this->code_buffer += ", " + constants[i]->name + "(\"" + constants[i]->name + "\", " + to_string(constants[i]->value) + ")";
            check_buffer_size(&code_buffer, &output_file);
        }
        this->code_buffer += ";";
    }
}


void tiramisu_code::write_footer(){
    this->code_buffer += "\n\n    return 0;"
                         "\n}";
    this->output_file << code_buffer;
    this->output_file.close();
}


void tiramisu_code::write_computations() {
    for (int i = 0; i < computations.size(); ++i) {
        if(computations[i]->type <= 3){
            new_line(2, indentation_level, &code_buffer);
            code_buffer += "computation " + computations[i]->name + "(\"" + computations[i]->name
                           + "\", " + computations[i]->variables_to_string() + ", " + computations[i]->expression + ");";
            check_buffer_size(&code_buffer, &output_file);
        }

        else {
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
    code_buffer += "}, \"build/generated/generated_" + function_name + ".o\");";
}

void tiramisu_code::write_buffers() {
    new_line(1, indentation_level, &code_buffer);
    for (int i = 0; i < buffers.size(); ++i) {
        new_line(1, indentation_level, &code_buffer);
        code_buffer += "buffer " + buffers[i]->name + "(\"" + buffers[i]->name + "\", {"
                       + buffers[i]->dimensions_to_string(false).substr(1, buffers[i]->dimensions_to_string(false).size() - 2)
                       + "}, " + default_type + ", ";
        if (buffers[i]->type == INPUT_BUFFER){
            code_buffer += "a_input";
        }
        else{
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
            if (buffers[i]->dimensions.size() != buffers[i]->computations[j]->variables.size()){
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
computation_abstract* tiramisu_code::generate_tiramisu_conv_layer(vector<int> *input_dimensions, computation_abstract *input_comp, vector<int> *filter_dimensions, int padding_type, int layer_number) {
    //input_dimensions : #channels, height, width
    //filter_dimensions : #filters, #channels, height, width
    int padding, output_height, output_width;

   int nb_channels = (*input_dimensions)[0], input_height = (*input_dimensions)[1], input_width = (*input_dimensions)[2], nb_filters = (*filter_dimensions)[0], filter_height = (*filter_dimensions)[2], filter_width = (*filter_dimensions)[3];



    switch (padding_type){
        case SAME:
            padding = ((*filter_dimensions)[2] - 1) / 2;
            output_height = input_height;
            output_width = input_width;
            break;
        case VALID:
            padding = 0;
  
            output_height = (*input_dimensions)[1] + 2 * padding - (*filter_dimensions)[2] + 1;
            output_width = (*input_dimensions)[2] + 2 * padding - (*filter_dimensions)[3] + 1;
            break;
    }


    //variables
    vector <variable*> filter_vars, output_vars;
    variable *h_var = new variable("h" + to_string(layer_number), 0, output_height);
    variable *w_var = new variable("w" + to_string(layer_number), 0, output_width);
    variable *c_var = new variable("c" + to_string(layer_number), 0, nb_channels);
    variable *f_h_var = new variable("f_h" + to_string(layer_number), 0, filter_height);
    variable *f_w_var = new variable("f_w" + to_string(layer_number), 0, filter_width);
    variable *f_c_var = new variable("f_c" + to_string(layer_number), 0, nb_filters);

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
    input *filter_comp = new input("filter" + to_string(layer_number), &filter_vars);
    inputs.push_back(filter_comp);

    computation_abstract *init_input;

    if (padding_type == SAME){     //computation for adding the padding
        vector <variable*> init_same_variables;
        init_same_variables.push_back(new variable("chan" + to_string(layer_number), 0, nb_channels));
        init_same_variables.push_back(new variable("i" + to_string(layer_number), padding, padding + (*input_dimensions)[1]));
        init_same_variables.push_back(new variable("j" + to_string(layer_number), padding, padding + (*input_dimensions)[2]));


        variables.push_back(new variable("chan" + to_string(layer_number), 0, nb_channels));
        variables.push_back(new variable("i" + to_string(layer_number), padding, padding + (*input_dimensions)[1]));
        variables.push_back(new variable("j" + to_string(layer_number), padding, padding + (*input_dimensions)[2]));


        computation *init_input_same = new computation("init_same" + to_string(layer_number),  ASSIGNMENT_INPUTS,&init_same_variables);
        init_input_same->expression = input_comp->name + "(chan" + to_string(layer_number) + ", i" + to_string(layer_number) +" - " + to_string(padding) +", j" + to_string(layer_number) + "- " + to_string(padding);
        if(input_comp->variables.size() > 3){
            init_input_same->expression += ", " + to_string(nb_channels) + " - 1, " + to_string(filter_height) + " - 1, " + to_string(filter_width) + " - 1";
        }
        init_input_same->expression += ")";
        init_input = init_input_same;
        computations.push_back(init_input_same);
    }
    else {
        init_input = input_comp;
    }

    computation *output_comp = new computation("output" + to_string(layer_number), CONV, &output_vars);
    string output_comp_vars = output_comp->vars_to_string();
    output_comp->expression = output_comp->name + output_comp_vars.substr(0, output_comp_vars.size() - 1) + " - 1) + " +
                              init_input->name + "(c" + to_string(layer_number) +
                              ", h" + to_string(layer_number) +" + f_h" + to_string(layer_number) +
                              ", w" + to_string(layer_number) +" + f_w" + to_string(layer_number);
    if(init_input->variables.size() > 3){
        output_comp->expression += ", " + to_string(nb_channels) + " - 1, " + to_string(filter_height) + " - 1, " + to_string(filter_width) + " - 1";
    }
    output_comp->expression += ") * " + filter_comp->name + filter_comp->vars_to_string();

    computations.push_back(output_comp);

    //buffers
    vector <computation_abstract*> comps;
    comps.push_back(init_input);

    if (padding_type == SAME){
        buffer *b_input = new buffer("buf_input" + to_string(layer_number), {(*input_dimensions)[0], (*input_dimensions)[1] + 2 * padding, (*input_dimensions)[2] + 2 * padding}, INPUT_INIT_0, &comps);
        buffers.push_back(b_input);
    }

    comps.clear();
    comps.push_back(filter_comp);
    buffer *b_filter = new buffer("buf_filter" + to_string(layer_number), *filter_dimensions, INPUT_BUFFER, &comps);

    comps.clear();
    comps.push_back(output_comp);
    buffer *b_output = new buffer("buf_output" + to_string(layer_number), {(*filter_dimensions)[0], output_height, output_width}, INPUT_INIT_0, &comps);

    buffers.push_back(b_filter);
    buffers.push_back(b_output);

    return output_comp;

}

tiramisu_code::tiramisu_code(string function_name, vector <int> *padding_types, string *default_type) {
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
    this->output_file.open("samples/" + function_name + "/" + function_name +"_file.cpp");

    int nb_layers = (*padding_types).size();

    //creating the first input "layer"
    vector <int> inits = {3, 1200, 1200};
    vector<vector <int>> inits_filter;
    for (int j = 0; j < nb_layers; ++j) {
        inits_filter.push_back({3, 3, 3, 3});
    }

    vector <variable*> variables_test;
    vector <computation_abstract*> comps;

    int init_nb_channels = inits[0], init_input_height = inits[1], init_input_width = inits[2];

    variables_test.push_back(new variable("chan0", 0, init_nb_channels));
    variables_test.push_back(new variable("i0", 0, init_input_height));
    variables_test.push_back(new variable("j0", 0, init_input_width));

    variables.push_back(new variable("chan0", 0, init_nb_channels));
    variables.push_back(new variable("i0", 0, init_input_height));
    variables.push_back(new variable("j0", 0, init_input_width));

    inputs.push_back(new input("input0", &variables_test));
    comps.push_back(inputs[0]);

    buffers.push_back(new buffer("buf0", inits, INPUT_BUFFER,&comps));

    computation_abstract *new_input = inputs[0];


    for (int i = 1; i < nb_layers; ++i) {
        new_input = generate_tiramisu_conv_layer(&inits, new_input, &inits_filter[i], (*padding_types)[i - 1], i);
        //f_c1, h1, w1, c1, f_h1, f_w1
        inits = {inits_filter[i][0], new_input->variables[1]->sup_value, new_input->variables[2]->sup_value};

    }

    write_variables();
    write_inputs();
    write_computations();
    write_buffers();
    generate_code();
    write_footer();
}



//=====================================================================variable class==========================================================================================================
variable::variable(string name, int inf_value, int sup_value) {
    this->name = name;
    this->inf_value = inf_value;
    this->sup_value = sup_value;
}


//=====================================================================wrapper==========================================================================================================
void generate_h_wrapper(string function_name, vector <buffer *> buffers){
    ofstream output;
    output.open("samples/" + function_name + "/" + function_name +"_wrapper_file.h");
    string code_buffer = "#ifndef HALIDE__generated_" + function_name + "_h\n"
                                                                        "#define HALIDE__generated_" + function_name + "_h\n"
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


void generate_cpp_wrapper(string function_name, vector <buffer*> buffers, string *default_type_wrapper){
    ofstream output;
    output.open("samples/" + function_name + "/" + function_name + "_wrapper_file.cpp");
    string code_buffer = "#include \"Halide.h\"\n"
                         "#include \"" + function_name + "_wrapper_file.h\"\n"
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
    code_buffer += "exec_times_file.open(\"exec_times.txt\", std::ios_base::app);";
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
}


string random_array_initialization_same_dimensions(vector<buffer*> same_size_input_buffers, int *indentation_level, string *default_type_wrapper){
    string initialize_array;
    for (int i = 0; i < same_size_input_buffers.size(); ++i) {
        new_line(1, *indentation_level, &initialize_array);
        initialize_array += "Halide::Buffer<" + *default_type_wrapper + "> " + same_size_input_buffers[i]->name + same_size_input_buffers[i]->dimensions_to_string(
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

string random_array_initialization(buffer *buffer, int *indentation_level, string *default_type_wrapper){
    string initialize_array;
    new_line(1, *indentation_level, &initialize_array);
    initialize_array += "Halide::Buffer<" + *default_type_wrapper + "> " + buffer->name + buffer->dimensions_to_string(true) + ";";

    new_line(1, *indentation_level, &initialize_array);
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
    }
    return initialize_array;
}
