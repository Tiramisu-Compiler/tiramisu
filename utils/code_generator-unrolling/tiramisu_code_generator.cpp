#include <random>
#include <algorithm>
#include "tiramisu_code_generator.h"

//=====================================================================tiramisu_code_generator==========================================================================================================


bool inf(int i, int j) { return (i < j); }



vector<variable *> generate_variables(int nb_variables, int from, int *inf_values, vector<constant *> constants) {
    vector<variable *> variables;
    for (int i = from; i < nb_variables + from; ++i) {
        variables.push_back(new variable("i" + to_string(i), i, inf_values[i - from], constants[i - from]));
    }
    return variables;
}


//automatically generating a tiramisu code with convolutional layers
void generate_tiramisu_code_conv(int code_id, int nb_layers, double *padding_probs, string *default_type_tiramisu,
                                 string *default_type_wrapper) {
    //initializations
    vector<int> padding_types = generate_padding_types(nb_layers, padding_probs);
    tiramisu_code *code = new tiramisu_code("function" + to_string(code_id), &padding_types, default_type_tiramisu);

    generate_cpp_wrapper(code->function_name, code->buffers, default_type_wrapper, code_id, 0);
    generate_h_wrapper(code->function_name, code->buffers, code_id, 0);
}


//automatically generating a multiple computations tiramisu code with the associated wrapper
void generate_tiramisu_code_multiple_computations(int code_id, int nb_stages, double *probs,
                                                  int max_nb_dims, vector<int> scheduling_commands, bool all_schedules,
                                                  int nb_inputs, string *default_type_tiramisu,
                                                  string *default_type_wrapper, int offset,
                                                  vector<int> *tile_sizes, vector<int> *unrolling_factors,
                                                  int nb_rand_schedules, vector<dim_stats *> *stats) {


    //initializations
    srand(code_id);
    cout << "_________________code " + to_string(code_id) + "________________" << endl;
    offset = rand() % offset + 1;
    int id = 0, nb, nb_dims = (rand() % (max_nb_dims - 1)) + 2, sum = 0, const_sum = MAX_MEMORY_SIZE - nb_dims * MIN_LOOP_DIM;
    vector<int> computation_dims;
    for (int i = 0; i < nb_dims; ++i) {
        computation_dims.push_back(rand() % MAX_CONST_VALUE);
        sum += computation_dims[i];
    }

    for (int i = 0; i < nb_dims; ++i) {
        computation_dims[i] *= const_sum;
        computation_dims[i] /= sum;
        computation_dims[i] += MIN_LOOP_DIM;
        computation_dims[i] = (int) pow(2.0, computation_dims[i]);
    }
    (*stats)[nb_dims - 2]->data_sizes[const_sum]++;


    (*stats)[nb_dims - 2]->nb_progs++;

    string function_name = "function" + to_string(code_id);
    vector<vector<variable *>> variables, variables_stencils;
    vector<variable*> all_vars;
    vector<computation *> computations;
    vector<buffer *> buffers;
    vector<input *> inputs;
    vector<computation_abstract *> abs, abs1;

    int *variables_min_values = new int[computation_dims.size()];
    vector<constant *> variable_max_values;
    int *variables_min_values_stencils = new int[computation_dims.size()];
    vector<constant *> variable_max_values_stencils;

    //nb = rand() % nb_stages + 1;
    nb = nb_stages;
    bool st = false;

    int inp, random_index;

    map<int, vector<int>> indexes;
    vector<vector<variable *>> variables_inputs;
    vector<variable *> vars_inputs;

    vector<int> types = computation_types(nb_stages, probs), new_indices;
    computation *stage_computation;

    for (int i = 0; i < computation_dims.size(); ++i) {
        variables_min_values[i] = 0;
        variable_max_values.push_back(new constant("c" + to_string(i), computation_dims[i]));
        variables_min_values_stencils[i] = offset;
        variable_max_values_stencils.push_back(
                new constant("c" + to_string(i) + " - " + to_string(offset), computation_dims[i] - offset));
    }

    for (int i = 0; i < nb; ++i) {
        variables.push_back(generate_variables(computation_dims.size(), i * computation_dims.size(), variables_min_values, variable_max_values));
        variables_stencils.push_back(generate_variables(computation_dims.size(), 100 + i * computation_dims.size(),
                                                        variables_min_values_stencils, variable_max_values_stencils));
        inp = rand() % nb_inputs + 1;
        switch (types[i]) {
            case ASSIGNMENT:
                stage_computation = generate_computation("comp" + to_string(i), variables[i], ASSIGNMENT, {}, {}, 0,
                                                         *default_type_tiramisu, id++);
                computations.push_back(stage_computation);
                abs = {stage_computation};
                buffers.push_back(new buffer("buf" + to_string(i), computation_dims, OUTPUT_BUFFER, &abs));
                break;
            case ASSIGNMENT_INPUTS:
                abs1.clear();
                variables_inputs.clear();
                for (int j = 0; j < inp; ++j) {
                    int nb_vars = rand() % (variables[i].size() - 1) + 1;
                    vars_inputs.clear();
                    if ((double) rand() / (RAND_MAX) < SHUFFLE_SAME_SIZE_PROB) {
                        indexes = indexes_by_size(variables[i]);
                        for (int k = 0; k < variables[i].size(); ++k) {
                            auto var = indexes.find(variables[i][k]->sup_value->value)->second;
                            random_index = rand() % var.size();
                            vars_inputs.push_back(variables[i][var[random_index]]);
                            indexes.find(variables[i][k]->sup_value->value)->second.erase(
                                    indexes.find(variables[i][k]->sup_value->value)->second.begin() + random_index);
                        }
                    }
                    else {
                        vars_inputs = variables[i];
                    }
                    int n_erase = variables[i].size() - nb_vars;
                    vector<int> input_dims = computation_dims;
                    for (int l = 0; l < n_erase; ++l) {
                        random_index = rand() % vars_inputs.size();
                        vars_inputs.erase(vars_inputs.begin() + random_index);
                        input_dims.erase(input_dims.begin() + random_index);
                    }
                    variables_inputs.push_back(vars_inputs);


                    input *in = new input("input" + to_string(i) + to_string(j), &(variables_inputs[j]),
                                          *default_type_tiramisu, id++);
                    inputs.push_back(in);
                    abs = {in};
                    buffers.push_back(
                            new buffer("buf" + to_string(i) + to_string(j), input_dims, INPUT_BUFFER, &abs));
                    abs1.push_back(in);
                }
                //TODO: use previous stages as inputs in abs1
                stage_computation = generate_computation("comp" + to_string(i), variables[i], ASSIGNMENT_INPUTS, abs1, {},
                                                         0, *default_type_tiramisu, id++);
                computations.push_back(stage_computation);
                abs = {stage_computation};
                buffers.push_back(new buffer("buf" + to_string(i), computation_dims, OUTPUT_BUFFER, &abs));


                break;
            case STENCIL:
                vector<int> var_nums_copy;
                vector<int> var_nums_copy_copy;

                for (int l = 0; l < variables_stencils.size(); ++l) {
                    var_nums_copy_copy.push_back(i);
                }

                auto rng = default_random_engine{};
                shuffle(begin(var_nums_copy_copy), end(var_nums_copy_copy), rng);


                int stencil_size;
                (variables_stencils.size() > MAX_STENCIL_SIZE) ? stencil_size = MAX_STENCIL_SIZE
                                                               : stencil_size = variables_stencils.size();

                for (int i = 0; i < rand() % stencil_size + 1; ++i) {
                    var_nums_copy.push_back(var_nums_copy_copy[i]);
                }

                sort(var_nums_copy.begin(), var_nums_copy.end(), inf);

                st = true;
                if (abs.empty()) {
                    input *in = new input("input" + to_string(i), &(variables[i]), *default_type_tiramisu, id++);
                    abs = {in};
                    inputs.push_back(in);
                    abs = {in};
                    buffers.push_back(
                            new buffer("buf" + to_string(i) + to_string(i), computation_dims, INPUT_BUFFER, &abs));
                    abs1.push_back(in);
                }
                stage_computation = generate_computation("comp" + to_string(i), variables_stencils[i], STENCIL, abs,
                                                         var_nums_copy, offset, *default_type_tiramisu, id++);
                computations.push_back(stage_computation);
                abs = {stage_computation};
                buffers.push_back(new buffer("buf" + to_string(i), computation_dims, OUTPUT_BUFFER, &abs));
                break;
        }
        (*stats)[computation_dims.size() - 2]->types[types[i]]->nb_assignments++;
    }

    for (int k = 0; (k < variables_stencils.size()) && st; ++k) {
        variables.push_back(variables_stencils[k]);
    }

    for (int i = 0; i < variables.size(); ++i) {
        for (int j = 0; j < variables[i].size(); ++j) {
            all_vars.push_back(variables[i][j]);
        }
    }

    vector<schedule *> schedules;

    node_class *n = comp_to_node(computations[0], code_id);
    generate_json_one_node(n, code_id);

    vector<schedule_params> schedules_parameters;
    for (int i = 0; i < scheduling_commands.size(); ++i) {
        switch(scheduling_commands[i]){
            case INTERCHANGE:
                schedules_parameters.push_back({INTERCHANGE, {}, 1});
                break;
            case TILING:
                schedules_parameters.push_back({TILING, *tile_sizes, 1});
                break;
            case UNROLL:
                schedules_parameters.push_back({UNROLL, *unrolling_factors, 1});
                break;
            default:
                cout << "The command should be either interchange, tiling or unrolling." << endl;
                return;
        }
    }


    vector<vector<schedule*>> schedules_exhaustive = {};
    vector<vector<variable*>> variables_exhaustive;
    vector<schedules_class*> schedule_classes;

    vector<variable*> all_schedule_variables;

    if (all_schedules) {
        generate_all_schedules(schedules_parameters, computations[0], &schedules_exhaustive, &variables_exhaustive,
                               &schedule_classes);
    }
    else {
        generate_random_schedules(nb_rand_schedules, schedules_parameters, computations[0], &schedules_exhaustive,
                                  &variables_exhaustive, &schedule_classes);
    }


    tiramisu_code *code;
    int cpt_schedule = -1, cpt_unrolling = -1;
    string fpath;

    for (int i = 0; i < variables_exhaustive.size(); ++i) {
        all_schedule_variables = all_vars;
        for (int j = 0; j < variables_exhaustive[i].size(); ++j) {
            if (!(contains(all_schedule_variables, variables_exhaustive[i][j]))){
                all_schedule_variables.push_back(variables_exhaustive[i][j]);
            }
        }
       // all_schedule_variables.push_back(new variable("i_vec", 21));
       // all_schedule_variables.push_back(new variable("i_vec1", 22));
        //schedules_exhaustive[i].push_back(new schedule({computations[0], computations[1]}, AFTER, {0}, {}));

        fpath = "samples/function" + to_string(code_id);
        mkdir(fpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if (find_schedule(schedules_exhaustive[i], UNROLL) == -1){
            cpt_schedule++;
            fpath = "samples/function" + to_string(code_id) + "/function" + to_string(code_id) + "_schedule_" + to_string(cpt_schedule);
            mkdir(fpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            cpt_unrolling = -1;
        }

        cpt_unrolling++;

        code = new tiramisu_code(code_id,
                                 function_name + "_schedule_" + to_string(cpt_schedule) + "_unroll_" +
                                 to_string(cpt_unrolling), cpt_schedule,
                                 &computations, &all_schedule_variables,
                                 &variable_max_values, &inputs, &buffers,
                                 default_type_tiramisu, &schedules_exhaustive[i]);
        generate_cpp_wrapper(code->function_name, buffers, default_type_wrapper, code_id, cpt_schedule);
        generate_h_wrapper(code->function_name, buffers, code_id, cpt_schedule);
        generate_json_schedules(schedule_classes[i], code_id, cpt_schedule, code->function_name);
    }



}



node_class *comp_to_node(computation *comp, int seed) {

    vector<iterator_class *> iterators_array;
    for (int i = 0; i < comp->variables.size(); ++i) {
        iterators_array.push_back(new iterator_class(comp->variables[i]->id, comp->variables[i]->inf_value,
                                                     comp->variables[i]->sup_value->value));
    }

    iterators_class *iterators = new iterators_class(iterators_array.size(), iterators_array);

    vector<mem_access_class *> mem_accesses_array;
    for (int i = 0; i < comp->used_comps.size(); ++i) {
        mem_accesses_array.push_back(new mem_access_class(comp->used_comps[i]->id, comp->used_comps_accesses[i]));
    }

    mem_accesses_class *mem_accesses = new mem_accesses_class(mem_accesses_array.size(), mem_accesses_array);

    vector<int> its, its_stencils;

    for (int i = 0; i < iterators_array.size(); ++i) {
        its.push_back(iterators_array[i]->id);
    }

    vector<input_class *> inputs;
    //not flexible, based on the fact that only stencils use different iterators
    if (comp->type == STENCIL) {
        for (int i = 0; i < comp->used_comps[0]->variables.size(); ++i) {
            iterators->n++;
            iterators->it_array.push_back(new iterator_class(comp->used_comps[0]->variables[i]->id,
                                                             comp->used_comps[0]->variables[i]->inf_value,
                                                             comp->used_comps[0]->variables[i]->sup_value->value));
            its_stencils.push_back(comp->used_comps[0]->variables[i]->id);
        }
        inputs.push_back(new input_class(comp->used_comps[0]->id, comp->used_comps[0]->data_type, its_stencils));
    } else {
        for (int i = 0; i < comp->used_comps.size(); ++i) {
            inputs.push_back(new input_class(comp->used_comps[i]->id, comp->used_comps[i]->data_type, its));
        }
    }

    inputs_class *all_inputs = new inputs_class(inputs.size(), inputs);


    computation_class *computation = new computation_class(comp->id, comp->data_type, its, comp->op_stats,
                                                           mem_accesses);

    computations_class *computations = new computations_class(1, {computation});

    assignment_class *assignment = new assignment_class(comp->id, 0);

    assignments_class *assignments = new assignments_class(1, {assignment});

    vector<loop_class *> loops_array;
    for (int i = 0; i < comp->variables.size(); ++i) {
        loops_array.push_back(new loop_class(i, i - 1, 0, iterators_array[i]->id, new assignments_class(0, {})));
    }
    loops_array[comp->variables.size() - 1]->assignments = assignments;

    loops_class *loops = new loops_class(comp->variables.size(), loops_array);

    node_class *node = new node_class(seed, loops, computations, iterators, all_inputs);

    node->code_type = comp->type;

    return node;
}


//=====================================================================random_generator==========================================================================================================
//automatically generating computation
computation *generate_computation(string name, vector<variable *> computation_variables, int computation_type,
                                  vector<computation_abstract *> inputs, vector<int> var_nums, int offset,
                                  string data_type, int id) {
    computation *c = new computation(name, computation_type, &computation_variables, data_type, id);
    vector<int> vect;
    vector<vector<int>> access;
    if (computation_type == ASSIGNMENT) {
        c->expression = assignment_expression_generator(c->op_stats);
        c->used_comps = {};
    }

    if (computation_type == ASSIGNMENT_INPUTS) {
        c->expression = assignment_expression_generator_inputs(inputs, c->op_stats);
        c->used_comps = inputs;
        c->used_comps_accesses.clear();
        for (int i = 0; i < inputs.size(); ++i) {
            access.clear();
            for (int j = 0; j < c->used_comps[i]->variables.size(); ++j) {
                vect.clear();
                for (int k = 0; k < c->variables.size() + 1; ++k) {
                    if (k < c->variables.size() && (c->used_comps[i]->variables[j]->id == c->variables[k]->id)) {
                        vect.push_back(1);
                    } else
                        vect.push_back(0);
                }
                access.push_back(vect);
            }
            c->used_comps_accesses.push_back(access);
        }
    }

    if (computation_type == STENCIL) {
        c->used_comps_accesses.clear();
        c->used_comps.clear();
        for (int i = 0; i < offset * (pow(3.0, var_nums.size()) - 1) + 1; ++i) {
            access.clear();
            for (int j = 0; j < c->variables.size(); ++j) {
                vect.clear();
                for (int k = 0; k < c->variables.size() + 1; ++k) {
                    if (j == k) {
                        vect.push_back(1);
                    } else vect.push_back(0);
                }
                access.push_back(vect);
            }
            c->used_comps_accesses.push_back(access);
            c->used_comps.push_back(inputs[0]);
        }
        c->expression = stencil_expression_generator(inputs[0]->name, c, &var_nums, offset, c->op_stats,
                                                     &c->used_comps_accesses);
    }
    return c;
}


//automatically generating computation expression in case of a simple assignment
string assignment_expression_generator(int **op_stats) {
    string expr = to_string(rand() % MAX_ASSIGNMENT_VAL);
    for (int i = 0; i < rand() % (MAX_NB_OPERATIONS_ASSIGNMENT - 1) + 1; ++i) {
        int op = rand() % 3;
        switch (op) {
            case 0:
                expr += " + " + to_string(rand() % MAX_ASSIGNMENT_VAL);
                op_stats[0][0]++;
                break;
            case 1:
                expr += " - " + to_string(rand() % MAX_ASSIGNMENT_VAL);
                op_stats[0][1]++;
                break;
            case 2:
                expr += " * " + to_string(rand() % MAX_ASSIGNMENT_VAL);
                op_stats[0][2]++;
                break;
                //       case 3:
                //         expr += " / " + to_string(rand() % (MAX_ASSIGNMENT_VAL - 1) + 1);
                //         op_stats[3]++;
                //       break;

        }
    }
    return expr;
}


//automatically generating computation expression in case of an assignment using other computations
string assignment_expression_generator_inputs(vector<computation_abstract *> inputs, int **op_stats) {
    string vars = inputs[0]->vars_to_string();
    string expr = inputs[0]->name + vars;
    for (int i = 1; i < inputs.size(); ++i) {
        int op = rand() % 3;
        switch (op) {
            case 0:
                expr += " + " + inputs[i]->name + inputs[i]->vars_to_string();
                op_stats[0][0]++;
                break;
            case 1:
                expr += " - " + inputs[i]->name + inputs[i]->vars_to_string();
                op_stats[0][1]++;
                break;
            case 2:
                expr += " * " + inputs[i]->name + inputs[i]->vars_to_string();
                op_stats[0][2]++;
                break;
                // case 3:
                //   expr += " / " + inputs[i]->name + vars + " + 1";
                //   op_stats[3]++;
                // break;

        }
    }
    return expr;
}


//automatically generating computation expression in case of stencils
string stencil_expression_generator(string input_name, computation_abstract *in, vector<int> *var_nums, int offset,
                                    int **op_stats, vector<vector<vector<int>>> *accesses) {
    vector<string> vars = in->for_stencils(*var_nums, offset, accesses);
    string expr = "(";
    for (int i = 0; i < vars.size() - 1; ++i) {
        expr += input_name + vars[i];
        if (rand() % 2) {
            expr += " + ";
            op_stats[0][0]++;
        } else {
            expr += " - ";
            op_stats[0][1]++;
        }
    }
    expr += input_name + vars[vars.size() - 1] + ")";

    return expr;
}


//returns a vector of computation types according to the probability of their occurence (used in multiple computations codes)
vector<int> computation_types(int nb_comps, double *probs) {
    vector<int> types;
    double num;
    for (int i = 0; i < nb_comps; ++i) {
        num = (double) rand() / (RAND_MAX);
        if (num < probs[0]) {
            types.push_back(ASSIGNMENT);
        } else if (num < probs[0] + probs[1]) {
            types.push_back(ASSIGNMENT_INPUTS);
        } else {
            types.push_back(STENCIL);
        }
    }
    return types;

}


//returns a vector of padding types according to the probability of their occurence (used in convolutions)
vector<int> generate_padding_types(int nb_layers, double *padding_probs) {
    vector<int> types;
    double num;
    for (int i = 0; i < nb_layers; ++i) {
        num = (double) rand() / (RAND_MAX);
        if (num < padding_probs[0]) {
            types.push_back(SAME);
        } else {
            types.push_back(VALID);
        }
    }
    return types;
}

//=============================================Managing schedules=====================================================================



vector<configuration> generate_configurations(int schedule, vector<int> factors, vector<variable*> variables, int computation_type){
    //variables : all computation variables before applying schedule
    //in_variables : variables used in schedule
    //out_variables : all computation variables after applying schedule

    vector<configuration> configs;
    variable *v1, *v2, *v3, *v4, *v5, *v6;
    configuration conf;
    conf.computation_type = computation_type;
    conf.out_variables = variables;
    switch(schedule){
        case INTERCHANGE:{
            conf.schedule = NONE;
            configs.push_back(conf);
            conf.schedule = INTERCHANGE;
            for(int i = 0; i <variables.size(); i++){
                for(int j = i + 1; j < variables.size(); j++){
                    conf.in_variables = {variables[i], variables[j]};
                    conf.out_variables = variables;
                    iter_swap(conf.out_variables.begin() + i, conf.out_variables.begin() + j);
                    configs.push_back(conf);
                }
            }
            break;
        }
        case TILING:{
            v1 = new variable("i01", 11);
            v2 = new variable("i02", 12);
            v3 = new variable("i03", 13);
            v4 = new variable("i04", 14);
            conf.schedule = NONE;
            configs.push_back(conf);
            conf.schedule = TILE_2;
            for(int i = 0; i <factors.size(); i++){
                for(int j = 0; j <factors.size(); j++){
                    for(int l = 0; l <variables.size() - 1; l++){
                        conf.factors = {factors[i], factors[j]};
                        conf.in_variables = {variables[l], variables[l + 1], v1, v2, v3, v4};
                        conf.out_variables = variables;
                        conf.out_variables.insert(conf.out_variables.begin() + l, v1);
                        conf.out_variables.insert(conf.out_variables.begin() + l + 1, v2);
                        conf.out_variables.insert(conf.out_variables.begin() + l + 2, v3);
                        conf.out_variables.insert(conf.out_variables.begin() + l + 3, v4);
                        conf.out_variables.erase(conf.out_variables.begin() + l + 4, conf.out_variables.begin() + l + 6);
                        configs.push_back(conf);
                    }
                }
            }
            if (variables.size() > 2){
                conf.schedule = TILE_3;
                v5 = new variable("i05", 15);
                v6 = new variable("i06", 16);
                for(int i = 0; i <factors.size(); i++){
                    for(int j = 0; j <factors.size(); j++){
                        for(int k = 0; k <factors.size(); k++){
                            for(int l = 0; l <variables.size() - 2; l++){
                                conf.factors = {factors[i], factors[j], factors[k]};
                                conf.in_variables = {variables[l], variables[l + 1], variables[l + 2], v1, v2, v3, v4, v5, v6};
                                conf.out_variables = variables;
                                conf.out_variables.insert(conf.out_variables.begin() + l, v1);
                                conf.out_variables.insert(conf.out_variables.begin() + l + 1, v2);
                                conf.out_variables.insert(conf.out_variables.begin() + l + 2, v3);
                                conf.out_variables.insert(conf.out_variables.begin() + l + 3, v4);
                                conf.out_variables.insert(conf.out_variables.begin() + l + 4, v5);
                                conf.out_variables.insert(conf.out_variables.begin() + l + 5, v6);
                                conf.out_variables.erase(conf.out_variables.begin() + l + 6, conf.out_variables.begin() + l + 9);
                                configs.push_back(conf);
                            }
                        }
                    }
                }
            }
            break;
        }
        case UNROLL:{
            conf.schedule = NONE_UNROLL;
            configs.push_back(conf);
            conf.schedule = UNROLL;
            conf.in_variables = {variables.back()};
            for(int i = 0; i <factors.size(); i++){
            
                conf.factors = {factors[i]};
                configs.push_back(conf);
            }
            break;
        }
    }
    return configs;
}



//possible ways of generating schedules :
//*exhaustively(list of (schedule, list of parameters)) :
//explore all combinations with all possible factors
//*randomly(number of schedules, list of(schedule, list of parameters, probability for schedule))

void generate_all_schedules(vector<schedule_params> schedules, computation *comp, vector<vector <schedule*>> *generated_schedules, vector<vector<variable*>> *generated_variables, vector<schedules_class*> *schedule_classes){
    deque<state*> q;
    state *current_state;
    int current_level = 0, cpt = 0;
    vector<vector<schedule*>> schedules_exhaustive;
    vector<configuration> stage_configurations = generate_configurations(schedules[current_level].schedule, schedules[current_level].factors, comp->variables, comp->type),
            passed_configurations;
    for (int i = 0; i < stage_configurations.size(); i++){
        q.insert(q.begin() + i, new state({stage_configurations[i]}, current_level));
    }
    while (!q.empty()){
        current_state = q.front();
        q.pop_front();
        if (current_state->is_extendable(schedules.size())){
            passed_configurations = current_state->schedules;
            current_level = current_state->level + 1;
            stage_configurations = generate_configurations(schedules[current_level].schedule, schedules[current_level].factors, current_state->schedules.back().out_variables, comp->type);
            for (int i = 0; i < stage_configurations.size(); i++){
                passed_configurations.push_back(stage_configurations[i]);
                q.insert(q.begin() + i, new state(passed_configurations, current_level));
                passed_configurations.pop_back();
            }
        }
        if (current_state->is_appliable(schedules.size())){
            (*generated_schedules).push_back(current_state->apply(comp));
            (*generated_variables).push_back(current_state->schedules.back().out_variables);
            (*schedule_classes).push_back(confs_to_sc(current_state->schedules));
            cpt++;
        }
    }
}



void generate_random_schedules(int nb_schedules, vector<schedule_params> schedules, computation *comp, vector<vector <schedule*>> *generated_schedules, vector<vector<variable*>> *generated_variables, vector<schedules_class*> *schedule_classes){
    int num, n_attempts = 5, attempts = 0;
    string s;
    vector<variable*> computation_variables;
    vector<schedule*> generated_schedule;
    vector<configuration> configs;
    configuration conf;
    (*generated_schedules).push_back(generated_schedule);
    (*generated_variables).push_back(computation_variables);
    conf.schedule = NONE;
    configs.push_back(conf);
    configs.push_back(conf);
    configs.push_back(conf);
    (*schedule_classes).push_back(confs_to_sc(configs));
    for (int i = 0; i < nb_schedules; ++i) {
        generated_schedule.clear();
        computation_variables = comp->variables;
        num = rand() % ((int) pow(2.0, schedules.size()) - 1) + 1;
        s = to_base_2(num, schedules.size());
        for (int j = 0; j < schedules.size(); ++j) {
            if (s[j] == '1') {
                attempts = 0;
                conf = random_conf(schedules[j], comp->type, computation_variables);
                while ((!is_valid(conf)) && (attempts < n_attempts)){
                    conf = random_conf(schedules[j], comp->type, computation_variables);
                    attempts++;
                }
                if (attempts == n_attempts) continue;
                computation_variables = conf.out_variables;
                if (conf.schedule != NONE) {
                    generated_schedule.push_back(new schedule({comp}, conf.schedule, conf.factors, conf.in_variables));
                }
                configs.push_back(conf);
            }
        }
        (*generated_schedules).push_back(generated_schedule);
        (*generated_variables).push_back(computation_variables);
        (*schedule_classes).push_back(confs_to_sc(configs));
    }
}


schedules_class *confs_to_sc(vector<configuration> schedules){
    tiling_class *tc;
    vector<int> interchange_dims;
    int unrolling_factor;
    (schedules[0].schedule == INTERCHANGE) ?
            interchange_dims = {schedules[0].in_variables[0]->id, schedules[0].in_variables[1]->id} :
            interchange_dims = {};
    (schedules[1].schedule == TILE_2) ?
            tc = new tiling_class(2, {schedules[1].in_variables[0]->id, schedules[1].in_variables[1]->id}, {schedules[1].factors[0], schedules[1].factors[1]}) :
            (schedules[1].schedule == TILE_3) ?
                tc = new tiling_class(3, {schedules[1].in_variables[0]->id, schedules[1].in_variables[1]->id, schedules[1].in_variables[2]->id}, {schedules[1].factors[0], schedules[1].factors[1], schedules[1].factors[2]}) :
                tc = nullptr;
    (schedules[2].schedule == UNROLL) ? unrolling_factor = schedules[2].factors[0] : unrolling_factor = (int)NULL;
    return new schedules_class(interchange_dims, unrolling_factor, tc);
}

string to_base_2(int num, int nb_pos){
    string vals = "01";
    string num_to_base_2;
    while (num > 0) {
        num_to_base_2 = vals[num % 2] + num_to_base_2;
        num /= 2;
    }
    int nb_zeros = nb_pos - num_to_base_2.size();
    for (int i = 0; i < nb_zeros; ++i) {
        num_to_base_2 = "0" + num_to_base_2;
    }
    return num_to_base_2;
}
