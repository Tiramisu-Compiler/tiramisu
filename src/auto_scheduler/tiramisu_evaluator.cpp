#include <tiramisu/auto_scheduler/evaluator.h>

#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tiramisu::auto_scheduler
{

evaluate_by_execution::evaluate_by_execution(tiramisu::function *fct, 
                                             std::vector<tiramisu::buffer*> const& arguments, 
                                             std::string const& obj_filename, 
                                             std::string const& wrapper_cmd)
    : evaluator(), fct(fct), obj_filename(obj_filename), wrapper_cmd(wrapper_cmd)
{
    // Set Halide compilation features
    halide_target = Halide::get_host_target();
    halide_target.set_features(halide_features);
    
    // Set input and output buffers
    fct->set_arguments(arguments);
    for (auto const& buf : arguments)
    {
        Halide::Argument buffer_arg(
                buf->get_name(),
                halide_argtype_from_tiramisu_argtype(buf->get_argument_type()),
                halide_type_from_tiramisu_type(buf->get_elements_type()),
                buf->get_n_dims());
                
       halide_arguments.push_back(buffer_arg);
    }
}

float evaluate_by_execution::evaluate(syntax_tree& ast)
{
    // Apply all the optimizations
    apply_optimizations(ast);
    parallelize_outermost_levels(ast.computations_list);
    
    // Compile the program to an object file
    fct->lift_dist_comps();
    fct->gen_time_space_domain();
    fct->gen_isl_ast();
    fct->gen_halide_stmt();
    
    Halide::Module m = lower_halide_pipeline(fct->get_name(), halide_target, halide_arguments,
                                             Halide::Internal::LoweredFunc::External,
                                             fct->get_halide_stmt());
                                             
    m.compile(Halide::Outputs().object(obj_filename));
    
    // Turn the object file to a shared library
    std::string gcc_cmd = "g++ -shared -o " + obj_filename + ".so " + obj_filename;
    int status = system(gcc_cmd.c_str());
    
    // Execute the wrapper and get execution time
    double exec_time = 0.f;
    FILE *pipe = popen(wrapper_cmd.c_str(), "r");
    
    fscanf(pipe, "%lf", &exec_time);
    pclose(pipe);
    
    // Remove all the optimizations
    fct->reset_schedules();
    
    return exec_time;
}

simple_rnn_evaluator::simple_rnn_evaluator(std::string const& model_path) 
    : evaluator()
{
    model = torch::jit::load(model_path);
}

float simple_rnn_evaluator::evaluate(syntax_tree& ast)
{
    std::vector<tiramisu::computation*> const& computations = ast.get_computations();
    int nb_computations = computations.size();
    
    at::Tensor dnn_input = torch::zeros({1, nb_computations, COMPUTATION_REPR_SIZE});
    at::Tensor dnn_length = nb_computations * torch::ones({1});
    
    // Get information about the schedule
    dnn_schedule sched(MAX_NB_ITERATORS, ast.new_optims);
    
    // Create the vector representation of each computation
    int comp_index = 0;
    for (ast_node *node : ast.roots)
        comp_index = represent_node(node, sched, comp_index, dnn_input);
    
    // Call the DNN model
    std::vector<torch::jit::IValue> params = {dnn_input, dnn_length};
    at::Tensor output = model.forward(params).toTensor();
    
    return -output.item().to<float>();
}

int simple_rnn_evaluator::represent_node(ast_node *node, dnn_schedule const& sched, int comp_index, at::Tensor& dnn_input)
{    
    for (int i = 0; i < node->computations.size(); ++i)
    {
        std::vector<dnn_iterator> iters = dnn_iterator::get_iterators_from_computation(*node->computations[i]);
        dnn_input[0][comp_index] = get_computation_repr(iters, sched, node->comps_accesses[i]);
        comp_index++;
    }
    
    for (ast_node *child : node->children)
        comp_index = represent_node(child, sched, comp_index, dnn_input);
    
    return comp_index;
}

at::Tensor simple_rnn_evaluator::get_computation_repr(std::vector<dnn_iterator> const& iters, dnn_schedule const& sched, dnn_accesses const& accesses)
{
    at::Tensor ret = torch::zeros({COMPUTATION_REPR_SIZE});
    int offset = 0;
    
    // Add iterators, interchange and tiling
    for (int i = 0; i < iters.size(); ++i)
    {
        ret[offset + 0] = iters[i].low_bound;
        ret[offset + 1] = iters[i].up_bound + 1;
        
        ret[offset + 2] = sched.interchanged[i];
        ret[offset + 3] = sched.tiled[i];
        ret[offset + 4] = sched.tiling_fact[i];
        
        offset += ITERATORS_REPR_SIZE;
    }
    
    offset = MAX_NB_ITERATORS * ITERATORS_REPR_SIZE;
    
    // Add accesses
    for (int i = 0; i < accesses.accesses_list.size(); ++i)
    {
        dnn_access_matrix const& access = accesses.accesses_list[i];
        
        ret[offset] = access.buffer_id;
        offset++;
        
        for (int j = 0; j < access.matrix.size(); ++j)
        {
            for (int k = 0; k < access.matrix[j].size(); ++k)
                ret[offset + k] = access.matrix[j][k];
            
            offset += MAX_NB_ITERATORS + 1;
        }
        
        offset += (MAX_NB_ITERATORS - access.matrix.size()) * MAX_NB_ITERATORS * (MAX_NB_ITERATORS + 1);
    }
    
    offset = MAX_NB_ITERATORS * ITERATORS_REPR_SIZE + MAX_NB_ACCESSES * ACCESS_REPR_SIZE;
    
    // Add unrolling factor
    if (sched.unrolling_fact > 0)
        ret[offset] = 1.0;
        
    ret[offset + 1] = sched.unrolling_fact;
    
    return ret;
}

tree_lstm_evaluator::tree_lstm_evaluator(std::string const& cmd_path, std::vector<std::string> const& cmd_args)
    : evaluator()
{
    pid_t pid = 0;
    int inpipe_fd[2];
    int outpipe_fd[2];
    
    pipe(inpipe_fd);
    pipe(outpipe_fd);
    
    pid = fork();
    if (pid == 0)
    {
        dup2(outpipe_fd[0], STDIN_FILENO);
        dup2(inpipe_fd[1], STDOUT_FILENO);
        
        close(outpipe_fd[1]);
        close(inpipe_fd[0]);
        
        char* argv[cmd_args.size() + 2];
        argv[0] = (char*)malloc(sizeof(char) * (cmd_path.size() + 1));
        strcpy(argv[0], cmd_path.c_str());
        argv[cmd_args.size() + 1] = NULL;
        
        for (int i = 0; i < cmd_args.size(); ++i) {
            argv[i + 1] = (char*)malloc(sizeof(char) * (cmd_args[i].size() + 1));
            strcpy(argv[i + 1], cmd_args[i].c_str());
        }
        
        std::cout << getpid() << std::endl;
        execv(cmd_path.c_str(), argv);
        
        exit(1);
    }
    
    close(outpipe_fd[0]);
    close(inpipe_fd[1]);
    
    model_write = fdopen(outpipe_fd[1], "w");
    model_read = fdopen(inpipe_fd[0], "r");
}

float tree_lstm_evaluator::evaluate(syntax_tree& ast)
{
    tiramisu::computation *comp = ast.computations_list[0];
    ast_node *node = ast.roots[0]->get_leftmost_node();
    std::vector<dnn_iterator> iterators_list = dnn_iterator::get_iterators_from_computation(*comp);
    dnn_accesses& accesses = node->comps_accesses[0];
    
    std::string iterators_str = "\"iterators\" : {";
    
    for (int i = 0; i < iterators_list.size(); ++i)
    {
        dnn_iterator const& iter = iterators_list[i];
        
        iterators_str += "\"" + iter.name + "\" : {";
        
        iterators_str += "\"lower_bound\" : " + std::to_string(iter.low_bound) + ",";
        iterators_str += "\"upper_bound\" : " + std::to_string(iter.up_bound + 1) + ",";
        
        iterators_str += "\"parent_iterator\" : ";
        if (i == 0)
            iterators_str += "null,";
        else
            iterators_str += "\"" + iterators_list[i - 1].name + "\",";
            
        iterators_str += "\"child_iterators\" : [";
        if (i != iterators_list.size() - 1)
            iterators_str += "\"" + iterators_list[i + 1].name + "\"";
        
        iterators_str += "],";
        
        iterators_str += "\"computations_list\" : [";
        if (i == iterators_list.size() - 1)
            iterators_str += "\"" + comp->get_name() + "\"";
        iterators_str += "]";
        
        iterators_str += "}";
        
        if (i != iterators_list.size() - 1)
            iterators_str += ",";
    }
    
    iterators_str += "},";
    
    std::string computations_str = "\"computations\" : {";
    
    computations_str += "\"" + comp->get_name() + "\" : {";
    computations_str += "\"iterators\" : [";
    
    for (int i = 0; i < iterators_list.size(); ++i)
    {
        computations_str += "\"" + iterators_list[i].name + "\"";
        if (i != iterators_list.size() - 1)
            computations_str += ",";
    }
    
    computations_str += "],";
    
    computations_str += "\"real_dimensions\" : [";
    
    for (int i = 0; i < real_nb_iters; ++i)
    {
        computations_str += "\"" + iterators_list[i].name + "\"";
        if (i != real_nb_iters - 1)
            computations_str += ",";
    }
    
    computations_str += "],";
    
    computations_str += "\"comp_is_reduction\" : ";
    if (is_reduction)
        computations_str += "true,";
    else
        computations_str += "false,";
        
    computations_str += "\"number_of_additions\" : " + std::to_string(nb_additions) + ",";
    computations_str += "\"number_of_subtraction\" : " + std::to_string(nb_substractions) + ",";
    computations_str += "\"number_of_multiplication\" : " + std::to_string(nb_multiplications) + ",";
    computations_str += "\"number_of_division\" : " + std::to_string(nb_divisions) + ",";
    
    computations_str += "\"accesses\" : [";
    
    for (int i = 0; i < accesses.accesses_list.size(); ++i)
    {
        dnn_access_matrix const& matrix  = accesses.accesses_list[i];
        
        computations_str += "{";
        
        computations_str += "\"access_is_reduction\" : ";
        if (i == 0)
            computations_str += "true,";
        else
            computations_str += "false,";
            
        computations_str += "\"buffer_id\" : " + std::to_string(matrix.buffer_id) + ",";
        computations_str += "\"access_matrix\" : [";
        
        for (int x = 0; x < matrix.matrix.size(); ++x)
        {
            computations_str += "[";
            for (int y = 0; y < matrix.matrix[x].size(); ++y)
            {
                computations_str += std::to_string(matrix.matrix[x][y]);
                if (y != matrix.matrix[x].size() - 1)
                    computations_str += ", ";
            }
            
            computations_str += "]";
            if (x != matrix.matrix.size() - 1)
                computations_str += ",";
        }
        
        computations_str += "]";
        
        computations_str += "}";
        
        if (i != accesses.accesses_list.size() - 1)
            computations_str += ",";
    }
    
    computations_str += "]";

    computations_str += "}";
    
    computations_str += "}";
    
    std::string prog_json = "{" + iterators_str + computations_str + "}\n";
    
    bool interchanged = false;
    bool tiled = false;
    bool unrolled = false;
    
    int int_l0, int_l1;
    int tile_nb_l, tile_l0, tile_l0_fact, tile_l1_fact, tile_l2_fact;
    int unrolling_fact;
    
    for (optimization_info const& optim_info : ast.new_optims)
    {
        switch (optim_info.type)
        {
            case optimization_type::TILING:
                tiled = true;
                if (optim_info.nb_l == 2)
                {
                    tile_nb_l = 2;
                    tile_l0 = optim_info.l0;
                    tile_l0_fact = optim_info.l0_fact;
                    tile_l1_fact = optim_info.l1_fact;
                }
                
                else if (optim_info.nb_l == 3)
                {
                    tile_nb_l = 3;
                    tile_l0 = optim_info.l0;
                    tile_l0_fact = optim_info.l0_fact;
                    tile_l1_fact = optim_info.l1_fact;
                    tile_l2_fact = optim_info.l2_fact;
                }
                break;
                
            case optimization_type::INTERCHANGE:
                interchanged = true;
                int_l0 = optim_info.l0;
                int_l1 = optim_info.l1;
                break;
                
            case optimization_type::UNROLLING:
                unrolled = true;
                unrolling_fact = optim_info.l0_fact;
                break;
                
            default:
                break;
        }
    }
    
    std::string sched_json = "{";
    
    sched_json += "\"" + comp->get_name() + "\" : {";
    
    sched_json += "\"interchange_dims\" : [";
    
    if (interchanged)
    {
        sched_json += "\"" + iterators_list[int_l0].name + "\", \"" + iterators_list[int_l1].name + "\"";
        
        dnn_iterator dnn_it = iterators_list[int_l0];
        iterators_list[int_l0] = iterators_list[int_l1];
        iterators_list[int_l1] = dnn_it;
    }
    
    sched_json += "],";
    
    sched_json += "\"tiling\" : {";
    
    if (tiled)
    {
        if (tile_nb_l == 2)
        {
            sched_json += "\"tiling_depth\" : 2,";
            sched_json += "\"tiling_dims\" : [";
            
            sched_json += "\"" + iterators_list[tile_l0].name + "\", " + "\"" + iterators_list[tile_l0 + 1].name + "\"";
            
            sched_json += "],";
            
            sched_json += "\"tiling_factors\" : [";
            
            sched_json += "\"" + std::to_string(tile_l0_fact) + "\", " + "\"" + std::to_string(tile_l1_fact) + "\"";
            
            sched_json += "]";
        }
        
        else
        {
            sched_json += "\"tiling_depth\" : 2,";
            sched_json += "\"tiling_dims\" : [";
            
            sched_json += "\"" + iterators_list[tile_l0].name + "\", " + "\"" + iterators_list[tile_l0 + 1].name + "\", " + "\"" + iterators_list[tile_l0 + 2].name + "\"";
            
            sched_json += "],";
            
            sched_json += "\"tiling_factors\" : [";
            
            sched_json += "\"" + std::to_string(tile_l0_fact) + "\", " + "\"" + std::to_string(tile_l1_fact) + "\", " + "\"" + std::to_string(tile_l2_fact) + "\"";
            
            sched_json += "]";
        }
    }
    
    sched_json += "},";
    
    sched_json += "\"unrolling_factor\" : ";
    
    if (unrolled)
    {
        sched_json += "\"" + std::to_string(unrolling_fact) + "\"";
    }
    
    else
    {
        sched_json += "null";
    }
    
    sched_json += "}";
    
    sched_json += "}\n";
    
    fputs(prog_json.c_str(), model_write);
    fputs(sched_json.c_str(), model_write);
    fflush(model_write);
    
    float speedup = 0.f;
    fscanf(model_read, "%f", &speedup);
    
    return speedup;
}

}
