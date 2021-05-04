#include <tiramisu/auto_scheduler/evaluator.h>

#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tiramisu::auto_scheduler
{

evaluate_by_execution::evaluate_by_execution(std::vector<tiramisu::buffer*> const& arguments, 
                                             std::string const& obj_filename, 
                                             std::string const& wrapper_cmd,
                                             tiramisu::function *fct)
    : evaluation_function(), fct(fct), obj_filename(obj_filename), wrapper_cmd(wrapper_cmd)
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
//    parallelize_outermost_levels(ast.computations_list);
    
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
    double exec_time = std::numeric_limits<double>::infinity();
    FILE *pipe = popen(wrapper_cmd.c_str(), "r");
    
    fscanf(pipe, "%lf", &exec_time);
    pclose(pipe);
    
    // Remove all the optimizations
    fct->reset_schedules();
    
    return exec_time;
}

std::vector<float> evaluate_by_execution::get_measurements(syntax_tree& ast, bool exit_on_timeout, float timeout)
{
    // Apply all the optimizations
    apply_optimizations(ast);

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

    // define the execution command of the wrapper
    std::string cmd = wrapper_cmd;
    if (timeout!=0) {// check if a timeout is defined for the execution time
        int nb_exec = 30; //by default
        if (std::getenv("NB_EXEC")!=NULL)
            nb_exec = std::stoi(std::getenv("NB_EXEC"));
        float cumulative_timeout = timeout * nb_exec; // the timeout for the total number of executions
        cmd = std::string("timeout ") + std::to_string(cumulative_timeout) + std::string(" ") + wrapper_cmd;
    }


    // execute the command
    FILE *pipe = popen(cmd.c_str(), "r");

    // read the output into a string
    char buf[100];
    std::string output;
    while (fgets(buf, 100, pipe))
        output += buf;

    // close the pipe and check if the timeout has been reached
    auto returnCode = pclose(pipe)/256;
    if (exit_on_timeout && (timeout!=0) && (returnCode == 124)){ // a potential issue here is that the 124 exit code is returned by another error
        std::cerr << "error: Execution time exceeded the defined timeout "<< timeout << "s *"<< std::getenv("NB_EXEC") << "execution" << std::endl;
        exit(1);
    }

    // parse the output into a vector of floats
    std::vector<float> measurements;
    std::istringstream iss(output);
    std::copy(std::istream_iterator<float>(iss), std::istream_iterator<float>(), std::back_inserter(measurements));

    if (measurements.empty()) // if there is no output this means that the execution failed
        measurements.push_back(std::numeric_limits<float>::infinity());

    // Remove all the optimizations
    fct->reset_schedules();

    return measurements;
}

evaluate_by_learning_model::evaluate_by_learning_model(std::string const& cmd_path, std::vector<std::string> const& cmd_args)
    : evaluation_function()
{
    // Create the pipe
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
        
        // Here we are in a new process.
        // Launch the program that evaluates schedules with the command cmd_path,
        // and arguments cmd_args.
        char* argv[cmd_args.size() + 2];
        
        argv[0] = (char*)malloc(sizeof(char) * (cmd_path.size() + 1));
        strcpy(argv[0], cmd_path.c_str());
        argv[cmd_args.size() + 1] = NULL;
        
        for (int i = 0; i < cmd_args.size(); ++i) {
            argv[i + 1] = (char*)malloc(sizeof(char) * (cmd_args[i].size() + 1));
            strcpy(argv[i + 1], cmd_args[i].c_str());
        }
        
        execv(cmd_path.c_str(), argv);
        exit(1);
    }
    
    close(outpipe_fd[0]);
    close(inpipe_fd[1]);
    
    model_write = fdopen(outpipe_fd[1], "w");
    model_read = fdopen(inpipe_fd[0], "r");
}

float evaluate_by_learning_model::evaluate(syntax_tree& ast)
{
    // Get JSON representations for the program, and for the schedule
    std::string prog_json = get_program_json(ast);
    std::string sched_json = get_schedule_json(ast);
    
    // Write the program JSON and the schedule JSON to model_write
    fputs(prog_json.c_str(), model_write);
    fputs(sched_json.c_str(), model_write);
    fflush(model_write);
    
    // Read the evaluation from model_read.
    float speedup = 0.f;
    fscanf(model_read, "%f", &speedup);
    
    return -speedup;
}

std::string evaluate_by_learning_model::get_program_json(syntax_tree const& ast)
{
    // Get the memory size allocated by the program, if declared
    std::string mem_size = "\"undeclared\"";
    if (std::getenv("MEM_SIZE")!=NULL)
        mem_size = std::string(std::getenv("MEM_SIZE"));
    std::string mem_size_json = "\"memory_size\" : " + mem_size + " ";

    // Get JSON for iterators from ast.iterators_json
    std::string iterators_json = "\"iterators\" : {" + ast.iterators_json + "}";
    
    // Use represent_computations_from_nodes to get JSON for computations
    std::string computations_json = "\"computations\" : {";
    int comp_absolute_order = 1;
    
    for (ast_node *node : ast.roots)
        represent_computations_from_nodes(node, computations_json, comp_absolute_order);
        
    computations_json.pop_back();
    computations_json += "}";
    
    // Return JSON of the program
    return "{" + mem_size_json + "," + iterators_json + "," + computations_json + "}\n";
}

void evaluate_by_learning_model::represent_computations_from_nodes(ast_node *node, std::string& computations_json, int& comp_absolute_order)
{
    // Build the JSON for the computations stored in "node".
    for (computation_info const& comp_info : node->computations)
    {
        std::string comp_json = "\"absolute_order\" : " + std::to_string(comp_absolute_order) + ","; 
        comp_absolute_order++;
        
        comp_json += "\"iterators\" : [";
        
        for (int i = 0; i < comp_info.iters.size(); ++i)
        {
            comp_json += "\"" + comp_info.iters[i].name + "\"";
            if (i != comp_info.iters.size() - 1)
                comp_json += ",";
        }
        
        comp_json += "],";
        
        comp_json += "\"real_dimensions\" : [";
        
        for (int i = 0; i < comp_info.buffer_nb_dims; ++i)
        {
            comp_json += "\"" + comp_info.iters[i].name + "\"";
            if (i != comp_info.buffer_nb_dims - 1)
                comp_json += ",";
        }
        
        comp_json += "],";
        
        comp_json += "\"comp_is_reduction\" : ";
        if (comp_info.is_reduction)
            comp_json += "true,";
        else
            comp_json += "false,";
            
        comp_json += "\"number_of_additions\" : " + std::to_string(comp_info.nb_additions) + ",";
        comp_json += "\"number_of_subtraction\" : " + std::to_string(comp_info.nb_substractions) + ",";
        comp_json += "\"number_of_multiplication\" : " + std::to_string(comp_info.nb_multiplications) + ",";
        comp_json += "\"number_of_division\" : " + std::to_string(comp_info.nb_divisions) + ",";

        comp_json += "\"write_access_relation\" : \"" +  comp_info.write_access_relation + "\",";
        comp_json += "\"write_buffer_id\" : " +  std::to_string(comp_info.storage_buffer_id) + ",";
        comp_json += "\"data_type\" : \"" +  comp_info.data_type_str + "\",";
        comp_json += "\"data_type_size\" : " +  std::to_string(comp_info.data_type_size) + ",";
        
        // Build JSON for the accesses of this computation
        comp_json += "\"accesses\" : [";

        for (int i = 0; i < comp_info.accesses.accesses_list.size(); ++i)
        {
            dnn_access_matrix const& matrix  = comp_info.accesses.accesses_list[i];
            
            comp_json += "{";
            
            comp_json += "\"access_is_reduction\" : ";
            if (i == 0 && comp_info.is_reduction)
                comp_json += "true,";
            else
                comp_json += "false,";
                
            comp_json += "\"buffer_id\" : " + std::to_string(matrix.buffer_id) + ",";
            comp_json += "\"access_matrix\" : [";
            
            for (int x = 0; x < matrix.matrix.size(); ++x)
            {
                comp_json += "[";
                for (int y = 0; y < matrix.matrix[x].size(); ++y)
                {
                    comp_json += std::to_string(matrix.matrix[x][y]);
                    if (y != matrix.matrix[x].size() - 1)
                        comp_json += ", ";
                }
                
                comp_json += "]";
                if (x != matrix.matrix.size() - 1)
                    comp_json += ",";
            }
            
            comp_json += "]";
            
            comp_json += "}";
            
            if (i != comp_info.accesses.accesses_list.size() - 1)
                comp_json += ",";
        }
        
        comp_json += "]";
    
        computations_json += "\"" + comp_info.comp_ptr->get_name() + "\" : {" + comp_json + "},";
    }
    
    // Recursively get JSON for the rest of computations
    for (ast_node *child : node->children)
        represent_computations_from_nodes(child, computations_json, comp_absolute_order);
}

std::string evaluate_by_learning_model::get_schedule_json(syntax_tree const& ast)
{
    bool interchanged = false;
    bool tiled = false;
    bool unrolled = false;
    bool skewed = false;
    bool parallelized = false;
    
    int unfuse_l0 = -1;
    int int_l0, int_l1;
    int tile_nb_l, tile_l0, tile_l0_fact, tile_l1_fact, tile_l2_fact;
    int unrolling_fact;
    int skewing_fact_l0, skewing_fact_l1;
    int skewing_l0, skewing_l1;
    int skew_extent_l0, skew_extent_l1;
    int parallelized_level;
    
    // Get information about the schedule
    for (optimization_info const& optim_info : ast.new_optims)
    {
        switch (optim_info.type)
        {
            case optimization_type::UNFUSE:
                unfuse_l0 = optim_info.l0;
                break;
                
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

            case optimization_type::PARALLELIZE:
                parallelized = true;
                parallelized_level = optim_info.l0;
                break;

            case optimization_type::SKEWING:
                skewed = true;
                skewing_fact_l0 = optim_info.l0_fact;
                skewing_fact_l1 = optim_info.l1_fact;
                skewing_l0 = optim_info.l0;
                skewing_l1 = optim_info.l1;
                skew_extent_l0 = optim_info.node->up_bound -optim_info.node->low_bound;
                assert(optim_info.node->children.size()==1); // only shared nodes are currently skewable
                skew_extent_l1 = optim_info.node->children[0]->up_bound -optim_info.node->children[0]->low_bound;
                break;
                
            default:
                break;
        }
    }
    
    // Transform the schedule to JSON
    std::vector<dnn_iterator> iterators_list;
    std::string sched_json = "{";
    
    // Set the schedule for every computation
    // For the moment, all computations share the same schedule
    for (tiramisu::computation *comp : ast.computations_list)
    {
        std::string comp_sched_json;
        iterators_list = dnn_iterator::get_iterators_from_computation(*comp);
        
        // JSON for interchange
        comp_sched_json += "\"interchange_dims\" : [";
        
        if (interchanged)
        {
            comp_sched_json += "\"" + iterators_list[int_l0].name + "\", \"" + iterators_list[int_l1].name + "\"";
            
            dnn_iterator dnn_it = iterators_list[int_l0];
            iterators_list[int_l0] = iterators_list[int_l1];
            iterators_list[int_l1] = dnn_it;
        }
        
        comp_sched_json += "],";
        
        // JSON for tiling
        comp_sched_json += "\"tiling\" : {";
        
        if (tiled)
        {
            if (tile_nb_l == 2)
            {
                comp_sched_json += "\"tiling_depth\" : 2,";
                comp_sched_json += "\"tiling_dims\" : [";
                
                comp_sched_json += "\"" + iterators_list[tile_l0].name + "\", " + "\"" + iterators_list[tile_l0 + 1].name + "\"";
                
                comp_sched_json += "],";
                
                comp_sched_json += "\"tiling_factors\" : [";
                
                comp_sched_json += "\"" + std::to_string(tile_l0_fact) + "\", " + "\"" + std::to_string(tile_l1_fact) + "\"";
                
                comp_sched_json += "]";
            }
            
            else
            {
                comp_sched_json += "\"tiling_depth\" : 3,";
                comp_sched_json += "\"tiling_dims\" : [";
                
                comp_sched_json += "\"" + iterators_list[tile_l0].name + "\", " + "\"" + iterators_list[tile_l0 + 1].name + "\", " + "\"" + iterators_list[tile_l0 + 2].name + "\"";
                
                comp_sched_json += "],";
                
                comp_sched_json += "\"tiling_factors\" : [";
                
                comp_sched_json += "\"" + std::to_string(tile_l0_fact) + "\", " + "\"" + std::to_string(tile_l1_fact) + "\", " + "\"" + std::to_string(tile_l2_fact) + "\"";
                
                comp_sched_json += "]";
            }
        }
        
        comp_sched_json += "},";
        
        // JSON for unrolling
        comp_sched_json += "\"unrolling_factor\" : ";
        
        if (unrolled)
        {
            comp_sched_json += "\"" + std::to_string(unrolling_fact) + "\",";
        }
        else
        {
            comp_sched_json += "null,";
        }

        // Parallelization tag
        comp_sched_json += "\"parallelized_dim\" : ";
        if (parallelized)
        {
            comp_sched_json += "\"" + iterators_list[parallelized_level].name + "\",";
        }
        else
        {
            comp_sched_json += "null, ";

        }

        // Skewing info
        comp_sched_json += "\"skewing\" : ";
        if (skewed)
        {
            comp_sched_json += "{\"skewed_dims\" : [\""+ iterators_list[skewing_l0].name + "\", " + "\"" + iterators_list[skewing_l1].name + "\"],";
            comp_sched_json += "\"skewing_factors\" : ["+std::to_string(skewing_fact_l0)+","+std::to_string(skewing_fact_l1)+"],";
            comp_sched_json += "\"average_skewed_extents\" : ["+std::to_string(skew_extent_l0)+","+std::to_string(skew_extent_l1)+"], ";

            // Adding the access matrices transformed by skewing

            // get the comp_info corresponding to the current computation
            ast_node* comp_node = ast.computations_mapping.at(comp);
            std::vector<dnn_access_matrix> comp_accesses_list;
            for (auto comp_i: comp_node->computations)
            {
                if (comp_i.comp_ptr == comp)
                {
                    comp_accesses_list = comp_i.accesses.accesses_list;
                    break;
                }
            }

            // Build JSON of the transformed accesses
            comp_sched_json += "\"transformed_accesses\" : [";

            for (int i = 0; i < comp_accesses_list.size(); ++i)
            {
                dnn_access_matrix const& matrix  = comp_accesses_list[i];
                comp_sched_json += "{";

                comp_sched_json += "\"buffer_id\" : " + std::to_string(matrix.buffer_id) + ",";
                comp_sched_json += "\"access_matrix\" : [";

                for (int x = 0; x < matrix.matrix.size(); ++x)
                {
                    comp_sched_json += "[";
                    for (int y = 0; y < matrix.matrix[x].size(); ++y)
                    {
                        comp_sched_json += std::to_string(matrix.matrix[x][y]);
                        if (y != matrix.matrix[x].size() - 1)
                            comp_sched_json += ", ";
                    }

                    comp_sched_json += "]";
                    if (x != matrix.matrix.size() - 1)
                        comp_sched_json += ",";
                }

                comp_sched_json += "]";

                comp_sched_json += "}";

                if (i != comp_accesses_list.size() - 1)
                    comp_sched_json += ",";
            }

            comp_sched_json += "]}";

        }
        else
        {
            comp_sched_json += "null";
        }
        
        sched_json += "\"" + comp->get_name() + "\" : {" + comp_sched_json + "},";
    }
    
    // Write JSON information about unfused iterators (not specific to a computation)
    sched_json += "\"unfuse_iterators\" : ["; 
    if (unfuse_l0 != -1)
        sched_json += "\"" + iterators_list[unfuse_l0].name + "\"";
        
    sched_json += "],";
    
    // Write the structure of the tree
    sched_json += "\"tree_structure\": {";
    sched_json += ast.tree_structure_json;
    sched_json += "}";
    
    // End of JSON
    sched_json += "}\n";
    return sched_json;
}

// ------------------------------------------------------------------------------------------ //

void evaluate_by_learning_model::represent_iterators_from_nodes(ast_node *node, std::string& iterators_json)
{
    if (node->get_extent() <= 1)
        return;
        
    std::string iter_json;
    
    // Represent basic information about this iterator
    iter_json += "\"lower_bound\" : " + std::to_string(node->low_bound) + ",";
    iter_json += "\"upper_bound\" : " + std::to_string(node->up_bound + 1) + ",";
        
    iter_json += "\"parent_iterator\" : ";
    if (node->parent == nullptr)
        iter_json += "null,";
    else
        iter_json += "\"" + node->parent->name + "\",";
            
    iter_json += "\"child_iterators\" : [";
    bool printed_child = false;
    
    for (int i = 0; i < node->children.size(); ++i)
    {
        if (node->children[i]->get_extent() <= 1)
            continue;
            
        iter_json += "\"" + node->children[i]->name + "\",";
        printed_child = true;
    }
        
    if (printed_child)
        iter_json.pop_back();
    iter_json += "],";
        
    // Add the names of the computations computed at this loop level
    iter_json += "\"computations_list\" : [";
    bool has_computations = false;
    
    for (int i = 0; i < node->computations.size(); ++i)
    {
        iter_json += "\"" + node->computations[i].comp_ptr->get_name() + "\",";
        has_computations = true;
    }
    
    for (int i = 0; i < node->children.size(); ++i)
    {
        if (node->children[i]->get_extent() > 1)
            continue;
            
        ast_node *dummy_child = node->children[i];
        for (int j = 0; j < dummy_child->computations.size(); ++j)
        {
            iter_json += "\"" + dummy_child->computations[j].comp_ptr->get_name() + "\",";
            has_computations = true;
        }
    }
       
    if (has_computations)
        iter_json.pop_back();
        
    iter_json += "]";
        
    iterators_json += "\"" + node->name + "\" : {" + iter_json + "},";
    
    // Recursively represent other iterators
    for (ast_node *child : node->children)
        represent_iterators_from_nodes(child, iterators_json);
}

std::string evaluate_by_learning_model::get_tree_structure_json(syntax_tree const& ast)
{
    // For the moment, this only supports ASTs with one root node.
    ast_node *node = ast.roots[0];
    return get_tree_structure_json(node);
}

std::string evaluate_by_learning_model::get_tree_structure_json(ast_node *node)
{
    std::string json;
    
    json += "\"loop_name\" : \"" + node->name + "\",";
    
    // Add the name of the computations computed at this loop level
    json += "\"computations_list\" : [";
    
    std::vector<std::string> comps_list;
    for (computation_info const& comp_info : node->computations)
        comps_list.push_back(comp_info.comp_ptr->get_name());
        
    for (ast_node *child : node->children)
    {
        if (child->get_extent() > 1)
            continue;
            
        for (int j = 0; j < child->computations.size(); ++j)
            comps_list.push_back(child->computations[j].comp_ptr->get_name());
    }
    
    for (std::string comp_name : comps_list)
        json += "\"" + comp_name + "\",";
        
    if (!comps_list.empty())
        json.pop_back();
        
    json += "],";
    
    // Get tree structure for children of this node
    json += "\"child_list\" : [";
    
    bool has_children = false;
    for (ast_node *child : node->children)
    {
        if (child->get_extent() == 1)
            continue;
            
        json += "{" + get_tree_structure_json(child) + "},";
        has_children = true;
    }
        
    if (has_children)
        json.pop_back();
    
    json += "]";
    
    return json;
}

}
