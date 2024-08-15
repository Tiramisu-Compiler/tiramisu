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
    : evaluation_function(), obj_filename(obj_filename), wrapper_cmd(wrapper_cmd), fct(fct)
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
            buf->get_n_dims(),
            Halide::ArgumentEstimates{});

        halide_arguments.push_back(buffer_arg);
    }
}
//TODO remove this function and change the whole structure of the evaluator classes
float evaluate_by_execution::evaluate(syntax_tree& ast, std::string no_sched_json)
{
    return -1;
}
float evaluate_by_execution::evaluate(syntax_tree& ast)
{
    fct->reset_schedules();
    // Apply all the optimizations
    apply_optimizations(ast);
    
    // Compile the program to an object file
    fct->lift_dist_comps();
    fct->gen_time_space_domain();
    fct->gen_isl_ast();
    fct->gen_halide_stmt();
    
    Halide::Module m = lower_halide_pipeline(fct->get_name(), halide_target, halide_arguments,
                                             Halide::LinkageType::External,
                                             fct->get_halide_stmt());
                                             
    // m.compile(Halide::Outputs().object(obj_filename));
    std::map<Halide::OutputFileType, std::string> omap = {
        {Halide::OutputFileType::object, obj_filename}
    };

    m.compile(omap);

    std::string gpp_command = read_env_var("GXX");

    if (gpp_command.empty())
    {
       gpp_command = "g++";
    }

    // Turn the object file to a shared library
    std::string gcc_cmd = gpp_command + " -shared -o " + obj_filename + ".so " + obj_filename;
    // run the command and retrieve the execution status
    int status = system(gcc_cmd.c_str());
    assert(status != 139 && "Segmentation Fault when trying to execute schedule");
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
                                             Halide::LinkageType::External,
                                             fct->get_halide_stmt());

    // m.compile(Halide::Outputs().object(obj_filename));
    std::map<Halide::OutputFileType, std::string> omap = {
        {Halide::OutputFileType::object, obj_filename}
    };

    m.compile(omap);

    std::string gpp_command = read_env_var("GXX");

    if (gpp_command.empty())
    {
       gpp_command = "g++";
    }

    // Turn the object file to a shared library
    std::string gcc_cmd = gpp_command + " -shared -o " + obj_filename + ".so " + obj_filename;
    int status = system(gcc_cmd.c_str());
    assert(status != 139 && "Segmentation Fault when trying to execute schedule");
    // define the execution command of the wrapper
    std::string cmd = wrapper_cmd;

    float cumulative_timeout;
    if (timeout!=0) {// check if a timeout is defined for the execution time
        int nb_exec = 30; //by default
        if (std::getenv("MAX_RUNS")!=NULL)
            nb_exec = std::stoi(std::getenv("MAX_RUNS"));
        cumulative_timeout = timeout * nb_exec; // the timeout for the total number of executions
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
        std::cerr << "error: Execution time exceeded the defined timeout "<< timeout << "s *"<< std::getenv("MAX_RUNS") << "execution" << std::endl;
        exit(1);
    }

    // parse the output into a vector of floats
    std::vector<float> measurements;
    std::istringstream iss(output);
    std::copy(std::istream_iterator<float>(iss), std::istream_iterator<float>(), std::back_inserter(measurements));

    if (measurements.empty() && (returnCode != 124)) // if there is no output and the cmd didn't timeout, this means that the execution failed
        measurements.push_back(std::numeric_limits<float>::infinity());

    else if (measurements.empty() && (returnCode == 124) && (timeout!=0)){  //if there is no output and the cmd timed out, this means that no execution finished before timeout
        measurements.push_back(cumulative_timeout*1000); // converted to ms
        std::cout<< "Execution timed out"<< std::endl;
    }
    
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

//TODO remove this function and change the whole structure of the evaluator classes
float evaluate_by_learning_model::evaluate(syntax_tree& ast)
{
    return -1;
}
float evaluate_by_learning_model::evaluate(syntax_tree& ast, std::string no_sched_json)
{
    // Get JSON representations for the program, and for the schedule
    std::string prog_json = get_program_json(ast);
    std::string sched_json = get_schedule_json(ast);

    // Write the program JSON and the schedule JSON to model_write
    fputs(prog_json.c_str(), model_write);
    fputs(no_sched_json.c_str(), model_write);
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
    std::string mem_size_json = "\"memory_size\" : \"" + std::string(read_env_var("MEM_SIZE")) + "\" ";

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
        
        
        comp_json += "\"comp_is_reduction\" : ";
        if (comp_info.is_reduction)
            comp_json += "true,";
        else
            comp_json += "false,";

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
            if (comp_info.storage_buffer_id==matrix.buffer_id)
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
        
        comp_json += "],";
    
        tiramisu::expr comp_info_expr = comp_info.comp_ptr->get_expr();
        comp_json += "\"expression_representation\" : " +  comp_info_expr.to_json();

        computations_json += "\"" + comp_info.comp_ptr->get_name() + "\" : {" + comp_json + "},";
    }
    
    // Recursively get JSON for the rest of computations
    for (ast_node *child : node->children)
        represent_computations_from_nodes(child, computations_json, comp_absolute_order);
}
/*
multiply two matrices AxB
*/
std::vector<std::vector<int>>  mat_mul(const std::vector<std::vector<int>> & m1, const std::vector<std::vector<int>> & m2)
{
std::vector<std::vector<int>> result(m1.size(), std::vector<int>(m2.at(0).size()));

    for(std::size_t row = 0; row < result.size(); ++row) {
        for(std::size_t col = 0; col < result.at(0).size(); ++col) {
            for(std::size_t inner = 0; inner < m2.size(); ++inner) {
                result.at(row).at(col) += m1.at(row).at(inner) * m2.at(inner).at(col);
            }
        }
    }
    return result;
}
std::vector<int> get_transformation_vector_from_optimization(optimization_info opt){
        //TODOF generalize to MAX_TAGS
        std::vector<int> result(16);
        assert(opt.unimodular_transformation_type != 0);

        result.at(0) = opt.unimodular_transformation_type;
        switch(opt.unimodular_transformation_type){
            // Interchange
            case 1:
                result.at(1) = opt.l0;
                result.at(2) = opt.l1;
                break;

            // Rversal
            case 2:
                result.at(3) = opt.l0;
                break;

            // Skewing
            case 3:
                result.at(4) = opt.l0;
                result.at(5) = opt.l1;
                result.at(6) = opt.l2;
                result.at(7) = opt.l0_fact;
                result.at(8) = opt.l1_fact;
                result.at(9) = opt.l2_fact;
                result.at(10) = opt.l3_fact;
                result.at(11) = opt.l4_fact;
                result.at(12) = opt.l5_fact;
                result.at(13) = opt.l6_fact;
                result.at(14) = opt.l7_fact;
                result.at(15) = opt.l8_fact;
                break;

        }
    return result;
}
std::string evaluate_by_learning_model::get_schedule_json(syntax_tree & ast)
{
    std::string sched_json = "{";
    std::vector<computation_info*> all_comps_info;

    // retrieve all computation infos to get the iterators for each computation later
    for (auto root : ast.roots){
        root->collect_all_computation(all_comps_info);
    }
    for (tiramisu::computation *comp : ast.computations_list)
    {
        bool tiled = false;
        bool unrolled = false;
        bool parallelized = false;
        bool shifted = false;
        bool transformed_by_matrix = false;

        int tile_nb_l, tile_l0, tile_l0_fact, tile_l1_fact, tile_l2_fact;
        int unrolling_fact;
        int parallelized_level;
        std::vector < std::vector<int> > matrix;
        std::vector <optimization_info > transformations;
        std::vector<std::pair<int,int>> shiftings; //pairs of loop_level,shift_factor

        
        // Get information about the schedule
        for (optimization_info const& optim_info : ast.get_schedule())
        {
            if(std::find(optim_info.comps.begin(), optim_info.comps.end(), comp) == optim_info.comps.end()) {
                // if the current computation isn't affected by the current optim_info
                continue;
            }
            switch (optim_info.type)
            {
                case optimization_type::SHIFTING:
                    shifted = true;
                    shiftings.emplace_back(optim_info.l0,optim_info.l0_fact);
                    break;
                case optimization_type::TILING:
                    tiled = true;
                    if (optim_info.nb_l == 1)
                    {
                        tile_nb_l = 1;
                        tile_l0 = optim_info.l0;
                        tile_l0_fact = optim_info.l0_fact;
                    }
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

                case optimization_type::MATRIX:
                    transformed_by_matrix = true;
                    if (optim_info.unimodular_transformation_type != 0){
                        transformations.push_back(optim_info);
                    }
                    
                    break;
                    
                case optimization_type::UNROLLING:
                    unrolled = true;
                    unrolling_fact = optim_info.l0_fact;
                    break;

                case optimization_type::PARALLELIZE:
                    parallelized = true;
                    parallelized_level = optim_info.l0;
                    break;
                    
                default:
                    break;
            }
        }
        
        // Transform the schedule to JSON
        std::vector<dnn_iterator> iterators_list;
        
        // Set the schedule for every computation
        std::string comp_sched_json;

        // look for the computation and assign the correct iterators list
        for (auto comp_info : all_comps_info){
                if(comp_info->comp_ptr == comp){
                    iterators_list = comp_info->iters;
                }
        }
        // Check if fusion was applied on this coomputation
        for (optimization_info const& optim_info : ast.get_schedule())
            if (optim_info.type==optimization_type::FUSION && optim_info.comps[1] == comp){
                // Retrieve the iterators list of the computation that it was fused with
                for (auto comp_info : all_comps_info){
                    if(comp_info->comp_ptr == optim_info.comps[0]){
                        // Assign the same iterators list to both computations
                        iterators_list = comp_info->iters;
                    }
                }
            }
        assert(!iterators_list.empty() && "couldn't find the list of iterators for this computation");

        comp_sched_json += "\"shiftings\" : ";
        if (shifted){
            comp_sched_json+= "[";
            for (auto shifting:shiftings)
                comp_sched_json+= "[\"" + iterators_list[std::get<0>(shifting)].name + "\","+std::to_string(std::get<1>(shifting))+"],";
            comp_sched_json.pop_back(); //remove last comma
            comp_sched_json += "], ";
        }
        else
            comp_sched_json += "null,";
    
        // JSON for tiling
        comp_sched_json += "\"tiling\" : {";
        
        if (tiled)
        {
            if (tile_nb_l == 1)
            {
                comp_sched_json += "\"tiling_depth\" : 1,";
                comp_sched_json += "\"tiling_dims\" : [";
                
                comp_sched_json += "\"" + iterators_list[tile_l0].name + "\"";
                
                comp_sched_json += "],";
                
                comp_sched_json += "\"tiling_factors\" : [";
                
                comp_sched_json += "\"" + std::to_string(tile_l0_fact) + "\"";
                
                comp_sched_json += "]";
            }
            else if (tile_nb_l == 2)
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

        comp_sched_json += "\"transformations_list\" : [";
        std::vector<int> transformation_vector;
        if (transformed_by_matrix)
        {
            for(int i = 0; i < transformations.size(); i++){
                comp_sched_json += "[";
                transformation_vector = get_transformation_vector_from_optimization(transformations.at(i));

                for(int j = 0; j < transformation_vector.size(); j++){
                    comp_sched_json += std::to_string(transformation_vector.at(j));
                    if(!(j==transformation_vector.size()-1)) comp_sched_json += ", ";
                }

                comp_sched_json += "] ";
                if(i!=transformations.size()-1) comp_sched_json += ", ";
            }
        }
        comp_sched_json += "]";

        sched_json += "\"" + comp->get_name() + "\" : {" + comp_sched_json + "}, ";
    }
    bool has_fusions = false;
    sched_json += "\"fusions\" : [";
    for (optimization_info const& optim_info : ast.get_schedule())
        if (optim_info.type==optimization_type::FUSION) {
            sched_json += " [\"" + optim_info.comps[0]->get_name() + "\",\"" + optim_info.comps[1]->get_name() + "\"," +
                          std::to_string(optim_info.l0) + "],"; //Fusion ordered with the .then semantic
            has_fusions=true;
        }
    sched_json.pop_back(); //drop the last comma or '['
    if (has_fusions)
        sched_json +="], ";
    else
        sched_json += "null, ";

    sched_json += "\"sched_str\": \"" + ast.get_schedule_str() + "\", ";

    // Write the structure of the tree
    sched_json += "\"tree_structure\": {";
    sched_json += ast.tree_structure_json;
    sched_json += "}, ";
    
    sched_json += "\"legality_check\": true, ";
    sched_json += "\"exploration_method\": 1";
    // End of JSON
    sched_json += "}\n";
    return sched_json;
}

// ------------------------------------------------------------------------------------------ //

void evaluate_by_learning_model::represent_iterators_from_nodes(ast_node *node, std::string& iterators_json)
{
    // We skip dummy nodes
    if (node->name.compare("dummy_iter")==0)
        return;
        
    std::string iter_json;
    
    // Represent basic information about this iterator
    iter_json += "\"lower_bound\" : \"" + node->low_bound + "\",";
    if(check_if_number(node->up_bound)){
        iter_json += "\"upper_bound\" : \"" + std::to_string(stoi(node->up_bound) + 1) + "\",";
    }else{
        iter_json += "\"upper_bound\" : \"" + node->up_bound + "+1" + "\",";
    }
        
    iter_json += "\"parent_iterator\" : ";
    if (node->parent == nullptr)
        iter_json += "null,";
    else
        iter_json += "\"" + node->parent->name + "\",";            
            
    iter_json += "\"child_iterators\" : [";
    bool printed_child = false;
    
    for (int i = 0; i < node->children.size(); ++i)
    {
        if (node->children[i]->name.compare("dummy_iter")==0)
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
        if (node->children[i]->name.compare("dummy_iter")!=0)
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
    std::string roots_jsons = "\"roots\" : [";

    for (ast_node *node : ast.roots)
    {
        roots_jsons += "{" + get_tree_structure_json(node) + "},";
    }
    roots_jsons.pop_back();
    roots_jsons += "]";

    return roots_jsons;
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
        if (child->name.compare("dummy_iter")!=0)
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
        if (child->name.compare("dummy_iter")==0)
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
