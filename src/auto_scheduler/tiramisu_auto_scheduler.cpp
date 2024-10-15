#include <tiramisu/auto_scheduler/auto_scheduler.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

#include <chrono>

namespace tiramisu::auto_scheduler
{

auto_scheduler::auto_scheduler(search_method *searcher, evaluation_function *eval_func, std::vector<optimization_info> transformations, 
                               tiramisu::function *fct )
                               
    : fct(fct), ast(fct, transformations), searcher(searcher), eval_func(eval_func)
{
    searcher->set_eval_func(eval_func);
}

void auto_scheduler::sample_search_space(std::string filename, bool timeout_schedules)
{
    std::chrono::steady_clock::time_point sampling_start = std::chrono::steady_clock::now();
    fct->reset_all_static_dims_to_zero();
    fct->set_original_number_of_computations();
    float initial_timeout = std::atof(read_env_var("INITIAL_TIMEOUT"));
    std::vector<float> initial_measurements;
    initial_exec_time = 0;

    if (std::atoi(read_env_var("EXPLORE_BY_EXECUTION"))==1 || std::atoi(read_env_var("EXECUTE_INITIAL_SCHED"))==1) // If the exploratin is done by excecution or we need to execute the initial schedule
    {
        initial_measurements =  exec_evaluator->get_measurements(ast, true, initial_timeout);
        initial_exec_time = min_eval(initial_measurements);
        if (std::isinf(initial_exec_time)){
            std::cerr << "error: Evaluation of the non scheduled version of the program failed "<< std::endl;
            exit(1);
        }
        if (std::atoi(read_env_var("AS_VERBOSE"))==1)
            std::cout << "Initial exec time : " << initial_exec_time << std::endl;

        // If we're exploring using the model, measuring the initial exec time shouldn't be counted in the searh time. So the we reset the timer.
        if (std::atoi(read_env_var("EXPLORE_BY_EXECUTION"))==0)
            sampling_start = std::chrono::steady_clock::now();
    }

    // The exploraton assumes that the schedules are reset and especially that the scheduling graph has been cleared before the candidate generation
    fct->reset_schedules();

    ast.program_json = evaluate_by_learning_model::get_program_json(ast);
    std::vector<std::string> schedules_annotations;

    // prepare the schedule_json of the initial program
    std::string empty_schedule_json = evaluate_by_learning_model::get_schedule_json(ast);

    if (std::atoi(read_env_var("EXPLORE_BY_EXECUTION"))==1) // If we are evaluating by execution, set up the evaluation of the initial ast as the initial execution time
        ast.evaluation = initial_exec_time;
    else // If we are evaluating using the model, let the model evaluate the initial AST
        ast.evaluation = eval_func->evaluate(ast, empty_schedule_json);

    // Set the initial ast as the best we have so far
    searcher->set_best_evaluation(ast.evaluation);
    searcher->set_best_ast(&ast);
    searcher->set_exec_eval(exec_evaluator);

    // Add the no_schedule version to the schedule list
    empty_schedule_json.erase(empty_schedule_json.length()-2); // remove the last two characters "}\n"
    if (!initial_measurements.empty()) // if initial measurements have been collected
        empty_schedule_json += ", \"execution_times\" : " + measurements_to_str(initial_measurements) + "}\n";
    else // else just report the evaluation
        empty_schedule_json += ", \"execution_times\" : " + measurements_to_str({ast.evaluation}) + "}\n";
    schedules_annotations.push_back(empty_schedule_json);

    // initialize the exploration trace root
    candidate_trace exploration_trace_root = candidate_trace(&ast, 0);

    float schedule_timeout = 0;
    float schedule_timeout_factor = 50;
    if (std::getenv("SCHED_TIMEOUT_FACTOR")!=nullptr)
        schedule_timeout_factor = std::stof(std::getenv("SCHED_TIMEOUT_FACTOR"));
    if (timeout_schedules) {
        //define a timeout for scheduler evaluation, the max between schedule_timeout_factor times the initial exec_time (converted to seconds) and 3s per run
        schedule_timeout = std::max(initial_exec_time * schedule_timeout_factor / 1000, (float) 3.0);
        std::cout << "Schedule measurements timeout set to " << schedule_timeout << "*" << read_env_var("MAX_RUNS") << "(MAX_RUNS) s" << std::endl;
    }

    // start exploration with fusion and explore other transformations recursivly
    searcher->explore_schedules(ast, &schedules_annotations, &exploration_trace_root, schedule_timeout);

    std::chrono::steady_clock::time_point sampling_end = std::chrono::steady_clock::now();
    auto search_time  = std::chrono::duration_cast<std::chrono::milliseconds>(sampling_end - sampling_start).count();

    std::string output_json;
    output_json = "{\n\t\"filename\" : \"" + filename + "\"," +
                  "\n\t\"node_name\" : \"" + read_env_var("SLURMD_NODENAME") + "\"," +
                  "\n\t\"parameters\" : {" +
                  "\n\t\t\"beam_size\" : " + read_env_var("BEAM_SIZE") + "," +
                  "\n\t\t\"eval_mode\" : " + (std::atoi(read_env_var("EXPLORE_BY_EXECUTION"))==1 ? "\"Execution\"" : "\"Model\"") +
                  "\n\t}, " +
                  "\n\t\"program_annotation\" : " + ast.program_json + ", " +
                  "\n\t\"initial_execution_time\" : " + std::to_string(initial_exec_time) + ", " +
                  "\n\t\"schedules_list\" : [\n" ;

    for (std::string schedules_annot : schedules_annotations)
        output_json += schedules_annot + ",\n";
    if (!schedules_annotations.empty()){
        // remove the last comma
        output_json.pop_back();
        output_json.pop_back();
        output_json += "\n";
    }
    output_json += "\t], \n";

    output_json += "\"exploration_trace\": " + exploration_trace_root.get_exploration_trace_json() + ",\n";
    output_json += "\"search_time\": " + std::to_string(search_time);

    float best_execution_time = searcher->get_best_evaluation();
    std::cout << "Search time : " << search_time << " ms" << std::endl;
    std::cout << "Best evaluation : " << best_execution_time << std::endl;

    if(std::atoi(read_env_var("SAVE_BEST_SCHED_IN_FILE"))==1){
        syntax_tree* best_ast = searcher->get_best_ast();
        std::ofstream myfile;

        myfile.open(read_env_var("LOG_FILE_PATH"), std::ios_base::app);
        myfile<<"\""<<filename.substr(2,filename.size()-26)<<"\",";
        myfile << "\""<< initial_exec_time<<"\",";
        
        if (std::atoi(read_env_var("EXPLORE_BY_EXECUTION"))==0 && std::atoi(read_env_var("EXECUTE_BEST_SCHED"))==1)
        {
            best_execution_time = min_eval(exec_evaluator->get_measurements(*best_ast, false, 0));
            std::cout << "Best execution time : " << best_execution_time << std::endl;
        }

        myfile << "\""<<best_execution_time<<"\",";
        myfile << "\"" << best_ast->get_schedule_str() <<"\""<< std::endl;
        myfile.close();

        output_json += ",\n";
        output_json += "\"best_schedule\":{";
        output_json += "\"actual_exec_time\": " + std::to_string(best_execution_time)+ ", ";
        output_json += "\"sched_str\": \"" + best_ast->get_schedule_str() + "\"}";
    }

    output_json += " \n}\n";

    std::ofstream file(filename);
    file << output_json;
    file.close();

}

void auto_scheduler::find_schedule()
{
    //fct->reset_schedules();
    fct->reset_all_static_dims_to_zero();
    if (exec_evaluator != nullptr)
        initial_exec_time = exec_evaluator->evaluate(ast);
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    // Get the initial evaluation, and start the search.
    ast.evaluation = eval_func->evaluate(ast);
    searcher->search(ast);
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    
    // Print some info about the search
    std::cout << "NB explored schedules : " << searcher->get_nb_explored_schedules() << std::endl;
    std::cout << "Best evaluation : " << searcher->get_best_evaluation() << std::endl;
    
    if (exec_evaluator != nullptr)
        std::cout << "Initial exec time : " << initial_exec_time << std::endl;
        
    std::cout << "Initial evaluation : " << ast.evaluation << std::endl;
    std::cout << "Search time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms " << std::endl;  
}

void auto_scheduler::apply_best_schedule()
{
    syntax_tree *best_ast = searcher->get_best_ast();
    best_ast->print_ast();
    
    // To apply the best schedule, we need to use exec_evaluator.
    // Note : this should be improved, meaning no need to use exec_evaluator
    // to apply the best schedule.
    if (exec_evaluator != nullptr)
    {
        float best_sched_exec_time = exec_evaluator->evaluate(*best_ast);
        std::cout << "Best schedule exec time : " << best_sched_exec_time << std::endl;
        std::cout << "Speedup : " << initial_exec_time / best_sched_exec_time << std::endl;
    }
}

}
