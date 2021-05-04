#include <tiramisu/auto_scheduler/auto_scheduler.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

#include <chrono>

namespace tiramisu::auto_scheduler
{

auto_scheduler::auto_scheduler(search_method *searcher, evaluation_function *eval_func, 
                               tiramisu::function *fct)
                               
    : fct(fct), ast(fct), searcher(searcher), eval_func(eval_func)
{
    searcher->set_eval_func(eval_func);
}

void auto_scheduler::sample_search_space(std::string filename)
{
    std::chrono::steady_clock::time_point sampling_start = std::chrono::steady_clock::now();
    fct->reset_schedules();
    float initial_timeout = 0;
    if (std::getenv("INITIAL_TIMEOUT")!=NULL)
        initial_timeout = std::stof(std::getenv("INITIAL_TIMEOUT"));
    std::vector<float> initial_measurements = exec_evaluator->get_measurements(ast, true, initial_timeout);
    initial_exec_time = min_eval(initial_measurements);
    ast.evaluation = initial_exec_time;
    if (std::getenv("AS_VERBOSE")!=NULL)
        if (std::stoi(std::getenv("AS_VERBOSE"))==1)
            std::cout << "Initial exec time : " << initial_exec_time << std::endl;
    std::string program_json = evaluate_by_learning_model::get_program_json(ast);
    std::vector<std::string> schedules_annotations;

    // add the no_schedule version to the schedule list
    std::string empty_schedule_json = evaluate_by_learning_model::get_schedule_json(ast);
    empty_schedule_json.pop_back(); // remove the last two characters }\n
    empty_schedule_json.pop_back();
    empty_schedule_json += ", \n\"execution_times\" : " + measurements_to_str(initial_measurements) + "\n}\n";
    schedules_annotations.push_back(empty_schedule_json);

    //define a timeout for scheduler evaluation, the max between 100times the initial exec_time (converted to seconds) and 3s per run
    float schedule_timeout = std::max(initial_exec_time*100/1000, (float)3.0);

    searcher->set_exec_eval(exec_evaluator);
    searcher->search_save(ast, &schedules_annotations, schedule_timeout);

    std::string output_json;

    std::string nb_exec = "\"default\"";
    if (std::getenv("NB_EXEC")!=NULL)
        nb_exec = std::string(std::getenv("NB_EXEC"));

    output_json = "{\n\t\"filename\" : \"" + filename + "\"," +
                  "\n\t\"parameters\" : {" +
                  "\n\t\t\"beam_size\" : " + std::getenv("BEAM_SIZE") + ", " +
                  "\n\t\t\"max_depth\" : " + std::getenv("MAX_DEPTH") + ", " +
                  "\n\t\t\"nb_exec\" : " + nb_exec +
                  "\n\t}, " +
                  "\n\t\"program_annotation\" : " + program_json + ", " +
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
    output_json += "\t] \n}\n";

    std::ofstream file(filename);
    file << output_json;
    file.close();

    std::chrono::steady_clock::time_point sampling_end = std::chrono::steady_clock::now();
    if (std::getenv("AS_VERBOSE")!=NULL)
        if (std::stoi(std::getenv("AS_VERBOSE"))==1){
            std::cout << "Search time : " << std::chrono::duration_cast<std::chrono::milliseconds>(sampling_end - sampling_start).count() << " ms" << std::endl;
            std::cout << "Best execution time : " << searcher->get_best_evaluation() << std::endl;
        }
}

void auto_scheduler::find_schedule()
{
    fct->reset_schedules();
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
