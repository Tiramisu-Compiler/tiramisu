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
    fct->reset_schedules();
    initial_exec_time = exec_evaluator->evaluate_timeout(ast);
    ast.evaluation = initial_exec_time;
    std::string program_json = evaluate_by_learning_model::get_program_json(ast);
    std::vector<std::string> schedules_annotations;

    searcher->search_save(ast, &schedules_annotations);

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
