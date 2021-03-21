#ifndef _H_TIRAMISU_AUTO_SCHEDULER_AUTO_SCHEDULER_
#define _H_TIRAMISU_AUTO_SCHEDULER_AUTO_SCHEDULER_

#include <tiramisu/core.h>
#include <tiramisu/expr.h>
#include "ast.h"
#include "utils.h"

namespace tiramisu::auto_scheduler
{

class evaluation_function;
class search_method;

/**
  * The core class for the autoscheduler.
  * The user must provide the program to optimize, the evaluation
  * function and the search method.
  */
class auto_scheduler
{
private:
        
protected:
    /**
     * The function (the program) to optimize.
     */
    tiramisu::function *fct;
    
    /**
     * The AST obtained by transforming fct to an AST.
     */
    syntax_tree ast;
        
    /**
     * The search method to use.
     */
    search_method *searcher;
    
    /**
     * The evaluation function used by the search method.
     */
    evaluation_function *eval_func;
    
    /**
     * An evaluation function that measures execution time.
     * Used to automatically measure the speedup of the final optimizations found.
     * Used also by apply_best_schedule() to apply the final optimizations.
     */
    evaluate_by_execution *exec_evaluator = nullptr;
    
    /**
     * Initial execution time of the program (the only optimization applied is parallelization).
     * It is measured using "exec_evaluator".
     */
    float initial_exec_time;
        
public:
    /**
     * Create an autoscheduler with the given search method and the given evaluation function
     * for the given program.
     */
    auto_scheduler(search_method *searcher, evaluation_function *eval_func,
                   tiramisu::function *fct = tiramisu::global::get_implicit_function());
              
    /**
     * If you want the autoscheduler to measure the speedup of the final optimizations,
     * or if you want the autoscheduler to apply the final optimizations,
     * provide it with an evaluation function that measures execution time.
     */     
    void set_exec_evaluator(evaluate_by_execution *exec_evaluator) { this->exec_evaluator = exec_evaluator; }
    
    /**
     * Use the search method to find a set of optimizations.
     */
    void find_schedule();
    
    /**
     * Use the Tiramisu API to apply the schedule found by
     * find_schedule().
     */
    void apply_best_schedule();

    /**
     * Explores the search space and saves the explored schedules on a json file along with the measured
     * execution time of each schedule
     */
    void sample_search_space(std::string filename = "./schedules_sample.json");
};

}

#endif
