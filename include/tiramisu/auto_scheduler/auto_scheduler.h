#ifndef _H_TIRAMISU_AUTO_SCHEDULER_AUTO_SCHEDULER_
#define _H_TIRAMISU_AUTO_SCHEDULER_AUTO_SCHEDULER_

#include <tiramisu/core.h>
#include <tiramisu/expr.h>
#include "ast.h"
#include "utils.h"

namespace tiramisu::auto_scheduler
{

class evaluator;
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
    tiramisu::function *fct;
    syntax_tree ast;
        
    search_method *searcher;
    evaluator *eval_func;
    
    evaluate_by_execution *exec_evaluator = nullptr;
        
public:
    auto_scheduler(search_method *searcher, evaluator *eval_func,
                   tiramisu::function *fct = tiramisu::global::get_implicit_function());
                   
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
};

}

#endif
