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
        
public:
    auto_scheduler(search_method *searcher, evaluator *eval_func,
                   tiramisu::function *fct = tiramisu::global::get_implicit_function());
};

}

#endif
