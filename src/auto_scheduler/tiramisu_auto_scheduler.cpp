#include <tiramisu/auto_scheduler/auto_scheduler.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{

auto_scheduler::auto_scheduler(search_method *searcher, evaluator *eval_func, 
                               tiramisu::function *fct)
                               
    : fct(fct), ast(fct), searcher(searcher), eval_func(eval_func)
{
    searcher->set_eval_func(eval_func);
    searcher->search(ast);
}

}
