#include <tiramisu/auto_scheduler/core.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{

program_repr::program_repr(tiramisu::function *fct)
{
    const std::vector<computation*> comps = fct->get_computations();
    
    for (computation* c : comps) 
    {
        c->dump_schedule();
    }
}

auto_scheduler::auto_scheduler(search_method *searcher, evaluator *eval_func, 
                               tiramisu::function *fct)
                               
    : prog_repr(fct), searcher(searcher), eval_func(eval_func), fct(fct)
{
    searcher->set_eval_func(eval_func);
}

}
