#include <tiramisu/auto_scheduler/core.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{

computation_graph::computation_graph(tiramisu::function *fct)
{
    const std::vector<computation*> comps = fct->get_computations();
    
    for (computation* c : comps) 
    {
        c->dump_iteration_domain();
    }
}

auto_scheduler::auto_scheduler(search_method *searcher, evaluator *eval_func, 
                               tiramisu::function *fct)
                               
    : cg(fct), searcher(searcher), eval_func(eval_func), fct(fct)
{
    searcher->set_eval_func(eval_func);
}

}
