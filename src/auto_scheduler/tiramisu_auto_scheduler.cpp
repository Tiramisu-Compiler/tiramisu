#include <tiramisu/auto_scheduler/core.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{

computation_graph::computation_graph(tiramisu::function *fct)
{
    const std::vector<computation*> computations = fct->get_computations();
    
    for (computation* comp : computations) 
    {
        cg_node* node = new cg_node();
        node->comp = comp;
        
        // Get computation iterators
        
        roots.push_back(node);
    }
    
    // Build the computation graph
}

auto_scheduler::auto_scheduler(search_method *searcher, evaluator *eval_func, 
                               tiramisu::function *fct)
                               
    : cg(fct), searcher(searcher), eval_func(eval_func), fct(fct)
{
    searcher->set_eval_func(eval_func);
}

}
