#include <tiramisu/auto_scheduler/auto_scheduler.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{

auto_scheduler::auto_scheduler(search_method *searcher, evaluator *eval_func, 
                               tiramisu::function *fct)
                               
    : cg(fct), searcher(searcher), eval_func(eval_func), fct(fct)
{
    //searcher->set_eval_func(eval_func);
    //searcher->search(cg);

    states_generator* g = new exhaustive_generator(true, true, true, true);
    std::vector<computation_graph*> foo = g->generate_states(cg);
    
    std::cout << foo.size() << std::endl;

    for (computation_graph* c : foo)
    {
        for (cg_node *node : c->roots)
        {
            cg_node *bar = node;
            
            while (true)
            {
                std::cout << bar->fused << " " << bar->fused_with << std::endl;
                if (bar->children.size() == 0)
                    break;
                    
                bar = bar->children[0];
            }
            
            std::cout << std::endl;
        }
        
        std::cout << "END" << std::endl;
    }
}

}
