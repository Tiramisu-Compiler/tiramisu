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
    
    for (computation_graph *cg : foo) {
        cg->print_graph();
        std::cout << std::endl;
    }

    for (computation_graph* cg : foo)
        cg->transform_computation_graph();

    for (computation_graph *cg : foo) {
        cg->print_graph();
        std::cout << std::endl;
    }
    
    for (computation_graph *cg : foo) {
        cg->next_optim_index = 0;
        std::vector<computation_graph*> foo2 = g->generate_states(*cg);
        std::cout << cg->roots.size() << std::endl;
        
        for (computation_graph *cg2 : foo2) {
            cg2->print_graph();
            std::cout << std::endl;
        }

        for (computation_graph* cg2 : foo2)
            cg2->transform_computation_graph();

        for (computation_graph *cg2 : foo2) {
            cg2->print_graph();
            std::cout << std::endl;
        }
        
        for (computation_graph *cg3 : foo2) {
            cg3->next_optim_index = 1;
            std::vector<computation_graph*> foo3 = g->generate_states(*cg3);
            
            for (computation_graph *cg4 : foo3) {
                cg4->print_graph();
                std::cout << std::endl;
            }

            for (computation_graph* cg4 : foo3)
                cg4->transform_computation_graph();

            for (computation_graph *cg4 : foo3) {
                cg4->print_graph();
                std::cout << std::endl;
            }
        }
    }
}

}
