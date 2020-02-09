#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{

void beam_search::search(computation_graph& cg)
{
    // Generate children and evaluate them
    std::vector<computation_graph*> children = states_gen->generate_states(cg);
    for (computation_graph* child : children)
        child->evaluation = eval_func->evaluate(*child);

    // Sort children from smallest to highest evaluation
    std::sort(children.begin(), children.end(), [](computation_graph* a, computation_graph* b) {
        return a->evaluation < b->evaluation;
    });

    // Search recursively on the best children
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];
        
    children.resize(beam_size);
    
    for (computation_graph* child : children) 
    {
        search(*child);
    }
}

}
