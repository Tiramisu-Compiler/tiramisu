#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{

void beam_search::search(syntax_tree const& ast)
{
    // Generate children and evaluate them
    std::vector<syntax_tree*> children = states_gen->generate_states(ast);
    for (syntax_tree* child : children)
        child->evaluation = eval_func->evaluate(*child);

    // Sort children from smallest to highest evaluation
    std::sort(children.begin(), children.end(), [](syntax_tree* a, syntax_tree* b) {
        return a->evaluation < b->evaluation;
    });

    // Search recursively on the best children
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];
        
    children.resize(beam_size);
    
    for (syntax_tree* child : children) 
    {
        search(*child);
    }
}

}
