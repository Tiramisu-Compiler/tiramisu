#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{

void beam_search::search(syntax_tree const& ast)
{
    // Generate children and evaluate them
    std::vector<syntax_tree*> children;
    optimization_type optim_type;
    int optim_index = ast.search_depth % NB_OPTIMIZATIONS;
    
    do
    {
        optim_type = DEFAULT_OPTIMIZATIONS_ORDER[optim_index];
        children = states_gen->generate_states(ast, optim_type);
        
        optim_index = (optim_index + 1) % NB_OPTIMIZATIONS;
        
    } while (children.size() == 0 && optim_index != ast.search_depth % NB_OPTIMIZATIONS);
       
    // Stop if no more optimizations can be applied
    if (children.size() == 0)
        return ;
       
    for (syntax_tree *child : children)
        child->evaluation = eval_func->evaluate(*child);

    // Sort children from smallest to highest evaluation
    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });
    
    // Stop if we reached the maximum depth
    if (ast.search_depth >= max_depth)
        return ;

    // Search recursively on the best children
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];
        
    children.resize(std::min(beam_size, (int)children.size()));

    for (syntax_tree *child : children) 
    {
        child->search_depth = ast.search_depth + 1;
        child->transform_ast();
        
        search(*child);
    }
}

}
