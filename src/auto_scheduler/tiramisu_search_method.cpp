#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{

void beam_search::search(syntax_tree const& ast)
{
    std::cout << ast.search_depth << std::endl;
    ast.print_ast();
    std::cout << "------------------------------------------------" << std::endl;
    
    std::vector<syntax_tree*> children;
    optimization_type optim_type;
    int optim_index = ast.search_depth % NB_OPTIMIZATIONS;
    
    // Look for an optimization that can be applied
    do
    {
        optim_type = DEFAULT_OPTIMIZATIONS_ORDER[optim_index];
        children = states_gen->generate_states(ast, optim_type);
        
        optim_index = (optim_index + 1) % NB_OPTIMIZATIONS;
        
    } while (children.size() == 0 && optim_index != ast.search_depth % NB_OPTIMIZATIONS);
       
    // Stop if no more optimizations can be applied
    if (children.size() == 0)
        return ;
       
    // Evaluate children and sort them from smallest to highest evaluation
    for (syntax_tree *child : children)
    {
        child->evaluation = eval_func->evaluate(*child);
        if (child->evaluation < best_evaluation)
        {
            best_evaluation = child->evaluation;
            best_schedule = child->optims_info;
        }
    }

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
