#include <tiramisu/auto_scheduler/search_method.h>

namespace tiramisu::auto_scheduler
{

void beam_search::search(syntax_tree& ast)
{
    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
        ast.clear_new_optimizations();
       
    std::vector<syntax_tree*> children;
        
    // Look for an optimization that can be applied
    int nb_optims_tried = 0;
    int nb_explored_optims = ast.nb_explored_optims;
    
    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS)
    {
        optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[nb_explored_optims % NB_OPTIMIZATIONS];
        children = states_gen->generate_states(ast, optim_type);
        
        nb_explored_optims++;
        nb_optims_tried++;
    }
       
    // Stop if no more optimizations can be applied
    if (children.size() == 0)
        return ;
       
    // Evaluate children and sort them from smallest to highest evaluation
    for (syntax_tree *child : children)
    {
        child->print_ast();
        child->nb_explored_optims = nb_explored_optims;
        child->evaluation = eval_func->evaluate(*child);
        
        if (child->evaluation < best_evaluation)
        {
            best_evaluation = child->evaluation;
            best_schedule = child->get_schedule();
        }
        
        nb_explored_schedules++;
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
        if (transform_ast)
            child->transform_ast();
        
        search(*child);
    }
}

}
