#include <tiramisu/auto_scheduler/search_method.h>
#include <random>

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
    
    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
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
        child->nb_explored_optims = nb_explored_optims;
        if (eval_func->should_transform_ast(*child))
            child->transform_ast();
            
        child->evaluation = eval_func->evaluate(*child);
        
        if (child->evaluation < best_evaluation)
        {
            best_evaluation = child->evaluation;
            best_ast = child;
        }
        
        nb_explored_schedules++;
    }
    
    // Stop if we reached the maximum depth
    if (nb_explored_optims >= max_depth)
        return ;
        
    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
    ast_copy->nb_explored_optims = nb_explored_optims;
    children.push_back(ast_copy);

    // Sort children from smallest evaluation to largest
    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });

    // Search recursively on the best children
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];
        
    children.resize(std::min(beam_size, (int)children.size()));

    for (syntax_tree *child : children)
    {
        child->search_depth = ast.search_depth + 1;        
        search(*child);
    }
}

void beam_search_topk::search(syntax_tree& ast)
{
    beam_search_subroutine(ast);
    
    std::sort(schedules.begin(), schedules.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });
    
    for (int i = 0; i < topk; ++i)
    {
        float exec_time = exec_eval->evaluate(*schedules[i]);
        if (exec_time < best_evaluation)
        {
            best_evaluation = exec_time;
            best_ast = schedules[i];
        }
    }
}
    
void beam_search_topk::beam_search_subroutine(syntax_tree& ast)
{
    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
        ast.clear_new_optimizations();
       
    std::vector<syntax_tree*> children;
        
    // Look for an optimization that can be applied
    int nb_optims_tried = 0;
    int nb_explored_optims = ast.nb_explored_optims;
    
    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
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
        child->nb_explored_optims = nb_explored_optims;
        if (eval_func->should_transform_ast(*child))
            child->transform_ast();
            
        child->evaluation = eval_func->evaluate(*child);
        
        nb_explored_schedules++;
    }
        
    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
    ast_copy->nb_explored_optims = nb_explored_optims;
    children.push_back(ast_copy);

    // Sort children from smallest evaluation to largest
    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });
    
    for (int i = 0; i < beam_size; ++i)
        schedules.push_back(children[i]);
    
    // Stop if we reached the maximum depth
    if (nb_explored_optims >= max_depth)
        return ;

    // Search recursively on the best children
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];
        
    children.resize(std::min(beam_size, (int)children.size()));

    for (syntax_tree *child : children)
    {
        child->search_depth = ast.search_depth + 1;        
        search(*child);
    }
}

void beam_search_accuracy_evaluator::search(syntax_tree& ast)
{
    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
        ast.clear_new_optimizations();
       
    std::vector<syntax_tree*> children;
        
    // Look for an optimization that can be applied
    int nb_optims_tried = 0;
    int nb_explored_optims = ast.nb_explored_optims;
    
    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
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
        child->nb_explored_optims = nb_explored_optims;
        if (eval_func->should_transform_ast(*child))
            child->transform_ast();
            
        child->evaluation = eval_func->evaluate(*child);
        
        model_evals_list.push_back(child->evaluation);
        exec_evals_list.push_back(exec_eval->evaluate(*child));
        
        if (child->evaluation < best_evaluation)
        {
            best_evaluation = child->evaluation;
            best_ast = child;
        }
        
        nb_explored_schedules++;
    }
    
    // Stop if we reached the maximum depth
    if (nb_explored_optims >= max_depth)
        return ;
        
    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
    ast_copy->nb_explored_optims = nb_explored_optims;
    children.push_back(ast_copy);

    // Sort children from smallest evaluation to largest
    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });

    // Search recursively on the best children
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];
        
    children.resize(std::min(beam_size, (int)children.size()));

    for (syntax_tree *child : children)
    {
        child->search_depth = ast.search_depth + 1;        
        search(*child);
    }
}

void mcts::search(syntax_tree& ast)
{
    std::default_random_engine rand_generator;
    
    std::vector<syntax_tree*> samples;
    std::vector<syntax_tree*> children;
    std::vector<double> children_evals;
    
    for (int epoch = 0; epoch < nb_samples; ++epoch)
    {
        syntax_tree *ast_sample = &ast;
        for (int depth = 0; depth < max_depth; ++depth)
        {
            optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[depth % NB_OPTIMIZATIONS];
            children = states_gen->generate_states(*ast_sample, optim_type);
                        
            if (children.empty())
                continue;
                
            children_evals.clear();
            
            for (syntax_tree *child : children)
            {
                if (eval_func->should_transform_ast(*child))
                    child->transform_ast();
                    
                child->evaluation = eval_func->evaluate(*child);
                children_evals.push_back(child->evaluation);
                
                nb_explored_schedules++;
            }
            
            children.push_back(ast_sample->copy_ast());
            children_evals.push_back(ast_sample->evaluation);
            
            std::discrete_distribution<int> dist(children_evals.begin(), children_evals.end());
            ast_sample = children[dist(rand_generator)];
            
            samples.push_back(ast_sample);
            ast_sample->print_ast();
        }
    }
    
    if (samples.empty())
        return ;
    
    std::sort(samples.begin(), samples.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });
    
    for (int i = 0; i < topk; ++i)
    {
        float exec_time = exec_eval->evaluate(*samples[i]);
        if (exec_time < best_evaluation)
        {
            best_evaluation = exec_time;
            best_ast = samples[i];
        }
    }
}

}
