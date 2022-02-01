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

    if(generator_state::initialized == false)
    {
        
        ast.initialize_search_space_optimizations(DEFAULT_OPTIMIZATIONS_ORDER);
        // the optimizations are specified along with the parameters in the generator_state attribute inside the AST.
        assert(generator_state::initialized == true);
    }

    std::cout<<"TESTED";
    
    while (children.size() == 0 && (!ast.is_search_space_empty()))
    {
        // schedule generation based on generator_state attribute in the AST.
        children = scheds_gen->generate_schedules(ast);

        std::cout<<"not empty";

        // move to next optimization
        //explores next optimization/alternative
        ast.move_to_next_optimization_target();

        for(auto& child:children)
        {
            child->move_to_next_optimization_target();
        }
        
        nb_explored_optims++;
        nb_optims_tried++;
    }
       
    // Stop if no more optimizations can be applied
    if (children.size() == 0)
        return ;
       
    // Evaluate children and sort them from smallest to highest evaluation
    

   // evaluate while removing illegal versions
    auto iterator = children.begin();
    while (iterator != children.end())
    {
        (*iterator)->nb_explored_optims = nb_explored_optims;
        (*iterator)->transform_ast();

        if ((*iterator)->ast_is_legal() == false) {

            // print deleted Ast 
            (*iterator)->print_previous_optims();
            std::cout << "\n-----------" << std::endl;
            (*iterator)->print_new_optims();
            (*iterator)->print_ast();
            (*iterator)->print_isl_states();
            std::cout << "\n<illegal>\n";
            delete (*iterator);
            iterator = children.erase(iterator);
        }
        else {

            // evaluate and print Ast
            (*iterator)->print_previous_optims();
            std::cout << "\n-----------" << std::endl;
            (*iterator)->print_new_optims();
            (*iterator)->print_ast();
//            (*iterator)->print_isl_states();
//            (*iterator)->print_computations_accesses();
            std::cout << "\n<legal>\n";

            (*iterator)->evaluation = eval_func->evaluate(*(*iterator));
            std::cout << "Evaluation : " << (*iterator)->evaluation << std::endl << std::endl;


            std::cout << "\n============================================================================================================" << std::endl;

            if ((*iterator)->evaluation < best_evaluation)
            {
                best_evaluation = (*iterator)->evaluation;
                best_ast = (*iterator);
            }

            ++iterator;

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
    
    std::cout<<"\noriginal list\n" ;
    for (syntax_tree *child : children)
    {
        std::cout<<child->evaluation<<"+";
    }

    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });

    // keep the top 'beam_size' children and delete the rest
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];
    
        
    children.resize(std::min(beam_size, (int)children.size()));

    std::cout<<"\nremaining list\n" ;
    for (syntax_tree *child : children)
    {
        std::cout<<child->evaluation<<"+";
    }

    // Search recursively on the best children
    for (syntax_tree *child : children)
    {
        child->search_depth = ast.search_depth + 1;        
        search(*child);
    }
}
void beam_search::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
    std::cerr<< "search_save::search_save temporarily removed" << std::endl;
    exit(1);
}
//void beam_search::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
//{
//    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
//        ast.clear_new_optimizations();
//
//    std::vector<syntax_tree*> children;
//
//    // Look for an optimization that can be applied
//    int nb_optims_tried = 0;
//    int nb_explored_optims = ast.nb_explored_optims;
//
//    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
//    {
//        optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[nb_explored_optims % NB_OPTIMIZATIONS];
//        children = scheds_gen->generate_schedules(ast, optim_type);
//
//        nb_explored_optims++;
//        nb_optims_tried++;
//    }
//
//    // Stop if no more optimizations can be applied
//    if (children.size() == 0)
//        return ;
//
//    // Evaluate children and sort them from smallest to highest evaluation
//    // evaluate while removing illegal versions
//    auto iterator = children.begin();
//    while (iterator != children.end())
//    {
//        syntax_tree *child = *iterator;
//        child->nb_explored_optims = nb_explored_optims;
//        child->transform_ast();
//
//        if (child->schedule_is_prunable()){
//            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
//                // print deleted Ast
//                child->print_previous_optims();
//                std::cout << "\n-----------" << std::endl;
//                child->print_new_optims();
//                child->print_ast();
//                std::cout << "\n<Schedule pruned>\n";
//            }
//            delete child;
//            iterator = children.erase(iterator);
//        }
//
//        else if (!child->ast_is_legal()) {
//            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
//                // print deleted Ast
//                child->print_previous_optims();
//                std::cout << "\n-----------" << std::endl;
//                child->print_new_optims();
//                child->print_ast();
//                child->print_isl_states();
//                std::cout << "\n<illegal>\n";
//            }
//            delete child;
//            iterator = children.erase(iterator);
//        }
//        else {
//
//            // print and evaluate Ast
//
//            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
//                child->print_previous_optims();
//                std::cout << "\n-----------" << std::endl;
//                child->print_new_optims();
//                child->print_ast();
//                child->print_isl_states();
//                std::cout << "\n<legal>\n";
//                child->print_computations_accesses();
//            }
//
//            std::vector<float> measurements;
//            if (child->can_set_default_evaluation()) { // if yes the child's evaluation is set to a default value
//                measurements = {child->evaluation};
//            }
//            else{
//                measurements = exec_eval->get_measurements(*child, false, schedule_timeout);
//                child->evaluation = min_eval(measurements);
//            }
//
//            parent_trace->add_child_path(child, schedules_annotations->size());
//
//            std::string schedule_annot = evaluate_by_learning_model::get_schedule_json(*child);
//
//            //remove the last two characters }\n
//            schedule_annot.pop_back();
//            schedule_annot.pop_back();
//
//            if (std::isfinite(child->evaluation)) // the evaluation is not finite mean that the schedule didn't run
//                schedule_annot += ", \n\"execution_times\" : " + measurements_to_str(measurements) + "\n}\n";
//            else
//                schedule_annot += ", \n\"execution_times\" : null\n}\n";
//
//            schedules_annotations->push_back(schedule_annot);
//
//            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
//                std::cout << "Schedule number "<< schedules_annotations->size() << std::endl;
//                std::cout << "Evaluation : " << child->evaluation << std::endl;
//                std::cout << "Number of measurements : " << measurements.size() << std::endl;
//                std::cout << "===================================" << std::endl << std::endl;
//            }
//
//            if (std::isinf(child->evaluation))
//                std::cerr<< "Evaluation of schedule "<< schedules_annotations->size() <<" failed "<< std::endl;
//
//            if (child->evaluation < best_evaluation)
//            {
//                best_evaluation = child->evaluation;
//                best_ast = child;
//            }
//
//            ++iterator;
//
//        }
//
//        nb_explored_schedules++;
//    }
//
//    // Stop if we reached the maximum depth
//    if (nb_explored_optims >= max_depth)
//        return ;
//
//    // Add the current AST to the list of children
//    syntax_tree *ast_copy = ast.copy_ast();
//    ast_copy->nb_explored_optims = nb_explored_optims;
//    children.push_back(ast_copy);
//    parent_trace->add_child_path(ast_copy, parent_trace->get_candidate_id()); // keeps the same id since it's just copy
//
//    // Sort children from smallest evaluation to largest
//    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
//        return a->evaluation < b->evaluation;
//    });
//
//    // keep the top 'beam_size' children and delete the rest
//    for (int i = beam_size; i < children.size(); ++i)
//        delete children[i];
//
//    children.resize(std::min(beam_size, (int)children.size()));
//
//    // Search recursively on the best children
//    for (syntax_tree *child : children)
//    {
//        child->search_depth = ast.search_depth + 1;
//        search_save(*child, schedules_annotations, parent_trace->child_mappings[child], schedule_timeout);
//    }
//}

//void mcts::search(syntax_tree& ast)
//{
//    std::default_random_engine rand_generator;
//
//    std::vector<syntax_tree*> samples;
//    std::vector<syntax_tree*> children;
//    std::vector<double> children_evals;
//
//    for (int epoch = 0; epoch < nb_samples; ++epoch)
//    {
//        // Starting from the initial ast, generate optimizations until reaching max_depth
//        syntax_tree *ast_sample = &ast;
//        for (int depth = 0; depth < max_depth; ++depth)
//        {
//            optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[depth % NB_OPTIMIZATIONS];
//            children = scheds_gen->generate_schedules(*ast_sample, optim_type);
//
//            if (children.empty())
//                continue;
//
//            children_evals.clear();
//
//            for (syntax_tree *child : children)
//            {
//                child->transform_ast();
//
//                child->evaluation = eval_func->evaluate(*child);
//                children_evals.push_back(child->evaluation);
//
//                nb_explored_schedules++;
//            }
//
//            // Add the current AST to the list of children
//            children.push_back(ast_sample->copy_ast());
//            children_evals.push_back(ast_sample->evaluation);
//
//            // Sample an AST
//            std::discrete_distribution<int> dist(children_evals.begin(), children_evals.end());
//            ast_sample = children[dist(rand_generator)];
//
//            samples.push_back(ast_sample);
//        }
//    }
//
//    if (samples.empty())
//        return ;
//
//    // Sort schedules with respect to evaluations
//    std::sort(samples.begin(), samples.end(), [](syntax_tree *a, syntax_tree *b) {
//        return a->evaluation < b->evaluation;
//    });
//
//    // Execute top-k schedules and return the best
//    for (int i = 0; i < topk; ++i)
//    {
//        float exec_time = exec_eval->evaluate(*samples[i]);
//        if (exec_time < best_evaluation)
//        {
//            best_evaluation = exec_time;
//            best_ast = samples[i];
//        }
//    }
//}
//
//void mcts::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
//{
//    std::cerr<< "mcts::search_save not yet implemented" << std::endl;
//    exit(1);
//}
//
//// -------------------------------------------------------------------------- //
//
//void beam_search_topk::search(syntax_tree& ast)
//{
//    // Do a beam search
//    beam_search_subroutine(ast);
//
//    // Sort schedules found
//    std::sort(schedules.begin(), schedules.end(), [](syntax_tree *a, syntax_tree *b) {
//        return a->evaluation < b->evaluation;
//    });
//
//    // Execute top-k schedules to find the best
//    for (int i = 0; i < topk; ++i)
//    {
//        float exec_time = exec_eval->evaluate(*schedules[i]);
//        if (exec_time < best_evaluation)
//        {
//            best_evaluation = exec_time;
//            best_ast = schedules[i];
//        }
//    }
//}
//
//void beam_search_topk::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
//{
//    std::cerr<< "beam_search_topk::search_save not yet implemented" << std::endl;
//    exit(1);
//}
//
//void beam_search_topk::beam_search_subroutine(syntax_tree& ast)
//{
//    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
//        ast.clear_new_optimizations();
//
//    std::vector<syntax_tree*> children;
//
//    // Look for an optimization that can be applied
//    int nb_optims_tried = 0;
//    int nb_explored_optims = ast.nb_explored_optims;
//
//    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
//    {
//        optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[nb_explored_optims % NB_OPTIMIZATIONS];
//        children = scheds_gen->generate_schedules(ast, optim_type);
//
//        nb_explored_optims++;
//        nb_optims_tried++;
//    }
//
//    // Stop if no more optimizations can be applied
//    if (children.size() == 0)
//        return ;
//
//    // Evaluate children and sort them from smallest to highest evaluation
//    for (syntax_tree *child : children)
//    {
//        child->nb_explored_optims = nb_explored_optims;
//        child->transform_ast();
//
//        child->evaluation = eval_func->evaluate(*child);
//
//        nb_explored_schedules++;
//    }
//
//    // Add the current AST to the list of children
//    syntax_tree *ast_copy = ast.copy_ast();
//    ast_copy->nb_explored_optims = nb_explored_optims;
//    children.push_back(ast_copy);
//
//    // Sort children from smallest evaluation to largest
//    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
//        return a->evaluation < b->evaluation;
//    });
//
//    for (int i = 0; i < beam_size; ++i)
//        schedules.push_back(children[i]);
//
//    // Stop if we reached the maximum depth
//    if (nb_explored_optims >= max_depth)
//        return ;
//
//    // Search recursively on the best children
//    for (int i = beam_size; i < children.size(); ++i)
//        delete children[i];
//
//    children.resize(std::min(beam_size, (int)children.size()));
//
//    for (syntax_tree *child : children)
//    {
//        child->search_depth = ast.search_depth + 1;
//        search(*child);
//    }
//}
//
//void beam_search_accuracy_evaluator::search(syntax_tree& ast)
//{
//    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
//        ast.clear_new_optimizations();
//
//    std::vector<syntax_tree*> children;
//
//    // Look for an optimization that can be applied
//    int nb_optims_tried = 0;
//    int nb_explored_optims = ast.nb_explored_optims;
//
//    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
//    {
//        optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[nb_explored_optims % NB_OPTIMIZATIONS];
//        children = scheds_gen->generate_schedules(ast, optim_type);
//
//        nb_explored_optims++;
//        nb_optims_tried++;
//    }
//
//    // Stop if no more optimizations can be applied
//    if (children.size() == 0)
//        return ;
//
//    // Evaluate children and sort them from smallest to highest evaluation
//    for (syntax_tree *child : children)
//    {
//        child->nb_explored_optims = nb_explored_optims;
//        child->transform_ast();
//
//        child->evaluation = eval_func->evaluate(*child);
//
//        // We evaluate both by the model and by execution
//        model_evals_list.push_back(child->evaluation);
//        exec_evals_list.push_back(exec_eval->evaluate(*child));
//
//        if (child->evaluation < best_evaluation)
//        {
//            best_evaluation = child->evaluation;
//            best_ast = child;
//        }
//
//        nb_explored_schedules++;
//    }
//
//    // Stop if we reached the maximum depth
//    if (nb_explored_optims >= max_depth)
//        return ;
//
//    // Add the current AST to the list of children
//    syntax_tree *ast_copy = ast.copy_ast();
//    ast_copy->nb_explored_optims = nb_explored_optims;
//    children.push_back(ast_copy);
//
//    // Sort children from smallest evaluation to largest
//    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
//        return a->evaluation < b->evaluation;
//    });
//
//    // Search recursively on the best children
//    for (int i = beam_size; i < children.size(); ++i)
//        delete children[i];
//
//    children.resize(std::min(beam_size, (int)children.size()));
//
//    for (syntax_tree *child : children)
//    {
//        child->search_depth = ast.search_depth + 1;
//        search(*child);
//    }
//}

}
