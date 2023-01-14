#include <sys/wait.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include <random>

#include <random>
#include <functional>
#include <exception>

#include <stdexcept>

namespace tiramisu::auto_scheduler
{
    // list of hashes of matrices we explored before to avoid repeating schedules. Used in search_save_matrix
std::vector<std::size_t> hashes;
void beam_search::explore_schedules(syntax_tree &ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout ){
    
    std::queue<syntax_tree*> exploration_queue;
    exploration_queue.push(&ast);
    std::unordered_map<syntax_tree*, candidate_trace*> trace_map;
    while(!exploration_queue.empty()){
        
        trace_map[&ast] = parent_trace;
        
        std::vector<syntax_tree*> level_schedules;
        while(!exploration_queue.empty()){
            
            syntax_tree *ast_to_explore = exploration_queue.front();
            exploration_queue.pop();
            std::vector<syntax_tree*> intermediate_schedules ;
            
            switch(ast_to_explore->ast_search_phase) {

                case search_phase::FUSION:
                    intermediate_schedules = search_save(*ast_to_explore, schedules_annotations, trace_map[ast_to_explore], schedule_timeout);
                    break;

                case search_phase::UNIMODULAR:
                    intermediate_schedules = search_save_matrix(*ast_to_explore, schedules_annotations, trace_map[ast_to_explore], schedule_timeout);
                    break;  

                case search_phase::NON_UNIMODULAR:
                    intermediate_schedules = search_save(*ast_to_explore, schedules_annotations, trace_map[ast_to_explore], schedule_timeout);
                    break;
                
                default:
                    return;
            }
            for(auto sched: intermediate_schedules){
                    trace_map[sched] = trace_map[ast_to_explore]->child_mappings[sched];
            }
            level_schedules.insert(level_schedules.end(), intermediate_schedules.begin(), intermediate_schedules.end());
        }
        //Sort children from smallest evaluation to largest
        std::sort(level_schedules.begin(), level_schedules.end(), [](syntax_tree *a, syntax_tree *b) {
            return a->evaluation < b->evaluation;
        });
        
        //keep the top 'beam_size' children and delete the rest
        for (int i = beam_size; i < level_schedules.size(); ++i)
            delete level_schedules[i];
        level_schedules.resize(std::min(beam_size, (int)level_schedules.size()));
        for (syntax_tree *child : level_schedules)
        {   
            exploration_queue.push(child);
        }

    }
}
/*
returns identity matrix
*/
std::vector <  std::vector<int> > get_identity(int depth){
    std::vector <  std::vector<int> >  matrix(depth);
        for(int l = 0; l<matrix.size(); l++){
            matrix.at(l)= std::vector<int>(depth);
            for(int c = 0; c<matrix.size(); c++){
                            if (l!=c ){
                                matrix.at(l).at(c) = 0;
                            }else{
                                matrix.at(l).at(c) = 1;
                            }
            }
        }
        return matrix;
}

std::vector<syntax_tree*> beam_search::search_save_matrix(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
        
    std::default_random_engine rand_generator;
    std::vector<syntax_tree*> children;
    // list of ASTs to be explored for next level 
    std::vector<syntax_tree*> to_be_explored;

    std::hash<std::string> hasher;

    // at this stage we only explore matrices
    std::vector<optimization_type> optims;
    optims.push_back(optimization_type::MATRIX);
    ast.initialize_search_space_optimizations(optims);

    // if this is the root of the exploration tree 
    // we want to create the original schedules which will include identity matrices only
    // to know the size of the matrix for each computation, we go through all the tree looking for nodes that have computations. 
    // the depth of this node is the size of the matrix for all the computations it contains 
    if (ast.search_depth==1){
        
        std::vector<ast_node*> nodes;
        // go through each root of the tree to recover all computations
        for(auto root: ast.roots){
            std::vector<ast_node*> nodes;
            root->get_all_nodes(nodes);
            for(auto node : nodes){
                if(node->computations.size()>0){
                    optimization_info optim_info;
                    optim_info.type = optimization_type::MATRIX;
                    node->get_node_computations(optim_info.comps);

                    // for the original schedule, the transformation matrix is the identity
                    optim_info.matrix = get_identity(node->depth+1);
                    ast.new_optims.push_back(optim_info);
                }   
            }
        }    
    }
    // add the hash of this tree to avoid exploring the same schedules twice
    hashes.push_back(hasher(ast.get_schedule_str()));
    while ((!ast.is_search_space_empty()))
    {
        // schedule generation based on generator_state attribute in the AST.
        auto new_children = scheds_gen->generate_matrices(ast);
        
        for(auto& child:new_children)
            child->move_to_next_head();
        
        children.insert(children.end(), new_children.begin(), new_children.end()); // concatenate

        if  (ast.search_state.is_current_optimization_fully_explored() && !children.empty()) {
            // move to next optimization
            // explores next optimization/alternative
            ast.move_to_next_head();
            
            break;
        }
        else
            ast.move_to_next_head();
    }

    // if no candidates were generated, return an empty list
    if (children.size() == 0) return children;
 
    // hash the parent 
    std::size_t parent_hash=hasher(ast.get_schedule_str());

    auto iterator = children.begin();
    std::vector<std::vector<std::vector<int>>> repeated;
    
    syntax_tree *child;
    // evaluate the legal children and sort them from smallest to highest evaluation
    while (iterator != children.end())
    {
        child = *iterator;
        child->transform_ast();
        if(child->ast_is_prunable()){
            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                    // print deleted Ast
                    child->print_previous_optims();
                    std::cout << "\n-----------" << std::endl;
                    child->print_new_optims();
                    
                    child->print_ast();
                    child->print_isl_states();
                    std::cout << "\n<surpassed MAX_MAT_DEPTH>\n";
                }
                delete child;
                iterator = children.erase(iterator);
        }else{
            if (!child->ast_is_legal()) {
                if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                    // print deleted Ast
                    child->print_previous_optims();
                    std::cout << "\n-----------" << std::endl;
                    child->print_new_optims();
                    
                    child->print_ast();
                    child->print_isl_states();
                    std::cout << "\n<illegal>\n";
                }
                delete child;
                iterator = children.erase(iterator);
            }
            else {
                // hash the legal schedule
                std::size_t hash=hasher(child->get_schedule_str());
                
                bool repeated = false;
                // check if we explored this matrix before  
                for(std::size_t hashe:hashes){
                    if(hashe==hash){
                        //if that's the case remove the child from the exploration tree
                        delete child;
                        iterator = children.erase(iterator);
                        repeated = true;
                        break;
                    }
                }
                if(repeated) continue;

                // if the matrix is legal and not repeated we add its hash to the list of seen hashes and we start the evaluation 
                hashes.push_back(hash);
                
                // print and evaluate Ast
                if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                    child->print_previous_optims();
                    std::cout << "\n-----------" << std::endl;
                    child->print_new_optims();
                    child->print_ast();
                    child->print_isl_states();
                    std::cout << "\n<legal>\n";
                    child->print_computations_accesses();
                }

                std::vector<float> measurements;
                // check the environment variable EXPLORE_BY_EXECUTION to decide the evaluation method
                if(std::atoi(read_env_var("EXPLORE_BY_EXECUTION"))==1){
                    measurements = exec_eval->get_measurements(*child, false, schedule_timeout);
                }else{
                    std::string no_sched_json = schedules_annotations->at(0);
                    measurements.push_back(eval_func->evaluate(*child, no_sched_json));
                }
                child->evaluation = min_eval(measurements);
                
                if(hash != parent_hash) child->nb_explored_matrices = child->nb_explored_matrices +1; 
                
                // add the child to the exploration trace
                parent_trace->add_child_path(child, schedules_annotations->size());
                
                std::string schedule_annot = evaluate_by_learning_model::get_schedule_json(*child);
                
                //remove the last two characters }\n
                schedule_annot.pop_back();
                schedule_annot.pop_back();
                
                if (std::isfinite(child->evaluation)) // the evaluation is not finite mean that the schedule didn't run
                    schedule_annot += ", \n\"execution_times\" : " + measurements_to_str(measurements) + "\n}\n";
                else
                    schedule_annot += ", \n\"execution_times\" : null\n}\n";

                schedules_annotations->push_back(schedule_annot);
                
                if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                    std::cout << "Schedule number "<< schedules_annotations->size() << std::endl;
                    std::cout << "Evaluation : " << child->evaluation << std::endl;
                    std::cout << "Number of measurements : " << measurements.size() << std::endl;
                    std::cout << "===================================" << std::endl << std::endl;
                }

                if (std::isinf(child->evaluation))
                    std::cerr<< "Evaluation of schedule "<< schedules_annotations->size() <<" failed "<< std::endl;

                if (child->evaluation < best_evaluation)
                {
                    best_evaluation = child->evaluation;
                    best_ast = child;
                }
                
                to_be_explored.push_back(child);
                
                ++iterator;  
                
            }
        }
    }
    // add the possibility to explore no transformation at this level by adding a copy of the parent to the list of candidates
    syntax_tree *ast_copy = ast.copy_ast();
    to_be_explored.push_back(ast_copy);
    parent_trace->add_child_path(ast_copy, parent_trace->get_candidate_id());
    
    // we explore MAX_MAT_DEPTH matrices per computations
    int nb_comps= ast.get_innermost_nodes().size();
    for (syntax_tree *child : to_be_explored)
    {
        // increment the search depth for the recursive call
        child->search_depth = child->search_depth + 1;
        // if we are NOT under the maximum depth of matrices to explore then call search_move on to the next exploration phase
        if (!(child->search_depth< MAX_MAT_DEPTH * nb_comps && child->search_depth-1 <= child->nb_explored_matrices)){
            child->initialize_search_space_optimizations(DEFAULT_OPTIMIZATIONS_ORDER);
            // if we surpassed the MAX_MAT_DEPTH amount of matrices to explore OR we detected the parent of this level through
            // the child->search_depth<=child->nb_explored_matrices condition which means that the search level is greater than the number of applied matrices
            // reinitialize current index to zero for the next level of exploration
            child->search_state.current_index = 0;
            child->search_state.optimization_index = 0;
            child->ast_search_phase = search_phase::NON_UNIMODULAR;
        }
    }
    return to_be_explored;
}
std::vector<syntax_tree*> beam_search::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
    
    std::vector<syntax_tree*> children;
    std::vector<optimization_type> transformations_to_explore;
    if(ast.ast_search_phase == search_phase::FUSION){
        transformations_to_explore.push_back(optimization_type::FUSION);
    }else{
        transformations_to_explore = DEFAULT_OPTIMIZATIONS_ORDER;
    }

    if(generator_state::initialized == false)
    {
        ast.initialize_search_space_optimizations(transformations_to_explore);
        // the optimizations are specified along with the parameters in the generator_state attribute inside the AST.
        assert(generator_state::initialized == true);
    }
    
    
    while ((!ast.is_search_space_empty()))
    {
        // schedule generation based on generator_state attribute in the AST.
        auto new_children = scheds_gen->generate_schedules(ast);
        
        for(auto& child:new_children)
            child->move_to_next_optimization_target();

        children.insert(children.end(), new_children.begin(), new_children.end()); // concatenate

        if  (ast.search_state.is_current_optimization_fully_explored() && !children.empty()) {
            // move to next optimization
            // explores next optimization/alternative
            ast.move_to_next_optimization_target();
            break;
        }
        else
            ast.move_to_next_optimization_target();
    }
    
    
    // Stop if no more optimizations can be applied
    // Unless we are exploring fusion. SInce Fusion is seperated from the other transformations, even if no fusion candidates are available, we explore the root.
    if (children.size() == 0 && ast.ast_search_phase != search_phase::FUSION)
        return children;
    

    // Evaluate children and sort them from smallest to highest evaluation
    // evaluate while removing illegal versions
    auto iterator = children.begin();
    while (iterator != children.end())
    {
        bool unrolling_exception_thrown = false;
        if ((*iterator)->schedule_is_prunable()){
        
            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                // print deleted Ast
                (*iterator)->print_previous_optims();
                std::cout << "\n-----------" << std::endl;
                (*iterator)->print_new_optims();
                (*iterator)->print_ast();
                std::cout << "\n<Schedule pruned>\n";
            }
            delete (*iterator);
            iterator = children.erase(iterator);

        }else{
            
            (*iterator)->transform_ast();
            if ((*iterator)->ast_is_legal() == false) {
                // print deleted Ast
                if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                    (*iterator)->print_previous_optims();
                    std::cout << "\n-----------" << std::endl;
                    (*iterator)->print_new_optims();
                    (*iterator)->print_ast();
                    (*iterator)->print_isl_states();
                    std::cout << "\n<illegal>\n";
                }
                delete (*iterator);
                iterator = children.erase(iterator);
                
            }
            else {
                // evaluate and print Ast
                if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                    (*iterator)->print_previous_optims();
                    std::cout << "\n-----------" << std::endl;
                    (*iterator)->print_new_optims();
                    (*iterator)->print_ast();
                    std::cout << "\n<legal>\n";
                }

                std::vector<float> measurements;
                try{
                        if ((*iterator)->can_set_default_evaluation()){ // if yes the child's evaluation is set to a default value
                            measurements = {(*iterator)->evaluation};
                        }else{
                            if(std::atoi(read_env_var("EXPLORE_BY_EXECUTION"))==1){
                                measurements = exec_eval->get_measurements(**iterator, false, schedule_timeout);
                            }else{
                                std::string no_sched_json = schedules_annotations->at(0);
                                measurements.push_back(eval_func->evaluate(*(*iterator), no_sched_json));
                            }
                        }
                }
                catch(NonForLoopBoundExtractionException e){ 
                    // Remove all the optimizations
                    exec_eval->fct->reset_schedules();
                    measurements.clear();
                    measurements.push_back(std::numeric_limits<float>::infinity());
                    unrolling_exception_thrown = true;
                    
                }             

                (*iterator)->evaluation = min_eval(measurements);
                parent_trace->add_child_path((*iterator), schedules_annotations->size());

                std::string schedule_annot = evaluate_by_learning_model::get_schedule_json(*(*iterator));

                //remove the last two characters }\n
                schedule_annot.pop_back();
                schedule_annot.pop_back();

                if (std::isfinite((*iterator)->evaluation)) // the evaluation is not finite mean that the schedule didn't run
                    schedule_annot += ", \n\"execution_times\" : " + measurements_to_str(measurements) + "\n}\n";
                else
                    schedule_annot += ", \n\"execution_times\" : null\n}\n";

                if(!unrolling_exception_thrown){
                    schedules_annotations->push_back(schedule_annot);

                    std::cout << "Schedule number "<< schedules_annotations->size() << std::endl;
                    std::cout << "Evaluation : " << (*iterator)->evaluation << std::endl;
                    std::cout << "Number of measurements : " << measurements.size() << std::endl;
                    std::cout << "===================================" << std::endl << std::endl;

                    if (std::isinf((*iterator)->evaluation))
                        std::cerr<< "Evaluation of schedule "<< schedules_annotations->size() <<" failed "<< std::endl;

                    if ((*iterator)->evaluation < best_evaluation)
                    {
                        best_evaluation = (*iterator)->evaluation;
                        best_ast = (*iterator);
                    }
                }
                ++iterator;

            }

            nb_explored_schedules++;
        }
    }

    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
    children.push_back(ast_copy);

    parent_trace->add_child_path(ast_copy, parent_trace->get_candidate_id()); // keeps the same id since it's just copy

    // Sort children from smallest evaluation to largest
    for (syntax_tree *child : children)
    {
        if(child->ast_search_phase == search_phase::FUSION){
            // reinitialize current index to zero for the next level of exploration
            child->search_state.current_index = 0;
            child->ast_search_phase = search_phase::UNIMODULAR;
        }
        child->search_depth = ast.search_depth + 1;
    }
    return children;
}

void beam_search::search(syntax_tree& ast)
{
    std::vector<syntax_tree*> children;
    // Look for an optimization that can be applied
    if(generator_state::initialized == false)
    {
        ast.initialize_search_space_optimizations(DEFAULT_OPTIMIZATIONS_ORDER);
        // the optimizations are specified along with the parameters in the generator_state attribute inside the AST.
        assert(generator_state::initialized == true);
    }
    
    while ((!ast.is_search_space_empty()))
    {
        // schedule generation based on generator_state attribute in the AST.
        auto new_children = scheds_gen->generate_schedules(ast);

        for(auto& child:new_children)
        {
            child->move_to_next_optimization_target();
        }
        
        children.insert(children.end(), new_children.begin(), new_children.end()); // concatenate
        if  (ast.search_state.is_current_optimization_fully_explored() && !children.empty()) {
            // move to next optimization
            // explores next optimization/alternative
            ast.move_to_next_optimization_target();
            break;
        }
        else
            ast.move_to_next_optimization_target();
    }
       
    // Stop if no more optimizations can be applied
    if (children.size() == 0)
        return ;
       
    // Evaluate children and sort them from smallest to highest evaluation
    // evaluate while removing illegal versions
    auto iterator = children.begin();
    while (iterator != children.end())
    {

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

        
    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
    children.push_back(ast_copy);

    // Sort children from smallest evaluation to largest


    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });

    // keep the top 'beam_size' children and delete the rest
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];
    
        
    children.resize(std::min(beam_size, (int)children.size()));

    // Search recursively on the best children
    for (syntax_tree *child : children)
    {
        child->search_depth = ast.search_depth + 1;        
        search(*child);
    }
}
}
