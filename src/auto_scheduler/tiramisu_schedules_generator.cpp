#include <tiramisu/auto_scheduler/schedules_generator.h>
#include <tiramisu/auto_scheduler/evaluator.h>

namespace tiramisu::auto_scheduler
{

    std::vector<syntax_tree *> exhaustive_generator::generate_schedules(syntax_tree const &ast, optimization_type optim)
    {
        std::vector<syntax_tree *> states;

        switch (optim)
        {
        case optimization_type::FUSION:
            generate_fusions(ast.roots, states, ast);
            break;

        case optimization_type::TILING:
            for (ast_node *root : ast.roots)
                generate_tilings(root, states, ast);

            break;

        case optimization_type::INTERCHANGE:
            for (ast_node *root : ast.roots)
                generate_interchanges(root, states, ast);

            break;

        case optimization_type::UNROLLING:
            for (ast_node *root : ast.roots)
                generate_unrollings(root, states, ast);

            break;

        default:
            break;
        }

        return states;
    }

    void exhaustive_generator::generate_fusions(std::vector<ast_node *> const &tree_level, std::vector<syntax_tree *> &states, syntax_tree const &ast)
    {
        for (int i = 0; i < tree_level.size(); ++i)
        {
            if (tree_level[i]->unrolled || tree_level[i]->get_extent() == "0-0+1")
                continue;

            for (int j = i + 1; j < tree_level.size(); ++j)
            {
                if (tree_level[j]->unrolled || tree_level[j]->get_extent() == "0-0+1")
                    continue;

                if (tree_level[i]->name == tree_level[j]->name &&
                    tree_level[i]->low_bound == tree_level[j]->low_bound &&
                    tree_level[i]->up_bound == tree_level[j]->up_bound)
                {
                    // Copy the AST, and add fusion to the list of optimizations
                    syntax_tree *new_ast = new syntax_tree();
                    ast_node *new_node = ast.copy_and_return_node(*new_ast, tree_level[i]);

                    optimization_info optim_info;
                    optim_info.type = optimization_type::FUSION;
                    optim_info.node = new_node;

                    optim_info.nb_l = 2;
                    optim_info.l0 = i;
                    optim_info.l1 = j;
                    new_ast->new_optims.push_back(optim_info);

                    states.push_back(new_ast);
                }
            }
        }

        for (ast_node *node : tree_level)
            generate_fusions(node->children, states, ast);
    }

    void exhaustive_generator::generate_tilings(ast_node *node, std::vector<syntax_tree *> &states, syntax_tree const &ast)
    {
        int branch_depth = node->get_loop_levels_chain_depth();

        // Generate tiling with dimension 2
        if (node->depth + 1 < branch_depth)
        {
            for (int tiling_size1 : tiling_factors_list)
            {
                if (!can_split_iterator(node->up_bound, node->low_bound, tiling_size1))
                    continue;

                ast_node *node2 = node->children[0];
                for (int tiling_size2 : tiling_factors_list)
                {
                    if (!can_split_iterator(node2->up_bound, node2->low_bound, tiling_size2))
                        continue;

                    // Copy the AST, and add tiling to the list of optimizations
                    syntax_tree *new_ast = new syntax_tree();
                    ast_node *new_node = ast.copy_and_return_node(*new_ast, node);

                    optimization_info optim_info;
                    optim_info.type = optimization_type::TILING;
                    optim_info.node = new_node;

                    optim_info.nb_l = 2;
                    optim_info.l0 = node->depth;
                    optim_info.l1 = node->depth + 1;

                    optim_info.l0_fact = tiling_size1;
                    optim_info.l1_fact = tiling_size2;

                    new_node->get_all_computations(optim_info.comps);

                    new_ast->new_optims.push_back(optim_info);
                    states.push_back(new_ast);

                    // Generate tiling with dimension 3
                    if (node->depth + 2 < branch_depth)
                    {
                        ast_node *node3 = node2->children[0];
                        for (int tiling_size3 : tiling_factors_list)
                        {
                            if (!can_split_iterator(node3->up_bound, node3->low_bound, tiling_size3))
                                continue;

                            // Copy the AST, and add tiling to the list of optimizations
                            syntax_tree *new_ast = new syntax_tree();
                            ast_node *new_node = ast.copy_and_return_node(*new_ast, node);

                            optimization_info optim_info;
                            optim_info.type = optimization_type::TILING;
                            optim_info.node = new_node;

                            optim_info.nb_l = 3;
                            optim_info.l0 = node->depth;
                            optim_info.l1 = node->depth + 1;
                            optim_info.l2 = node->depth + 2;

                            optim_info.l0_fact = tiling_size1;
                            optim_info.l1_fact = tiling_size2;
                            optim_info.l2_fact = tiling_size3;

                            new_node->get_all_computations(optim_info.comps);

                            new_ast->new_optims.push_back(optim_info);
                            states.push_back(new_ast);
                        }
                    }
                }
            }
        }

        for (ast_node *child : node->children)
            generate_tilings(child, states, ast);
    }

    void exhaustive_generator::generate_interchanges(ast_node *node, std::vector<syntax_tree *> &states, syntax_tree const &ast)
    {
        // if (!node->unrolled && node->get_extent() > 1)
        // {
        //     int branch_depth = node->get_loop_levels_chain_depth();

        //     for (int i = node->depth + 1; i < branch_depth; ++i)
        //     {
        //         // Copy the AST, and add interchange to the list of optimizations
        //         syntax_tree *new_ast = new syntax_tree();
        //         ast_node *new_node = ast.copy_and_return_node(*new_ast, node);

        //         optimization_info optim_info;
        //         optim_info.type = optimization_type::INTERCHANGE;
        //         optim_info.node = new_node;

        //         optim_info.nb_l = 2;
        //         optim_info.l0 = node->depth;
        //         optim_info.l1 = i;
        //         new_node->get_all_computations(optim_info.comps);

        //         new_ast->new_optims.push_back(optim_info);
        //         states.push_back(new_ast);
        //     }
        // }

        // for (ast_node *child : node->children)
        //     generate_interchanges(child, states, ast);
    }
    

    void exhaustive_generator::generate_unrollings(ast_node *node, std::vector<syntax_tree *> &states, syntax_tree const &ast)
    {
        if (!node->unrolled && node->get_extent() != "0-0+1" )
        {
            for (int unrolling_factor : unrolling_factors_list)
            {
                if (!can_split_iterator(node->up_bound, node->low_bound, unrolling_factor, optimization_type::UNROLLING))
                    continue;

                // Copy the AST, and add unrolling to the list of optimizations
                syntax_tree *new_ast = new syntax_tree();
                ast_node *new_node = ast.copy_and_return_node(*new_ast, node);

                optimization_info optim_info;
                optim_info.type = optimization_type::UNROLLING;
                optim_info.node = new_node;

                optim_info.nb_l = 1;
                optim_info.l0 = node->depth;
                optim_info.l0_fact = unrolling_factor;
                new_node->get_all_computations(optim_info.comps);

                new_ast->new_optims.push_back(optim_info);
                states.push_back(new_ast);
            }
        }

        for (ast_node *child : node->children)
            generate_unrollings(child, states, ast);
    }

std::vector<syntax_tree *> ml_model_schedules_generator::generate_schedules(syntax_tree &ast)
{
    //this method uses the AST custom schedule generator

    std::vector<syntax_tree *> states;

    ast_node *node = std::get<0>(ast.get_current_optimization_target());

    std::vector<std::string> shared_levels_extents;
    std::vector<int> innermost_extents;
    std::vector<ast_node *> innermost_nodes;

    std::vector<ast_node *> shared_nodes;
    std::vector<tiramisu::computation *> involved_computations;
    int nb_try = 0;
    std::cout<<"working on optimization: "<<ast.get_current_optimization_type()<<std::endl;
    // Generate the specified optimization
    switch (ast.get_current_optimization_type())
    {
        case optimization_type::MATRIX:
        {
            std::vector<syntax_tree *> suggested_transformations =  generate_matrices(ast);
            for (auto schedule: suggested_transformations){
                states.push_back(schedule);
            }
        }
        break;    
        case optimization_type::FUSION:
        /* iteration of the ast in done preorder  */
        {
            if (ast.search_state.current_index == 0)
            { // cant fuze the first computation
                return states;
            }

            // 2 computations's nodes that need to fuze together
            auto node_computation = ast.get_current_optimization_target();
            auto previous_node_computation = ast.get_previous_optimization_target();

            ast_node * current_node = node_computation.first;
            ast_node * previous_node = previous_node_computation.first;

            // adjusted nodes are the target nodes or their parents with same depth in the AST
            auto potentiel_fusion = current_node->get_possible_fusion_candidate(previous_node);
            ast_node * previous_node_adjusted = std::get<0>(potentiel_fusion);

            auto involved_iterators = previous_node_adjusted->get_all_iterators();

            auto real_iterators = current_node->computations[node_computation.second].comp_ptr->get_iteration_domain_dimension_names();

            real_iterators.resize(involved_iterators.size());

            // create a vector of involved tiramisu vars
            std::vector<tiramisu::var> loop_levels;
            // loop levels used for shifting solver
            std::cout<<"computation schedule is "<<isl_map_to_str(current_node->computations[node_computation.second].comp_ptr->get_schedule())<<std::endl;
            for (auto const &str : real_iterators)
            {
                std::cout<<"variable name is: "<<str<<std::endl;
                loop_levels.push_back(tiramisu::var(str));
            }
            std::vector<computation *> seen_computations;
            // computations list used for shifting solver
            ast.get_previous_computations(seen_computations, node, std::get<1>(node_computation));



            if ((previous_node != current_node) && (previous_node_adjusted == previous_node)
                    && (previous_node != nullptr) && (current_node != nullptr))
            {
                // the fusion that will generate dummy itr is removed by previous_node_adjusted == previous_node
                if (previous_node->is_candidate_for_fusion(current_node))
                { // similar itr domains
                    // create AST copy to falsely fuze and check legality
                    syntax_tree *new_ast = new syntax_tree();
                    ast_node *new_node = ast.copy_and_return_node(*new_ast, previous_node);
                    
                    // creating new sched graph
                    ast.stage_local_sched_graph();
                    new_ast->create_new_sched_graph();
                    ast.recover_local_sched_graph();
                    
                    new_ast->stage_isl_states();
                    // modify the schedule graph now using after
                    current_node->computations[node_computation.second].comp_ptr->after(
                        *previous_node->computations[previous_node_computation.second].comp_ptr,
                        previous_node_adjusted->depth
                    );

                    new_ast->fct->prepare_schedules_for_legality_checks(true);
                    
                    int depth = previous_node_adjusted->depth;
                    optimization_info optim_info;
                    optim_info.type = optimization_type::FUSION;

                    optim_info.node = new_node->find_node_by_depth(depth);
                    optim_info.nb_l = 1;
                    optim_info.l0 =  depth;
                    optim_info.l1 =-1;
                    optim_info.l0_fact = -1;
                    optim_info.l1_fact = -1;
                    optim_info.comps = {previous_node->computations[previous_node_computation.second].comp_ptr,current_node->computations[node_computation.second].comp_ptr};
                    new_ast->new_optims.push_back(optim_info);
                    

                    auto shifting_res = ast.get_function()->correcting_loop_fusion_with_shifting(
                        seen_computations,
                        *current_node->computations[node_computation.second].comp_ptr,
                        loop_levels);
                    

                    if (shifting_res.size() > 0)
                    {
                        // fusion accepted
                        // must generate shifting optim + transforme a copy of the ast

                        for(auto& shifting:shifting_res)
                        {
                            if(std::get<1>(shifting) > 0)
                            {
                                int depth = current_node->computations[node_computation.second].
                                    comp_ptr->get_loop_level_number_from_dimension_name(std::get<0>(shifting).get_name());
                                optimization_info optim_info;
                                optim_info.type = optimization_type::SHIFTING;

                                optim_info.node = new_node->find_node_by_depth(depth);
                                optim_info.nb_l = 1;
                                optim_info.l0 =  depth;
                                optim_info.l1 =0;
                                optim_info.l0_fact = std::get<1>(shifting);
                                optim_info.l1_fact = -1;
                                optim_info.comps = {current_node->computations[node_computation.second].comp_ptr};
                                new_ast->new_optims.push_back(optim_info);

                            }

                        }

                        new_ast->recover_isl_states();

                        // APPLY changes to the AST it self
                        new_ast->move_in_computation(new_node,current_node->computations[node_computation.second].comp_ptr);

                        // recompute the states vector because locations changed.
                        new_ast->refresh_states();
                        new_ast->tree_structure_json = evaluate_by_learning_model::get_tree_structure_json(*new_ast);

                        states.push_back(new_ast);
                    }
                    else
                    {
                        new_ast->recover_isl_states();
                        delete new_ast;
                        // This is added for the scheduling graph to be cleared. Since there is an after call that will cause an issue otherwise.
                        ast.get_function()->reset_schedules();
                    }

                }
            }
        }
        break;

    case optimization_type::TILING:
        shared_levels_extents = ast.get_shared_levels_extents();

        //shared nodes minus last shared node
        shared_nodes = node->collect_shared_nodes_from_head();
        shared_nodes.pop_back(); // remove the last because we can't start a tiling from the last iterator

        if (!shared_nodes.empty())
            shared_nodes[0]->get_all_computations(involved_computations); // since we are trying tilings on the shared nodes, we can directly get the involved comps from here.
        if (ast.optim_already_applied_on_comps(involved_computations,optimization_type::TILING)) // check if one of the involved computations is already tiled
            return states;

        // use nb try as to count if we reached last commun possible node (to disable 3layers tiling);
        nb_try = 0;

        for (auto &node_iterator : shared_nodes)
        {
            for (int tiling_size1 : tiling_factors_list)
            {
                // Check if tiling_size1 splits perfectly this iterator
                if (can_split_iterator_sup(node_iterator->up_bound, node_iterator->low_bound, tiling_size1))
                {
                    for (int tiling_size2 : tiling_factors_list)
                    {
                        if (can_split_iterator_sup(node_iterator->children[0]->up_bound, node_iterator->children[0]->low_bound, tiling_size2))
                        {
                            // Copy the AST and add tiling with 2 dimensions to the list of optimizations
                            syntax_tree *new_ast = new syntax_tree();
                            ast_node *new_node = ast.copy_and_return_node(*new_ast, node_iterator);
                            involved_computations = {};
                            node_iterator->get_all_computations(involved_computations);
                            optimization_info optim_info;
                            optim_info.type = optimization_type::TILING;
                            optim_info.node = new_node;
                            optim_info.nb_l = 2;
                            optim_info.l0 = node_iterator->depth;
                            optim_info.l1 = node_iterator->depth + 1;
                            optim_info.l0_fact = tiling_size1;
                            optim_info.l1_fact = tiling_size2;
                            optim_info.comps = involved_computations;
                            new_ast->new_optims.push_back(optim_info);
                            states.push_back(new_ast);

                            // Cannot apply tiling with 3 dimensions,
                            // continue to apply tiling with 2 dimensions.
                            /*if ((nb_try + 2) >= shared_nodes.size())
                                continue;*/

                            if ((nb_try + 1) < shared_nodes.size())
                            {
                                for (int tiling_size3 : tiling_factors_list)
                                {
                                    if (can_split_iterator_sup(node_iterator->children[0]->children[0]->up_bound, node_iterator->children[0]->children[0]->low_bound, tiling_size3))
                                    {
                                        // Copy the AST and add tiling with 3 dimensions to the list of optimizations
                                        syntax_tree *new_ast = new syntax_tree();
                                        ast_node *new_node = ast.copy_and_return_node(*new_ast, node_iterator);

                                        optimization_info optim_info;
                                        optim_info.type = optimization_type::TILING;
                                        optim_info.node = new_node;

                                        optim_info.nb_l = 3;
                                        optim_info.l0 = node_iterator->depth;
                                        optim_info.l1 = node_iterator->depth + 1;
                                        optim_info.l2 = node_iterator->depth + 2;

                                        optim_info.l0_fact = tiling_size1;
                                        optim_info.l1_fact = tiling_size2;
                                        optim_info.l2_fact = tiling_size3;

                                        optim_info.comps = involved_computations;
                                        new_ast->new_optims.push_back(optim_info);
                                        states.push_back(new_ast);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            nb_try++;
        }
        break;

    case optimization_type::INTERCHANGE:

        shared_nodes = node->collect_shared_nodes_from_head();

        if (shared_nodes.size() > 0)
        {
            node->get_all_computations(involved_computations);
        }
        else
        {
            return states;
        }
        if (ast.optim_already_applied_on_comps(involved_computations,optimization_type::INTERCHANGE)) // check if one of the involved computations is already interchanged
            return states;

        // To apply interchange, we pick all combinations of two iterators
        // in the shared loop levels.
        for (int i = 0; i < shared_nodes.size(); ++i)
        {
            for (int j = i + 1; j < shared_nodes.size(); ++j)
            {
                // Copy the AST and add interchange to the list of optimizations
                syntax_tree *new_ast = new syntax_tree();
                ast_node *new_node = ast.copy_and_return_node(*new_ast, shared_nodes[i]);

                optimization_info optim_info;
                optim_info.type = optimization_type::INTERCHANGE;
                optim_info.node = new_node;

                optim_info.nb_l = 2;
                optim_info.l0 = shared_nodes[i]->depth;
                optim_info.l1 = shared_nodes[j]->depth;

                optim_info.comps = involved_computations;
                new_ast->new_optims.push_back(optim_info);
                states.push_back(new_ast);
            }
        }
        break;

    case optimization_type::UNROLLING: {
        ast.stage_isl_states();
        node->get_innermost_nodes(innermost_nodes);
        //check that the sbutree starting from node is a branch (has no splits)
                ast_node *temp_node = node;
        while (temp_node->children.size() == 1 && temp_node->computations.size() == 0)
            temp_node = temp_node->children[0];
        if (temp_node->children.size() > 0) // if branch splits, return
        {
            ast.recover_isl_states();
            return states;
        }
        
        //search for possible unrolling from the bottom loop until one is found
        // Apply all possible unrolling factors to all innermost iterators
        //test unrolling for all inner nodes until we find a valid
        bool result = true;
        for (ast_node *inner_most_node: innermost_nodes) {
            std::vector<tiramisu::computation *> involved_computations;
            inner_most_node->get_innermost_computations(involved_computations);
            for (auto comp: involved_computations){
                std::vector<std::string> loop_names = comp->get_loop_level_names();
                std::vector<tiramisu::computation *> involved_comp_first; 
                involved_comp_first.push_back(comp);
                std::string loop_name = loop_names[inner_most_node->depth];
                result = result && (!inner_most_node->is_optimized_by_tag()) &&
                                ast.fct->loop_unrolling_is_legal(var(loop_name), involved_comp_first);
            }
            if (result) // unrollable: test all possible values
            {
                ast.recover_isl_states();

                for (int unrolling_fact: unrolling_factors_list) {
                    if (can_split_iterator(inner_most_node->up_bound, inner_most_node->low_bound, unrolling_fact, optimization_type::UNROLLING)) {
                        // Copy the AST and add unrolling to the list of optimizations
                        syntax_tree *new_ast = new syntax_tree();
                        ast_node *new_node = ast.copy_and_return_node(*new_ast, inner_most_node);

                        optimization_info optim_info;
                        optim_info.type = optimization_type::UNROLLING;
                        optim_info.nb_l = 1;

                        // When l0 is set to -1, unrolling is applied to all innermost levels, (1 to avoid that)
                        optim_info.l0 = new_node->depth;
                        optim_info.l0_fact = unrolling_fact;
                        // select this node
                        optim_info.node = new_node;
                        new_node->get_innermost_computations(optim_info.comps);
                        new_ast->new_optims.push_back(optim_info);
                        states.push_back(new_ast);
                    }
                }
                ast.stage_isl_states();
            }

            nb_try++;
        }
        ast.recover_isl_states();
    }
        break;

    case optimization_type::PARALLELIZE:

        ast.stage_isl_states();

        //for shared nodes the list of involved computations is always the same.
        // that's only the case when we compute test shared loop levels only (not always the case).

        shared_nodes = node->collect_shared_nodes_from_head();

        if (shared_nodes.size() > 0)
        {
            shared_nodes[0]->get_all_computations(involved_computations);
        }
        else
        {
            ast.recover_isl_states();
            return states;
        }

        if (ast.optim_already_applied_on_comps(involved_computations,optimization_type::PARALLELIZE)) // check if one of the involved computations is already parallelized
        {
            ast.recover_isl_states();
            return states;
        }

        for (ast_node *commun_node : shared_nodes)
        {
            std::vector<std::string> loop_names = involved_computations[0]->get_loop_level_names();

            std::string loop_name = loop_names[commun_node->depth];

            bool result = ast.fct->loop_parallelization_is_legal(var(loop_name), involved_computations);

            if (result) 
            {
                ast.recover_isl_states();

                // Copy the AST and add unrolling to the list of optimizations
                syntax_tree *new_ast = new syntax_tree();
                ast_node *new_node = ast.copy_and_return_node(*new_ast, commun_node);

                optimization_info optim_info;
                optim_info.type = optimization_type::PARALLELIZE;
                optim_info.nb_l = 1;

                optim_info.l0 = new_node->depth;
                optim_info.l0_fact = 0;
                // select this node
                optim_info.node = new_node;

                optim_info.comps = involved_computations;
                new_ast->new_optims.push_back(optim_info);
                states.push_back(new_ast);

                ast.stage_isl_states();
            }

            nb_try++;

            if (nb_try == this->parallelism_search_deapth)
            {
                break;
            }
        }

        ast.recover_isl_states();
        break;

    case optimization_type::SKEWING:

        ast.stage_isl_states();

        //for shared nodes the list of involved computations is always the same.
        // that's only the case when we compute test shared loop levels only (not always the case).
        shared_nodes = node->collect_shared_nodes_from_head();

        if (shared_nodes.size() > 1)
        {
            shared_nodes[0]->get_all_computations(involved_computations);
            shared_nodes.pop_back(); //removes 2nd loop level, first is enough
        }
        else
        {
            ast.recover_isl_states();
            return states;
        }

        if (ast.optim_already_applied_on_comps(involved_computations,optimization_type::SKEWING)) // check if one of the involved computations is already skewed
        {
            ast.recover_isl_states();
            return states;
        }

        for (ast_node *commun_node : shared_nodes)
        {
            std::vector<std::string> loop_names = involved_computations[0]->get_loop_level_names();

            std::string loop_name = loop_names[commun_node->depth];
            std::string loop_name_inner = loop_names[commun_node->depth + 1];
            // Compute the possible skewings parameters
            // Legacy skewing 2 parameters
            auto result_skewing = ast.fct->skewing_local_solver(involved_computations,
                                                                var(loop_name), var(loop_name_inner),
                                                                skewing_inner_parallelism_number);
            // Positive skewing 4 parameters in one vector
            auto positive_skewing = ast.fct->skewing_local_solver_positive(involved_computations,
                    var(loop_name),var(loop_name_inner),
                    skewing_inner_parallelism_number
                    );
            
            std::vector<std::tuple<int, int, int, int>> positive_param_vector = std::get<1>(positive_skewing);

            if(std::get<0>(positive_skewing).size() > 0){
                positive_param_vector.push_back(std::get<0>(positive_skewing)[0]);
            }
            if(std::get<2>(positive_skewing).size() > 0){
                positive_param_vector.push_back(std::get<2>(positive_skewing)[0]);
            }

            // Explore the legacy skewing
            if (std::get<1>(result_skewing).size() > 0) // inner parallelism has solutions
            {
                ast.recover_isl_states();
                for (auto &param : std::get<1>(result_skewing))
                {
                    // Copy the AST and add unrolling to the list of optimizations
                    syntax_tree *new_ast = new syntax_tree();
                    ast_node *new_node = ast.copy_and_return_node(*new_ast, commun_node);

                    optimization_info optim_info;
                    optim_info.type = optimization_type::SKEWING;
                    optim_info.node = new_node;

                    optim_info.nb_l = 2;
                    optim_info.l0 = new_node->depth;
                    optim_info.l1 = new_node->depth + 1;
                    optim_info.l0_fact = std::get<0>(param);
                    optim_info.l1_fact = std::get<1>(param);

                    optim_info.comps = involved_computations;
                    new_ast->new_optims.push_back(optim_info);
                    states.push_back(new_ast);
                }
                ast.stage_isl_states();
            }

            if (std::get<0>(result_skewing).size() > 0) // outer parallelism has solutions
            {
                ast.recover_isl_states();
                for (auto &param : std::get<0>(result_skewing))
                {
                    // Copy the AST and add unrolling to the list of optimizations
                    syntax_tree *new_ast = new syntax_tree();
                    ast_node *new_node = ast.copy_and_return_node(*new_ast, commun_node);

                    optimization_info optim_info;
                    optim_info.type = optimization_type::SKEWING;
                    optim_info.node = new_node;

                    optim_info.nb_l = 2;
                    optim_info.l0 = new_node->depth;
                    optim_info.l1 = new_node->depth + 1;
                    optim_info.l0_fact = std::get<0>(param);
                    optim_info.l1_fact = std::get<1>(param);

                    optim_info.comps = involved_computations;
                    new_ast->new_optims.push_back(optim_info);
                    states.push_back(new_ast);
                }
                ast.stage_isl_states();
            }

            if (std::get<2>(result_skewing).size() > 0) // locality has solutions
            {
                ast.recover_isl_states();
                for (auto &param : std::get<2>(result_skewing))
                {
                    // Copy the AST and add unrolling to the list of optimizations
                    syntax_tree *new_ast = new syntax_tree();
                    ast_node *new_node = ast.copy_and_return_node(*new_ast, commun_node);

                    optimization_info optim_info;
                    optim_info.type = optimization_type::SKEWING;
                    optim_info.node = new_node;

                    optim_info.nb_l = 2;
                    optim_info.l0 = new_node->depth;
                    optim_info.l1 = new_node->depth + 1;
                    optim_info.l0_fact = std::get<0>(param);
                    optim_info.l1_fact = std::get<1>(param);

                    if ((optim_info.l0 > 0) && (optim_info.l1 > 0))
                    { //require loop reversal for correctness
                        optim_info.l2_fact = -1;
                    }

                    optim_info.comps = involved_computations;
                    new_ast->new_optims.push_back(optim_info);
                    states.push_back(new_ast);
                }
                ast.stage_isl_states();
            }

            // Explore the positive skewing
            ast.recover_isl_states();
            for(auto& param : positive_param_vector)
            {
                // Copy the AST and add Skewing to the list of optimizations
                syntax_tree* new_ast = new syntax_tree();
                ast_node *new_node = ast.copy_and_return_node(*new_ast, commun_node);

                optimization_info optim_info;
                optim_info.type = optimization_type::MATRIX;
                optim_info.node = new_node;

                optim_info.nb_l = 2;
                optim_info.l0 = new_node->depth;
                optim_info.l1 = new_node->depth+1;
                optim_info.l0_fact = std::get<0>(param);
                optim_info.l1_fact = std::get<1>(param);
                optim_info.l2_fact = std::get<2>(param);
                optim_info.l3_fact = std::get<3>(param);


                optim_info.comps = involved_computations;
                new_ast->new_optims.push_back(optim_info);
                states.push_back(new_ast);
            }
            ast.stage_isl_states();
        }

        ast.recover_isl_states();
        break;



    default:
        break;
    }

    return states;
}

int gcd_extend(int a, int b,
               int& x, int& y)
{
    // Base Case
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
 
    // Recursively find the gcd
    else {
        int g = gcd_extend(b,
                           a % b, x, y);
        int x1 = x, y1 = y;
        x = y1;
        y = x1 - (a / b) * y1;
        return g;
    }
}
bool is_repeated( std::vector < std::vector<int> >  matrix,std::vector < std::vector < std::vector<int> > > matrices)
{
    //if there are no matrices to compare to then we return false
    if(matrices.size()==0) return false;
    //if there are no matrices to compare to then we return false
    if(matrices.at(0).size()==0) return false;

    int depth = matrix.size();
    int i=0;
    while(i<matrices.size() && matrices.at(i).size()!=0){ 
        //for each matrix that we already explored  
        bool diffrent = false;
        for (int s=0;s<depth;s++){
                for (int d=0;d<depth;d++){
                            // set diffrent to true if we find one element that is not the same
                            if (matrix.at(s).at(d)!=matrices.at(i).at(s).at(d)) diffrent =true ;
                    }
        }
        //if we found the same matrix return true
        if (!diffrent) return true;
    i++;
    }


    return false;
}
std::vector<int> get_skew_params(int f_i, int f_j)
{
        int gamma = 0;
        int sigma = 1;
        bool found = false;

        
        int n1 = abs(f_i);
        int n2 = abs(f_j);

        while (n1 != n2)
        {
            if (n1 > n2)
                n1 -= n2;
            else
                n2 -= n1;
        }

        DEBUG(3, tiramisu::str_dump("The gcd of f_i = " + std::to_string(f_i) + " and fj = " + std::to_string(f_j) + " is pgcd = " + std::to_string(n1)));

        // Update f_i and f_j to equivalent but prime between themselfs value
        f_i = f_i / n1;
        f_j = f_j / n1;

        

        if ((f_j == 1) || (f_i == 1))
        {
            gamma = f_i - 1;
            sigma = 1;
            /* Since sigma = 1  then
            f_i - gamma * f_j = 1 & using the previous condition :
             - f_i = 1 : then gamma = 0 (f_i-1) is enough
             - f_j = 1 : then gamma = f_i -1  */
        }
        else
        {
            if ((f_j == -1) && (f_i > 1))
            {
                gamma = 1;
                sigma = 0;
            }
            else
            { //General case : solving the Linear Diophantine equation & finding basic solution (sigma & gamma) for : f_i* sigma - f_j*gamma = 1
                int i = 0;
                while ((i < 100) && (!found))
                {
                    if (((sigma * f_i) % abs(f_j)) == 1)
                    {
                        found = true;
                    }
                    else
                    {
                        sigma++;
                        i++;
                    }
                };

                if (!found)
                {
                    // Detect infinite loop and prevent it in case where f_i and f_j are not prime between themselfs
                    ERROR(" Error in solving the Linear Diophantine equation f_i* sigma - f_j*gamma = 1  ", true);
                }

                gamma = ((sigma * f_i) - 1) / f_j;
            }
        }
        std::vector<int> result;
        result.push_back(gamma);
        result.push_back(sigma);
        return result;
}

std::vector<syntax_tree *> ml_model_schedules_generator::generate_matrices(syntax_tree &ast)
{
    //this method uses the AST custom schedule generator

    std::vector<syntax_tree *> states;
    bool explre_interchange = true;
    bool explre_solver_skew = true;
    std::vector<std::vector<std::vector<int>>> matrices;
    ast_node *node = std::get<0>(ast.get_current_optimization_target());
    
    std::vector<int> shared_levels_extents;
    std::vector<int> innermost_extents;
    std::vector<ast_node *> innermost_nodes;

    std::vector<ast_node *> shared_nodes;
    std::vector<tiramisu::computation *> involved_computations;
    
    shared_nodes = node->collect_shared_nodes_from_head();
    int depth = node->depth + shared_nodes.size();
    
    shared_nodes = node->collect_shared_nodes_from_head();
    
    if (shared_nodes.size() > 0)
    {
        node->get_all_computations(involved_computations);
    }
    else
    {
        explre_interchange = false;
    }
    // std::cout<<"before staging"<<std::endl;
    // for (auto sched: involved_computations){
    //     std::cout<<isl_map_to_str(sched->get_schedule())<<std::endl;
    // }
    // ast.stage_isl_states();
    // std::cout<<"after staging"<<std::endl;
    // for (auto sched: involved_computations){
    //     std::cout<<isl_map_to_str(sched->get_schedule())<<std::endl;
    // }
    // To apply interchange, we pick all combinations of two iterators
    // in the shared loop levels.
    if(explre_interchange){

        for (int i = 0; i < shared_nodes.size(); ++i)
        {
            for (int j = i + 1; j < shared_nodes.size(); ++j)
            {
                int first_loop = shared_nodes[i]->depth;
                int second_loop = shared_nodes[j]->depth;
                bool valid_interchange = true;
                // bool valid_interchange = ast.fct->loop_interchnage_is_legal(first_loop, second_loop, involved_computations);
                if(valid_interchange){
                    // Copy the AST and add interchange to the list of optimizations
                    syntax_tree *new_ast = new syntax_tree();
                    ast_node *new_node = ast.copy_and_return_node(*new_ast, shared_nodes[i]);

                    optimization_info optim_info;
                    optim_info.type = optimization_type::MATRIX;
                    optim_info.node = new_node;
                    
                    optim_info.nb_l = 2;
                    
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
                        
                    optim_info.l0 = first_loop;
                    optim_info.l1 = second_loop;
                    matrix.at(first_loop).at(second_loop) = 1;
                    matrix.at(second_loop).at(first_loop) = 1;
                    matrix.at(second_loop).at(second_loop) = 0;
                    matrix.at(first_loop).at(first_loop) = 0;

                    optim_info.comps = involved_computations;
                    optim_info.matrix = matrix;
                    optim_info.unimodular_transformation_type = 1;
                    new_ast->new_optims.push_back(optim_info);
                    states.push_back(new_ast);
                }
                
            }
        }
    }

    // Add reversal matriecs
    std::vector<tiramisu::computation *> involved_computations_reversal;
    // For shared nodes the list of involved computations is always the same.
    // that's only the case when we compute test shared loop levels only (not always the case).
    shared_nodes = node->collect_shared_nodes_from_head();

    if (shared_nodes.size() > 0)
    {
        shared_nodes[0]->get_all_computations(involved_computations_reversal);
    }
    
    for(int i=0;i<shared_nodes.size();i++){
        // Copy the AST and add reversal to the list of optimizations
        syntax_tree *new_ast = new syntax_tree();
        ast_node *new_node = ast.copy_and_return_node(*new_ast, shared_nodes[i]);

        optimization_info optim_info;
        optim_info.type = optimization_type::MATRIX;
        optim_info.node = new_node;
        
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
        optim_info.l0 = shared_nodes[i]->depth;
        matrix.at(shared_nodes[i]->depth).at(shared_nodes[i]->depth) = -1; 
        optim_info.comps = involved_computations_reversal;
        optim_info.matrix = matrix;
        optim_info.unimodular_transformation_type = 2;
        new_ast->new_optims.push_back(optim_info);
        states.push_back(new_ast);
    }
    
    
    //  Add skweing
    ast.stage_isl_states();

    std::vector<tiramisu::computation *> involved_computations_skew;
    // for shared nodes the list of involved computations is always the same.
    // that's only the case when we compute test shared loop levels only (not always the case).
    shared_nodes = node->collect_shared_nodes_from_head();

    if (shared_nodes.size() > 1)
    {
        shared_nodes[0]->get_all_computations(involved_computations_skew);
        shared_nodes.pop_back(); //removes 2nd loop level, first is enough
    }
    else
    {
        ast.recover_isl_states();
        explre_solver_skew = false;
    }

    if(explre_solver_skew){
        for (ast_node *commun_node : shared_nodes)
        {
            std::vector<std::string> loop_names = involved_computations_skew[0]->get_loop_level_names();

            std::string loop_name = loop_names[commun_node->depth];
            std::string loop_name_inner = loop_names[commun_node->depth + 1];
            // Compute the possible skewings parameters
            // Legacy skewing 2 parameters
            auto result_skewing = ast.fct->skewing_local_solver(involved_computations,
                                                                var(loop_name), var(loop_name_inner),
                                                                skewing_inner_parallelism_number);
            // Positive skewing 4 parameters in one vector
            auto positive_skewing = ast.fct->skewing_local_solver_positive(involved_computations,
                    var(loop_name),var(loop_name_inner),
                    skewing_inner_parallelism_number
                    );

            std::vector<std::tuple<int, int, int, int>> positive_param_vector = std::get<1>(positive_skewing);

            if(std::get<0>(positive_skewing).size() > 0){
                positive_param_vector.push_back(std::get<0>(positive_skewing)[0]);
            }
            if(std::get<2>(positive_skewing).size() > 0){
                positive_param_vector.push_back(std::get<2>(positive_skewing)[0]);
            }
            // Explore the legacy skewing
            if (std::get<1>(result_skewing).size() > 0) // inner parallelism has solutions
            {

                ast.recover_isl_states();
                for (auto &param : std::get<1>(result_skewing))
                {
                    // Copy the AST and add skewing to the list of optimizations
                    syntax_tree *new_ast = new syntax_tree();
                    ast_node *new_node = ast.copy_and_return_node(*new_ast, commun_node);

                    optimization_info optim_info;
                    optim_info.type = optimization_type::MATRIX;
                    optim_info.node = new_node;


                    optim_info.nb_l = 2;
                    optim_info.l0 = new_node->depth;
                    optim_info.l1 = new_node->depth + 1;
                    optim_info.l2 = 0;
                    optim_info.l0_fact = std::get<0>(param);
                    optim_info.l1_fact = std::get<1>(param);
                    optim_info.l2_fact = 0;
                    optim_info.l3_fact = 1;
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
                    matrix.at(optim_info.l0).at(optim_info.l1) = optim_info.l1_fact;
                    matrix.at(optim_info.l0).at(optim_info.l0) = optim_info.l0_fact;

                    if(optim_info.l0_fact!=1){
                        // This skewing would cause the increment to be !=1 which is not supported by Tiramisu.
                        // We modify the skewing matrix so that we can still apply skewing.
                        std::vector<int> solutions=get_skew_params(optim_info.l0_fact, optim_info.l1_fact);
                        optim_info.l2_fact = solutions.at(0);
                        optim_info.l3_fact = solutions.at(1);
                        matrix.at(optim_info.l1).at(optim_info.l1-1) =  optim_info.l2_fact;
                        matrix.at(optim_info.l1).at(optim_info.l1) =  optim_info.l3_fact;

                    }

                    optim_info.matrix = matrix;
                    optim_info.comps = involved_computations_skew;
                    optim_info.unimodular_transformation_type = 3;
                    new_ast->new_optims.push_back(optim_info);
                    states.push_back(new_ast);
                }
                ast.stage_isl_states();
            }

            if (std::get<0>(result_skewing).size() > 0) // outer parallelism has solutions
            {
                ast.recover_isl_states();
                for (auto &param : std::get<0>(result_skewing))
                {
                    // Copy the AST and add skewing to the list of optimizations
                    syntax_tree *new_ast = new syntax_tree();
                    ast_node *new_node = ast.copy_and_return_node(*new_ast, node);

                    optimization_info optim_info;
                    optim_info.type = optimization_type::MATRIX;
                    optim_info.node = new_node;

                    optim_info.nb_l = 2;
                    optim_info.l0 = new_node->depth;
                    optim_info.l1 = new_node->depth + 1;
                    optim_info.l2 = 0;
                    optim_info.l0_fact = std::get<0>(param);
                    optim_info.l1_fact = std::get<1>(param);
                    optim_info.l2_fact = 0;
                    optim_info.l3_fact = 1;

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
                    matrix.at(optim_info.l0).at(optim_info.l1) = optim_info.l1_fact;
                    matrix.at(optim_info.l0).at(optim_info.l0) = optim_info.l0_fact;

                    if(optim_info.l0_fact!=1){
                        // This skewing would cause the increment to be !=1 which is not supported by Tiramisu.
                        // We modify the skewing matrix so that we can still apply skewing.
                        std::vector<int> solutions=get_skew_params(optim_info.l0_fact, optim_info.l1_fact);
                        optim_info.l2_fact = solutions.at(0);
                        optim_info.l3_fact = solutions.at(1);
                        matrix.at(optim_info.l1).at(optim_info.l1-1) =  optim_info.l2_fact;
                        matrix.at(optim_info.l1).at(optim_info.l1) =  optim_info.l3_fact;
                    }
                    optim_info.matrix = matrix;
                    optim_info.comps = involved_computations_skew;
                    optim_info.unimodular_transformation_type = 3;
                    new_ast->new_optims.push_back(optim_info);
                    states.push_back(new_ast);
                }
                ast.stage_isl_states();
            }

            if (std::get<2>(result_skewing).size() > 0) // locality has solutions
            {
                ast.recover_isl_states();
                for (auto &param : std::get<2>(result_skewing))
                {
                    // Copy the AST and add skewing to the list of optimizations
                    syntax_tree *new_ast = new syntax_tree();
                    ast_node *new_node = ast.copy_and_return_node(*new_ast, node);

                    optimization_info optim_info;
                    optim_info.type = optimization_type::MATRIX;
                    optim_info.node = new_node;

                    optim_info.nb_l = 2;
                    optim_info.l0 = new_node->depth;
                    optim_info.l1 = new_node->depth + 1;
                    optim_info.l2 = 0;
                    optim_info.l0_fact = std::get<0>(param);
                    optim_info.l1_fact = std::get<1>(param);
                    optim_info.l2_fact = 0;
                    optim_info.l3_fact = 1;
                    if ((optim_info.l0 > 0) && (optim_info.l1 > 0))
                    { //require loop reversal for correctness
                        optim_info.l2_fact = -1;
                    }
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
                    matrix.at(optim_info.l0).at(optim_info.l1) = optim_info.l1_fact;
                    matrix.at(optim_info.l0).at(optim_info.l0) = optim_info.l0_fact;

                    if(optim_info.l0_fact!=1){
                        // This skewing would cause the increment to be !=1 which is not supported by Tiramisu.
                        // We modify the skewing matrix so that we can still apply skewing.
                        std::vector<int> solutions=get_skew_params(optim_info.l0_fact, optim_info.l1_fact);
                        optim_info.l2_fact = solutions.at(0);
                        optim_info.l3_fact = solutions.at(1);
                        matrix.at(optim_info.l1).at(optim_info.l1-1) =  optim_info.l2_fact;
                        matrix.at(optim_info.l1).at(optim_info.l1) =  optim_info.l3_fact;
                    }
                    optim_info.matrix = matrix;

                    optim_info.comps = involved_computations_skew;
                    optim_info.unimodular_transformation_type = 3;
                    new_ast->new_optims.push_back(optim_info);

                    if ((optim_info.l0 > 0) && (optim_info.l1 > 0))
                    { // require loop reversal for correctness
                        optimization_info optim_info_reversal;
                        optim_info_reversal.type = optimization_type::MATRIX;
                        optim_info_reversal.node = new_node;
                        
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
                        optim_info_reversal.l0 = optim_info.l1;
                        matrix.at(optim_info.l1).at(optim_info.l1) = -1; 
                        optim_info_reversal.comps = involved_computations_skew;
                        optim_info_reversal.matrix = matrix;
                        optim_info_reversal.unimodular_transformation_type = 2;
                        new_ast->new_optims.push_back(optim_info_reversal);
                    }
                    states.push_back(new_ast);
                }
                ast.stage_isl_states();
            }
            // Explore the positive skewing
            ast.recover_isl_states();
            for(auto& param : positive_param_vector)
            {
                bool identity_params = false;
                // Check if the solver has returned identity parameters
                if(std::get<0>(param) == 1 && std::get<1>(param) == 0 &&
                   std::get<2>(param) == 0 && std::get<3>(param) == 1){
                    identity_params = true;
                }
                if(!identity_params){
                    // Copy the AST and add the positive skewing to the list of optimizations
                    syntax_tree* new_ast = new syntax_tree();
                    ast_node *new_node = ast.copy_and_return_node(*new_ast, commun_node);

                    optimization_info optim_info;
                    optim_info.type = optimization_type::MATRIX;
                    optim_info.node = new_node;

                    optim_info.nb_l = 2;
                    optim_info.l0 = new_node->depth;
                    optim_info.l1 = new_node->depth+1;
                    optim_info.l2 = 0;
                    optim_info.l0_fact = std::get<0>(param);
                    optim_info.l1_fact = std::get<1>(param);
                    optim_info.l2_fact = std::get<2>(param);
                    optim_info.l3_fact = std::get<3>(param);

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
                    matrix.at(optim_info.l0).at(optim_info.l0) = optim_info.l0_fact;
                    matrix.at(optim_info.l0).at(optim_info.l1) = optim_info.l1_fact;
                    matrix.at(optim_info.l1).at(optim_info.l0) = optim_info.l2_fact;
                    matrix.at(optim_info.l1).at(optim_info.l1) = optim_info.l3_fact;

                    optim_info.matrix = matrix;
                    optim_info.comps = involved_computations;
                    optim_info.unimodular_transformation_type = 3;
                    new_ast->new_optims.push_back(optim_info);
                    states.push_back(new_ast);
                }

            }
            ast.stage_isl_states();
        }
        ast.recover_isl_states();
    }
    // Removed 3d skewing until further tests
    // ast.stage_isl_states();
    // bool explre_3d_solver_skew = true;

    // shared_nodes = node->collect_shared_nodes_from_head();
    // if (shared_nodes.size() > 2)
    // {
    //     shared_nodes[0]->get_all_computations(involved_computations_skew);
    //     shared_nodes.pop_back(); //removes 2nd loop level, first is enough
    //     shared_nodes.pop_back(); //removes 3rd loop level, first is enough
    // }
    // else
    // {
    //     ast.recover_isl_states();
    //     explre_3d_solver_skew = false;
    // }
    // if(explre_3d_solver_skew){
    //     for (ast_node *commun_node : shared_nodes)
    //     {
    //         std::vector<std::string> loop_names = involved_computations_skew[0]->get_loop_level_names();

    //         std::string loop_name = loop_names[commun_node->depth];
    //         std::string loop_name_middle = loop_names[commun_node->depth + 1];
    //         std::string loop_name_inner = loop_names[commun_node->depth + 2];
    //         auto local_3d_skewing = ast.fct->skewing_local_3D_solver_positive(involved_computations,
    //                 var(loop_name),var(loop_name_middle),var(loop_name_inner)
    //                 );
    //         std::vector<std::tuple<int,int,int,int,int,int,int,int,int>> local_3d_param_vector = std::get<1>(local_3d_skewing);

    //         if(std::get<0>(local_3d_skewing).size() > 0){
    //             local_3d_param_vector.push_back(std::get<0>(local_3d_skewing)[0]);
    //         }
    //         if(std::get<2>(local_3d_skewing).size() > 0){
    //             local_3d_param_vector.push_back(std::get<2>(local_3d_skewing)[0]);
    //         }
    //         // Explore 3D skewing
    //         ast.recover_isl_states();
    //         for(auto& param : local_3d_param_vector)
    //         {
    //             bool identity_params = false;
    //             // Check if the solver has returned identity parameters
    //             if(std::get<0>(param) == 1 && std::get<1>(param) == 0 && std::get<2>(param) == 0 &&
    //                std::get<3>(param) == 0 && std::get<4>(param) == 1 && std::get<5>(param) == 0 &&
    //                std::get<6>(param) == 0 && std::get<7>(param) == 0 && std::get<8>(param) == 1 ){
    //                 identity_params = true;
    //             }
    //             if(!identity_params){
    //                 // Copy the AST and add Skewing to the list of optimizations
    //                 syntax_tree* new_ast = new syntax_tree();
    //                 ast_node *new_node = ast.copy_and_return_node(*new_ast, commun_node);

    //                 optimization_info optim_info;
    //                 optim_info.type = optimization_type::MATRIX;
    //                 optim_info.node = new_node;

    //                 optim_info.nb_l = 2;
    //                 optim_info.l0 = new_node->depth;
    //                 optim_info.l1 = new_node->depth+1;
    //                 optim_info.l2 = new_node->depth+2;
    //                 optim_info.l0_fact = std::get<0>(param);
    //                 optim_info.l1_fact = std::get<1>(param);
    //                 optim_info.l2_fact = std::get<2>(param);
    //                 optim_info.l3_fact = std::get<3>(param);
    //                 optim_info.l4_fact = std::get<4>(param);
    //                 optim_info.l5_fact = std::get<5>(param);
    //                 optim_info.l6_fact = std::get<6>(param);
    //                 optim_info.l7_fact = std::get<7>(param);
    //                 optim_info.l8_fact = std::get<8>(param);

    //                 std::vector <  std::vector<int> >  matrix(depth);
    //                 for(int l = 0; l<matrix.size(); l++){
    //                     matrix.at(l)= std::vector<int>(depth);
    //                     for(int c = 0; c<matrix.size(); c++){
    //                         if (l!=c ){
    //                             matrix.at(l).at(c) = 0;
    //                         }else{
    //                             matrix.at(l).at(c) = 1;
    //                         }
    //                     }
    //                 }

    //                 matrix.at(optim_info.l0).at(optim_info.l0) = optim_info.l0_fact;
    //                 matrix.at(optim_info.l0).at(optim_info.l1) = optim_info.l1_fact;
    //                 matrix.at(optim_info.l0).at(optim_info.l2) = optim_info.l2_fact;
    //                 matrix.at(optim_info.l1).at(optim_info.l0) = optim_info.l3_fact;
    //                 matrix.at(optim_info.l1).at(optim_info.l1) = optim_info.l4_fact;
    //                 matrix.at(optim_info.l1).at(optim_info.l2) = optim_info.l5_fact;
    //                 matrix.at(optim_info.l2).at(optim_info.l0) = optim_info.l6_fact;
    //                 matrix.at(optim_info.l2).at(optim_info.l1) = optim_info.l7_fact;
    //                 matrix.at(optim_info.l2).at(optim_info.l2) = optim_info.l8_fact;

    //                 optim_info.matrix = matrix;
    //                 optim_info.comps = involved_computations;
    //                 optim_info.unimodular_transformation_type = 3;
    //                 new_ast->new_optims.push_back(optim_info);
    //                 states.push_back(new_ast);
    //             }
    //         }
    //         ast.stage_isl_states();
    //     }
    //     ast.recover_isl_states();
    // }
    return states;
}

}
