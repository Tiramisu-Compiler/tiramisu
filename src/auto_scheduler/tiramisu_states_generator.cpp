#include <tiramisu/auto_scheduler/states_generator.h>

namespace tiramisu::auto_scheduler
{

std::vector<computation_graph*> exhaustive_generator::generate_states(computation_graph const& cg)
{
    std::vector<computation_graph*> states;
    optimization_type next_optimization = optimizations_order[cg.next_optim_index];
    
    switch(next_optimization)
    {
        case optimization_type::FUSION:
            if (apply_fusion)
                generate_fusions(cg.roots, states, cg);

            break;

        case optimization_type::TILING:
            if (apply_tiling)
                for (cg_node *root : cg.roots)
                    generate_tilings(root, states, cg);
            
            break;

        case optimization_type::INTERCHANGE:
            if (apply_interchange)
                for (cg_node *root : cg.roots)
                    generate_interchanges(root, states, cg);
                    
            break;

        case optimization_type::UNROLLING:
            if (apply_unrolling)
                for (cg_node *root : cg.roots)
                    generate_unrollings(root, states, cg);
                    
            break;

        default:
            break;
    }
    
    return states;
}

void exhaustive_generator::generate_fusions(std::vector<cg_node*> const& tree_level, std::vector<computation_graph*>& states, computation_graph const& cg)
{
    for (int i = 0; i < tree_level.size(); ++i)
    {
        if (tree_level[i]->unrolling_factor > 0)
            continue;

        for (int j = i + 1; j < tree_level.size(); ++j)
        {
            if (tree_level[j]->unrolling_factor > 0)
                continue;

            int depth = tree_level[i]->depth;
            if (tree_level[i]->iterators[depth].name == tree_level[j]->iterators[depth].name)
            {
                computation_graph* new_cg = new computation_graph();
                cg_node *new_node = cg.copy_and_return_node(*new_cg, tree_level[i]);
                
                new_node->fused = true;
                new_node->fused_with = j;
                
                new_cg->next_optim_index = (new_cg->next_optim_index + 1) % NB_OPTIMIZATIONS;
                states.push_back(new_cg);
            }
        }
    }

    for (cg_node* node : tree_level)
        generate_fusions(node->children, states, cg);
}

void exhaustive_generator::generate_tilings(cg_node *node, std::vector<computation_graph*>& states, computation_graph const& cg)
{
    int branch_depth = node->get_branch_depth();
    
    // Generate tiling with dimension 2
    if (node->depth + 1 < branch_depth && node->unrolling_factor == 0)
    {
        iterator it1 = node->iterators.back();
        for (int tiling_size1 : tiling_factors_list)
        {
            if (!can_split_iterator(it1.up_bound - it1.low_bound + 1, tiling_size1))
                continue;
                
            iterator it2 = node->children[0]->iterators.back();
            for (int tiling_size2 : tiling_factors_list)
            {
                if (!can_split_iterator(it2.up_bound - it2.low_bound + 1, tiling_size2))
                    continue;
                    
                computation_graph *new_cg = new computation_graph();
                cg_node *new_node = cg.copy_and_return_node(*new_cg, node);
                    
                new_node->tiled = true;
                new_node->tiling_dim = 2;
                
                new_node->tiling_size1 = tiling_size1;
                new_node->tiling_size2 = tiling_size2;
                
                new_cg->next_optim_index = (new_cg->next_optim_index + 1) % NB_OPTIMIZATIONS;
                states.push_back(new_cg);
                
                // Generate tiling with dimension 3
                if (node->depth + 2 < branch_depth)
                {
                    iterator it3 = node->children[0]->children[0]->iterators.back();
                    for (int tiling_size3 : tiling_factors_list)
                    {
                        if (!can_split_iterator(it3.up_bound - it3.low_bound + 1, tiling_size3))
                            continue;
                            
                        computation_graph* new_cg = new computation_graph();
                        cg_node *new_node = cg.copy_and_return_node(*new_cg, node);
                            
                        new_node->tiled = true;
                        new_node->tiling_dim = 3;
                        
                        new_node->tiling_size1 = tiling_size1;
                        new_node->tiling_size2 = tiling_size2;
                        new_node->tiling_size3 = tiling_size3;
                            
                        new_cg->next_optim_index = (new_cg->next_optim_index + 1) % NB_OPTIMIZATIONS;
                        states.push_back(new_cg);
                    }
                }
            }
        }
    }
    
    for (cg_node *child : node->children)
        generate_tilings(child, states, cg);
}

void exhaustive_generator::generate_interchanges(cg_node *node, std::vector<computation_graph*>& states, computation_graph const& cg)
{
    if (node->unrolling_factor == 0)
    {
        int branch_depth = node->get_branch_depth();
        
        for (int i = node->depth + 1; i < branch_depth; ++i)
        {
            computation_graph* new_cg = new computation_graph();
            cg_node *new_node = cg.copy_and_return_node(*new_cg, node);
            
            new_node->interchanged = true;
            new_node->interchanged_with = i;
            
            new_cg->next_optim_index = (new_cg->next_optim_index + 1) % NB_OPTIMIZATIONS;
            states.push_back(new_cg);
        }
    }
    
    for (cg_node *child : node->children)
        generate_interchanges(child, states, cg);
}

void exhaustive_generator::generate_unrollings(cg_node *node, std::vector<computation_graph*>& states, computation_graph const& cg)
{
    if (node->unrolling_factor == 0)
    {
        for (int unrolling_factor : unrolling_factors_list)
        {
            computation_graph* new_cg = new computation_graph();
            cg_node *new_node = cg.copy_and_return_node(*new_cg, node);
            
            new_node->unrolling_factor = unrolling_factor;

            new_cg->next_optim_index = (new_cg->next_optim_index + 1) % NB_OPTIMIZATIONS;
            states.push_back(new_cg);
        }
    }
    
    for (cg_node *child : node->children)
        generate_unrollings(child, states, cg);
}

}
