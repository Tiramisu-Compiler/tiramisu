#include <tiramisu/auto_scheduler/states_generator.h>

namespace tiramisu::auto_scheduler
{

std::vector<syntax_tree*> exhaustive_generator::generate_states(syntax_tree const& ast, optimization_type optim)
{
    std::vector<syntax_tree*> states;
    
    switch(optim)
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

void exhaustive_generator::generate_fusions(std::vector<ast_node*> const& tree_level, std::vector<syntax_tree*>& states, syntax_tree const& ast)
{
    for (int i = 0; i < tree_level.size(); ++i)
    {
        if (tree_level[i]->unrolling_factor > 0)
            continue;

        for (int j = i + 1; j < tree_level.size(); ++j)
        {
            if (tree_level[j]->unrolling_factor > 0)
                continue;

            if (tree_level[i]->name == tree_level[j]->name)
            {
                syntax_tree* new_ast = new syntax_tree();
                ast_node *new_node = ast.copy_and_return_node(*new_ast, tree_level[i]);
                
                new_node->fused = true;
                new_node->fused_with = j;
                
                states.push_back(new_ast);
            }
        }
    }

    for (ast_node* node : tree_level)
        generate_fusions(node->children, states, ast);
}

void exhaustive_generator::generate_tilings(ast_node *node, std::vector<syntax_tree*>& states, syntax_tree const& ast)
{
    int branch_depth = node->get_branch_depth();
    
    // Generate tiling with dimension 2
    if (node->depth + 1 < branch_depth && node->unrolling_factor == 0)
    {
        for (int tiling_size1 : tiling_factors_list)
        {
            if (!can_split_iterator(node->up_bound - node->low_bound + 1, tiling_size1))
                continue;
                
            ast_node *node2 = node->children[0];
            for (int tiling_size2 : tiling_factors_list)
            {
                if (!can_split_iterator(node2->up_bound - node2->low_bound + 1, tiling_size2))
                    continue;
                    
                syntax_tree *new_ast = new syntax_tree();
                ast_node *new_node = ast.copy_and_return_node(*new_ast, node);
                    
                new_node->tiled = true;
                new_node->tiling_dim = 2;
                
                new_node->tiling_size1 = tiling_size1;
                new_node->tiling_size2 = tiling_size2;
                
                states.push_back(new_ast);
                
                // Generate tiling with dimension 3
                if (node->depth + 2 < branch_depth)
                {
                    ast_node *node3 = node2->children[0];
                    for (int tiling_size3 : tiling_factors_list)
                    {
                        if (!can_split_iterator(node3->up_bound - node3->low_bound + 1, tiling_size3))
                            continue;
                            
                        syntax_tree* new_ast = new syntax_tree();
                        ast_node *new_node = ast.copy_and_return_node(*new_ast, node);
                            
                        new_node->tiled = true;
                        new_node->tiling_dim = 3;
                        
                        new_node->tiling_size1 = tiling_size1;
                        new_node->tiling_size2 = tiling_size2;
                        new_node->tiling_size3 = tiling_size3;
                            
                        states.push_back(new_ast);
                    }
                }
            }
        }
    }
    
    for (ast_node *child : node->children)
        generate_tilings(child, states, ast);
}

void exhaustive_generator::generate_interchanges(ast_node *node, std::vector<syntax_tree*>& states, syntax_tree const& ast)
{
    if (node->unrolling_factor == 0)
    {
        int branch_depth = node->get_branch_depth();
        
        for (int i = node->depth + 1; i < branch_depth; ++i)
        {
            syntax_tree* new_ast = new syntax_tree();
            ast_node *new_node = ast.copy_and_return_node(*new_ast, node);
            
            new_node->interchanged = true;
            new_node->interchanged_with = i;
            
            states.push_back(new_ast);
        }
    }
    
    for (ast_node *child : node->children)
        generate_interchanges(child, states, ast);
}

void exhaustive_generator::generate_unrollings(ast_node *node, std::vector<syntax_tree*>& states, syntax_tree const& ast)
{
    if (node->unrolling_factor == 0)
    {
        for (int unrolling_factor : unrolling_factors_list)
        {
            if (node->up_bound - node->low_bound + 1 != unrolling_factor && 
                !can_split_iterator(node->up_bound - node->low_bound + 1, unrolling_factor))
                continue;
                
            syntax_tree* new_ast = new syntax_tree();
            ast_node *new_node = ast.copy_and_return_node(*new_ast, node);
            
            new_node->unrolling_factor = unrolling_factor;

            states.push_back(new_ast);
        }
    }
    
    for (ast_node *child : node->children)
        generate_unrollings(child, states, ast);
}

}
