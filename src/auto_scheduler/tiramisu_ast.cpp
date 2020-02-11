#include <tiramisu/auto_scheduler/ast.h>

namespace tiramisu::auto_scheduler
{
syntax_tree::syntax_tree(tiramisu::function *fct)
    : fct(fct)
{
    const std::vector<computation*> computations = fct->get_computations();
    
    for (computation *comp : computations) 
    {
        if (comp->get_expr().get_expr_type() == e_none)
            continue;

        roots.push_back(computation_to_ast_node(comp));
    }
}

ast_node* syntax_tree::computation_to_ast_node(tiramisu::computation *comp)
{
    std::vector<ast_node*> nodes;

    // Get computation iterators
    isl_set *iter_domain = comp->get_iteration_domain();
    int nb_iterators = isl_set_dim(iter_domain, isl_dim_set);
        
    for (int i = 0; i < nb_iterators; ++i)
    {
        ast_node *node = new ast_node();
        
        node->depth = i;
        node->name = isl_set_get_dim_name(iter_domain, isl_dim_set, i);
        node->low_bound = utility::get_bound(iter_domain, i, false).get_int_val();
        node->up_bound = utility::get_bound(iter_domain, i, true).get_int_val();

        nodes.push_back(node);
    }

    for (int i = 0; i < nodes.size() - 1; ++i)
        nodes[i]->children.push_back(nodes[i + 1]);

    nodes.back()->computations.push_back(comp);

    return nodes[0];
}

ast_node* syntax_tree::copy_and_return_node(syntax_tree& new_ast, ast_node *node_to_find) const
{
    ast_node *ret_node = nullptr;

    for (ast_node *root : roots) 
    {
        ast_node *new_node = new ast_node();

        ast_node *tmp = root->copy_and_return_node(new_node, node_to_find);
        if (tmp != nullptr)
            ret_node = tmp;

        new_ast.roots.push_back(new_node);
    }

    new_ast.fct = fct;
    new_ast.next_optim_index = next_optim_index;

    return ret_node;
}

ast_node* ast_node::copy_and_return_node(ast_node *new_node, ast_node *node_to_find) const
{
    ast_node *ret_node = nullptr;

    if (this == node_to_find)
        ret_node = new_node;

    for (ast_node *child : children)
    {
        ast_node *new_child = new ast_node();

        ast_node *tmp = child->copy_and_return_node(new_child, node_to_find);
        if (tmp != nullptr)
            ret_node = tmp;

        new_node->children.push_back(new_child);
    }

    new_node->depth = depth;
    new_node->name = name;
    new_node->low_bound = low_bound;
    new_node->up_bound = up_bound;
    new_node->computations = computations;
    
    new_node->fused = fused;
    new_node->fused_with = fused_with;
    
    new_node->tiled = tiled;
    new_node->tiling_dim = tiling_dim;
    new_node->tiling_size1 = tiling_size1;
    new_node->tiling_size2 = tiling_size2;
    new_node->tiling_size3 = tiling_size3;
    
    new_node->interchanged = interchanged;
    new_node->interchanged_with = interchanged_with;
    
    new_node->unrolling_factor = unrolling_factor;

    return ret_node;
}

void syntax_tree::transform_ast()
{
    transform_ast_by_fusion(roots);

    for (ast_node *node : roots)
    {
        transform_ast_by_tiling(node);
        transform_ast_by_interchange(node);
    }
}

void syntax_tree::transform_ast_by_fusion(std::vector<ast_node*>& tree_level)
{
    for (int i = 0; i < tree_level.size(); ++i)
    {
        ast_node *node_a = tree_level[i];
        if (!node_a->fused)
            continue;

        ast_node *node_b = tree_level[node_a->fused_with];

        for (ast_node *child : node_b->children)
            node_a->children.push_back(child);

        for (tiramisu::computation *comp : node_b->computations)
            node_a->computations.push_back(comp);

        node_a->fused = false;
        tree_level.erase(tree_level.begin() + node_a->fused_with);
        --i;
    }
    
    for (ast_node *node : tree_level)
        transform_ast_by_fusion(node->children);
}

void syntax_tree::transform_ast_by_tiling(ast_node *node)
{
    if (node->tiled)
    {
        if (node->tiling_dim == 2)
        {
            ast_node *i_outer = node;
            ast_node *j_outer = new ast_node();
            
            ast_node *i_inner = new ast_node();
            ast_node *j_inner = node->children[0];
            
            i_outer->children[0] = j_outer;
            j_outer->children.push_back(i_inner);
            i_inner->children.push_back(j_inner);
            
            i_inner->name = i_outer->name + "_inner";
            i_outer->name = i_outer->name + "_outer";
            
            j_outer->name = j_inner->name + "_outer";
            j_inner->name = j_inner->name + "_inner";
            
            i_outer->low_bound = 0;
            i_outer->up_bound = (i_outer->up_bound - i_outer->low_bound + 1) / node->tiling_size1 - 1;
            
            j_outer->low_bound = 0;
            j_outer->up_bound = (j_inner->up_bound - j_inner->low_bound + 1) / node->tiling_size2 - 1;
            
            i_inner->low_bound = 0;
            i_inner->up_bound = node->tiling_size1 - 1;
            
            j_inner->low_bound = 0;
            j_inner->up_bound = node->tiling_size2 - 1;
        }
        
        else if (node->tiling_dim == 3)
        {
            ast_node *i_outer = node;
            ast_node *j_outer = new ast_node();
            ast_node *k_outer = new ast_node();
            
            ast_node *i_inner = new ast_node();
            ast_node *j_inner = node->children[0];
            ast_node *k_inner = j_inner->children[0];
            
            i_outer->children[0] = j_outer;
            j_outer->children.push_back(k_outer);
            k_outer->children.push_back(i_inner);
            i_inner->children.push_back(j_inner);
            j_inner->children[0] = k_inner;
            
            i_inner->name = i_outer->name + "_inner";
            i_outer->name = i_outer->name + "_outer";
            
            j_outer->name = j_inner->name + "_outer";
            j_inner->name = j_inner->name + "_inner";
            
            k_outer->name = k_inner->name + "_outer";
            k_inner->name = k_inner->name + "_inner";
            
            i_outer->low_bound = 0;
            i_outer->up_bound = (i_outer->up_bound - i_outer->low_bound + 1) / node->tiling_size1 - 1;
            
            j_outer->low_bound = 0;
            j_outer->up_bound = (j_inner->up_bound - j_inner->low_bound + 1) / node->tiling_size2 - 1;
            
            k_outer->low_bound = 0;
            k_outer->up_bound = (k_inner->up_bound - k_inner->low_bound + 1) / node->tiling_size2 - 1;
            
            i_inner->low_bound = 0;
            i_inner->up_bound = node->tiling_size1 - 1;
            
            j_inner->low_bound = 0;
            j_inner->up_bound = node->tiling_size2 - 1;
            
            k_inner->low_bound = 0;
            k_inner->up_bound = node->tiling_size3 - 1;
        }
        
        node->tiled = false;
        node->update_depth(node->depth);
    }
    
    for (ast_node *child : node->children)
        transform_ast_by_tiling(child);
}

void syntax_tree::transform_ast_by_interchange(ast_node *node)
{
    if (node->interchanged)
    {
        ast_node *tmp_node = node;
        for (int i = node->depth; i < node->interchanged_with; ++i)
            tmp_node = tmp_node->children[0];
            
        std::string tmp_str;
        tmp_str = node->name;
        node->name = tmp_node->name;
        tmp_node->name = tmp_str;
            
        int tmp_int;
        tmp_int = node->low_bound;
        node->low_bound = tmp_node->low_bound;
        tmp_node->low_bound = tmp_int;
        
        tmp_int = node->up_bound;
        node->up_bound = tmp_node->up_bound;
        tmp_node->up_bound = tmp_int;
            
        node->interchanged = false;
    }
    
    for (ast_node *child : node->children)
        transform_ast_by_interchange(child);
}

int ast_node::get_branch_depth() const
{
    int ret = depth;
    const ast_node *node = this;
    
    while (node->children.size() == 1 && node->computations.size() == 0)
    {
        ret++;
        node = node->children[0];
    }
    
    if (node->children.size() == 0 || node->computations.size() > 0)
        return ret + 1;
    
    return ret;
}

void ast_node::update_depth(int depth)
{
    this->depth = depth;
    
    for (ast_node *child : children)
        child->update_depth(depth + 1);
}

void syntax_tree::print_graph() const
{
    for (ast_node *root : roots)
        root->print_node();
}

void ast_node::print_node() const
{
    for (int i = 0; i < depth; ++i)
        std::cout << "\t";

    std::cout << "for " << low_bound << " <= " << name << " < " << up_bound + 1 << std::endl;
    
    for (tiramisu::computation* comp : computations) 
    {
        for (int i = 0; i < depth + 1; ++i)
            std::cout << "\t";
            
        std::cout << comp->get_name() << std::endl;
    }

    for (ast_node *child : children)
        child->print_node();
}

}
