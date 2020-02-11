#include <tiramisu/auto_scheduler/computation_graph.h>

namespace tiramisu::auto_scheduler
{
computation_graph::computation_graph(tiramisu::function *fct)
    : fct(fct)
{
    const std::vector<computation*> computations = fct->get_computations();
    
    for (computation *comp : computations) 
    {
        if (comp->get_expr().get_expr_type() == e_none)
            continue;

        roots.push_back(computation_to_cg_node(comp));
    }
}

cg_node* computation_graph::computation_to_cg_node(tiramisu::computation *comp)
{
    std::vector<cg_node*> nodes;

    // Get computation iterators
    isl_set *iter_domain = comp->get_iteration_domain();
    int nb_iterators = isl_set_dim(iter_domain, isl_dim_set);
    std::vector<iterator> iterators;
        
    for (int i = 0; i < nb_iterators; ++i)
    {
        std::string iter_name = isl_set_get_dim_name(iter_domain, isl_dim_set, i);
        int low_bound = utility::get_bound(iter_domain, i, false).get_int_val();
        int up_bound = utility::get_bound(iter_domain, i, true).get_int_val();
            
        iterators.push_back(iterator(iter_name, low_bound, up_bound));

        cg_node *node = new cg_node();
        node->depth = i;
        node->iterators = iterators;

        nodes.push_back(node);
    }

    for (int i = 0; i < nodes.size() - 1; ++i)
        nodes[i]->children.push_back(nodes[i + 1]);

    nodes.back()->computations.push_back(comp);

    return nodes[0];
}

cg_node* computation_graph::copy_and_return_node(computation_graph& new_cg, cg_node *node_to_find) const
{
    cg_node *ret_node = nullptr;

    for (cg_node *root : roots) 
    {
        cg_node *new_node = new cg_node();

        cg_node *tmp = root->copy_and_return_node(new_node, node_to_find);
        if (tmp != nullptr)
            ret_node = tmp;

        new_cg.roots.push_back(new_node);
    }

    new_cg.fct = fct;
    new_cg.next_optim_index = next_optim_index;

    return ret_node;
}

cg_node* cg_node::copy_and_return_node(cg_node *new_node, cg_node *node_to_find) const
{
    cg_node *ret_node = nullptr;

    if (this == node_to_find)
        ret_node = new_node;

    for (cg_node *child : children)
    {
        cg_node *new_child = new cg_node();

        cg_node *tmp = child->copy_and_return_node(new_child, node_to_find);
        if (tmp != nullptr)
            ret_node = tmp;

        new_node->children.push_back(new_child);
    }

    new_node->depth = depth;
    new_node->iterators = iterators;
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

void computation_graph::transform_computation_graph()
{
    transform_computation_graph_by_fusion(roots);

    for (cg_node *node : roots)
    {
        transform_computation_graph_by_tiling(node);
        transform_computation_graph_by_interchange(node);
    }
}

void computation_graph::transform_computation_graph_by_fusion(std::vector<cg_node*>& tree_level)
{
    for (int i = 0; i < tree_level.size(); ++i)
    {
        cg_node *node_a = tree_level[i];
        if (!node_a->fused)
            continue;

        cg_node *node_b = tree_level[node_a->fused_with];

        for (cg_node *child : node_b->children)
            node_a->children.push_back(child);

        for (tiramisu::computation *comp : node_b->computations)
            node_a->computations.push_back(comp);

        node_a->fused = false;
        tree_level.erase(tree_level.begin() + node_a->fused_with);
        --i;
    }
    
    for (cg_node *node : tree_level)
        transform_computation_graph_by_fusion(node->children);
}

void computation_graph::transform_computation_graph_by_tiling(cg_node *node)
{
    if (node->tiled)
    {
        if (node->tiling_dim == 2)
        {
            cg_node *i_outer = node;
            cg_node *j_outer = new cg_node();
            cg_node *i_inner = new cg_node();
            cg_node *j_inner = node->children[0];
            
            i_outer->children[0] = j_outer;
            j_outer->children.push_back(i_inner);
            i_inner->children.push_back(j_inner);
            
            i_inner->name = i_outer->name + "_inner";
            i_outer->name = i_outer->name + "_outer";
            j_outer->name = j_inner->name + "_outer";
            j_inner->name = j_inner->name + "_inner";
        }
        
        else if (node->tiling_dim == 3)
        {
        
        }
        
        node->tiled = false;
    }
    
    for (cg_node *child : node->children)
        transform_computation_graph_by_tiling(node);
}

void computation_graph::transform_computation_graph_by_interchange(cg_node *node)
{
    if (node->interchanged)
    {
        int it1_depth = node->depth,
            it2_depth = node->interchanged_with;
            
        cg_node *tmp_node = node;
        for (int i = it1_depth; i < it2_depth; ++i)
            tmp_node = tmp_node->children[0];
            
        iterator it1 = node->iterators.back();
        iterator it2 = tmp_node->iterators.back();
            
        interchange_iterators(node, it1, it1_depth, it2, it2_depth);
        node->interchanged = false;
    }
    
    for (cg_node *child : node->children)
        transform_computation_graph_by_interchange(child);
}

void computation_graph::interchange_iterators(cg_node *node, iterator const& it1, int it1_depth, iterator const& it2, int it2_depth)
{
    if (node->depth >= it1_depth)
        node->iterators[it1_depth] = it2;
        
    if (node->depth >= it2_depth)
        node->iterators[it2_depth] = it1;
        
    for (cg_node *child : node->children)
        interchange_iterators(child, it1, it1_depth, it2, it2_depth);
}

int cg_node::get_branch_depth() const
{
    int ret = depth;
    const cg_node *node = this;
    
    while (node->children.size() == 1 && node->computations.size() == 0)
    {
        ret++;
        node = node->children[0];
    }
    
    if (node->children.size() == 0 || node->computations.size() > 0)
        return ret + 1;
    
    return ret;
}

void computation_graph::print_graph() const
{
    for (cg_node *root : roots)
        root->print_node();
}

void cg_node::print_node() const
{
    for (int i = 0; i < depth; ++i)
        std::cout << "\t";

    iterator it = iterators.back();

    std::cout << "for " << it.low_bound << " <= " << it.name << " < " << it.up_bound + 1 << std::endl;
    
    for (tiramisu::computation* comp : computations) 
    {
        for (int i = 0; i < depth + 1; ++i)
            std::cout << "\t";
            
        std::cout << comp->get_name() << std::endl;
    }

    for (cg_node *child : children)
        child->print_node();
}

}