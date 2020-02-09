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

    nodes.back()->comp.push_back(comp);

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
    new_cg.next_optimization = next_optimization;

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
    new_node->comp = comp;
    
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

int cg_node::get_branch_depth() const
{
    int ret = depth;
    const cg_node *node = this;
    
    while (node->children.size() == 1)
    {
        ret++;
        node = node->children[0];
    }
    
    if (node->children.size() == 0)
        return ret + 1;
    
    return ret;
}

}
