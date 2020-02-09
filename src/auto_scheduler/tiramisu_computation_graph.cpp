#include <tiramisu/auto_scheduler/computation_graph.h>

namespace tiramisu::auto_scheduler
{
computation_graph::computation_graph(tiramisu::function *fct)
{
    const std::vector<computation*> computations = fct->get_computations();
    
    for (computation *comp : computations) 
    {
        if (comp->get_expr().get_expr_type() == e_none)
            continue;
            
        cg_node *node = new cg_node();
        node->comp = comp;
        
        // Get computation iterators
        isl_set *iter_domain = comp->get_iteration_domain();
        
        for (int i = 0; i < isl_set_dim(iter_domain, isl_dim_set); ++i)
        {
            std::string iter_name = isl_set_get_dim_name(iter_domain, isl_dim_set, i);
            int low_bound = utility::get_bound(iter_domain, i, false).get_int_val();
            int up_bound = utility::get_bound(iter_domain, i, true).get_int_val();
            
            node->iterators.push_back(iterator(iter_name, low_bound, up_bound));
        }

        roots.push_back(node);
    }
}
}