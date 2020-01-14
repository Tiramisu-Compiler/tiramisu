#include <tiramisu/auto_scheduler/core.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>

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
        
        // Build the access matrices
        tiramisu::expr const& c_expr = comp->get_expr();
        build_access_matrices(c_expr, node);
        
        roots.push_back(node);
    }
}

void computation_graph::build_access_matrices(tiramisu::expr const& e, cg_node *node)
{
    // Not an operation, stop the search
    if (e.get_expr_type() != tiramisu::e_op)
        return ;
        
    // We have an access, so we add its access matrix
    if (e.get_op_type() == tiramisu::o_access || 
        e.get_op_type() == tiramisu::o_lin_index ||
        e.get_op_type() == tiramisu::o_address_of || 
        e.get_op_type() == tiramisu::o_dummy ||
        e.get_op_type() == tiramisu::o_buffer)
    {
        add_access_matrix(e, node);
        return ;
    }
    
    // We have an operation, we explore its operands
    for (int i = 0; i < e.get_n_arg(); ++i)
        build_access_matrices(e.get_operand(i), node);
}

void computation_graph::add_access_matrix(tiramisu::expr const& acc_expr, cg_node *node)
{
    std::vector<tiramisu::expr> const& acc_vector = acc_expr.get_access();
    
    int nb_iterators = node->iterators.size();
    int nb_dims = acc_vector.size();
    access_matrix acc_matrix(nb_iterators, nb_dims);
    
    // Fill the access matrix
    for (int i = 0; i < acc_vector.size(); ++i)
        fill_matrix_raw(acc_vector[i], node, acc_matrix.matrix[i]);
    
    // Get the access ID
    auto it = std::find(accesses_names.begin(), accesses_names.end(), acc_expr.get_name());
    
    if (it == accesses_names.end())
    {
        acc_matrix.buffer_id = accesses_names.size();
        accesses_names.push_back(acc_expr.get_name());
    }
    
    else
        acc_matrix.buffer_id = std::distance(accesses_names.begin(), it);
     
    node->accesses.push_back(acc_matrix);
}

void computation_graph::fill_matrix_raw(tiramisu::expr const& e, cg_node *node, std::vector<int>& acc_raw, bool minus)
{
    if (e.get_expr_type() == tiramisu::e_op)
    {
        // We got : access1 +- access2
        if (e.get_op_type() == o_add || e.get_op_type() == o_sub)
        {
            bool minus = false;
            if (e.get_op_type() == o_sub)
                minus = true;
                
            fill_matrix_raw(e.get_operand(0), node, acc_raw, minus);
            fill_matrix_raw(e.get_operand(1), node, acc_raw, minus);
        }
        
        // We got : coefficient * iterator
        else if (e.get_op_type() == o_mul)
        {
            int coeff = e.get_operand(0).get_int32_value();
            int iterator_pos = node->get_iterator_pos_by_name(e.get_operand(1).get_name());
            
            if (minus)
                acc_raw[iterator_pos] = -coeff;
            else
                acc_raw[iterator_pos] = coeff;
        }
        
        else
            return;
    }
    
    // Access coefficient == 1
    else if (e.get_expr_type() == tiramisu::e_var)
    {
        int iterator_pos = node->get_iterator_pos_by_name(e.get_name());
        acc_raw[iterator_pos] = 1;
        
        if (minus)
            acc_raw[iterator_pos] = -1;
    }
    
    // Constant increment
    else if (e.get_expr_type() == tiramisu::e_val)
    {
        if (minus)
            acc_raw[acc_raw.size() - 1] = -e.get_int32_value();
        else
            acc_raw[acc_raw.size() - 1] = e.get_int32_value();
    }
    
    else
        return ;
}

auto_scheduler::auto_scheduler(search_method *searcher, evaluator *eval_func, 
                               tiramisu::function *fct)
                               
    : cg(fct), searcher(searcher), eval_func(eval_func), fct(fct)
{
    searcher->set_eval_func(eval_func);
    
    std::cout << eval_func->evaluate(cg, schedule_info(3)) << std::endl;
}

}
