#include <tiramisu/auto_scheduler/dnn_accesses.h>

namespace tiramisu::auto_scheduler
{

dnn_accesses::dnn_accesses(tiramisu::computation *comp, int nb_iterators, tiramisu::function *fct)
    : comp(comp), nb_iterators(nb_iterators)
{
    create_accesses(comp->get_expr());
}

void dnn_accesses::create_accesses(tiramisu::expr const& e)
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
        accesses_list.push_back(dnn_access_matrix(nb_iterators, e, comp));
        return ;
    }
    
    // We have an operation, we explore its operands
    for (int i = 0; i < e.get_n_arg(); ++i)
        create_accesses(e.get_operand(i));
}

dnn_access_matrix::dnn_access_matrix(int nb_iterators, int nb_dims)
    : nb_iterators(nb_iterators), nb_dims(nb_dims), matrix(nb_dims), buffer_id(0)
{
    for (int i = 0; i < nb_dims; ++i)
        matrix[i] = std::vector<int>(nb_iterators + 1, 0);
}
    
dnn_access_matrix::dnn_access_matrix(int nb_iterators, tiramisu::expr const& e, tiramisu::computation *comp)
    : dnn_access_matrix(nb_iterators, e.get_access().size())
{
    this->comp = comp;
    std::vector<tiramisu::expr> const& acc_vector = e.get_access();

    for (int i = 0; i < acc_vector.size(); ++i)
        fill_matrix_row(i, acc_vector[i]);
    
    buffer_name = e.get_name();
}

void dnn_access_matrix::fill_matrix_row(int i, tiramisu::expr const& e, bool minus)
{
    if (e.get_expr_type() == tiramisu::e_op)
    {
        // We got : access1 +- access2
        if (e.get_op_type() == o_add || e.get_op_type() == o_sub)
        {
            minus = false;
            if (e.get_op_type() == o_sub)
                minus = true;
                
            fill_matrix_row(i, e.get_operand(0), false);
            fill_matrix_row(i, e.get_operand(1), minus);
        }
        
        // We got : coefficient * iterator
        else if (e.get_op_type() == o_mul)
        {
            int coeff = e.get_operand(0).get_int32_value();
            int it_pos = comp->get_loop_level_number_from_dimension_name(e.get_operand(1).get_name());
            
            if (minus)
                matrix[i][it_pos] = -coeff;
            else
                matrix[i][it_pos] = coeff;
        }
    }
    
    // Access coefficient == 1
    else if (e.get_expr_type() == tiramisu::e_var)
    {
        int it_pos = comp->get_loop_level_number_from_dimension_name(e.get_name());
        matrix[i][it_pos] = 1;
        
        if (minus)
            matrix[i][it_pos] = -1;
    }
    
    // Constant increment
    else if (e.get_expr_type() == tiramisu::e_val)
    {
        if (minus)
            matrix[i][nb_iterators] = -e.get_int32_value();
        else
            matrix[i][nb_iterators] = e.get_int32_value();
    }
}

dnn_schedule::dnn_schedule(int nb_iterators, std::vector<optimization_info> const& optims_list)
    : dnn_schedule(nb_iterators)
{
    for (optimization_info const& optim_info : optims_list)
    {
        switch (optim_info.type)
        {
            case optimization_type::TILING:
                if (optim_info.nb_l == 2)
                {
                    tiled[optim_info.l0] = true;
                    tiled[optim_info.l1] = true;
                    
                    tiling_fact[optim_info.l0] = optim_info.l0_fact;
                    tiling_fact[optim_info.l1] = optim_info.l1_fact;
                }
                
                else if (optim_info.nb_l == 3)
                {
                    tiled[optim_info.l0] = true;
                    tiled[optim_info.l1] = true;
                    tiled[optim_info.l2] = true;
                    
                    tiling_fact[optim_info.l0] = optim_info.l0_fact;
                    tiling_fact[optim_info.l1] = optim_info.l1_fact;
                    tiling_fact[optim_info.l2] = optim_info.l2_fact;
                }
                break;
                
            case optimization_type::INTERCHANGE:
                interchanged[optim_info.l0] = true;
                interchanged[optim_info.l1] = true;
                break;
                
            case optimization_type::UNROLLING:
                unrolling_fact = optim_info.l0_fact;
                break;
                
            default:
                break;
        }
    }
}

std::vector<dnn_iterator> 
dnn_iterator::get_iterators_from_computation(tiramisu::computation const& comp)
{
    std::vector<dnn_iterator> iters_list;
    
    isl_set *iter_domain = comp.get_iteration_domain();
    int nb_iterators = isl_set_dim(iter_domain, isl_dim_set);
    
    for (int i = 0; i < nb_iterators; ++i)
    {
        std::string name = isl_set_get_dim_name(iter_domain, isl_dim_set, i);
        int low_bound = utility::get_bound(iter_domain, i, false).get_int_val();
        int up_bound = utility::get_bound(iter_domain, i, true).get_int_val();
        
        iters_list.push_back(dnn_iterator(name, low_bound, up_bound));
    }
    
    return iters_list;
}

}
