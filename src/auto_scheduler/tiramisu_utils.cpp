#include <tiramisu/auto_scheduler/utils.h>
#include <tiramisu/auto_scheduler/ast.h>
#include <tiramisu/block.h>

namespace tiramisu::auto_scheduler
{

void dnn_access_matrix::create_accesses(tiramisu::expr const& e, int nb_iterators,
                                        std::vector<dnn_access_matrix>& accesses, 
                                        tiramisu::computation *comp)
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
        accesses.push_back(dnn_access_matrix(nb_iterators, e, comp));
        return ;
    }
    
    // We have an operation, we explore its operands
    for (int i = 0; i < e.get_n_arg(); ++i)
        create_accesses(e.get_operand(i), nb_iterators, accesses, comp);
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
                
            fill_matrix_row(i, e.get_operand(0), minus);
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

void dnn_access_matrix::set_buffer_id(tiramisu::function *fct)
{
    buffer_id = 0;
    for (auto const& map_el : fct->get_buffers())
    {
        if (map_el.first == buffer_name)
            break;
            
        buffer_id++;
    }
}

void apply_optimizations(syntax_tree const& ast)
{
    for (optimization_info const& optim_info : ast.previous_optims)
        apply_optimizations(optim_info);
        
    for (optimization_info const& optim_info : ast.new_optims)
        apply_optimizations(optim_info);
            
    // Apply fusions
    if (ast.new_optims.size() > 0 && ast.new_optims.back().type == optimization_type::FUSION)
    {
        syntax_tree *ast_copy = ast.copy_ast();
        ast_copy->transform_ast_by_fusion(ast.new_optims.back());
        
        apply_fusions(*ast_copy);
        delete ast_copy;
    }
    
    else
        apply_fusions(ast);
}

void apply_optimizations(optimization_info const& optim_info)
{
    // tiramisu::block can be used to apply the same optimization to a set of computations
    tiramisu::block block(optim_info.comps);
        
    switch (optim_info.type)
    {
        case optimization_type::TILING:
            if (optim_info.nb_l == 2)
                block.tile(optim_info.l0, optim_info.l1, 
                           optim_info.l0_fact, optim_info.l1_fact);
                
            else if (optim_info.nb_l == 3)
                block.tile(optim_info.l0, optim_info.l1, optim_info.l2,
                           optim_info.l0_fact, optim_info.l1_fact, optim_info.l2_fact);
            break;
                
        case optimization_type::INTERCHANGE:
            block.interchange(optim_info.l0, optim_info.l1);
            break;
                
        case optimization_type::UNROLLING:
            block.unroll(optim_info.l0, optim_info.l0_fact);
            break;
                
        default:
            break;
    }
}

void apply_fusions(syntax_tree const& ast)
{
    tiramisu::computation *next_comp = nullptr;
    
    for (ast_node *root : ast.roots)
        next_comp = apply_fusions(root, next_comp, tiramisu::computation::root_dimension);
}

tiramisu::computation* apply_fusions(ast_node *node, tiramisu::computation *last_comp, int dimension)
{
    tiramisu::computation *next_comp;
    
    if (node->computations.size() > 0)
    {
        next_comp = node->computations[0];
        
        if (last_comp != nullptr)
            next_comp->after(*last_comp, dimension);
            
        last_comp = next_comp;
        for (int i = 1; i < node->computations.size(); ++i)
        {
            next_comp = node->computations[i];
            next_comp->after(*last_comp, node->depth);
        }
    }
    
    else
        next_comp = last_comp;
    
    int new_dimension = dimension;
    if (node->children.size() >= 2)
        new_dimension = node->depth;
    
    for (ast_node *child : node->children)
        next_comp = apply_fusions(child, next_comp, new_dimension);
    
    return next_comp;
}

}
