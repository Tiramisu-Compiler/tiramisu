#include <tiramisu/auto_scheduler/dnn_accesses.h>

namespace tiramisu::auto_scheduler
{

std::vector<dnn_iterator> 
dnn_iterator::get_iterators_from_computation(tiramisu::computation const& comp)
{
    std::vector<dnn_iterator> iters_list;
    
    // In the next lines, we use Tiramisu internals to get information about iterators
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
    
    // We get the buffer name that "e" accesses.
    // Note that we can't get the buffer ID at this level.
    // We get it in the constructor (see ast.h) :
    // computation_info::computation_info(tiramisu::computation *comp, syntax_tree *ast)
    buffer_name = e.get_name();
}

void dnn_access_matrix::print_access_matrix() const
{
    std::cout<<buffer_name<<":";
    for(auto& line:this->matrix)
    {

        for(int val:line)
        {
            std::cout<<val<<" ";
        }
        std::cout<<",";
        
    }
    std::cout<<"\n";
}

void dnn_access_matrix::transforme_matrix_by_skewing(int first_node_depth,int alpha,int beta,int gamma,int sigma)
{
    

    /**
     * We change bases as it is done in linear algebra
    */
    //we search for 

    int first_target = -1;
    int second_target = -1;

    for(int i=0;i<matrix.size();i++)
    { // for all the buffer dimensions
        if(matrix[i][first_node_depth] == 1)
        {
            if(first_target == -1)
            {
                first_target = i;
            }
            else
            {
                first_target = -2; //disable change
            }
        }
        if(matrix[i][first_node_depth+1] == 1)
        {
            if(second_target == -1)
            {
                second_target = i;
            }
            else
            {
                second_target = -2; //disable change
            }
        }   
    }

    if((first_target >=0) && (second_target >= 0))
    {// both have lines general case access change
        int cst_1 = matrix[first_target][matrix[first_target].size()-1];

        int cst_2 = matrix[second_target][matrix[second_target].size()-1];

        //transform
        int cst_new_1 = cst_1*alpha+cst_2*beta;
        int cst_new_2 = cst_1*gamma+cst_2*sigma;

        matrix[first_target][matrix[first_target].size()-1] = cst_new_1;
        matrix[second_target][matrix[second_target].size()-1] = cst_new_2;

    }
    else
    {
        if(first_target >=0)
        {// special case where only one is concrete buffer mapping (first)
            matrix[first_target][matrix[first_target].size()-1] *=alpha ;
        }

        if(second_target >=0)
        {// special case where only one is concrete buffer mapping (second)
            matrix[second_target][matrix[second_target].size()-1] *=sigma ;
        }

    }


}

/*
dnn_access_matrix::dnn_access_matrix(dnn_access_matrix const& reference)
{
    this->buffer_id = reference.buffer_id;
    this->buffer_name = reference.buffer_name;
    this->comp = reference.comp;
    this->nb_dims = reference.nb_dims;
    this->nb_iterators = reference.nb_iterators;
    
    for(auto& line:reference.matrix)
    {
        std::vector<int> line_in;
        for(int val:line)
        {
            line_in.push_back(val);
        }
        this->matrix.push_back(line_in);
    }
}
*/

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

dnn_accesses::dnn_accesses(tiramisu::computation *comp, int nb_iterators, tiramisu::function *fct)
    : comp(comp), nb_iterators(nb_iterators)
{
    // We recursively retrieve the accesses of "comp", by starting from the
    // root expression : comp->get_expr().
    create_accesses(comp->get_expr());
}

/*
dnn_accesses::dnn_accesses(dnn_accesses const& reference)
{
    this->comp = reference.comp;
    this->nb_iterators = reference.nb_iterators;
    for(auto access:reference.accesses_list)
    {
        this->accesses_list.push_back(access);
    }
}
*/

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
    
    // We have an operation, we recursively explore its operands
    for (int i = 0; i < e.get_n_arg(); ++i)
        create_accesses(e.get_operand(i));
}
void dnn_accesses::print_all_access() const
{
    for(auto& matrix:this->accesses_list)
    {
        matrix.print_access_matrix();
    }
}

void dnn_accesses::modify_accesses_by_skewing(int first_node_depth,int alpha,int beta,int gamma,int sigma)
{

    for(auto& access:this->accesses_list)
    {
        access.transforme_matrix_by_skewing(first_node_depth,alpha,beta,gamma,sigma);
    }
}

}
