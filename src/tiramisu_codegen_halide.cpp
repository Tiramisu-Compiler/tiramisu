#include <isl/aff.h>
#include <isl/set.h>
#include <isl/constraint.h>
#include <isl/space.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <tiramisu/type.h>
#include <tiramisu/expr.h>

#include <string>

namespace tiramisu
{

Halide::Argument::Kind halide_argtype_from_tiramisu_argtype(tiramisu::argument_t type);
Halide::Expr linearize_access(Halide::Buffer<> *buffer, isl_ast_expr *index_expr);
std::string generate_new_variable_name();

computation *function::get_computation_by_name(std::string name) const
{
    assert(name.size() > 0);

    DEBUG(10, tiramisu::str_dump ("Searching computation " + name));

    tiramisu::computation *res_comp = NULL;

    for (const auto &comp : this->get_computations())
    {
        if (name == comp->get_name())
        {
            res_comp = comp;
        }
    }

    if (res_comp == NULL)
    {
        DEBUG(10, tiramisu::str_dump ("Computation not found."));
    }
    else
    {
        DEBUG(10, tiramisu::str_dump ("Computation found."));
    }

    return res_comp;
}

/**
  * Get the computation associated with a node.
  */
tiramisu::computation *get_computation_by_node(tiramisu::function *fct, isl_ast_node *node)
{
    isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
    isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
    isl_id *id = isl_ast_expr_get_id(arg);
    isl_ast_expr_free(arg);
    std::string computation_name(isl_id_get_name(id));
    isl_id_free(id);
    tiramisu::computation *comp = fct->get_computation_by_name(computation_name);

    assert((comp != NULL) && "Computation not found for this node.");

    return comp;
}

isl_map* create_map_from_domain_and_range (isl_set* domain, isl_set* range)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump ("Domain:", isl_set_to_str(domain)));
    DEBUG(3, tiramisu::str_dump ("Range:", isl_set_to_str(range)));
    // Extracting the spaces and aligning them
    isl_space* sp1 = isl_set_get_space (domain);
    isl_space* sp2 = isl_set_get_space (range);
    sp1 = isl_space_align_params (sp1, isl_space_copy (sp2));
    sp2 = isl_space_align_params (sp2, isl_space_copy (sp1));
    // Create the space access_domain -> sched_range.
    isl_space* sp = isl_space_map_from_domain_and_range (
            isl_space_copy (sp1), isl_space_copy (sp2));
    isl_map* adapter = isl_map_universe (sp);
    DEBUG(3, tiramisu::str_dump ("Transformation map:", isl_map_to_str (adapter)));
    isl_space* sp_map = isl_map_get_space (adapter);
    isl_local_space* l_sp = isl_local_space_from_space (sp_map);
    // Add equality constraints.
    for (int i = 0; i < isl_space_dim (sp1, isl_dim_set); i++)
    {
        if (isl_space_has_dim_id(sp1, isl_dim_set, i) == true)
        {
            for (int j = 0; j < isl_space_dim (sp2, isl_dim_set); j++)
            {
                if (isl_space_has_dim_id(sp2, isl_dim_set, j) == true)
                {
                    isl_id* id1 = isl_space_get_dim_id (sp1, isl_dim_set, i);
                    isl_id* id2 = isl_space_get_dim_id (sp2, isl_dim_set, j);
                    if (strcmp (isl_id_get_name (id1), isl_id_get_name (id2)) == 0)
                    {
                        isl_constraint* cst = isl_equality_alloc (
                                        isl_local_space_copy (l_sp));
                        cst = isl_constraint_set_coefficient_si (cst,
                                                                 isl_dim_in,
                                                                 i, 1);
                        cst = isl_constraint_set_coefficient_si (
                                        cst, isl_dim_out, j, -1);
                        adapter = isl_map_add_constraint (adapter, cst);
                    }
                }
            }
        }
    }
    DEBUG(3, tiramisu::str_dump(
            "Transformation map after adding equality constraints:",
            isl_map_to_str (adapter)));

    DEBUG_INDENT(-4);

    return adapter;
}

isl_ast_expr* create_isl_ast_index_expression(isl_ast_build* build,
        isl_map* access)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_map *schedule = isl_map_from_union_map(isl_ast_build_get_schedule(build));
    DEBUG(3, tiramisu::str_dump("Schedule:", isl_map_to_str(schedule)));

    isl_map* map = isl_map_reverse(isl_map_copy(schedule));
    DEBUG(3, tiramisu::str_dump("Schedule reversed:", isl_map_to_str(map)));

    isl_pw_multi_aff* iterator_map = isl_pw_multi_aff_from_map(map);
    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("iterator_map (the iterator map of an AST leaf after scheduling):");
                        isl_pw_multi_aff_dump(iterator_map));
    DEBUG(3, tiramisu::str_dump("Access:", isl_map_to_str(access)));
    isl_pw_multi_aff* index_aff = isl_pw_multi_aff_from_map(isl_map_copy(access));
    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("index_aff = isl_pw_multi_aff_from_map(access):");
                        isl_pw_multi_aff_dump(index_aff));
    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("space(index_aff):");
                                isl_space_dump(isl_pw_multi_aff_get_space(index_aff)));
    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("space(iterator_map):");
                                isl_space_dump(isl_pw_multi_aff_get_space(iterator_map)));
    iterator_map = isl_pw_multi_aff_pullback_pw_multi_aff(index_aff, iterator_map);
    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("isl_pw_multi_aff_pullback_pw_multi_aff(index_aff,iterator_map):");
                        isl_pw_multi_aff_dump(iterator_map));
    isl_ast_expr* index_expr = isl_ast_build_access_from_pw_multi_aff(
                                    build,
                                    isl_pw_multi_aff_copy(iterator_map));
    DEBUG(3, tiramisu::str_dump("isl_ast_build_access_from_pw_multi_aff(build, iterator_map):",
                            (const char * ) isl_ast_expr_to_C_str(index_expr)));

    DEBUG_INDENT(-4);

    return index_expr;
}


bool access_has_id(const tiramisu::expr& exp)
{
    DEBUG_INDENT(4);
    DEBUG_FCT_NAME(10);

    bool has_id = false;

    // Traverse the expression tree and try to see if the expression has an ID.
    if (exp.get_expr_type() == tiramisu::e_val)
        has_id = false;
    else if (exp.get_expr_type() == tiramisu::e_var)
        has_id = true;
    else if (exp.get_expr_type() == tiramisu::e_op)
    {
            switch(exp.get_op_type())
            {
                case tiramisu::o_access:
                    has_id = false;
                    break;
                case tiramisu::o_minus:
                case tiramisu::o_logical_not:
                case tiramisu::o_floor:
                case tiramisu::o_cast:
                case tiramisu::o_sin:
                case tiramisu::o_cos:
                case tiramisu::o_tan:
                case tiramisu::o_asin:
                case tiramisu::o_acos:
                case tiramisu::o_atan:
                case tiramisu::o_abs:
                case tiramisu::o_sqrt:
                case tiramisu::o_expo:
                case tiramisu::o_log:
                case tiramisu::o_ceil:
                case tiramisu::o_round:
                case tiramisu::o_trunc:
                    has_id = access_has_id(exp.get_operand(0));
                    break;
                case tiramisu::o_logical_and:
                case tiramisu::o_logical_or:
                case tiramisu::o_max:
                case tiramisu::o_min:
                case tiramisu::o_add:
                case tiramisu::o_sub:
                case tiramisu::o_mul:
                case tiramisu::o_div:
                case tiramisu::o_mod:
                case tiramisu::o_le:
                case tiramisu::o_lt:
                case tiramisu::o_ge:
                case tiramisu::o_gt:
                case tiramisu::o_eq:
                case tiramisu::o_ne:
                case tiramisu::o_right_shift:
                case tiramisu::o_left_shift:
                    has_id = access_has_id(exp.get_operand(0)) ||
                             access_has_id(exp.get_operand(1));
                    break;
                case tiramisu::o_select:
                    has_id = access_has_id(exp.get_operand(0)) ||
                             access_has_id(exp.get_operand(1)) ||
                             access_has_id(exp.get_operand(2));
                    break;
                default:
                    tiramisu::error("Checking an unsupported tiramisu expression for whether it has an ID.", 1);
            }
        }

    DEBUG_INDENT(-4);

    return has_id;
}


bool access_is_affine(const tiramisu::expr& exp)
{
    DEBUG_INDENT(4);
    DEBUG_FCT_NAME(10);

    // We assume that the access is affine until we find the opposite.
    bool affine = true;

    // Traverse the expression tree and try to find expressions that are non-affine.
    if (exp.get_expr_type() == tiramisu::e_val ||
        exp.get_expr_type() == tiramisu::e_var)
    {
        affine = true;
    }
    else if (exp.get_expr_type() == tiramisu::e_op)
    {
        switch(exp.get_op_type())
        {
            case tiramisu::o_access:
                affine = false;
                break;
            case tiramisu::o_minus:
            case tiramisu::o_logical_not:
                affine = access_is_affine(exp.get_operand(0));
                break;
            case tiramisu::o_logical_and:
            case tiramisu::o_logical_or:
            case tiramisu::o_add:
            case tiramisu::o_sub:
                affine = access_is_affine(exp.get_operand(0)) && access_is_affine(exp.get_operand(1));
                break;
            case tiramisu::o_max:
            case tiramisu::o_min:
            case tiramisu::o_floor:
            case tiramisu::o_sin:
            case tiramisu::o_cos:
            case tiramisu::o_select:
            case tiramisu::o_tan:
            case tiramisu::o_asin:
            case tiramisu::o_acos:
            case tiramisu::o_atan:
            case tiramisu::o_abs:
            case tiramisu::o_sqrt:
            case tiramisu::o_expo:
            case tiramisu::o_log:
            case tiramisu::o_ceil:
            case tiramisu::o_round:
            case tiramisu::o_trunc:
                // For now we consider these expression to be non-affine expression (although they can be expressed
                // as affine contraints).
                // TODO: work on the expression parser to support parsing these expressions into an access relation
                // with affine constraints.
                affine = false;
                break;
            case tiramisu::o_right_shift:
            case tiramisu::o_left_shift:
            case tiramisu::o_cast:
                affine = false;
                break;
            case tiramisu::o_mul:
            case tiramisu::o_div:
            case tiramisu::o_mod:
            case tiramisu::o_le:
            case tiramisu::o_lt:
            case tiramisu::o_ge:
            case tiramisu::o_gt:
            case tiramisu::o_eq:
            case tiramisu::o_ne:
                if (access_has_id(exp.get_operand(0)) && access_has_id(exp.get_operand(1)))
                    affine = false;
                break;
            default:
                tiramisu::error("Extracting access function from an unsupported tiramisu expression.", 1);
        }
    }

    DEBUG_INDENT(-4);

    return affine;
}

/**
 * access_dimension:
 *      The dimension of the access. For example, the access
 *      C0(i0, i1, i2) have three access dimensions: i0, i1 and i2.
 * access_expression:
 *      The expression of the access.
 *      This expression is parsed recursively (by calling get_constraint_for_access)
 *      and is gradually used to update the constraint.
 * access_relation:
 *      The access relation that represents the access.
 * cst:
 *      The constraint that defines the access and that is being constructed.
 *      Different calls to get_constraint_for_access modify this constraint
 *      gradually until the final constraint is created. Only the final constraint
 *      is added to the access_relation.
 * coeff:
 *      The coefficient in which all the dimension coefficients of the constraint
 *      are going to be multiplied. This coefficient is used to implement o_minus,
 *      o_mul and o_sub.
 */
isl_constraint* get_constraint_for_access(int access_dimension,
                                          const tiramisu::expr& access_expression,
                                          isl_map*& access_relation,
                                          isl_constraint* cst,
                                          int coeff,
                                          tiramisu::function* fct)
{
    /*
     * An e_val can appear in an expression passed to this function in two cases:
     *  I- the e_val refers to the constant of the constraint (for example in
     *  "i + 1", the "+1" refers to the constant of the constraint).
     *  II- the e_val is a coefficient of a dimension. For example, in "2*i"
     *  the "+2" is a coefficient of "i".
     *
     *  The case (I) is handled in the following block, while the case (II)
     *  is handled in the block handling o_mul. The "+2" in that case is
     *  used to update coeff.
     */
    if (access_expression.get_expr_type () == tiramisu::e_val)
    {
        int64_t val = coeff * access_expression.get_int_val() + (-1)*isl_val_get_num_si(isl_constraint_get_constant_val(cst));
        cst = isl_constraint_set_constant_si(cst, (-1)*val);
        DEBUG(3, tiramisu::str_dump("Assigning (-1)*(coeff * access_expression.get_int_val() + (-1)*isl_val_get_num_si(isl_constraint_get_constant_val(cst))) to the cst dimension. The value assigned is : " + std::to_string(-val)));
    }
    else if (access_expression.get_expr_type () == tiramisu::e_var)
    {
        assert(access_expression.get_name().length() > 0);

        DEBUG(3, tiramisu::str_dump("Looking for a dimension named "); tiramisu::str_dump(access_expression.get_name()); tiramisu::str_dump(" in the domain of ", isl_map_to_str(access_relation)));
        int dim0 = isl_space_find_dim_by_name(isl_map_get_space (access_relation),
                                              isl_dim_in,
                                              access_expression.get_name().c_str());
        if (dim0 >= 0)
        {
            int current_coeff = (-1) * isl_val_get_num_si(isl_constraint_get_coefficient_val(cst, isl_dim_in, dim0));
            coeff = current_coeff + coeff;
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, dim0, (coeff) * (-1));
            DEBUG(3, tiramisu::str_dump("Dimension found. Assigning -1 to the input coefficient of dimension " + std::to_string (dim0)));
        }
        else
        {
            DEBUG(3, tiramisu::str_dump("Dimension not found.  Adding dimension as a parameter."));
            access_relation = isl_map_add_dims(access_relation, isl_dim_param, 1);
            int pos = isl_map_dim(access_relation, isl_dim_param);
            isl_id* param_id = isl_id_alloc(fct->get_ctx(), access_expression.get_name().c_str (), NULL);
            access_relation = isl_map_set_dim_id(access_relation, isl_dim_param, pos - 1, param_id);
            isl_local_space* ls2 = isl_local_space_from_space(isl_map_get_space(isl_map_copy(access_relation)));
            cst = isl_constraint_alloc_equality(isl_local_space_copy(ls2));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_param, pos - 1, (coeff) * (-1));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, access_dimension, 1);
            DEBUG(3, tiramisu::str_dump ("After adding a parameter:", isl_map_to_str (access_relation)));
        }
    }
    else if (access_expression.get_expr_type() == tiramisu::e_op)
    {
        if (access_expression.get_op_type() == tiramisu::o_add)
        {
            tiramisu::expr op0 = access_expression.get_operand (0);
            tiramisu::expr op1 = access_expression.get_operand (1);
            cst = get_constraint_for_access(access_dimension, op0, access_relation, cst, coeff, fct);
            isl_constraint_dump(cst);
            cst = get_constraint_for_access(access_dimension, op1, access_relation, cst, coeff, fct);
            isl_constraint_dump(cst);
        }
        else if (access_expression.get_op_type () == tiramisu::o_sub)
        {
            tiramisu::expr op0 = access_expression.get_operand(0);
            tiramisu::expr op1 = access_expression.get_operand(1);
            cst = get_constraint_for_access(access_dimension, op0, access_relation, cst, coeff, fct);
            cst = get_constraint_for_access(access_dimension, op1, access_relation, cst, -coeff, fct);
        }
        else if (access_expression.get_op_type () == tiramisu::o_minus)
        {
            tiramisu::expr op0 = access_expression.get_operand(0);
            cst = get_constraint_for_access(access_dimension, op0, access_relation, cst, -coeff, fct);
        }
        else if (access_expression.get_op_type () == tiramisu::o_mul)
        {
             tiramisu::expr op0 = access_expression.get_operand(0);
             tiramisu::expr op1 = access_expression.get_operand(1);
             if (op0.get_expr_type () == tiramisu::e_val)
             {
                 coeff = coeff * op0.get_int_val();
                 cst = get_constraint_for_access(access_dimension, op1, access_relation, cst, coeff, fct);
             }
             else if (op1.get_expr_type () == tiramisu::e_val)
             {
                 coeff = coeff * op1.get_int_val();
                 cst = get_constraint_for_access(access_dimension, op0, access_relation, cst, coeff, fct);
             }
        }
        else
            tiramisu::error ("Currently only Add and Sub operations for accesses are supported.", true);
    }

    return cst;
}

/**
 * Traverse the tiramisu expression and extract accesses.
 */
void traverse_expr_and_extract_accesses(tiramisu::function *fct,
                                        tiramisu::computation *comp,
                                        const tiramisu::expr &exp,
                                        std::vector<isl_map *> &accesses,
                                        bool return_buffer_accesses)
{
    assert(fct != NULL);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if ((exp.get_expr_type() == tiramisu::e_op) && (exp.get_op_type() == tiramisu::o_access))
    {
        DEBUG(3, tiramisu::str_dump("Extracting access from o_access."));

        // Get the corresponding computation
        tiramisu::computation *comp2 = fct->get_computation_by_name(exp.get_name());
        DEBUG(3, tiramisu::str_dump("The computation corresponding to the access: "
                                + comp2->get_name()));

        isl_map *access_function = isl_map_copy(comp2->get_access_relation());

        DEBUG(3, tiramisu::str_dump("The original access function of this computation (before transforming its domain into time-space) : ",
                                isl_map_to_str(access_function)));

        isl_set *domain = isl_set_copy(isl_set_universe(
                    isl_set_get_space(
                        isl_set_copy(comp->get_iteration_domain()))));

        isl_set *range = isl_set_copy(isl_set_universe(
                            isl_set_get_space(
                                isl_set_copy(comp2->get_iteration_domain()))));

        isl_map* identity = create_map_from_domain_and_range(isl_set_copy(domain),
                                                             isl_set_copy(range));

        identity = isl_map_universe(isl_map_get_space(identity));

        DEBUG(3, tiramisu::str_dump("Transformation map before adding constraints:",
                                isl_map_to_str(identity)));

        // The dimension_number is a counter that indicates to which dimension
        // is the access associated.
        int access_dimension = 0;
        for (const auto &access: exp.get_access())
        {
            DEBUG(3, tiramisu::str_dump ("Assigning 1 to the coefficient of output dimension " + std::to_string (access_dimension)));
            isl_constraint* cst = isl_constraint_alloc_equality(isl_local_space_copy(isl_local_space_from_space(isl_map_get_space(isl_map_copy(identity)))));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, access_dimension, 1);
            cst = get_constraint_for_access(access_dimension, access, identity, cst, +1, fct);
            identity = isl_map_add_constraint(identity, cst);
            DEBUG(3, tiramisu::str_dump("After adding a constraint:", isl_map_to_str(identity)));
            access_dimension++;
        }

        DEBUG(3, tiramisu::str_dump("Access function:", isl_map_to_str(access_function)));
        DEBUG(3, tiramisu::str_dump("Transformation function after adding constraints:", isl_map_to_str(identity)));

        if (return_buffer_accesses == true)
        {
            access_function = isl_map_apply_range(isl_map_copy(identity), isl_map_copy(access_function));
            DEBUG(3, tiramisu::str_dump("Applying access function on the range of transformation function:", isl_map_to_str(access_function)));
        }
        else
            access_function = isl_map_copy(identity);

        // Run the following block (i.e., apply the schedule on the access function) only if
        // we are looking for the buffer access functions (i.e., return_buffer_accesses == true)
        // otherwise return the access function that is not transformed into time-processor space
        // this is mainly because the function that calls this function expects the access function
        // to be in the iteration domain.
        if ((global::is_auto_data_mapping_set() == true) && (return_buffer_accesses == true))
        {
            DEBUG(3, tiramisu::str_dump("Apply the schedule on the domain of the access function. Access functions:", isl_map_to_str(access_function)));
            DEBUG(3, tiramisu::str_dump("Trimmed schedule:", isl_map_to_str(comp->get_trimmed_union_of_schedules())));
            access_function = isl_map_apply_domain(access_function, isl_map_copy(comp->get_trimmed_union_of_schedules()));
            DEBUG(3, tiramisu::str_dump("Result: ", isl_map_to_str(access_function)));
        }
        accesses.push_back(access_function);
    }
    else if (exp.get_expr_type() == tiramisu::e_op)
    {
            DEBUG(3, tiramisu::str_dump("Extracting access from e_op."));

            switch(exp.get_op_type())
            {
                case tiramisu::o_minus:
                case tiramisu::o_logical_not:
                case tiramisu::o_floor:
                case tiramisu::o_cast:
                case tiramisu::o_sin:
                case tiramisu::o_cos:
                case tiramisu::o_tan:
                case tiramisu::o_asin:
                case tiramisu::o_acos:
                case tiramisu::o_atan:
                case tiramisu::o_abs:
                case tiramisu::o_sqrt:
                case tiramisu::o_expo:
                case tiramisu::o_log:
                case tiramisu::o_ceil:
                case tiramisu::o_round:
                case tiramisu::o_trunc:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses, return_buffer_accesses);
                    break;
                case tiramisu::o_logical_and:
                case tiramisu::o_logical_or:
                case tiramisu::o_max:
                case tiramisu::o_min:
                case tiramisu::o_add:
                case tiramisu::o_sub:
                case tiramisu::o_mul:
                case tiramisu::o_div:
                case tiramisu::o_mod:
                case tiramisu::o_le:
                case tiramisu::o_lt:
                case tiramisu::o_ge:
                case tiramisu::o_gt:
                case tiramisu::o_eq:
                case tiramisu::o_ne:
                case tiramisu::o_right_shift:
                case tiramisu::o_left_shift:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses, return_buffer_accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses, return_buffer_accesses);
                    break;
                case tiramisu::o_select:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses, return_buffer_accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses, return_buffer_accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(2), accesses, return_buffer_accesses);
                    break;
                default:
                    tiramisu::error("Extracting access function from an unsupported tiramisu expression.", 1);
            }
        }

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(3);
}


/**
 * Traverse the tiramisu expression and replace non-affine accesses by a constant.
 */
tiramisu::expr traverse_expr_and_replace_non_affine_accesses(tiramisu::computation *comp, const tiramisu::expr &exp)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Input expression: "));
    exp.dump(false); DEBUG_NEWLINE(10);

    tiramisu::expr output_expr;

    if (exp.get_expr_type() == tiramisu::e_val ||
        exp.get_expr_type() == tiramisu::e_var)
    {
        output_expr = exp;
    }
    else if ((exp.get_expr_type() == tiramisu::e_op) && (exp.get_op_type() == tiramisu::o_access))
    {
        tiramisu::expr exp2 = exp;

        DEBUG(10, tiramisu::str_dump("Looking for non-affine accesses in an o_access."));

        for (const auto &access: exp2.get_access())
            traverse_expr_and_replace_non_affine_accesses(comp, access);

        // Check if the access expressions of exp are affine (exp is an access operation).
        for (int i = 0; i < exp2.get_access().size(); i++)
        {
            // If the access is not affine, create a new constant that computes it
            // and use it as an access expression.
            if (access_is_affine(exp2.get_access()[i]) == false)
            {
                DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Access is not affine. Access: "));
                exp2.get_access()[i].dump(false); DEBUG_NEWLINE(10);
                std::string access_name = generate_new_variable_name();
                int at_loop_level = isl_set_dim(isl_set_copy(comp->get_iteration_domain()), isl_dim_set) - 1;
                tiramisu::constant *cons = new tiramisu::constant(access_name , exp2.get_access()[i],
                                                                  exp2.get_access()[i].get_data_type(),
                                                                  false, comp, at_loop_level, comp->get_function());
                exp2.set_access_dimension(i, tiramisu::var(exp2.get_access()[i].get_data_type(), access_name));
                DEBUG(10, tiramisu::str_dump("New access:")); exp2.get_access()[i].dump(false);
                DEBUG(10, tiramisu::str_dump("Constant created.  Constant body:")); cons->dump(false);
            }
        }

        output_expr = exp2;
    }
    else if (exp.get_expr_type() == tiramisu::e_op)
    {
            DEBUG(10, tiramisu::str_dump("Extracting access from e_op."));

            tiramisu::expr exp2, exp3, exp4;

            switch(exp.get_op_type())
            {
                case tiramisu::o_minus:
                case tiramisu::o_logical_not:
                case tiramisu::o_floor:
                case tiramisu::o_sin:
                case tiramisu::o_cos:
                case tiramisu::o_tan:
                case tiramisu::o_asin:
                case tiramisu::o_acos:
                case tiramisu::o_atan:
                case tiramisu::o_abs:
                case tiramisu::o_sqrt:
                case tiramisu::o_expo:
                case tiramisu::o_log:
                case tiramisu::o_ceil:
                case tiramisu::o_round:
                case tiramisu::o_trunc:
                    exp2 = traverse_expr_and_replace_non_affine_accesses(comp, exp.get_operand(0));
                    output_expr = tiramisu::expr(exp.get_op_type(), exp2);
                    break;
                case tiramisu::o_cast:
                    exp2 = traverse_expr_and_replace_non_affine_accesses(comp, exp.get_operand(0));
                    output_expr = expr(exp.get_op_type(), exp.get_data_type(), exp2);
                    break;
                case tiramisu::o_logical_and:
                case tiramisu::o_logical_or:
                case tiramisu::o_sub:
                case tiramisu::o_add:
                case tiramisu::o_max:
                case tiramisu::o_min:
                case tiramisu::o_mul:
                case tiramisu::o_div:
                case tiramisu::o_mod:
                case tiramisu::o_le:
                case tiramisu::o_lt:
                case tiramisu::o_ge:
                case tiramisu::o_gt:
                case tiramisu::o_eq:
                case tiramisu::o_ne:
                case tiramisu::o_right_shift:
                case tiramisu::o_left_shift:
                    exp2 = traverse_expr_and_replace_non_affine_accesses(comp, exp.get_operand(0));
                    exp3 = traverse_expr_and_replace_non_affine_accesses(comp, exp.get_operand(1));
                    output_expr = tiramisu::expr(exp.get_op_type(), exp2, exp3);
                    break;
                case tiramisu::o_select:
                    exp2 = traverse_expr_and_replace_non_affine_accesses(comp, exp.get_operand(0));
                    exp3 = traverse_expr_and_replace_non_affine_accesses(comp, exp.get_operand(1));
                    exp4 = traverse_expr_and_replace_non_affine_accesses(comp, exp.get_operand(2));
                    output_expr = tiramisu::expr(exp.get_op_type(), exp2, exp3, exp4);
                    break;
                default:
                    tiramisu::error("Extracting access function from an unsupported tiramisu expression.", 1);
            }
        }

    DEBUG_INDENT(-4);

    return output_expr;
}



/**
 * Compute the accesses of the RHS of the computation
 * \p comp and store them in the accesses vector.
 *
 * If \p return_buffer_accesses is set to true, this function returns access functions to
 * buffers. Otherwise it returns access functions to computations.
 */
void get_rhs_accesses(tiramisu::function *func, tiramisu::computation *comp, std::vector<isl_map *> &accesses, bool return_buffer_accesses)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    const tiramisu::expr &rhs = comp->get_expr();
    traverse_expr_and_extract_accesses(func, comp, rhs, accesses, return_buffer_accesses);

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(3);
}

/**
 * Retrieve the access function of the ISL AST leaf node (which represents a
 * computation).  Store the access in computation->access.
 */
isl_ast_node *stmt_code_generator(isl_ast_node *node, isl_ast_build *build, void *user)
{
    assert(build != NULL);
    assert(node != NULL);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::function *func = (tiramisu::function *) user;

    // Find the name of the computation associated to this AST leaf node.
    tiramisu::computation *comp = get_computation_by_node(func, node);
    assert((comp != NULL) && "Computation not found!");;

    DEBUG(3, tiramisu::str_dump("Computation:", comp->get_name().c_str()));

    // Get the accesses of the computation.  The first access is the access
    // for the LHS.  The following accesses are for the RHS.
    std::vector<isl_map *> accesses;
    isl_map *access = comp->get_access_relation_adapted_to_time_processor_domain();
    accesses.push_back(access);
    // Add the accesses of the RHS to the accesses vector
    get_rhs_accesses(func, comp, accesses, true);

    if (accesses.size() > 0)
    {
        DEBUG(3, tiramisu::str_dump("Generated RHS access maps:"));
        DEBUG_INDENT(4);
        for (int i = 0; i < accesses.size(); i++)
        {
            if (accesses[i] != NULL)
            {
                DEBUG(3, tiramisu::str_dump("Access " + std::to_string(i) + ":", isl_map_to_str(accesses[i])));
            }
            else
            {
                DEBUG(3, tiramisu::str_dump("Access " + std::to_string(i) + ": NULL"));
            }
        }

        DEBUG_INDENT(-4);

        std::vector<isl_ast_expr *> index_expressions;
        // For each access in accesses (i.e. for each access in the computation),
        // compute the corresponding isl_ast expression.
        for (auto &access: accesses)
        {
            if (access != NULL)
            {
                DEBUG(3, tiramisu::str_dump("Creating an isl_ast_index_expression for the access (isl_map *):", isl_map_to_str(access)));
                index_expressions.push_back(create_isl_ast_index_expression(build, access));
            }
            else
            {
                if (!comp->is_let_stmt()) // If this is not let stmt,
                                          // it should have an access function.
                    tiramisu::error("An access function should be provided before generating code.", true);
            }
        }

        // We want to insert the elements of index_expressions vector one by one in the beginning of comp->get_index_expr()
        for (int i=index_expressions.size()-1; i>=0; i--)
            comp->get_index_expr().insert(comp->get_index_expr().begin(), index_expressions[i]);

        for (const auto &i_expr : comp->get_index_expr())
        {
            DEBUG(3, tiramisu::str_dump("Generated Index expression:", (const char *)
                                    isl_ast_expr_to_C_str(i_expr)));
        }
    }
    else
        DEBUG(3, tiramisu::str_dump("Generated RHS empty."));

    DEBUG_FCT_NAME(3);
    DEBUG(3, tiramisu::str_dump("\n\n"));
    DEBUG_INDENT(-4);


    return node;
}

void print_isl_ast_expr_vector(
          const std::vector<isl_ast_expr*>& index_expr_cp)
{
    DEBUG(3, tiramisu::str_dump ("List of index expressions."));
    for (auto& i_expr : index_expr_cp)
        DEBUG(3, tiramisu::str_dump (" ", (const char * ) isl_ast_expr_to_C_str (i_expr)));
}

Halide::Expr halide_expr_from_tiramisu_expr(tiramisu::computation *comp,
                                            std::vector<isl_ast_expr *> &index_expr,
                                            const tiramisu::expr &tiramisu_expr)
{
    Halide::Expr result;

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if (tiramisu_expr.get_expr_type() == tiramisu::e_val)
    {
        DEBUG(3, tiramisu::str_dump("tiramisu expression of type tiramisu::e_val"));
        if (tiramisu_expr.get_data_type() == tiramisu::p_uint8)
            result = Halide::Expr(tiramisu_expr.get_uint8_value());
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int8)
            result = Halide::Expr(tiramisu_expr.get_int8_value());
        else if (tiramisu_expr.get_data_type() == tiramisu::p_uint16)
            result = Halide::Expr(tiramisu_expr.get_uint16_value());
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int16)
            result = Halide::Expr(tiramisu_expr.get_int16_value());
        else if (tiramisu_expr.get_data_type() == tiramisu::p_uint32)
            result = Halide::Expr(tiramisu_expr.get_uint32_value());
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int32)
            result = Halide::Expr(tiramisu_expr.get_int32_value());
        else if (tiramisu_expr.get_data_type() == tiramisu::p_uint64)
            result = Halide::Expr(tiramisu_expr.get_uint64_value());
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int64)
            result = Halide::Expr(tiramisu_expr.get_int64_value());
        else if (tiramisu_expr.get_data_type() == tiramisu::p_float32)
            result = Halide::Expr(tiramisu_expr.get_float32_value());
        else if (tiramisu_expr.get_data_type() == tiramisu::p_float64)
            result = Halide::Expr(tiramisu_expr.get_float64_value());
    }
    else if (tiramisu_expr.get_expr_type() == tiramisu::e_op)
    {
        Halide::Expr op0, op1, op2;

        DEBUG(3, tiramisu::str_dump("tiramisu expression of type tiramisu::e_op"));

        if (tiramisu_expr.get_n_arg() > 0)
            op0 = halide_expr_from_tiramisu_expr(comp, index_expr, tiramisu_expr.get_operand(0));

        if (tiramisu_expr.get_n_arg() > 1)
        {
            op1 = halide_expr_from_tiramisu_expr(comp, index_expr, tiramisu_expr.get_operand(1));
        }

        if (tiramisu_expr.get_n_arg() > 2)
            op2 = halide_expr_from_tiramisu_expr(comp, index_expr, tiramisu_expr.get_operand(2));

        switch(tiramisu_expr.get_op_type())
        {
            case tiramisu::o_logical_and:
                result = Halide::Internal::And::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_logical_and"));
                break;
            case tiramisu::o_logical_or:
                result = Halide::Internal::Or::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_logical_or"));
                break;
            case tiramisu::o_max:
                result = Halide::Internal::Max::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_max"));
                break;
            case tiramisu::o_min:
                result = Halide::Internal::Min::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_min"));
                break;
            case tiramisu::o_minus:
                result = Halide::Internal::Sub::make(Halide::Expr(0), op0);
                DEBUG(3, tiramisu::str_dump("op type: o_minus"));
                break;
            case tiramisu::o_add:
                result = Halide::Internal::Add::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_add"));
                break;
            case tiramisu::o_sub:
                result = Halide::Internal::Sub::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_sub"));
                break;
            case tiramisu::o_mul:
                result = Halide::Internal::Mul::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_mul"));
                break;
            case tiramisu::o_div:
                result = Halide::Internal::Div::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_div"));
                break;
            case tiramisu::o_mod:
                result = Halide::Internal::Mod::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_mod"));
                break;
            case tiramisu::o_select:
                result = Halide::Internal::Select::make(op0, op1, op2);
                DEBUG(3, tiramisu::str_dump("op type: o_select"));
                break;
            case tiramisu::o_le:
                result = Halide::Internal::LE::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_le"));
                break;
            case tiramisu::o_lt:
                result = Halide::Internal::LT::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_lt"));
                break;
            case tiramisu::o_ge:
                result = Halide::Internal::GE::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_ge"));
                break;
            case tiramisu::o_gt:
                result = Halide::Internal::GT::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_gt"));
                break;
            case tiramisu::o_logical_not:
                result = Halide::Internal::Not::make(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_not"));
                break;
            case tiramisu::o_eq:
                result = Halide::Internal::EQ::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_eq"));
                break;
            case tiramisu::o_ne:
                result = Halide::Internal::NE::make(op0, op1);
                DEBUG(3, tiramisu::str_dump("op type: o_ne"));
                break;
            case tiramisu::o_access:
            {
                DEBUG(3, tiramisu::str_dump("op type: o_access"));
                const char *access_comp_name = tiramisu_expr.get_name().c_str();
                DEBUG(3, tiramisu::str_dump("Computation being accessed: ");tiramisu::str_dump(access_comp_name));
                tiramisu::computation *access_comp = comp->get_function()->get_computation_by_name(access_comp_name);
                const char *buffer_name = isl_space_get_tuple_name(
                                            isl_map_get_space(access_comp->get_access_relation_adapted_to_time_processor_domain()), isl_dim_out);
                DEBUG(3, tiramisu::str_dump("Name of the associated buffer: ");tiramisu::str_dump(buffer_name));
                assert(buffer_name != NULL);

                auto buffer_entry = comp->get_function()->get_buffers().find(buffer_name);
                assert(buffer_entry != comp->get_function()->get_buffers().end());

                auto tiramisu_buffer = buffer_entry->second;

                // Tiramisu buffer is from outermost to innermost, whereas Halide buffer is from innermost
                // to outermost; thus, we need to reverse the order
                halide_dimension_t shape[tiramisu_buffer->get_dim_sizes().size()];
                int stride = 1;

                for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++) {
                    shape[i].min = 0;
                    shape[i].extent = (int) tiramisu_buffer->get_dim_sizes()[tiramisu_buffer->get_dim_sizes().size()- i - 1].get_int_val();
                    shape[i].stride = stride;
                    stride *= (int) tiramisu_buffer->get_dim_sizes()[tiramisu_buffer->get_dim_sizes().size()- i - 1].get_int_val();
                }

                Halide::Type type = halide_type_from_tiramisu_type(tiramisu_buffer->get_elements_type());

                Halide::Buffer<> *buffer =
                    new Halide::Buffer<>(
                            type,
                            tiramisu_buffer->get_data(),
                            tiramisu_buffer->get_dim_sizes().size(),
                            shape,
                            tiramisu_buffer->get_name());

                print_isl_ast_expr_vector(index_expr);

                Halide::Expr index = tiramisu::linearize_access(buffer, index_expr[0]);
                index_expr.erase(index_expr.begin());

                Halide::Internal::Parameter param(
                    buffer->type(), true, buffer->dimensions(), buffer->name());
                param.set_buffer(*buffer);

                result = Halide::Internal::Load::make(
                            type, tiramisu_buffer->get_name(),
                            index, *buffer, param, Halide::Internal::const_true(type.lanes()));
                }
                break;
            case tiramisu::o_right_shift:
                result = op0 >> op1;
                DEBUG(3, tiramisu::str_dump("op type: o_right_shift"));
                break;
            case tiramisu::o_left_shift:
                result = op0 << op1;
                DEBUG(3, tiramisu::str_dump("op type: o_left_shift"));
                break;
            case tiramisu::o_floor:
                result = Halide::floor(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_floor"));
                break;
            case tiramisu::o_cast:
                result = Halide::cast(halide_type_from_tiramisu_type(tiramisu_expr.get_data_type()), op0);
                DEBUG(3, tiramisu::str_dump("op type: o_cast"));
                break;
            case tiramisu::o_sin:
                result = Halide::sin(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_sin"));
                break;
            case tiramisu::o_cos:
                result = Halide::cos(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_cos"));
                break;
            case tiramisu::o_tan:
                result = Halide::tan(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_tan"));
                break;
            case tiramisu::o_asin:
                result = Halide::asin(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_asin"));
                break;
            case tiramisu::o_acos:
                result = Halide::acos(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_acos"));
                break;
            case tiramisu::o_atan:
                result = Halide::atan(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_atan"));
                break;
            case tiramisu::o_abs:
                result = Halide::abs(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_abs"));
                break;
            case tiramisu::o_sqrt:
                result = Halide::sqrt(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_sqrt"));
                break;
            case tiramisu::o_expo:
                result = Halide::exp(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_expo"));
                break;
            case tiramisu::o_log:
                result = Halide::log(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_log"));
                break;
            case tiramisu::o_ceil:
                result = Halide::ceil(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_ceil"));
                break;
            case tiramisu::o_round:
                result = Halide::round(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_round"));
                break;
            case tiramisu::o_trunc:
                result = Halide::trunc(op0);
                DEBUG(3, tiramisu::str_dump("op type: o_trunc"));
                break;
            default:
                tiramisu::error("Translating an unsupported ISL expression into a Halide expression.", 1);
        }
    }
    else if (tiramisu_expr.get_expr_type() == tiramisu::e_var)
        result = Halide::Internal::Variable::make(
                    halide_type_from_tiramisu_type(tiramisu_expr.get_data_type()),
                    tiramisu_expr.get_name());
    else
    {
        tiramisu::str_dump("tiramisu type of expr: ", str_from_tiramisu_type_expr(tiramisu_expr.get_expr_type()).c_str());
        tiramisu::error("\nTranslating an unsupported ISL expression in a Halide expression.", 1);
    }

    if (result.defined() == true)
        DEBUG(10, tiramisu::str_dump("Generated stmt: "); std::cout << result);

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(3);

    return result;
}

Halide::Expr halide_expr_from_isl_ast_expr(isl_ast_expr *isl_expr)
{
    Halide::Expr result;

    if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_int)
    {
        isl_val *init_val = isl_ast_expr_get_val(isl_expr);
        result = Halide::Expr((int32_t)isl_val_get_num_si(init_val));
    }
    else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_id)
    {
        isl_id *identifier = isl_ast_expr_get_id(isl_expr);
        std::string name_str(isl_id_get_name(identifier));
        result = Halide::Internal::Variable::make(Halide::Int(32), name_str);
    }
    else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_op)
    {
        Halide::Expr op0, op1, op2;

        op0 = halide_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(isl_expr, 0));

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 1)
            op1 = halide_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(isl_expr, 1));

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 2)
            op2 = halide_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(isl_expr, 2));

        switch(isl_ast_expr_get_op_type(isl_expr))
        {
            case isl_ast_op_and:
                result = Halide::Internal::And::make(op0, op1);
                break;
            case isl_ast_op_and_then:
                result = Halide::Internal::And::make(op0, op1);
                tiramisu::error("isl_ast_op_and_then operator found in the AST. This operator is not well supported.", 0);
                break;
            case isl_ast_op_or:
                result = Halide::Internal::Or::make(op0, op1);
                break;
            case isl_ast_op_or_else:
                result = Halide::Internal::Or::make(op0, op1);
                tiramisu::error("isl_ast_op_or_then operator found in the AST. This operator is not well supported.", 0);
                break;
            case isl_ast_op_max:
                result = Halide::Internal::Max::make(op0, op1);
                break;
            case isl_ast_op_min:
                result = Halide::Internal::Min::make(op0, op1);
                break;
            case isl_ast_op_minus:
                result = Halide::Internal::Sub::make(Halide::Expr(0), op0);
                break;
            case isl_ast_op_add:
                result = Halide::Internal::Add::make(op0, op1);
                break;
            case isl_ast_op_sub:
                result = Halide::Internal::Sub::make(op0, op1);
                break;
            case isl_ast_op_mul:
                result = Halide::Internal::Mul::make(op0, op1);
                break;
            case isl_ast_op_div:
                result = Halide::Internal::Div::make(op0, op1);
                break;
            case isl_ast_op_fdiv_q:
            case isl_ast_op_pdiv_q:
                result = Halide::Internal::Div::make(op0, op1);
                result = Halide::Internal::Cast::make(Halide::Int(32), Halide::floor(result));
                break;
            case isl_ast_op_zdiv_r:
            case isl_ast_op_pdiv_r:
                result = Halide::Internal::Mod::make(op0, op1);
                break;
            case isl_ast_op_select:
            case isl_ast_op_cond:
                result = Halide::Internal::Select::make(op0, op1, op2);
                break;
            case isl_ast_op_le:
                result = Halide::Internal::LE::make(op0, op1);
                break;
            case isl_ast_op_lt:
                result = Halide::Internal::LT::make(op0, op1);
                break;
            case isl_ast_op_ge:
                result = Halide::Internal::GE::make(op0, op1);
                break;
            case isl_ast_op_gt:
                result = Halide::Internal::GT::make(op0, op1);
                break;
            case isl_ast_op_eq:
                result = Halide::Internal::EQ::make(op0, op1);
                break;
            default:
                tiramisu::str_dump("Transforming the following expression",
                        (const char *) isl_ast_expr_to_C_str(isl_expr));
                tiramisu::str_dump("\n");
                tiramisu::error("Translating an unsupported ISL expression in a Halide expression.", 1);
        }
    }
    else
    {
        tiramisu::str_dump("Transforming the following expression",
                (const char *) isl_ast_expr_to_C_str(isl_expr));
        tiramisu::str_dump("\n");
        tiramisu::error("Translating an unsupported ISL expression in a Halide expression.", 1);
    }

    return result;
}

std::vector<std::pair<std::string, Halide::Expr>> let_stmts_vector;

/**
  * Generate a Halide statement from an ISL ast node object in the ISL ast
  * tree.
  * Level represents the level of the node in the schedule.  0 means root.
  */
Halide::Internal::Stmt *halide_stmt_from_isl_node(
    tiramisu::function fct, isl_ast_node *node,
    int level, std::vector<std::string> &tagged_stmts)
{
    assert(node != NULL);
    assert(level >= 0);


    Halide::Internal::Stmt *result = new Halide::Internal::Stmt();
    int i;

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if (isl_ast_node_get_type(node) == isl_ast_node_block)
    {
        DEBUG(3, tiramisu::str_dump("Generating code for a block"));

        isl_ast_node_list *list = isl_ast_node_block_get_children(node);
        isl_ast_node *child1;

        for (i=isl_ast_node_list_n_ast_node(list)-1; i>=0; i--)
        {
            child1 = isl_ast_node_list_get_ast_node(list, i);

            DEBUG(3, tiramisu::str_dump("Generating block."));

            Halide::Internal::Stmt *block1 =
                tiramisu::halide_stmt_from_isl_node(fct, child1, level, tagged_stmts);

            DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Generated block: "); std::cout << *block1);

            if (block1->defined() == false) // Probably block1 is a let stmt.
            {
                if (let_stmts_vector.empty() == false) // i.e. non-consumed let statements
                {
                    if (result->defined() == true) // if some stmts have already been created
                                                   // in this loop we can generate letStmt
                    {
                        for (auto l_stmt: let_stmts_vector)
                        {
                            DEBUG(3, tiramisu::str_dump("Generating the following let statement."));
                            DEBUG(3, tiramisu::str_dump("Name : " + l_stmt.first));
                            DEBUG(3, tiramisu::str_dump("Expression of the let statement: ");
                                     std::cout << l_stmt.second);

                            *result = Halide::Internal::LetStmt::make(
                                                l_stmt.first,
                                                l_stmt.second,
                                                *result);

                            DEBUG(10, tiramisu::str_dump("Generated let stmt:"));
                            DEBUG_NO_NEWLINE(10, std::cout << *result);
                        }
                        let_stmts_vector.clear();
                    }
                    // else, if (result->defined() == false), continue creating stmts
                    // until the first actual stmt (result) is created.
                }
                // else, if (let_stmts_vector.empty() == true), continue looping to
                // create more let stmts and to encounter a real statement
            }
            else // ((block1->defined() == true)
            {
                if (result->defined() == true)
                {
                    *result = Halide::Internal::Block::make(
                        *block1,
                        *result);
                }
                else // (result->defined() == false)
                    result = block1;
            }
        }
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_for)
    {
        DEBUG(3, tiramisu::str_dump("Generating code for Halide::For"));

        isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
        std::string iterator_str = std::string(isl_ast_expr_to_C_str(iter));

        isl_ast_expr *init = isl_ast_node_for_get_init(node);
        isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
        isl_ast_expr *inc  = isl_ast_node_for_get_inc(node);

        if (!isl_val_is_one(isl_ast_expr_get_val(inc)))
            tiramisu::error("The increment in one of the loops is not +1."
                        "This is not supported by Halide", 1);

        isl_ast_node *body = isl_ast_node_for_get_body(node);
        isl_ast_expr *cond_upper_bound_isl_format = NULL;

        /*
           Halide expects the loop bound to be of the form
            iter < bound
           where as ISL can generated loop bounds of the forms
            ite < bound
           and
            iter <= bound
           We need to transform the two ISL loop bounds into the Halide
           format.
        */
        if (isl_ast_expr_get_op_type(cond) == isl_ast_op_lt)
            cond_upper_bound_isl_format = isl_ast_expr_get_op_arg(cond, 1);
        else if (isl_ast_expr_get_op_type(cond) == isl_ast_op_le)
        {
            // Create an expression of "1".
            isl_val *one = isl_val_one(isl_ast_node_get_ctx(node));
            // Add 1 to the ISL ast upper bound to transform it into a strinct bound.
            cond_upper_bound_isl_format = isl_ast_expr_add(
                            isl_ast_expr_get_op_arg(cond, 1),
                            isl_ast_expr_from_val(one));
        }
        else
            tiramisu::error("The for loop upper bound is not an isl_est_expr of type le or lt" ,1);

        assert(cond_upper_bound_isl_format != NULL);
        DEBUG(3, tiramisu::str_dump("Creating for loop init expression."));

        Halide::Expr init_expr = halide_expr_from_isl_ast_expr(init);
        if (init_expr.type() != halide_type_from_tiramisu_type(global::get_loop_iterator_default_data_type()))
            init_expr = Halide::Internal::Cast::make(halide_type_from_tiramisu_type(global::get_loop_iterator_default_data_type()), init_expr);
        DEBUG(3, tiramisu::str_dump("init expression: "); std::cout << init_expr);
        Halide::Expr cond_upper_bound_halide_format =
                halide_expr_from_isl_ast_expr(cond_upper_bound_isl_format);
        cond_upper_bound_halide_format = simplify(cond_upper_bound_halide_format);
        if (cond_upper_bound_halide_format.type() != halide_type_from_tiramisu_type(global::get_loop_iterator_default_data_type()))
            cond_upper_bound_halide_format =
                Halide::Internal::Cast::make(halide_type_from_tiramisu_type(global::get_loop_iterator_default_data_type()), cond_upper_bound_halide_format);
        DEBUG(3, tiramisu::str_dump("Upper bound expression: "); std::cout << cond_upper_bound_halide_format);
        Halide::Internal::Stmt *halide_body = tiramisu::halide_stmt_from_isl_node(fct, body, level+1, tagged_stmts);
        Halide::Internal::ForType fortype = Halide::Internal::ForType::Serial;
        Halide::DeviceAPI dev_api = Halide::DeviceAPI::Host;

        // Change the type from Serial to parallel or vector if the
        // current level was marked as such.
        for (int tt = 0; tt < tagged_stmts.size(); tt++)
        {
            if (fct.should_parallelize(tagged_stmts[tt], level))
            {
                fortype = Halide::Internal::ForType::Parallel;
            }
            else if (fct.should_vectorize(tagged_stmts[tt], level))
            {
                DEBUG(3, tiramisu::str_dump("Trying to vectorize at level "); tiramisu::str_dump(std::to_string(level)));

                const Halide::Internal::IntImm *extent = cond_upper_bound_halide_format.as<Halide::Internal::IntImm>();
                if (extent) {
                    fortype = Halide::Internal::ForType::Vectorized;
                    DEBUG(3, tiramisu::str_dump("Loop vectorized"));
                }
                else
                {
                    DEBUG(3, tiramisu::str_dump("Loop not vectorized (extent is non constant)"));
                    // Currently we can only print Halide expressions using
                    // "std::cout << ".
                    DEBUG(3, std::cout << cond_upper_bound_halide_format << std::endl);
                }
            }
            else if (fct.should_map_to_gpu_thread(tagged_stmts[tt], level))
            {
                    fortype = Halide::Internal::ForType::Parallel;
                    dev_api = Halide::DeviceAPI::OpenCL;
                    std::string gpu_iter = fct.get_gpu_thread_iterator(
                        tagged_stmts[tt], level);
                    Halide::Expr new_iterator_var =
                        Halide::Internal::Variable::make(
                            Halide::Int(32),
                            gpu_iter);
                    *halide_body = Halide::Internal::LetStmt::make(
                        iterator_str,
                        new_iterator_var,
                        *halide_body);
                        iterator_str = gpu_iter;
                        DEBUG(3, tiramisu::str_dump("Loop over " + gpu_iter +
                             " created.\n"));
            }
            else if (fct.should_map_to_gpu_block(tagged_stmts[tt], level))
            {
                    fortype = Halide::Internal::ForType::Parallel;
                    dev_api = Halide::DeviceAPI::OpenCL;
                    std::string gpu_iter = fct.get_gpu_block_iterator(
                        tagged_stmts[tt], level);
                    Halide::Expr new_iterator_var =
                        Halide::Internal::Variable::make(
                            Halide::Int(32),
                            gpu_iter);
                    *halide_body = Halide::Internal::LetStmt::make(
                        iterator_str,
                        new_iterator_var,
                        *halide_body);
                        iterator_str = gpu_iter;
                        DEBUG(3, tiramisu::str_dump("Loop over " + gpu_iter +
                             " created.\n"));
            }
            else if (fct.should_unroll(tagged_stmts[tt], level))
            {
                DEBUG(3, tiramisu::str_dump("Trying to unroll at level "); tiramisu::str_dump(std::to_string(level)));

                const Halide::Internal::IntImm *extent = cond_upper_bound_halide_format.as<Halide::Internal::IntImm>();
                if (extent) {
                    fortype = Halide::Internal::ForType::Unrolled;
                    DEBUG(3, tiramisu::str_dump("Loop unrolled"));
                }
                else
                {
                    DEBUG(3, tiramisu::str_dump("Loop not unrolled (extent is non constant)"));
                    DEBUG(3, std::cout << cond_upper_bound_halide_format << std::endl);
                }
            }
        }

        DEBUG(3, tiramisu::str_dump("Creating the for loop."));
        *result = Halide::Internal::For::make(iterator_str, init_expr, cond_upper_bound_halide_format, fortype,
                dev_api, *halide_body);
        DEBUG(3, tiramisu::str_dump("For loop created."));
        DEBUG(10, std::cout<< *result);
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_user)
    {
        DEBUG(3, tiramisu::str_dump("Generating code for user node"));

        isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
        isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
        isl_id *id = isl_ast_expr_get_id(arg);
        isl_ast_expr_free(arg);
        std::string computation_name(isl_id_get_name(id));
        DEBUG(3, tiramisu::str_dump("Computation name: ");tiramisu::str_dump(computation_name));
        isl_id_free(id);

        // Check if any loop around this statement should be
        // parallelized, vectorized or mapped to GPU.
        for (int l = 0; l < level; l++)
        {
            if (fct.should_parallelize(computation_name, l) ||
                fct.should_vectorize(computation_name, l) ||
                fct.should_map_to_gpu_block(computation_name, l) ||
                fct.should_map_to_gpu_thread(computation_name, l) ||
                fct.should_unroll(computation_name, l))
            tagged_stmts.push_back(computation_name);
        }

        tiramisu::computation *comp = fct.get_computation_by_name(computation_name);
        DEBUG(10, comp->dump());

        comp->create_halide_assignment();

        *result = comp->get_halide_stmt();
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_if)
    {
        DEBUG(3, tiramisu::str_dump("Generating code for conditional"));

        isl_ast_expr *cond = isl_ast_node_if_get_cond(node);
        isl_ast_node *if_stmt = isl_ast_node_if_get_then(node);
        isl_ast_node *else_stmt = isl_ast_node_if_get_else(node);

        Halide::Expr c = halide_expr_from_isl_ast_expr(cond);

        DEBUG(3, tiramisu::str_dump("Condition: "); std::cout << c);
        DEBUG(3, tiramisu::str_dump("Generating code for the if branch."));

        Halide::Internal::Stmt *if_s =
                tiramisu::halide_stmt_from_isl_node(fct, if_stmt,
                                                level, tagged_stmts);

        DEBUG(10, tiramisu::str_dump("If branch: "); std::cout << *if_s);

        Halide::Internal::Stmt else_s;

        if (else_stmt != NULL)
        {
            DEBUG(3, tiramisu::str_dump("Generating code for the else branch."));

            else_s =
                 *tiramisu::halide_stmt_from_isl_node(fct, else_stmt,
                                                  level, tagged_stmts);

            DEBUG(10, tiramisu::str_dump("Else branch: "); std::cout << else_s);
        }
        else
            DEBUG(3, tiramisu::str_dump("Else statement is NULL."));

        *result = Halide::Internal::IfThenElse::make(
                    c,
                    *if_s,
                    else_s);

        DEBUG(10, tiramisu::str_dump("IfThenElse statement: "); std::cout << *result);

    }

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(3);

    return result;
}

void function::gen_halide_stmt()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, this->gen_c_code());

    // This vector is used in generate_Halide_stmt_from_isl_node to figure
    // out what are the statements that have already been visited in the
    // AST tree.
    std::vector<std::string> generated_stmts;
    Halide::Internal::Stmt *stmt = new Halide::Internal::Stmt();

    // Generate code to free buffers that are not passed as an argument to the function
    // TODO: Add Free().
    /*  for (const auto &b: this->get_buffers_list())
    {
        const tiramisu::buffer *buf = b.second;
        // Allocate only arrays that are not passed to the function as arguments.
        if (buf->is_argument() == false)
        {
            *stmt = Halide::Internal::Block::make(Halide::Internal::Free::make(buf->get_name()), *stmt);
        }
    }*/

    // Generate the statement that represents the whole function
    stmt = tiramisu::halide_stmt_from_isl_node(*this, this->get_isl_ast(), 0, generated_stmts);

    // Allocate buffers that are not passed as an argument to the function
    for (const auto &b : this->get_buffers())
    {
        const tiramisu::buffer *buf = b.second;
        // Allocate only arrays that are not passed to the function as arguments.
        if (buf->get_argument_type() == tiramisu::a_temporary)
        {
            std::vector<Halide::Expr> halide_dim_sizes;
            // Create a vector indicating the size that should be allocated.
            // Tiramisu buffer is defined from outermost to innermost, whereas Halide is from
            // innermost to outermost; thus, we need to reverse the order.
            for (int i = buf->get_dim_sizes().size() - 1; i >= 0; --i)
            {
                // TODO: if the size of an array is a computation access
                // this is not supported yet. Mainly because in the code below
                // we pass NULL pointers for parameters that are necessary
                // in case we are computing the halide expression from a tiramisu expression
                // that represents a computation access.
                const auto &sz = buf->get_dim_sizes()[i];
                std::vector<isl_ast_expr *> ie = {};
                halide_dim_sizes.push_back(halide_expr_from_tiramisu_expr(NULL, ie, sz));
            }
            *stmt = Halide::Internal::Allocate::make(
                        buf->get_name(),
                        halide_type_from_tiramisu_type(buf->get_elements_type()),
                        halide_dim_sizes, Halide::Internal::const_true(), *stmt);
        }
    }

    auto invariant_vector = this->get_invariants();

    // Generate the invariants of the function.
    // Traverse the vector of invariants in reverse order (this because
    // invariants are added at the beginning of the invariant vector so
    // the first vector element actually should be visited last because
    // it was added last).
    // We need to do this because usually for vectorization, the separation
    // invariant which is an expression that uses the loop parameters needs
    // to come after the initialization of those parameters, that is, it should
    // come last (when we are sure all the other parameters are already
    // initialized).
    for (int i = invariant_vector.size() - 1; i >= 0; i--)
    {
        const auto param = invariant_vector[i]; // Get the i'th invariant
        std::vector<isl_ast_expr *> ie = {};
        *stmt = Halide::Internal::LetStmt::make(
                    param.get_name(),
                    halide_expr_from_tiramisu_expr(NULL, ie, param.get_expr()),
                    *stmt);
    }

    this->halide_stmt = stmt;

    DEBUG(3, tiramisu::str_dump("\n\nGenerated Halide stmt before lowering:"));
    DEBUG(3, std::cout << (*stmt));

    DEBUG_INDENT(-4);
}

isl_ast_node *for_code_generator_after_for(isl_ast_node *node, isl_ast_build *build, void *user)
{
    return node;
}

/**
  * Linearize a multidimensional access to a Halide buffer.
  * Supposing that we have buf[N1][N2][N3], transform buf[i][j][k]
  * into buf[k + j*N3 + i*N3*N2].
  * Note that the first arg in index_expr is the buffer name.  The other args
  * are the indices for each dimension of the buffer.
  */
Halide::Expr linearize_access(Halide::Buffer<> *buffer,
        isl_ast_expr *index_expr)
{
    assert(isl_ast_expr_get_op_n_arg(index_expr) > 1);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int buf_dims = buffer->dimensions();

    // ISL dimension is ordered from outermost to innermost.

    isl_ast_expr *operand;
    Halide::Expr index = 0;
    for (int i = buf_dims; i >= 1; --i)
    {
        operand = isl_ast_expr_get_op_arg(index_expr, i);
        Halide::Expr operand_h = halide_expr_from_isl_ast_expr(operand);
        index += operand_h * Halide::Expr(buffer->stride(buf_dims - i));
    }

    DEBUG_INDENT(-4);

    return index;
}

/*
 * Create a Halide assign statement from a computation.
 * The statement will assign the computations to a memory buffer based on the
 * access function provided in access.
 */
void computation::create_halide_assignment()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Generating stmt for assignment."));

    if (this->is_let_stmt())
    {
        DEBUG(3, tiramisu::str_dump("This is a let statement."));
        DEBUG_NO_NEWLINE(10, tiramisu::str_dump("The expression associated with the let statement: "); this->expression.dump(false));
        DEBUG_NEWLINE(10);

        Halide::Expr result = halide_expr_from_tiramisu_expr(this,
                                                             this->get_index_expr(),
                                                             this->expression);

        Halide::Type l_type = halide_type_from_tiramisu_type(this->get_data_type());

        if (l_type != result.type())
            result = Halide::Internal::Cast::make(l_type, result);

        std::string let_stmt_name = this->get_name();

        let_stmts_vector.push_back(std::pair<std::string, Halide::Expr>(
                                            let_stmt_name,
                                            result));
        DEBUG(10, tiramisu::str_dump("A let statement was added to the vector of let statements."));
    }
    else
    {
        DEBUG(3, tiramisu::str_dump("This is not a let statement."));

        const char *buffer_name = isl_space_get_tuple_name(
                                    isl_map_get_space(this->get_access_relation_adapted_to_time_processor_domain()),
                                    isl_dim_out);
        assert(buffer_name != NULL);

        DEBUG(3, tiramisu::str_dump("Buffer name extracted from the access relation: ", buffer_name));

        isl_map *access = this->get_access_relation_adapted_to_time_processor_domain();
        isl_space *space = isl_map_get_space(access);
        // Get the number of dimensions of the ISL map representing
        // the access.
        int access_dims = isl_space_dim(space, isl_dim_out);

        // Fetch the actual buffer.
        auto buffer_entry = this->function->get_buffers().find(buffer_name);
        assert(buffer_entry != this->function->get_buffers().end());

        auto tiramisu_buffer = buffer_entry->second;
        DEBUG(3, tiramisu::str_dump("A Tiramisu buffer that corresponds to the buffer indicated in the access relation was found."));

        DEBUG(10, tiramisu_buffer->dump(true));

        // Tiramisu buffer is from outermost to innermost, whereas Halide buffer is
        // from innermost to outermost; thus, we need to reverse the order
        halide_dimension_t shape[tiramisu_buffer->get_dim_sizes().size()];
        int stride = 1;
        for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++) {
            shape[i].min = 0;
            shape[i].extent = (int) tiramisu_buffer->get_dim_sizes()[tiramisu_buffer->get_dim_sizes().size() - i - 1].get_int_val();
            shape[i].stride = stride;
            stride *= (int) tiramisu_buffer->get_dim_sizes()[tiramisu_buffer->get_dim_sizes().size() - i - 1].get_int_val();
        }

        Halide::Buffer<> *buffer =
            new Halide::Buffer<>(
                    halide_type_from_tiramisu_type(tiramisu_buffer->get_elements_type()),
                    tiramisu_buffer->get_data(),
                    tiramisu_buffer->get_dim_sizes().size(),
                    shape,
                    tiramisu_buffer->get_name());
        DEBUG(3, tiramisu::str_dump("Halide buffer object created.  This object will be passed to the Halide function that creates an assignment to a buffer."));

        int buf_dims = buffer->dimensions();

        // The number of dimensions in the Halide buffer should be equal to
        // the number of dimensions of the access function.
        assert(buf_dims == access_dims);
        assert(this->index_expr[0] != NULL);
        DEBUG(3, tiramisu::str_dump("Linearizing access of the LHS index expression."));
        Halide::Expr index = tiramisu::linearize_access(buffer, this->index_expr[0]);

        Halide::Internal::Parameter param(
              buffer->type(), true, buffer->dimensions(), buffer->name());
        param.set_buffer(*buffer);

        DEBUG(3, tiramisu::str_dump("Index expressions of this statement are (the first is the LHS and the others are the RHS) :"));
        print_isl_ast_expr_vector(this->index_expr);

        DEBUG(3, tiramisu::str_dump("Erasing the LHS index expression from the vector of index expressions (the LHS index has just been linearized)."));
        this->index_expr.erase(this->index_expr.begin());

        Halide::Type type = halide_type_from_tiramisu_type(this->get_data_type());

        DEBUG(3, tiramisu::str_dump("Calling the Halide::Internal::Store::make function which creates the store statement."));
        DEBUG(3, tiramisu::str_dump("The RHS index expressions are first transformed to Halide expressions then passed to the make function."));

        this->stmt = Halide::Internal::Store::make (
                        buffer_name,
                        halide_expr_from_tiramisu_expr(this, this->index_expr, this->expression),
                        index, param, Halide::Internal::const_true(type.lanes()));

        DEBUG(3, tiramisu::str_dump("Halide::Internal::Store::make statement created."));
    }

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("End of create_halide_stmt. Generated statement is: ");
             std::cout << this->stmt);

    DEBUG_INDENT(-4);
}

void function::gen_halide_obj(
    std::string obj_file_name, Halide::Target::OS os,
    Halide::Target::Arch arch, int bits) const
{
    Halide::Target target;
    target.os = os;
    target.arch = arch;
    target.bits = bits;
    std::vector<Halide::Target::Feature> x86_features;
    x86_features.push_back(Halide::Target::AVX);
    x86_features.push_back(Halide::Target::SSE41);
    target.set_features(x86_features);

    Halide::Module m(obj_file_name, target);

    std::vector<Halide::Argument> fct_arguments;

    for (const auto &buf : this->function_arguments)
    {
        Halide::Argument buffer_arg(
            buf->get_name(),
            halide_argtype_from_tiramisu_argtype(buf->get_argument_type()),
            halide_type_from_tiramisu_type(buf->get_elements_type()),
            buf->get_n_dims());

        fct_arguments.push_back(buffer_arg);
    }

    Halide::Internal::Stmt lowered = lower_halide_pipeline(target, this->get_halide_stmt());
    m.append(Halide::Internal::LoweredFunc(this->get_name(), fct_arguments, lowered, Halide::Internal::LoweredFunc::External));

    Halide::Outputs output = Halide::Outputs().object(obj_file_name);
    m.compile(output);
    m.compile(Halide::Outputs().c_header(obj_file_name + ".h"));
}

}
