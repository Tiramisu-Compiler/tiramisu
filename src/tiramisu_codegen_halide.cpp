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
#include "../include/tiramisu/expr.h"
#include "../Halide/src/Expr.h"
#include "../Halide/src/Parameter.h"
#include "../include/tiramisu/debug.h"
#include "../Halide/src/IR.h"
#include "../include/tiramisu/core.h"

namespace tiramisu
{

Halide::Argument::Kind halide_argtype_from_tiramisu_argtype(tiramisu::argument_t type);
Halide::Expr linearize_access(int dims, const halide_dimension_t *shape, isl_ast_expr *index_expr);
std::string generate_new_variable_name();

std::vector<computation *> function::get_computation_by_name(std::string name) const
{
    assert(!name.empty());

    DEBUG(10, tiramisu::str_dump("Searching computation " + name));

    std::vector<tiramisu::computation *> res_comp;

    for (const auto &comp : this->get_computations())
    {
        if (name == comp->get_name())
        {
            res_comp.push_back(comp);
        }
    }

    if (res_comp.empty())
    {
        DEBUG(10, tiramisu::str_dump("Computation not found."));
    }
    else
    {
        DEBUG(10, tiramisu::str_dump("Computation found."));
    }

    return res_comp;
}

/**
  * Get the computation associated with a node.
  */
std::vector<tiramisu::computation *> get_computation_by_node(tiramisu::function *fct,
        isl_ast_node *node)
{
    isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
    isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
    isl_id *id = isl_ast_expr_get_id(arg);
    std::string computation_name(isl_id_get_name(id));
    isl_ast_expr_free(expr);
    isl_ast_expr_free(arg);
    isl_id_free(id);
    std::vector<tiramisu::computation *> comp = fct->get_computation_by_name(computation_name);

    assert((comp.size() > 0) && "Computation not found for this node.");

    return comp;
}

isl_map *create_map_from_domain_and_range(isl_set *domain, isl_set *range)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Domain:", isl_set_to_str(domain)));
    DEBUG(3, tiramisu::str_dump("Range:", isl_set_to_str(range)));
    // Extracting the spaces and aligning them
    isl_space *sp1 = isl_set_get_space(domain);
    isl_space *sp2 = isl_set_get_space(range);
    sp1 = isl_space_align_params(sp1, isl_space_copy(sp2));
    sp2 = isl_space_align_params(sp2, isl_space_copy(sp1));
    // Create the space access_domain -> sched_range.
    isl_space *sp = isl_space_map_from_domain_and_range(
                        isl_space_copy(sp1), isl_space_copy(sp2));
    isl_map *adapter = isl_map_universe(sp);
    DEBUG(3, tiramisu::str_dump("Transformation map:", isl_map_to_str(adapter)));
    isl_space *sp_map = isl_map_get_space(adapter);
    isl_local_space *l_sp = isl_local_space_from_space(sp_map);
    // Add equality constraints.
    for (int i = 0; i < isl_space_dim(sp1, isl_dim_set); i++)
    {
        if (isl_space_has_dim_id(sp1, isl_dim_set, i) == true)
        {
            for (int j = 0; j < isl_space_dim (sp2, isl_dim_set); j++)
            {
                if (isl_space_has_dim_id(sp2, isl_dim_set, j) == true)
                {
                    isl_id *id1 = isl_space_get_dim_id(sp1, isl_dim_set, i);
                    isl_id *id2 = isl_space_get_dim_id(sp2, isl_dim_set, j);
                    if (strcmp(isl_id_get_name(id1), isl_id_get_name(id2)) == 0)
                    {
                        isl_constraint *cst = isl_equality_alloc(
                                                  isl_local_space_copy(l_sp));
                        cst = isl_constraint_set_coefficient_si(cst,
                                                                isl_dim_in,
                                                                i, 1);
                        cst = isl_constraint_set_coefficient_si(
                                  cst, isl_dim_out, j, -1);
                        adapter = isl_map_add_constraint(adapter, cst);
                    }
                    isl_id_free(id1);
                    isl_id_free(id2);
                }
            }
        }
    }

    isl_space_free(sp1);
    isl_space_free(sp2);
    isl_local_space_free(l_sp);

    DEBUG(3, tiramisu::str_dump(
              "Transformation map after adding equality constraints:",
              isl_map_to_str(adapter)));

    DEBUG_INDENT(-4);

    return adapter;
}

isl_ast_expr *create_isl_ast_index_expression(isl_ast_build *build,
        isl_map *access)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_map *schedule = isl_map_from_union_map(isl_ast_build_get_schedule(build));
    DEBUG(3, tiramisu::str_dump("Schedule:", isl_map_to_str(schedule)));

    isl_map *map = isl_map_reverse(isl_map_copy(schedule));
    DEBUG(3, tiramisu::str_dump("Schedule reversed:", isl_map_to_str(map)));

    isl_pw_multi_aff *iterator_map = isl_pw_multi_aff_from_map(map);
    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("iterator_map (the iterator map of an AST leaf after scheduling): "));
    DEBUG_NO_NEWLINE(3, isl_pw_multi_aff_dump(iterator_map));
    DEBUG(3, tiramisu::str_dump("Access:", isl_map_to_str(access)));
    isl_pw_multi_aff *index_aff = isl_pw_multi_aff_from_map(isl_map_copy(access));
    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("index_aff = isl_pw_multi_aff_from_map(access): "));
    DEBUG_NO_NEWLINE(3, isl_pw_multi_aff_dump(index_aff));
    isl_space *model2 = isl_pw_multi_aff_get_space(isl_pw_multi_aff_copy(iterator_map));
    index_aff = isl_pw_multi_aff_align_params(index_aff, model2);
    isl_space *model = isl_pw_multi_aff_get_space(isl_pw_multi_aff_copy(index_aff));
    iterator_map = isl_pw_multi_aff_align_params(iterator_map, model);
    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("space(index_aff): "));
    DEBUG_NO_NEWLINE(3, isl_space_dump(isl_pw_multi_aff_get_space(index_aff)));
    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("space(iterator_map): "));
    DEBUG_NO_NEWLINE(3, isl_space_dump(isl_pw_multi_aff_get_space(iterator_map)));
    iterator_map = isl_pw_multi_aff_pullback_pw_multi_aff(index_aff, iterator_map);
    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("isl_pw_multi_aff_pullback_pw_multi_aff(index_aff,iterator_map):"));
    DEBUG_NO_NEWLINE(3, isl_pw_multi_aff_dump(iterator_map));
    isl_ast_expr *index_expr = isl_ast_build_access_from_pw_multi_aff(
                                   build,
                                   iterator_map);
    DEBUG(3, tiramisu::str_dump("isl_ast_build_access_from_pw_multi_aff(build, iterator_map):",
                                (const char *)isl_ast_expr_to_C_str(index_expr)));

    DEBUG_INDENT(-4);

    isl_map_free(schedule);

    return index_expr;
}

bool access_has_id(const tiramisu::expr &exp)
{
    DEBUG_INDENT(4);
    DEBUG_FCT_NAME(10);

    bool has_id = false;

    // Traverse the expression tree and try to see if the expression has an ID.
    if (exp.get_expr_type() == tiramisu::e_val)
    {
        has_id = false;
    }
    else if (exp.get_expr_type() == tiramisu::e_var)
    {
        has_id = true;
    }
    else if (exp.get_expr_type() == tiramisu::e_op)
    {
        switch (exp.get_op_type())
        {
        case tiramisu::o_access:
        case tiramisu::o_call:
        case tiramisu::o_address:
        case tiramisu::o_allocate:
        case tiramisu::o_free:
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
        case tiramisu::o_cond:
        case tiramisu::o_lerp:
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

bool access_is_affine(const tiramisu::expr &exp)
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
        switch (exp.get_op_type())
        {
        case tiramisu::o_access:
        case tiramisu::o_call:
        case tiramisu::o_address:
        case tiramisu::o_allocate:
        case tiramisu::o_free:
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
        case tiramisu::o_lerp:
        case tiramisu::o_cond:
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
            {
                affine = false;
            }
            break;
        default:
            tiramisu::error("Unsupported tiramisu expression passed to access_is_affine().", 1);
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
isl_constraint *generator::get_constraint_for_access(int access_dimension,
        const tiramisu::expr &access_expression,
        isl_map *access_relation,
        isl_constraint *cst,
        int coeff,
        const tiramisu::function *fct)
{
    // An e_val can appear in an expression passed to this function in two cases:
    // I- the e_val refers to the constant of the constraint (for example in
    // "i + 1", the "+1" refers to the constant of the constraint).
    // II- the e_val is a coefficient of a dimension. For example, in "2*i"
    // the "+2" is a coefficient of "i".
    //
    // The case (I) is handled in the following block, while the case (II)
    // is handled in the block handling o_mul. The "+2" in that case is
    // used to update coeff.
    if (access_expression.get_expr_type() == tiramisu::e_val)
    {
        int64_t val = coeff * access_expression.get_int_val() -
                      isl_val_get_num_si(isl_constraint_get_constant_val(cst));
        cst = isl_constraint_set_constant_si(cst, -val);
        DEBUG(3, tiramisu::str_dump("Assigning -(coeff * access_expression.get_int_val() - isl_val_get_num_si(isl_constraint_get_constant_val(cst))) to the cst dimension. The value assigned is : "
                                    + std::to_string(-val)));
    }
    else if (access_expression.get_expr_type() == tiramisu::e_var)
    {
        assert(!access_expression.get_name().empty());

        DEBUG(3, tiramisu::str_dump("Looking for a dimension named ");
              tiramisu::str_dump(access_expression.get_name());
              tiramisu::str_dump(" in the domain of ", isl_map_to_str(access_relation)));
        int dim0 = isl_space_find_dim_by_name(isl_map_get_space(access_relation),
                                              isl_dim_in,
                                              access_expression.get_name().c_str());
        if (dim0 >= 0)
        {
            int current_coeff = -isl_val_get_num_si(isl_constraint_get_coefficient_val(cst, isl_dim_in, dim0));
            coeff = current_coeff + coeff;
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, dim0, -coeff);
            DEBUG(3, tiramisu::str_dump("Dimension found. Assigning -1 to the input coefficient of dimension " +
                                        std::to_string(dim0)));
        }
        else
        {
            DEBUG(3, tiramisu::str_dump("Dimension not found. Adding dimension as a parameter."));
            access_relation = isl_map_add_dims(access_relation, isl_dim_param, 1);
            int pos = isl_map_dim(access_relation, isl_dim_param);
            isl_id *param_id = isl_id_alloc(fct->get_isl_ctx(), access_expression.get_name().c_str (), NULL);
            access_relation = isl_map_set_dim_id(access_relation, isl_dim_param, pos - 1, param_id);
            isl_local_space *ls2 = isl_local_space_from_space(isl_map_get_space(access_relation));
            cst = isl_constraint_alloc_equality(ls2);
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_param, pos - 1, -coeff);
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, access_dimension, 1);
            DEBUG(3, tiramisu::str_dump("After adding a parameter:", isl_map_to_str(access_relation)));
        }
    }
    else if (access_expression.get_expr_type() == tiramisu::e_op)
    {
        if (access_expression.get_op_type() == tiramisu::o_add)
        {
            tiramisu::expr op0 = access_expression.get_operand(0);
            tiramisu::expr op1 = access_expression.get_operand(1);
            cst = generator::get_constraint_for_access(access_dimension, op0, access_relation, cst, coeff, fct);
            isl_constraint_dump(cst);
            cst = generator::get_constraint_for_access(access_dimension, op1, access_relation, cst, coeff, fct);
            isl_constraint_dump(cst);
        }
        else if (access_expression.get_op_type() == tiramisu::o_sub)
        {
            tiramisu::expr op0 = access_expression.get_operand(0);
            tiramisu::expr op1 = access_expression.get_operand(1);
            cst = generator::get_constraint_for_access(access_dimension, op0, access_relation, cst, coeff, fct);
            cst = generator::get_constraint_for_access(access_dimension, op1, access_relation, cst, -coeff, fct);
        }
        else if (access_expression.get_op_type() == tiramisu::o_minus)
        {
            tiramisu::expr op0 = access_expression.get_operand(0);
            cst = generator::get_constraint_for_access(access_dimension, op0, access_relation, cst, -coeff, fct);
        }
        else if (access_expression.get_op_type() == tiramisu::o_mul)
        {
            tiramisu::expr op0 = access_expression.get_operand(0);
            tiramisu::expr op1 = access_expression.get_operand(1);
            if (op0.get_expr_type() == tiramisu::e_val)
            {
                coeff = coeff * op0.get_int_val();
                cst = generator::get_constraint_for_access(access_dimension, op1, access_relation, cst, coeff, fct);
            }
            else if (op1.get_expr_type() == tiramisu::e_val)
            {
                coeff = coeff * op1.get_int_val();
                cst = generator::get_constraint_for_access(access_dimension, op0, access_relation, cst, coeff, fct);
            }
        }
        else
        {
            tiramisu::error ("Currently only Add and Sub operations for accesses are supported.", true);
        }
    }

    return cst;
}

/**
 * Traverse the vector of computations \p comp_vec and return the computations
 * that have a domain that intersects with \p domain.
 */
std::vector<tiramisu::computation *> generator::filter_computations_by_domain(std::vector<tiramisu::computation *> comp_vec,
        isl_union_set *node_domain)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    std::vector<tiramisu::computation *> res;

    DEBUG(10, tiramisu::str_dump("Filtering computations by ISL AST domain."));
    DEBUG(10, tiramisu::str_dump("ISL AST node domain:", isl_union_set_to_str(node_domain)));

    for (size_t i = 0; i < comp_vec.size(); i++)
    {
        isl_set *comp_domain = comp_vec[i]->get_iteration_domain();
        DEBUG(10, tiramisu::str_dump("Checking computation " + comp_vec[i]->get_name()));
        DEBUG(10, tiramisu::str_dump("Domain of the computation ", isl_set_to_str(comp_domain)));
        isl_map *sched = comp_vec[i]->get_trimmed_union_of_schedules();
        isl_set *scheduled_comp_domain = isl_set_apply(isl_set_copy(comp_domain), isl_map_copy(sched));
        DEBUG(10, tiramisu::str_dump("Domain of the computation in time-space domain ", isl_set_to_str(scheduled_comp_domain)));
        DEBUG(10, tiramisu::str_dump("Intersecting the set:", isl_set_to_str(scheduled_comp_domain)));
        DEBUG(10, tiramisu::str_dump("With the set:", isl_union_set_to_str(node_domain)));

        isl_space *space_model = isl_space_align_params(isl_space_copy(isl_set_get_space(scheduled_comp_domain)),
				isl_space_copy(isl_union_set_get_space(node_domain)));
        scheduled_comp_domain = isl_set_align_params(scheduled_comp_domain, space_model);

        isl_union_set *intersection =
            isl_union_set_intersect(isl_union_set_copy(node_domain),
                                    isl_union_set_from_set(scheduled_comp_domain));

        DEBUG(10, tiramisu::str_dump("Intersection", isl_union_set_to_str(intersection)));

        if (isl_union_set_is_empty(intersection) == isl_bool_false)
        {
            DEBUG(10, tiramisu::str_dump("This computation is accepted by the filter (intersection non-empty)."));
            res.push_back(comp_vec[i]);
        }
        isl_union_set_free(intersection);
    }

    assert((res.size() > 0) && "Computation not found.");

    DEBUG_INDENT(-4);

    return res;
}

/**
 * Traverse the tiramisu expression and extract the access of the access
 * operation passed in \p exp.
 * An access relation from the domain of the computation \p comp to the
 * computation accessed by the access operation \p exp is added to the
 * vector \p accesses.
 * If \p return_buffer_accesses = true, an access to a buffer is created
 * instead.
 * \p domain_of_accessed_computation is the domain of the current statement
 * (in ISL AST). Knowing this domain is important to retrieve the computation
 * that corresponds to the current statement if many computations that have
 * the name comp.get_name() exist.
 */
void generator::traverse_expr_and_extract_accesses(const tiramisu::function *fct,
                                        const tiramisu::computation *comp,
                                        const tiramisu::expr &exp,
                                        std::vector<isl_map *> &accesses,
                                        bool return_buffer_accesses)
{
    assert(fct != NULL);
    assert(comp != NULL);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if ((exp.get_expr_type() == tiramisu::e_op) && (exp.get_op_type() == tiramisu::o_access))
    {
        DEBUG(3, tiramisu::str_dump("Extracting access from o_access."));

        // Get the domain of the computation that corresponds to the access.
        // Even if there are many computations, we take the first because we are
        // only interested in getting the space of those computations and we assume
        // in Tiramisu that all the computations that have the same name, have the same
        // space.
        std::vector<tiramisu::computation *> computations_vector = fct->get_computation_by_name(exp.get_name());

        // Since we modify the names of update computations but do not modify the
        // expressions.  When accessing the expressions we find the old names, so
        // we need to look for the new names instead of the old names.
        // We do this instead of actually changing the expressions, because changing
        // the expressions will make the semantics of the printed program ambiguous,
        // since we do not have any way to distinguish between which update is the
        // consumer is consuming exactly.
        if (computations_vector.size() == 0)
        {
            // Search for update computations.
            computations_vector = fct->get_computation_by_name("_" + exp.get_name() + "_update_0");
            assert((computations_vector.size() > 0) && "Computation not found.");
        }
        tiramisu::computation *access_op_comp = computations_vector[0];

        DEBUG(10, tiramisu::str_dump("Obtained accessed computation."));

        isl_set *lhs_comp_domain = isl_set_universe(isl_set_get_space(comp->get_iteration_domain()));
        isl_set *rhs_comp_domain = isl_set_universe(isl_set_get_space(
                                       access_op_comp->get_iteration_domain()));
        isl_map *access_map = create_map_from_domain_and_range(lhs_comp_domain, rhs_comp_domain);
        isl_set_free(lhs_comp_domain);
        isl_set_free(rhs_comp_domain);

        isl_map *access_to_comp = isl_map_universe(isl_map_get_space(access_map));
        isl_map_free(access_map);

        DEBUG(3, tiramisu::str_dump("Transformation map before adding constraints:",
                                    isl_map_to_str(access_to_comp)));

        // The dimension_number is a counter that indicates to which dimension
        // is the access associated.
        int access_dimension = 0;
        for (const auto &access : exp.get_access())
        {
            DEBUG(3, tiramisu::str_dump("Assigning 1 to the coefficient of output dimension " +
                                        std::to_string (access_dimension)));
            isl_constraint *cst = isl_constraint_alloc_equality(isl_local_space_from_space(isl_map_get_space(
                                      access_to_comp)));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, access_dimension, 1);
            cst = generator::get_constraint_for_access(access_dimension, access, access_to_comp, cst, 1, fct);
            access_to_comp = isl_map_add_constraint(access_to_comp, cst);
            DEBUG(3, tiramisu::str_dump("After adding a constraint:", isl_map_to_str(access_to_comp)));
            access_dimension++;
        }

        DEBUG(3, tiramisu::str_dump("Transformation function after adding constraints:",
                                    isl_map_to_str(access_to_comp)));

        if (return_buffer_accesses)
        {
            isl_map *access_to_buff = isl_map_copy(access_op_comp->get_access_relation());

            DEBUG(3, tiramisu::str_dump("The access of this computation to buffers (before re-adapting its domain into the domain of the current access) : ",
                                        isl_map_to_str(access_to_buff)));

            access_to_buff = isl_map_apply_range(isl_map_copy(access_to_comp), access_to_buff);
            DEBUG(3, tiramisu::str_dump("Applying access function on the range of transformation function:",
                                        isl_map_to_str(access_to_buff)));

            // Run the following block (i.e. apply the schedule on the access function) only if
            // we are looking for the buffer access functions (i.e. return_buffer_accesses == true)
            // otherwise return the access function that is not transformed into time-processor space
            // this is mainly because the function that calls this function expects the access function
            // to be in the iteration domain.
            if (global::is_auto_data_mapping_set())
            {
                DEBUG(3, tiramisu::str_dump("Apply the schedule on the domain of the access function. Access functions:",
                                            isl_map_to_str(access_to_buff)));
                DEBUG(3, tiramisu::str_dump("Trimmed schedule:",
                                            isl_map_to_str(comp->get_trimmed_union_of_schedules())));
                access_to_buff = isl_map_apply_domain(access_to_buff,
                                                      isl_map_copy(comp->get_trimmed_union_of_schedules()));
                DEBUG(3, tiramisu::str_dump("Result: ", isl_map_to_str(access_to_buff)));
            }

            accesses.push_back(access_to_buff);
            isl_map_free(access_to_comp);
        }
        else
        {
            accesses.push_back(access_to_comp);
        }
    }
    else if (exp.get_expr_type() == tiramisu::e_op)
    {
        DEBUG(3, tiramisu::str_dump("Extracting access from e_op."));

        switch (exp.get_op_type())
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
        case tiramisu::o_address:
        {
            tiramisu::expr exp0 = exp.get_operand(0);
            generator::traverse_expr_and_extract_accesses(fct, comp, exp0, accesses, return_buffer_accesses);
            break;
        }
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
        {
            tiramisu::expr exp0 = exp.get_operand(0);
            tiramisu::expr exp1 = exp.get_operand(1);
            generator::traverse_expr_and_extract_accesses(fct, comp, exp0, accesses, return_buffer_accesses);
            generator::traverse_expr_and_extract_accesses(fct, comp, exp1, accesses, return_buffer_accesses);
            break;
        }
        case tiramisu::o_select:
        case tiramisu::o_cond:
        case tiramisu::o_lerp:
        {
            tiramisu::expr expr0 = exp.get_operand(0);
            tiramisu::expr expr1 = exp.get_operand(1);
            tiramisu::expr expr2 = exp.get_operand(2);
            generator::traverse_expr_and_extract_accesses(fct, comp, expr0, accesses, return_buffer_accesses);
            generator::traverse_expr_and_extract_accesses(fct, comp, expr1, accesses, return_buffer_accesses);
            generator::traverse_expr_and_extract_accesses(fct, comp, expr2, accesses, return_buffer_accesses);
            break;
        }
        case tiramisu::o_call:
        {
            for (const auto &e : exp.get_arguments())
            {
                generator::traverse_expr_and_extract_accesses(fct, comp, e, accesses, return_buffer_accesses);
            }
            break;
        }
        case tiramisu::o_allocate:
        case tiramisu::o_free:
            // They do not have any access.
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
tiramisu::expr traverse_expr_and_replace_non_affine_accesses(tiramisu::computation *comp,
        const tiramisu::expr &exp)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Input expression: "); exp.dump(false););
    DEBUG_NEWLINE(10);

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

        for (const auto &access : exp2.get_access())
        {
            traverse_expr_and_replace_non_affine_accesses(comp, access);
        }

        // Check if the access expressions of exp are affine (exp is an access operation).
        for (size_t i = 0; i < exp2.get_access().size(); i++)
        {
            // If the access is not affine, create a new constant that computes it
            // and use it as an access expression.
            if (access_is_affine(exp2.get_access()[i]) == false)
            {
                DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Access is not affine. Access: "));
                exp2.get_access()[i].dump(false); DEBUG_NEWLINE(10);
                std::string access_name = generate_new_variable_name();
                comp->add_associated_let_stmt(access_name, exp2.get_access()[i]);
                exp2.set_access_dimension(i, tiramisu::var(exp2.get_access()[i].get_data_type(), access_name));
                DEBUG(10, tiramisu::str_dump("New access:")); exp2.get_access()[i].dump(false);
            }
        }

        output_expr = exp2;
    }
    else if (exp.get_expr_type() == tiramisu::e_op)
    {
        DEBUG(10, tiramisu::str_dump("Extracting access from e_op."));

        tiramisu::expr exp2, exp3, exp4;
        std::vector<tiramisu::expr> new_arguments;

        switch (exp.get_op_type())
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
        case tiramisu::o_address:
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
        case tiramisu::o_cond:
        case tiramisu::o_lerp:
            exp2 = traverse_expr_and_replace_non_affine_accesses(comp, exp.get_operand(0));
            exp3 = traverse_expr_and_replace_non_affine_accesses(comp, exp.get_operand(1));
            exp4 = traverse_expr_and_replace_non_affine_accesses(comp, exp.get_operand(2));
            output_expr = tiramisu::expr(exp.get_op_type(), exp2, exp3, exp4);
            break;
        case tiramisu::o_call:
            for (const auto &e : exp.get_arguments())
            {
                exp2 = traverse_expr_and_replace_non_affine_accesses(comp, e);
                new_arguments.push_back(exp2);
            }
            output_expr = tiramisu::expr(o_call, exp.get_name(), new_arguments, exp.get_data_type());
            break;
        case tiramisu::o_allocate:
        case tiramisu::o_free:
            output_expr = exp;
            break;
        default:
            tiramisu::error("Unsupported tiramisu expression passed to traverse_expr_and_replace_non_affine_accesses().",
                            1);
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
void generator::get_rhs_accesses(const tiramisu::function *func, const tiramisu::computation *comp,
                      std::vector<isl_map *> &accesses, bool return_buffer_accesses)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    const tiramisu::expr &rhs = comp->get_expr();
    generator::traverse_expr_and_extract_accesses(func, comp, rhs, accesses, return_buffer_accesses);

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(3);
}

tiramisu::expr tiramisu_expr_from_isl_ast_expr(isl_ast_expr *isl_expr)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Input expression: ", isl_ast_expr_to_str(isl_expr)));
    DEBUG_NEWLINE(10);

    tiramisu::expr result;

    if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_int)
    {
        isl_val *init_val = isl_ast_expr_get_val(isl_expr);
        result = tiramisu::expr((int32_t) isl_val_get_num_si(init_val));
        isl_val_free(init_val);
    }
    else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_id)
    {
        isl_id *identifier = isl_ast_expr_get_id(isl_expr);
        std::string name_str(isl_id_get_name(identifier));
        isl_id_free(identifier);
        result = tiramisu::var(tiramisu::p_int32, name_str);
    }
    else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_op)
    {
        tiramisu::expr op0, op1, op2;
        std::vector<tiramisu::expr> new_arguments;

        isl_ast_expr *expr0 = isl_ast_expr_get_op_arg(isl_expr, 0);
        op0 = tiramisu_expr_from_isl_ast_expr(expr0);
        isl_ast_expr_free(expr0);

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 1)
        {
            isl_ast_expr *expr1 = isl_ast_expr_get_op_arg(isl_expr, 1);
            op1 = tiramisu_expr_from_isl_ast_expr(expr1);
            isl_ast_expr_free(expr1);
        }

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 2)
        {
            isl_ast_expr *expr2 = isl_ast_expr_get_op_arg(isl_expr, 2);
            op2 = tiramisu_expr_from_isl_ast_expr(expr2);
            isl_ast_expr_free(expr2);
        }

        switch (isl_ast_expr_get_op_type(isl_expr))
        {
            case isl_ast_op_and:
                result = tiramisu::expr(tiramisu::o_logical_and, op0, op1);
                break;
            case isl_ast_op_and_then:
                result = tiramisu::expr(tiramisu::o_logical_and, op0, op1);
                tiramisu::error("isl_ast_op_and_then operator found in the AST. This operator is not well supported.",
                                0);
                break;
            case isl_ast_op_or:
                result = tiramisu::expr(tiramisu::o_logical_or, op0, op1);
                break;
            case isl_ast_op_or_else:
                result = tiramisu::expr(tiramisu::o_logical_or, op0, op1);
                tiramisu::error("isl_ast_op_or_then operator found in the AST. This operator is not well supported.",
                                0);
                break;
            case isl_ast_op_max:
                result = tiramisu::expr(tiramisu::o_max, op0, op1);
                break;
            case isl_ast_op_min:
                result = tiramisu::expr(tiramisu::o_min, op0, op1);
                break;
            case isl_ast_op_minus:
                result = tiramisu::expr(tiramisu::o_minus, op0);
                break;
            case isl_ast_op_add:
                result = tiramisu::expr(tiramisu::o_add, op0, op1);
                break;
            case isl_ast_op_sub:
                result = tiramisu::expr(tiramisu::o_sub, op0, op1);
                break;
            case isl_ast_op_mul:
                result = tiramisu::expr(tiramisu::o_mul, op0, op1);
                break;
            case isl_ast_op_div:
                result = tiramisu::expr(tiramisu::o_div, op0, op1);
                break;
            case isl_ast_op_fdiv_q:
            case isl_ast_op_pdiv_q:
                result = tiramisu::expr(tiramisu::o_div, op0, op1);
                result = tiramisu::expr(tiramisu::o_floor, result);
                result = tiramisu::expr(tiramisu::o_cast, tiramisu::p_int32, result);
                break;
            case isl_ast_op_zdiv_r:
            case isl_ast_op_pdiv_r:
                result = tiramisu::expr(tiramisu::o_mod, op0, op1);
                break;
            case isl_ast_op_select:
            case isl_ast_op_cond:
                result = tiramisu::expr(tiramisu::o_select, op0, op1, op2);
                break;
            case isl_ast_op_le:
                result = tiramisu::expr(tiramisu::o_le, op0, op1);
                break;
            case isl_ast_op_lt:
                result = tiramisu::expr(tiramisu::o_lt, op0, op1);
                break;
            case isl_ast_op_ge:
                result = tiramisu::expr(tiramisu::o_ge, op0, op1);
                break;
            case isl_ast_op_gt:
                result = tiramisu::expr(tiramisu::o_gt, op0, op1);
                break;
            case isl_ast_op_eq:
                result = tiramisu::expr(tiramisu::o_eq, op0, op1);
                break;
            default:
                tiramisu::str_dump("Transforming the following expression",
                                   (const char *)isl_ast_expr_to_C_str(isl_expr));
                tiramisu::str_dump("\n");
                tiramisu::error("Translating an unsupported ISL expression into a Tiramisu expression.", 1);
        }
    }
    else
    {
        tiramisu::str_dump("Transforming the following expression",
                           (const char *)isl_ast_expr_to_C_str(isl_expr));
        tiramisu::str_dump("\n");
        tiramisu::error("Translating an unsupported ISL expression into a Tiramisu expression.", 1);
    }

    DEBUG_INDENT(-4);

    return result;
}

/**
 * Traverse the expression idx_expr and replace any occurrence of an iterator
 * (of the original loop) by its transformed form. Returned the transformed
 * expression.
 *
 * This would transform the occurences of the indices i, j by their equivalent
 * c0*10+c2 and c1*10+c3 for example.
 */
tiramisu::expr replace_original_indices_with_transformed_indices(tiramisu::expr exp,
                                                                 std::map<std::string, isl_ast_expr *> iterators_map)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Input expression: "); exp.dump(false));
    DEBUG_NEWLINE(10);

    tiramisu::expr output_expr;

    if (exp.get_expr_type() == tiramisu::e_val)
    {
        output_expr = exp;
    }
    else if (exp.get_expr_type() == tiramisu::e_var)
    {
        output_expr = tiramisu_expr_from_isl_ast_expr(iterators_map[exp.get_name()]);
    }
    else if ((exp.get_expr_type() == tiramisu::e_op) && (exp.get_op_type() == tiramisu::o_access))
    {
        DEBUG(10, tiramisu::str_dump("Replacing the occurrences of original iterators in an o_access."));

        for (const auto &access : exp.get_access())
        {
            replace_original_indices_with_transformed_indices(access, iterators_map);
        }

        output_expr = exp;
    }
    else if (exp.get_expr_type() == tiramisu::e_op)
    {
        DEBUG(10, tiramisu::str_dump("Replacing iterators in an e_op."));

        tiramisu::expr exp2, exp3, exp4;
        std::vector<tiramisu::expr> new_arguments;

        switch (exp.get_op_type())
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
            case tiramisu::o_address:
                exp2 = replace_original_indices_with_transformed_indices(exp.get_operand(0), iterators_map);
                output_expr = tiramisu::expr(exp.get_op_type(), exp2);
                break;
            case tiramisu::o_cast:
                exp2 = replace_original_indices_with_transformed_indices(exp.get_operand(0), iterators_map);
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
                exp2 = replace_original_indices_with_transformed_indices(exp.get_operand(0), iterators_map);
                exp3 = replace_original_indices_with_transformed_indices(exp.get_operand(1), iterators_map);
                output_expr = tiramisu::expr(exp.get_op_type(), exp2, exp3);
                break;
            case tiramisu::o_select:
            case tiramisu::o_cond:
            case tiramisu::o_lerp:
                exp2 = replace_original_indices_with_transformed_indices(exp.get_operand(0), iterators_map);
                exp3 = replace_original_indices_with_transformed_indices(exp.get_operand(1), iterators_map);
                exp4 = replace_original_indices_with_transformed_indices(exp.get_operand(2), iterators_map);
                output_expr = tiramisu::expr(exp.get_op_type(), exp2, exp3, exp4);
                break;
            case tiramisu::o_call:
                for (const auto &e : exp.get_arguments())
                {
                    exp2 = replace_original_indices_with_transformed_indices(e, iterators_map);
                    new_arguments.push_back(exp2);
                }
                output_expr = tiramisu::expr(o_call, exp.get_name(), new_arguments, exp.get_data_type());
                break;
            case tiramisu::o_allocate:
            case tiramisu::o_free:
                output_expr = exp;
                break;
            default:
                tiramisu::error("Unsupported tiramisu expression passed to replace_original_indices_with_transformed_indices().", 1);
        }
    }

    DEBUG_INDENT(-4);

    return output_expr;

}

std::map<std::string, isl_ast_expr *> generator::compute_iterators_map(tiramisu::computation *comp, isl_ast_build *build)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::map<std::string, isl_ast_expr *> iterators_map;

    /**
     * Create a fake access (that contains all the iterators).
     * Pass it to the function create_isl_ast_index_expression();
     * that creates the transformed index expressions.
     */
    isl_set *dom = isl_set_copy(comp->get_iteration_domain());
    isl_map *identity = isl_set_identity(isl_set_copy(dom));
    isl_map *schedule = isl_map_copy(comp->get_trimmed_union_of_schedules()); //isl_map_copy(isl_map_from_union_map(isl_ast_build_get_schedule(build)));
    isl_map_dump(identity);
    isl_map_dump(schedule);
    identity = isl_map_apply_domain(identity, schedule);

    DEBUG(3, tiramisu::str_dump("Creating an isl_ast_index_expression for the access :",
                                isl_map_to_str(identity)));
    isl_ast_expr *idx_expr = create_isl_ast_index_expression(build, identity);
    DEBUG(3, tiramisu::str_dump("The created isl_ast_expr expression for the index expression is :",
                                isl_ast_expr_to_str(idx_expr)));

    isl_space *dom_space = isl_set_get_space(dom);

    DEBUG(3, tiramisu::str_dump("The iterators map is :"));

    // Add each index in idx_expr to iterators_map to create the correspondence
    // between the names of indices and their transformed indices.
    // The first op_arg in idx_expr is the name of the buffer so we do not need it.
    for (int i = 1; i < isl_ast_expr_get_op_n_arg(idx_expr); i++)
    {
        if (isl_space_has_dim_name(dom_space, isl_dim_set, i-1))
        {
            std::string original_idx_name = isl_space_get_dim_name(dom_space, isl_dim_set, i - 1);
            isl_ast_expr *transformed_index = isl_ast_expr_get_op_arg(idx_expr, i);
            iterators_map.insert(std::pair<std::string, isl_ast_expr *>(original_idx_name, transformed_index));
            DEBUG(3, tiramisu::str_dump("Original index name = " + original_idx_name + ", Transformed index: ",
                                        isl_ast_expr_to_str(transformed_index)));
        }
    }

    DEBUG_INDENT(-4);

    return iterators_map;
}

/**
 * Retrieve the access function of the ISL AST leaf node (which represents a
 * computation). Store the access in computation->access.
 */
isl_ast_node *generator::stmt_code_generator(isl_ast_node *node, isl_ast_build *build, void *user)
{
    assert(node != NULL);
    assert(build != NULL);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::function *func = (tiramisu::function *)user;

    // Find the name of the computation associated to this AST leaf node.
    std::vector<tiramisu::computation *> comp_vec = get_computation_by_node(func, node);
    assert(!comp_vec.empty() && "get_computation_by_node() returned an empty vector!");
    isl_union_map *sched = isl_ast_build_get_schedule(build);
    isl_union_set *sched_range = isl_union_map_domain(sched);
    assert((sched_range != NULL) && "Range of schedule is NULL.");

    std::vector<tiramisu::computation *> filtered_comp_vec = generator::filter_computations_by_domain(comp_vec, sched_range);
    isl_union_set_free(sched_range);

    for (auto comp: filtered_comp_vec)
    {
        // Mark "comp" as the computation associated with this node.
        isl_id *annotation_id = isl_id_alloc(func->get_isl_ctx(), "", (void *)comp);
        node = isl_ast_node_set_annotation(node, annotation_id);

        assert((comp != NULL) && "Computation not found!");;

        DEBUG(3, tiramisu::str_dump("Computation:", comp->get_name().c_str()));

        // Get the accesses of the computation.  The first access is the access
        // for the LHS.  The following accesses are for the RHS.
        std::vector<isl_map *> accesses;
        if (comp->has_accesses() == true)
        {
            isl_map *access = comp->get_access_relation_adapted_to_time_processor_domain();
            accesses.push_back(access);
            // Add the accesses of the RHS to the accesses vector
            generator::get_rhs_accesses(func, comp, accesses, true);
        }

        if (!accesses.empty())
        {
            DEBUG(3, tiramisu::str_dump("Generated RHS access maps:"));
            DEBUG_INDENT(4);
            for (size_t i = 0; i < accesses.size(); i++)
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

            if (comp->has_accesses())
            {
                comp->dump();

                // For each access in accesses (i.e. for each access in the computation),
                // compute the corresponding isl_ast expression.
                for (size_t i = 0; i < accesses.size(); ++i)
                {
                    if (accesses[i] != NULL)
                    {
                        DEBUG(3, tiramisu::str_dump("Creating an isl_ast_index_expression for the access (isl_map *):",
                                                    isl_map_to_str(accesses[i])));
                        isl_ast_expr *idx_expr = create_isl_ast_index_expression(build, accesses[i]);
                        DEBUG(3, tiramisu::str_dump("The created isl_ast_expr expression for the index expression is :", isl_ast_expr_to_str(idx_expr)));
                        index_expressions.push_back(idx_expr);
                        isl_map_free(accesses[i]);
                    }
                    else
                    {
                        if ((!comp->is_let_stmt()))
                        // If this is not a let stmt and it is supposed to have accesses to other computations,
                        // it should have an access function.
                        {
                            tiramisu::error("An access function should be provided before generating code.", true);
                        }
                    }
                }
            }

            /*
             * Compute the iterators map.
             * The iterators map is map between the original names of the iterators of a computation
             * and their transformed form after schedule (also after renaming).
             *
             * If in the original computation, we had
             *
             * {C[i0, i1]: ...}
             *
             * And if in the generated code, the iterators are called c0, c1, c2 and c3 and
             * the loops are tiled, then the map will be
             *
             * {<i0, c0*10+c2>, <i1, c1*10+c3>}.
             **/
            std::map<std::string, isl_ast_expr *> iterators_map = generator::compute_iterators_map(comp, build);
            comp->set_iterators_map(iterators_map);

            // We want to insert the elements of index_expressions vector one by one in the beginning of comp->get_index_expr()
            for (int i = index_expressions.size() - 1; i >= 0; i--)
            {
                comp->get_index_expr().insert(comp->get_index_expr().begin(), index_expressions[i]);
            }

            for (const auto &i_expr : comp->get_index_expr())
            {
                DEBUG(3, tiramisu::str_dump("Generated Index expression:", (const char *)
                                            isl_ast_expr_to_C_str(i_expr)));
            }
        }
        else
        {
            DEBUG(3, tiramisu::str_dump("Generated RHS empty."));
        }
    }

    DEBUG_FCT_NAME(3);
    DEBUG(3, tiramisu::str_dump("\n\n"));
    DEBUG_INDENT(-4);

    return node;
}

void print_isl_ast_expr_vector(
    const std::vector<isl_ast_expr *> &index_expr_cp)
{
    DEBUG(3, tiramisu::str_dump("List of index expressions."));
    for (const auto &i_expr : index_expr_cp)
    {
        DEBUG(3, tiramisu::str_dump(" ", (const char *)isl_ast_expr_to_C_str(i_expr)));
    }
}

Halide::Expr halide_expr_from_isl_ast_expr(isl_ast_expr *isl_expr)
{
    Halide::Expr result;

    if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_int)
    {
        isl_val *init_val = isl_ast_expr_get_val(isl_expr);
        result = Halide::Expr((int32_t)isl_val_get_num_si(init_val));
        isl_val_free(init_val);
    }
    else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_id)
    {
        isl_id *identifier = isl_ast_expr_get_id(isl_expr);
        std::string name_str(isl_id_get_name(identifier));
        isl_id_free(identifier);
        result = Halide::Internal::Variable::make(Halide::Int(32), name_str);
    }
    else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_op)
    {
        Halide::Expr op0, op1, op2;

        isl_ast_expr *expr0 = isl_ast_expr_get_op_arg(isl_expr, 0);
        op0 = halide_expr_from_isl_ast_expr(expr0);
        isl_ast_expr_free(expr0);

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 1)
        {
            isl_ast_expr *expr1 = isl_ast_expr_get_op_arg(isl_expr, 1);
            op1 = halide_expr_from_isl_ast_expr(expr1);
            isl_ast_expr_free(expr1);
        }

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 2)
        {
            isl_ast_expr *expr2 = isl_ast_expr_get_op_arg(isl_expr, 2);
            op2 = halide_expr_from_isl_ast_expr(expr2);
            isl_ast_expr_free(expr2);
        }

        switch (isl_ast_expr_get_op_type(isl_expr))
        {
        case isl_ast_op_and:
            result = Halide::Internal::And::make(op0, op1);
            break;
        case isl_ast_op_and_then:
            result = Halide::Internal::And::make(op0, op1);
            tiramisu::error("isl_ast_op_and_then operator found in the AST. This operator is not well supported.",
                            0);
            break;
        case isl_ast_op_or:
            result = Halide::Internal::Or::make(op0, op1);
            break;
        case isl_ast_op_or_else:
            result = Halide::Internal::Or::make(op0, op1);
            tiramisu::error("isl_ast_op_or_then operator found in the AST. This operator is not well supported.",
                            0);
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
                               (const char *)isl_ast_expr_to_C_str(isl_expr));
            tiramisu::str_dump("\n");
            tiramisu::error("Translating an unsupported ISL expression in a Halide expression.", 1);
        }
    }
    else
    {
        tiramisu::str_dump("Transforming the following expression",
                           (const char *)isl_ast_expr_to_C_str(isl_expr));
        tiramisu::str_dump("\n");
        tiramisu::error("Translating an unsupported ISL expression in a Halide expression.", 1);
    }

    return result;
}

std::vector<std::pair<std::string, Halide::Expr>> let_stmts_vector;
std::vector<tiramisu::computation *> allocate_stmts_vector;

// For each node of the ISL AST, the corresponding computation is stored.
// This function retrieves that computation.
tiramisu::computation *get_computation_annotated_in_a_node(isl_ast_node *node)
{
    // Retrieve the computation of the node.
    isl_id *comp_id = isl_ast_node_get_annotation(node);
    tiramisu::computation *comp = (tiramisu::computation *)isl_id_get_user(comp_id);
    isl_id_free(comp_id);
    return comp;
}

Halide::Internal::Stmt tiramisu::generator::halide_stmt_from_isl_node(
    const tiramisu::function &fct, isl_ast_node *node,
    int level, std::vector<std::string> &tagged_stmts,
    bool is_a_child_block)
{
    assert(node != NULL);
    assert(level >= 0);

    Halide::Internal::Stmt result;

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if (isl_ast_node_get_type(node) == isl_ast_node_block)
    {
        DEBUG(3, tiramisu::str_dump("Generating code for a block"));

        isl_ast_node_list *list = isl_ast_node_block_get_children(node);

        for (int i = isl_ast_node_list_n_ast_node(list) - 1; i >= 0; i--)
        {
            isl_ast_node *child = isl_ast_node_list_get_ast_node(list, i);

            Halide::Internal::Stmt block;

            if ((isl_ast_node_get_type(child) == isl_ast_node_user) &&
                ((get_computation_annotated_in_a_node(child)->get_expr().get_op_type() == tiramisu::o_allocate) ||
                 (get_computation_annotated_in_a_node(child)->get_expr().get_op_type() == tiramisu::o_free)))
            {
                tiramisu::computation *comp = get_computation_annotated_in_a_node(child);
                if (get_computation_annotated_in_a_node(child)->get_expr().get_op_type() == tiramisu::o_allocate)
                {
                    DEBUG(3, tiramisu::str_dump("Adding a computation to vector of allocate stmts (for later construction)"));
                    allocate_stmts_vector.push_back(comp);
                }
                else
                    block = Halide::Internal::Free::make(comp->get_name());
            }
            else
            {
                DEBUG(3, tiramisu::str_dump("Generating block."));
                // Generate a child block
                block = tiramisu::generator::halide_stmt_from_isl_node(fct, child, level, tagged_stmts, true);
            }
            isl_ast_node_free(child);

            DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Generated block: "); std::cout << block);

            if (block.defined() == false) // Probably block is a let stmt.
            {
                if (!let_stmts_vector.empty()) // i.e. non-consumed let statements
                {
                    // if some stmts have already been created in this loop we can generate LetStmt
                    if (result.defined())
                    {
                        for (const auto &l_stmt : let_stmts_vector)
                        {
                            DEBUG(3, tiramisu::str_dump("Generating the following let statement."));
                            DEBUG(3, tiramisu::str_dump("Name : " + l_stmt.first));
                            DEBUG(3, tiramisu::str_dump("Expression of the let statement: ");
                                  std::cout << l_stmt.second);

                            result = Halide::Internal::LetStmt::make(
                                         l_stmt.first,
                                         l_stmt.second,
                                         result);

                            DEBUG(10, tiramisu::str_dump("Generated let stmt:"));
                            DEBUG_NO_NEWLINE(10, std::cout << result);
                        }
                        let_stmts_vector.clear();
                    }
                    // else, if (!result.defined()), continue creating stmts
                    // until the first actual stmt (result) is created.
                }
                // else, if (let_stmts_vector.empty()), continue looping to
                // create more let stmts and to encounter a real statement

            }
            else // ((block.defined())
            {
                if (result.defined())
                {
                    result = Halide::Internal::Block::make(block, result);
                }
                else // (!result.defined())
                {
                    result = block;
                }
            }
            DEBUG(3, std::cout << "Result is now: " << result);
        }

        /**
         *  Generate all the "allocate" statements (which should be declared on all the block)
            that's why they are left to be create last.
            We only generate "allocate" statements if we are not in a child block.
            In ISL is is possible to have the following blocks

            { // Main block
                { // child block
                    allocate
                    assignment
                }
                assignment
            }

            But in general we want the "allocate" to be declared in the main block
            because the two assignments are using the buffer.  So we should only
            generate the allocate when we are in the main block, not in a child block.
         */
        if ((allocate_stmts_vector.size() != 0) && (is_a_child_block == false))
        {
            Halide::Internal::Stmt block;
            for (auto comp: allocate_stmts_vector)
            {
                assert(comp != NULL);
                DEBUG(10, tiramisu::str_dump("The computation that corresponds to the child of this node: "); comp->dump());

                std::string buffer_name = comp->get_expr().get_name();
                DEBUG(10, tiramisu::str_dump("The computation of the node is an allocate or a free IR node."));
                DEBUG(10, tiramisu::str_dump("The buffer that should be allocated or freed is " + buffer_name));
                tiramisu::buffer *buf = comp->get_function()->get_buffers().find(buffer_name)->second;

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
                    const auto sz = buf->get_dim_sizes()[i];
                    std::vector<isl_ast_expr *> ie = {};
                    halide_dim_sizes.push_back(generator::halide_expr_from_tiramisu_expr(NULL, ie, sz));
                }

                if (comp->get_expr().get_op_type() == tiramisu::o_allocate)
                {
                    result = Halide::Internal::Allocate::make(
                           buf->get_name(),
                           halide_type_from_tiramisu_type(buf->get_elements_type()),
                           halide_dim_sizes, Halide::Internal::const_true(), result);

                    buf->mark_as_allocated();
                }
            }
            allocate_stmts_vector.clear();
        }

        isl_ast_node_list_free(list);
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_for)
    {
        DEBUG(3, tiramisu::str_dump("Generating code for Halide::For"));

        isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
        char *cstr = isl_ast_expr_to_C_str(iter);
        std::string iterator_str = std::string(cstr);

        isl_ast_expr *init = isl_ast_node_for_get_init(node);
        isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
        isl_ast_expr *inc  = isl_ast_node_for_get_inc(node);

        isl_val *inc_val = isl_ast_expr_get_val(inc);
        if (!isl_val_is_one(inc_val))
        {
            tiramisu::error("The increment in one of the loops is not +1."
                            "This is not supported by Halide", 1);
        }
        isl_val_free(inc_val);

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
        {
            cond_upper_bound_isl_format = isl_ast_expr_get_op_arg(cond, 1);
        }
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
        {
            tiramisu::error("The for loop upper bound is not an isl_est_expr of type le or lt" , 1);
        }
        assert(cond_upper_bound_isl_format != NULL);
        DEBUG(3, tiramisu::str_dump("Creating for loop init expression."));

        Halide::Expr init_expr = halide_expr_from_isl_ast_expr(init);
        if (init_expr.type() !=
                halide_type_from_tiramisu_type(global::get_loop_iterator_default_data_type()))
        {
            init_expr =
                Halide::Internal::Cast::make(
                    halide_type_from_tiramisu_type(global::get_loop_iterator_default_data_type()),
                    init_expr);
        }
        DEBUG(3, tiramisu::str_dump("init expression: "); std::cout << init_expr);
        Halide::Expr cond_upper_bound_halide_format =
            simplify(halide_expr_from_isl_ast_expr(cond_upper_bound_isl_format));
        if (cond_upper_bound_halide_format.type() !=
                halide_type_from_tiramisu_type(global::get_loop_iterator_default_data_type()))
        {
            cond_upper_bound_halide_format =
                Halide::Internal::Cast::make(
                    halide_type_from_tiramisu_type(global::get_loop_iterator_default_data_type()),
                    cond_upper_bound_halide_format);
        }
        DEBUG(3, tiramisu::str_dump("Upper bound expression: ");
              std::cout << cond_upper_bound_halide_format);
        Halide::Internal::Stmt halide_body =
                tiramisu::generator::halide_stmt_from_isl_node(fct, body, level + 1, tagged_stmts);
        Halide::Internal::ForType fortype = Halide::Internal::ForType::Serial;
        Halide::DeviceAPI dev_api = Halide::DeviceAPI::Host;

        // Change the type from Serial to parallel or vector if the
        // current level was marked as such.
        size_t tt = 0;
        while (tt < tagged_stmts.size())
        {
            if (tagged_stmts[tt] != "")
            {
                if (fct.should_parallelize(tagged_stmts[tt], level))
                {
                    fortype = Halide::Internal::ForType::Parallel;
                    // Since this statement is treated, remove it from the list of
                    // tagged statements so that it does not get treated again later.
                    tagged_stmts[tt] = "";
                    // As soon as we find one tagged statement that actually useful we exit
                    break;
                }
                else if (fct.should_vectorize(tagged_stmts[tt], level))
                {
                    DEBUG(3, tiramisu::str_dump("Trying to vectorize at level ");
                          tiramisu::str_dump(std::to_string(level)));

		    int vector_length = fct.get_vector_length(tagged_stmts[tt], level);

		    for (auto vd: fct.vector_dimensions)
			    std::cout << "stmt = " << std::get<0>(vd) << ", level = " << std::get<1>(vd) << ", length = " << std::get<2>(vd) << std::endl;

		    DEBUG(3, tiramisu::str_dump("Vector length = ");
                          tiramisu::str_dump(std::to_string(vector_length)));

		    // Currently we assume that when vectorization is used,
		    // then the original loop extent is > vector_length.
                    cond_upper_bound_halide_format = Halide::Expr(vector_length);
		    fortype = Halide::Internal::ForType::Vectorized;
	            DEBUG(3, tiramisu::str_dump("Loop vectorized"));

		    /*
			  The following code checks if the upper bound is a constant.

    		  	  const Halide::Internal::IntImm *extent =
                       	  	cond_upper_bound_halide_format.as<Halide::Internal::IntImm>();

  	                  if (!extent)
        	            {
                	        DEBUG(3, tiramisu::str_dump("Loop not vectorized (extent is non constant)"));
                        	// Currently we can only print Halide expressions using
	                        // "std::cout << ".
        	                DEBUG(3, std::cout << cond_upper_bound_halide_format << std::endl);
                	    }
		    */

                    // Since this statement is treated, remove it from the list of
                    // tagged statements so that it does not get treated again later.
                    tagged_stmts[tt] = "";
                    break;
                }
                else if (fct.should_map_to_gpu_thread(tagged_stmts[tt], level))
                {
                    // TODO(tiramisu): The for-type should have been "GPUThread"
                    fortype = Halide::Internal::ForType::Parallel;
                    dev_api = Halide::DeviceAPI::OpenCL;
                    std::string gpu_iter = fct.get_gpu_thread_iterator(tagged_stmts[tt], level);
                    Halide::Expr new_iterator_var =
                        Halide::Internal::Variable::make(Halide::Int(32), gpu_iter);
                    halide_body = Halide::Internal::LetStmt::make(
                                      iterator_str,
                                      new_iterator_var,
                                      halide_body);
                    iterator_str = gpu_iter;
                    DEBUG(3, tiramisu::str_dump("Loop over " + gpu_iter + " created.\n"));

                    // Since this statement is treated, remove it from the list of
                    // tagged statements so that it does not get treated again later.
                    tagged_stmts[tt] = "";
                    break;
                }
                else if (fct.should_map_to_gpu_block(tagged_stmts[tt], level))
                {
                    // TODO(tiramisu): The for-type should have been "GPUBlock"
                    fortype = Halide::Internal::ForType::Parallel;
                    dev_api = Halide::DeviceAPI::OpenCL;
                    std::string gpu_iter = fct.get_gpu_block_iterator(tagged_stmts[tt], level);
                    Halide::Expr new_iterator_var =
                        Halide::Internal::Variable::make(Halide::Int(32), gpu_iter);
                    halide_body = Halide::Internal::LetStmt::make(
                                      iterator_str,
                                      new_iterator_var,
                                      halide_body);
                    iterator_str = gpu_iter;
                    DEBUG(3, tiramisu::str_dump("Loop over " + gpu_iter + " created.\n"));

                    // Since this statement is treated, remove it from the list of
                    // tagged statements so that it does not get treated again later.
                    tagged_stmts[tt] = "";
                    break;
                }
                else if (fct.should_unroll(tagged_stmts[tt], level))
                {
                    DEBUG(3, tiramisu::str_dump("Trying to unroll at level ");
                          tiramisu::str_dump(std::to_string(level)));

                    const Halide::Internal::IntImm *extent =
                        cond_upper_bound_halide_format.as<Halide::Internal::IntImm>();
                    if (extent)
                    {
                        fortype = Halide::Internal::ForType::Unrolled;
                        DEBUG(3, tiramisu::str_dump("Loop unrolled"));
                    }
                    else
                    {
                        DEBUG(3, tiramisu::str_dump("Loop not unrolled (extent is non constant)"));
                        DEBUG(3, std::cout << cond_upper_bound_halide_format << std::endl);
                    }

                    // Since this statement is treated, remove it from the list of
                    // tagged statements so that it does not get treated again later.
                    tagged_stmts[tt] = "";
                    break;
                }
            }
            tt++;
        }

        DEBUG(10, tiramisu::str_dump("The full list of tagged statements is now:"));
        for (const auto &ts: tagged_stmts)
            DEBUG(10, tiramisu::str_dump(ts + " "));

        DEBUG(3, tiramisu::str_dump("Creating the for loop."));
        result = Halide::Internal::For::make(iterator_str, init_expr, cond_upper_bound_halide_format - init_expr,
                                             fortype, dev_api, halide_body);
        DEBUG(3, tiramisu::str_dump("For loop created."));
        DEBUG(10, std::cout << result);

        isl_ast_expr_free(iter);
        free(cstr);
        isl_ast_expr_free(init);
        isl_ast_expr_free(cond);
        isl_ast_expr_free(inc);
        isl_ast_node_free(body);
        isl_ast_expr_free(cond_upper_bound_isl_format);
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_user)
    {
        DEBUG(3, tiramisu::str_dump("Generating code for user node"));

        if ((isl_ast_node_get_type(node) == isl_ast_node_user) &&
            ((get_computation_annotated_in_a_node(node)->get_expr().get_op_type() == tiramisu::o_allocate) ||
             (get_computation_annotated_in_a_node(node)->get_expr().get_op_type() == tiramisu::o_free)))
          {
            if (get_computation_annotated_in_a_node(node)->get_expr().get_op_type() == tiramisu::o_allocate)
                tiramisu::error("Allocate node should not appear as a user ISL AST node. It should only appear with block construction (because of its scope).", true);
            else
            {
                tiramisu::computation *comp = get_computation_annotated_in_a_node(node);
                result = Halide::Internal::Free::make(comp->get_name());
            }
          }
        else
        {
            isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
            isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
            isl_id *id = isl_ast_expr_get_id(arg);
            isl_ast_expr_free(expr);
            isl_ast_expr_free(arg);
            std::string computation_name(isl_id_get_name(id));
            DEBUG(3, tiramisu::str_dump("Computation name: "); tiramisu::str_dump(computation_name));
            isl_id_free(id);

            // Check if any loop around this statement should be
            // parallelized, vectorized or mapped to GPU.
            for (int l = 0; l < level; l++)
            {
                if (fct.should_parallelize(computation_name, l))
                    tagged_stmts.push_back(computation_name);
                if (fct.should_vectorize(computation_name, l))
                    tagged_stmts.push_back(computation_name);
                if (fct.should_map_to_gpu_block(computation_name, l))
                    tagged_stmts.push_back(computation_name);
                if (fct.should_map_to_gpu_thread(computation_name, l))
                    tagged_stmts.push_back(computation_name);
                if (fct.should_unroll(computation_name, l))
                    tagged_stmts.push_back(computation_name);
            }

            DEBUG(10, tiramisu::str_dump("The full list of tagged statements is now"));
            for (const auto &ts: tagged_stmts)
                DEBUG(10, tiramisu::str_dump(ts + " "));

            // Retrieve the computation of the node.
            tiramisu::computation *comp = get_computation_annotated_in_a_node(node);
            DEBUG(10, tiramisu::str_dump("The computation that corresponds to this node: "); comp->dump());

            comp->create_halide_assignment();
            result = comp->get_generated_halide_stmt();


            for (const auto &l_stmt : comp->get_associated_let_stmts())
            {
                DEBUG(3, tiramisu::str_dump("Generating the following let statement."));
                DEBUG(3, tiramisu::str_dump("Name : " + l_stmt.first));
                DEBUG(3, tiramisu::str_dump("Expression of the let statement: "));

                l_stmt.second.dump(false);

                std::vector<isl_ast_expr *> ie = {}; // Dummy variable.
                tiramisu::expr tiramisu_let = replace_original_indices_with_transformed_indices(l_stmt.second, comp->get_iterators_map());
                Halide::Expr let_expr = halide_expr_from_tiramisu_expr(comp, ie, tiramisu_let);
                        result = Halide::Internal::LetStmt::make(
                             l_stmt.first,
                             let_expr,
                             result);

                DEBUG(10, tiramisu::str_dump("Generated let stmt:"));
                DEBUG_NO_NEWLINE(10, std::cout << result);
            }

            if (comp->get_predicate().is_defined())
            {
                std::vector<isl_ast_expr *> ie = {}; // Dummy variable.
                tiramisu::expr tiramisu_predicate = replace_original_indices_with_transformed_indices(comp->get_predicate(),
                                                                                                      comp->get_iterators_map());
                Halide::Expr predicate = halide_expr_from_tiramisu_expr(comp, ie, tiramisu_predicate);
                DEBUG(3, tiramisu::str_dump("Adding a predicate around the computation."); std::cout << predicate);
                DEBUG(3, tiramisu::str_dump("Generating code for the if branch."));

                Halide::Internal::Stmt if_s = result;
                DEBUG(10, tiramisu::str_dump("If branch: "); std::cout << if_s);

                Halide::Internal::Stmt else_s;

                result = Halide::Internal::IfThenElse::make(predicate, if_s, else_s);
                DEBUG(10, tiramisu::str_dump("The predicated statement is "); std::cout << result);
            }
        }
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_if)
    {
        DEBUG(3, tiramisu::str_dump("Generating code for conditional"));

        isl_ast_expr *cond = isl_ast_node_if_get_cond(node);
        isl_ast_node *if_stmt = isl_ast_node_if_get_then(node);
        isl_ast_node *else_stmt = isl_ast_node_if_get_else(node);


        if ((isl_ast_node_get_type(if_stmt) == isl_ast_node_user) &&
            ((get_computation_annotated_in_a_node(if_stmt)->get_expr().get_op_type() == tiramisu::o_allocate)))
        {
            tiramisu::computation *comp = get_computation_annotated_in_a_node(if_stmt);
            if (get_computation_annotated_in_a_node(if_stmt)->get_expr().get_op_type() == tiramisu::o_allocate)
            {
                DEBUG(3, tiramisu::str_dump("Adding a computation to vector of allocate stmts (for later construction)"));
                allocate_stmts_vector.push_back(comp);
            }
        }
        else
        {
            Halide::Expr c = halide_expr_from_isl_ast_expr(cond);

            DEBUG(3, tiramisu::str_dump("Condition: "); std::cout << c);
            DEBUG(3, tiramisu::str_dump("Generating code for the if branch."));

            Halide::Internal::Stmt if_s =
                    tiramisu::generator::halide_stmt_from_isl_node(fct, if_stmt,
                                                    level, tagged_stmts);

            DEBUG(10, tiramisu::str_dump("If branch: "); std::cout << if_s);

            Halide::Internal::Stmt else_s;

            if (else_stmt != NULL)
            {
                if ((isl_ast_node_get_type(else_stmt) == isl_ast_node_user) &&
                    ((get_computation_annotated_in_a_node(else_stmt)->get_expr().get_op_type() == tiramisu::o_allocate)))
                  {
                    tiramisu::computation *comp = get_computation_annotated_in_a_node(else_stmt);
                    if (get_computation_annotated_in_a_node(else_stmt)->get_expr().get_op_type() == tiramisu::o_allocate)
                    {
                      DEBUG(3, tiramisu::str_dump("Adding a computation to vector of allocate stmts (for later construction)"));
                      allocate_stmts_vector.push_back(comp);
                    }
                  }
                else
                {
                    DEBUG(3, tiramisu::str_dump("Generating code for the else branch."));
                    else_s = tiramisu::generator::halide_stmt_from_isl_node(fct, else_stmt, level, tagged_stmts);
                    DEBUG(10, tiramisu::str_dump("Else branch: "); std::cout << else_s);
                }
            }
            else
            {
                DEBUG(3, tiramisu::str_dump("Else statement is NULL."));
            }

            result = Halide::Internal::IfThenElse::make(c, if_s, else_s);
            DEBUG(10, tiramisu::str_dump("IfThenElse statement: "); std::cout << result);

            isl_ast_expr_free(cond);
            isl_ast_node_free(if_stmt);
            isl_ast_node_free(else_stmt);
        }
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
    Halide::Internal::Stmt stmt;

    // Generate the statement that represents the whole function
    stmt = tiramisu::generator::halide_stmt_from_isl_node(*this, this->get_isl_ast(), 0, generated_stmts);

    // Allocate buffers that are not passed as an argument to the function
    for (const auto &b : this->get_buffers())
    {
        tiramisu::buffer *buf = b.second;
        // Allocate only arrays that are not passed to the function as arguments.
        if (buf->get_argument_type() == tiramisu::a_temporary && buf->get_auto_allocate() == true)
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
                const auto sz = buf->get_dim_sizes()[i];
                std::vector<isl_ast_expr *> ie = {};
                halide_dim_sizes.push_back(generator::halide_expr_from_tiramisu_expr(NULL, ie, sz));
            }
            stmt = Halide::Internal::Allocate::make(
                       buf->get_name(),
                       halide_type_from_tiramisu_type(buf->get_elements_type()),
                       halide_dim_sizes, Halide::Internal::const_true(), stmt);

            buf->mark_as_allocated();
        }
    }

    const auto &invariant_vector = this->get_invariants();

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
        const auto &param = invariant_vector[i]; // Get the i'th invariant
        std::vector<isl_ast_expr *> ie = {};
        stmt = Halide::Internal::LetStmt::make(
                   param.get_name(),
                   generator::halide_expr_from_tiramisu_expr(NULL, ie, param.get_expr()),
                   stmt);
    }

    this->halide_stmt = stmt;

    DEBUG(3, tiramisu::str_dump("\n\nGenerated Halide stmt before lowering:"));
    DEBUG(3, std::cout << stmt);

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
Halide::Expr linearize_access(int dims, const halide_dimension_t *shape, isl_ast_expr *index_expr)
{
    assert(isl_ast_expr_get_op_n_arg(index_expr) > 1);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // ISL dimension is ordered from outermost to innermost.

    Halide::Expr index = 0;
    for (int i = dims; i >= 1; --i)
    {
        isl_ast_expr *operand = isl_ast_expr_get_op_arg(index_expr, i);
        Halide::Expr operand_h = halide_expr_from_isl_ast_expr(operand);
        index += operand_h * shape[dims - i].stride;
        isl_ast_expr_free(operand);
    }

    DEBUG_INDENT(-4);

    return index;
}

Halide::Expr linearize_access(int dims, std::vector<Halide::Expr> &strides, isl_ast_expr *index_expr)
{
    assert(isl_ast_expr_get_op_n_arg(index_expr) > 1);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // ISL dimension is ordered from outermost to innermost.

    Halide::Expr index = 0;
    for (int i = dims; i >= 1; --i)
    {
        isl_ast_expr *operand = isl_ast_expr_get_op_arg(index_expr, i);
        Halide::Expr operand_h = halide_expr_from_isl_ast_expr(operand);
        index += operand_h * strides[dims - i];
        isl_ast_expr_free(operand);
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
        DEBUG_NO_NEWLINE(10, tiramisu::str_dump("The expression associated with the let statement: ");
                         this->expression.dump(false));
        DEBUG_NEWLINE(10);

        Halide::Expr result = generator::halide_expr_from_tiramisu_expr(this,
                              this->get_index_expr(),
                              this->expression);

        Halide::Type l_type = halide_type_from_tiramisu_type(this->get_data_type());

        if (l_type != result.type())
        {
            result = Halide::Internal::Cast::make(l_type, result);
        }

        const std::string &let_stmt_name = this->get_name();

        let_stmts_vector.push_back(
            std::pair<std::string, Halide::Expr>(let_stmt_name, result));
        DEBUG(10, tiramisu::str_dump("A let statement was added to the vector of let statements."));
    }
    else
    {
        DEBUG(3, tiramisu::str_dump("This is not a let statement."));

        const char *buffer_name =
            isl_space_get_tuple_name(
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
        const auto &buffer_entry = this->function->get_buffers().find(buffer_name);
        assert(buffer_entry != this->function->get_buffers().end());

        const auto &tiramisu_buffer = buffer_entry->second;
        DEBUG(3, tiramisu::str_dump("A Tiramisu buffer that corresponds to the buffer indicated in the access relation was found."));

        DEBUG(10, tiramisu_buffer->dump(true));

        Halide::Type type = halide_type_from_tiramisu_type(this->get_data_type());
        int buf_dims = tiramisu_buffer->get_dim_sizes().size();

        // Tiramisu buffer is from outermost to innermost, whereas Halide buffer is
        // from innermost to outermost; thus, we need to reverse the order
        halide_dimension_t *shape = new halide_dimension_t[tiramisu_buffer->get_dim_sizes().size()];
        int stride = 1;
        std::vector<Halide::Expr> strides_vector;

        if (tiramisu_buffer->has_constant_extents())
        {
            for (int i = 0; i < buf_dims; i++)
            {
                shape[i].min = 0;
                int dim_idx = tiramisu_buffer->get_dim_sizes().size() - i - 1;
                shape[i].extent = (int) tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
                shape[i].stride = stride;
                stride *= (int) tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
            }
        }
        else
        {
            std::vector<isl_ast_expr *> empty_index_expr;
            Halide::Expr stride_expr = Halide::Expr(1);
            for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++)
            {
                int dim_idx = tiramisu_buffer->get_dim_sizes().size() - i - 1;
                strides_vector.push_back(stride_expr);
                stride_expr = stride_expr * generator::halide_expr_from_tiramisu_expr(this, empty_index_expr, tiramisu_buffer->get_dim_sizes()[dim_idx]);
            }
        }

        // The number of dimensions in the Halide buffer should be equal to
        // the number of dimensions of the access function.
        assert(buf_dims == access_dims);
        assert(this->index_expr[0] != NULL);
        DEBUG(3, tiramisu::str_dump("Linearizing access of the LHS index expression."));

        Halide::Expr index;
        if (tiramisu_buffer->has_constant_extents())
            index = tiramisu::linearize_access(buf_dims, shape, this->index_expr[0]);
        else
            index = tiramisu::linearize_access(buf_dims, strides_vector, this->index_expr[0]);

        DEBUG(3, tiramisu::str_dump("After linearization: "); std::cout << index << std::endl);

        DEBUG(3, tiramisu::str_dump("Index expressions of this statement are (the first is the LHS and the others are the RHS) :"));
        print_isl_ast_expr_vector(this->index_expr);

        DEBUG(3, tiramisu::str_dump("Erasing the LHS index expression from the vector of index expressions (the LHS index has just been linearized)."));
        this->index_expr.erase(this->index_expr.begin());

        Halide::Internal::Parameter param;

        if (tiramisu_buffer->get_argument_type() == tiramisu::a_output)
        {
            if (tiramisu_buffer->has_constant_extents())
            {
                Halide::Buffer<> buffer =
                    Halide::Buffer<>(
                        halide_type_from_tiramisu_type(tiramisu_buffer->get_elements_type()),
                        tiramisu_buffer->get_data(),
                        tiramisu_buffer->get_dim_sizes().size(),
                        shape,
                        tiramisu_buffer->get_name());
                param = Halide::Internal::Parameter(buffer.type(), true, buffer.dimensions(), buffer.name());
                param.set_buffer(buffer);
                DEBUG(3, tiramisu::str_dump("Halide buffer object created.  This object will be passed to the Halide function that creates an assignment to a buffer."));
            }
            else
            {
                param = Halide::Internal::Parameter(halide_type_from_tiramisu_type(tiramisu_buffer->get_elements_type()),
                                                    true,
                                                    tiramisu_buffer->get_dim_sizes().size(),
                                                    tiramisu_buffer->get_name());
                std::vector<isl_ast_expr *> empty_index_expr;
                Halide::Expr stride_expr = Halide::Expr(1);
                for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++)
                {
                    int dim_idx = tiramisu_buffer->get_dim_sizes().size() - i - 1;
                    param.set_min_constraint(i, Halide::Expr(0));
                    param.set_extent_constraint(i, generator::halide_expr_from_tiramisu_expr(this, empty_index_expr, tiramisu_buffer->get_dim_sizes()[dim_idx]));
                    param.set_stride_constraint(i, stride_expr);
                    stride_expr = stride_expr * generator::halide_expr_from_tiramisu_expr(this, empty_index_expr, tiramisu_buffer->get_dim_sizes()[dim_idx]);
                }
            }
        }

        DEBUG(3, tiramisu::str_dump("Calling the Halide::Internal::Store::make function which creates the store statement."));
        DEBUG(3, tiramisu::str_dump("The RHS index expressions are first transformed to Halide expressions then passed to the make function."));

        this->stmt = Halide::Internal::Store::make (
                         buffer_name,
                         generator::halide_expr_from_tiramisu_expr(this, this->index_expr, this->expression),
                         index, param, Halide::Internal::const_true(type.lanes()));

        DEBUG(3, tiramisu::str_dump("Halide::Internal::Store::make statement created."));
        delete[] shape;
    }

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("End of create_halide_stmt. Generated statement is: ");
                     std::cout << this->stmt);

    DEBUG_INDENT(-4);
}

Halide::Expr generator::halide_expr_from_tiramisu_expr(const tiramisu::computation *comp,
                                                       std::vector<isl_ast_expr *> &index_expr,
                                                       const tiramisu::expr &tiramisu_expr)
{
    Halide::Expr result;

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Input Tiramisu expression: "); tiramisu_expr.dump(false));

    if (tiramisu_expr.get_expr_type() == tiramisu::e_val)
    {
        DEBUG(3, tiramisu::str_dump("tiramisu expression of type tiramisu::e_val"));
        if (tiramisu_expr.get_data_type() == tiramisu::p_uint8)
        {
            result = Halide::Expr(tiramisu_expr.get_uint8_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int8)
        {
            result = Halide::Expr(tiramisu_expr.get_int8_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_uint16)
        {
            result = Halide::Expr(tiramisu_expr.get_uint16_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int16)
        {
            result = Halide::Expr(tiramisu_expr.get_int16_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_uint32)
        {
            result = Halide::Expr(tiramisu_expr.get_uint32_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int32)
        {
            result = Halide::Expr(tiramisu_expr.get_int32_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_uint64)
        {
            result = Halide::Expr(tiramisu_expr.get_uint64_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_int64)
        {
            result = Halide::Expr(tiramisu_expr.get_int64_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_float32)
        {
            result = Halide::Expr(tiramisu_expr.get_float32_value());
        }
        else if (tiramisu_expr.get_data_type() == tiramisu::p_float64)
        {
            result = Halide::Expr(tiramisu_expr.get_float64_value());
        }
    }
    else if (tiramisu_expr.get_expr_type() == tiramisu::e_op)
    {
        Halide::Expr op0, op1, op2;

        DEBUG(3, tiramisu::str_dump("tiramisu expression of type tiramisu::e_op"));

        if (tiramisu_expr.get_n_arg() > 0)
        {
            tiramisu::expr expr0 = tiramisu_expr.get_operand(0);
            op0 = generator::halide_expr_from_tiramisu_expr(comp, index_expr, expr0);
        }

        if (tiramisu_expr.get_n_arg() > 1)
        {
            tiramisu::expr expr1 = tiramisu_expr.get_operand(1);
            op1 = generator::halide_expr_from_tiramisu_expr(comp, index_expr, expr1);
        }

        if (tiramisu_expr.get_n_arg() > 2)
        {
            tiramisu::expr expr2 = tiramisu_expr.get_operand(2);
            op2 = generator::halide_expr_from_tiramisu_expr(comp, index_expr, expr2);
        }

        switch (tiramisu_expr.get_op_type())
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
            case tiramisu::o_lerp:
                result = Halide::lerp(op0, op1, op2);
                DEBUG(3, tiramisu::str_dump("op type: lerp"));
                break;
            case tiramisu::o_cond:
                tiramisu::error("Code generation for o_cond is not supported yet.", true);
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
            case tiramisu::o_address:
            {
                DEBUG(3, tiramisu::str_dump("op type: o_access or o_address"));

                const char *access_comp_name = NULL;

                if (tiramisu_expr.get_op_type() == tiramisu::o_access)
                {
                    access_comp_name = tiramisu_expr.get_name().c_str();
                }
                else if (tiramisu_expr.get_op_type() == tiramisu::o_address)
                {
                    access_comp_name = tiramisu_expr.get_operand(0).get_name().c_str();
                }
                else
                {
                    tiramisu::error("Unsupported operation.", true);
                }

                assert(access_comp_name != NULL);

                DEBUG(3, tiramisu::str_dump("Computation being accessed: "); tiramisu::str_dump(access_comp_name));

                // Since we modify the names of update computations but do not modify the
                // expressions.  When accessing the expressions we find the old names, so
                // we need to look for the new names instead of the old names.
                // We do this instead of actually changing the expressions, because changing
                // the expressions will make the semantics of the printed program ambiguous,
                // since we do not have any way to distinguish between which update is the
                // consumer is consuming exactly.
                std::vector<tiramisu::computation *> computations_vector
                        = comp->get_function()->get_computation_by_name(access_comp_name);
                if (computations_vector.size() == 0)
                {
                    // Search for update computations.
                    computations_vector
                            = comp->get_function()->get_computation_by_name("_" + std::string(access_comp_name) + "_update_0");
                    assert((computations_vector.size() > 0) && "Computation not found.");
                }

                // We assume that computations that have the same name write all to the same buffer
                // but may have different access relations.
                tiramisu::computation *access_comp = computations_vector[0];
                assert((access_comp != NULL) && "Accessed computation is NULL.");
                const char *buffer_name = isl_space_get_tuple_name(
                        isl_map_get_space(access_comp->get_access_relation_adapted_to_time_processor_domain()),
                        isl_dim_out);
                assert(buffer_name != NULL);
                DEBUG(3, tiramisu::str_dump("Name of the associated buffer: "); tiramisu::str_dump(buffer_name));

                const auto &buffer_entry = comp->get_function()->get_buffers().find(buffer_name);
                assert(buffer_entry != comp->get_function()->get_buffers().end());

                const auto &tiramisu_buffer = buffer_entry->second;

                Halide::Type type = halide_type_from_tiramisu_type(tiramisu_buffer->get_elements_type());

                // Tiramisu buffer is from outermost to innermost, whereas Halide buffer is from innermost
                // to outermost; thus, we need to reverse the order
                halide_dimension_t *shape = new halide_dimension_t[tiramisu_buffer->get_dim_sizes().size()];
                int stride = 1;
                std::vector<Halide::Expr> strides_vector;

                if (tiramisu_buffer->has_constant_extents())
                {
                    for (size_t i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++)
                    {
                        shape[i].min = 0;
                        int dim_idx = tiramisu_buffer->get_dim_sizes().size() - i - 1;
                        shape[i].extent = (int)tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
                        shape[i].stride = stride;
                        stride *= (int)tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
                    }
                }
                else
                {
                    std::vector<isl_ast_expr *> empty_index_expr;
                    Halide::Expr stride_expr = Halide::Expr(1);
                    for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++)
                    {
                        int dim_idx = tiramisu_buffer->get_dim_sizes().size() - i - 1;
                        strides_vector.push_back(stride_expr);
                        stride_expr = stride_expr * generator::halide_expr_from_tiramisu_expr(comp, empty_index_expr, tiramisu_buffer->get_dim_sizes()[dim_idx]);
                    }
                }

                if (tiramisu_expr.get_op_type() == tiramisu::o_access)
                {
                    print_isl_ast_expr_vector(index_expr);

                    Halide::Expr index;

                    if (tiramisu_buffer->has_constant_extents())
                        index = tiramisu::linearize_access(tiramisu_buffer->get_dim_sizes().size(), shape, index_expr[0]);
                    else
                        index = tiramisu::linearize_access(tiramisu_buffer->get_dim_sizes().size(), strides_vector, index_expr[0]);

                    index_expr.erase(index_expr.begin());

                    if (tiramisu_buffer->get_argument_type() == tiramisu::a_input)
                    {
                        /*Halide::Buffer<> buffer = Halide::Buffer<>(
                                                      type,
                                                      tiramisu_buffer->get_data(),
                                                      tiramisu_buffer->get_dim_sizes().size(),
                                                      shape,
                                                      tiramisu_buffer->get_name());*/

                        Halide::Internal::Parameter param =
                                Halide::Internal::Parameter(halide_type_from_tiramisu_type(tiramisu_buffer->get_elements_type()),
                                                            true,
                                                            tiramisu_buffer->get_dim_sizes().size(),
                                                            tiramisu_buffer->get_name());

                        // TODO(psuriana): ImageParam is not currently supported.
                        result = Halide::Internal::Load::make(
                                type, tiramisu_buffer->get_name(), index, Halide::Buffer<>(),
                                param, Halide::Internal::const_true(type.lanes()));
                    }
                    else
                    {
                        result = Halide::Internal::Load::make(
                                type, tiramisu_buffer->get_name(), index, Halide::Buffer<>(),
                                Halide::Internal::Parameter(), Halide::Internal::const_true(type.lanes()));
                    }
                }
                else if (tiramisu_expr.get_op_type() == tiramisu::o_address)
                {
                    // Create a pointer to Halide buffer.
                    result = Halide::Internal::Variable::make(Halide::type_of<struct halide_buffer_t *>(),
                                                              tiramisu_buffer->get_name() + ".buffer");
                }
                delete[] shape;
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
            case tiramisu::o_call:
            {
                std::vector<Halide::Expr> vec;
                for (const auto &e : tiramisu_expr.get_arguments())
                {
                    Halide::Expr he = generator::halide_expr_from_tiramisu_expr(comp, index_expr, e);
                    vec.push_back(he);
                }
                result = Halide::Internal::Call::make(halide_type_from_tiramisu_type(tiramisu_expr.get_data_type()),
                                                      tiramisu_expr.get_name(),
                                                      vec,
                                                      Halide::Internal::Call::CallType::Extern);
                DEBUG(3, tiramisu::str_dump("op type: o_call"));
                break;
            }
            case tiramisu::o_allocate:
            case tiramisu::o_free:
                tiramisu::error("An expression of type o_allocate or o_free "
                                        "should not be passed to this function", true);
                break;
            default:
                tiramisu::error("Translating an unsupported ISL expression into a Halide expression.", 1);
        }
    }
    else if (tiramisu_expr.get_expr_type() == tiramisu::e_var)
    {
        result = Halide::Internal::Variable::make(
                halide_type_from_tiramisu_type(tiramisu_expr.get_data_type()),
                tiramisu_expr.get_name());
    }
    else
    {
        tiramisu::str_dump("tiramisu type of expr: ",
                           str_from_tiramisu_type_expr(tiramisu_expr.get_expr_type()).c_str());
        tiramisu::error("\nTranslating an unsupported ISL expression in a Halide expression.", 1);
    }

    if (result.defined())
    {
        DEBUG(10, tiramisu::str_dump("Generated stmt: "); std::cout << result);
    }

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(3);

    return result;
}

void function::gen_halide_obj(const std::string &obj_file_name, Halide::Target::OS os,
                              Halide::Target::Arch arch, int bits) const
{
    // TODO(tiramisu): For GPU schedule, we need to set the features, e.g.
    // Halide::Target::OpenCL, etc.
    std::vector<Halide::Target::Feature> features =
    {
        Halide::Target::AVX, Halide::Target::SSE41
    };
    Halide::Target target(os, arch, bits, features);

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

    Halide::Module m = lower_halide_pipeline(this->get_name(), target, fct_arguments,
                                             Halide::Internal::LoweredFunc::External,
                                             this->get_halide_stmt());

    m.compile(Halide::Outputs().object(obj_file_name));
    m.compile(Halide::Outputs().c_header(obj_file_name + ".h"));
}

}
