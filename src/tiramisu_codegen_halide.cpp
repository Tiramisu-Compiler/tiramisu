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
#include "../3rdParty/Halide/src/Expr.h"
#include "../3rdParty/Halide/src/Parameter.h"
#include "../include/tiramisu/debug.h"
#include "../3rdParty/Halide/src/IR.h"
#include "../include/tiramisu/core.h"

namespace tiramisu
{

std::string generate_new_variable_name();
Halide::Expr make_comm_call(Halide::Type type, std::string func_name, std::vector<Halide::Expr> args);
Halide::Expr halide_expr_from_tiramisu_type(tiramisu::primitive_t ptype);

Halide::Argument::Kind halide_argtype_from_tiramisu_argtype(tiramisu::argument_t type)
{
    Halide::Argument::Kind res;

    if (type == tiramisu::a_temporary)
    {
        ERROR("Buffer type \"temporary\" can't be translated to Halide.\n", true);
    }

    if (type == tiramisu::a_input)
    {
        res = Halide::Argument::InputBuffer;
    }
    else
    {
        assert(type == tiramisu::a_output);
        res = Halide::Argument::OutputBuffer;
    }

    return res;
}

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

std::vector<tiramisu::computation *> generator::get_computation_by_node(tiramisu::function *fct,
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
                                              isl_map *access, int remove_level = -1)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_map *schedule = isl_map_from_union_map(isl_ast_build_get_schedule(build));
    DEBUG(3, tiramisu::str_dump("Schedule:", isl_map_to_str(schedule)));

    if (remove_level != -1) {
        DEBUG(3, tiramisu::str_dump("Dropping this level from the index computation :" + std::to_string(remove_level)));
        int dim_idx = loop_level_into_dynamic_dimension(remove_level) - 1; // subtract 1 b/c this includes the duplicate dim
        std::string sched_str = isl_map_to_str(schedule);
        std::string dim_name = isl_map_get_dim_name(schedule, isl_dim_in, dim_idx);
        std::string new_constraint = " and " + dim_name + " = 0 }";
        std::vector<std::string> parts;
        split_string(sched_str, "}", parts);
        sched_str = parts[0] + new_constraint;
        schedule = isl_map_read_from_str(isl_ast_build_get_ctx(build), sched_str.c_str());
    }

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
            case tiramisu::o_type:
            case tiramisu::o_address_of:
            case tiramisu::o_lin_index:
            case tiramisu::o_buffer:
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
            case tiramisu::o_sinh:
            case tiramisu::o_cosh:
            case tiramisu::o_tanh:
            case tiramisu::o_asinh:
            case tiramisu::o_acosh:
            case tiramisu::o_atanh:
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
                ERROR("Checking an unsupported tiramisu expression for whether it has an ID.", 1);
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
            case tiramisu::o_type:
            case tiramisu::o_address_of:
            case tiramisu::o_lin_index:
            case tiramisu::o_buffer:
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
            case tiramisu::o_select:
            case tiramisu::o_lerp:
            case tiramisu::o_cond:
            case tiramisu::o_sin:
            case tiramisu::o_cos:
            case tiramisu::o_tan:
            case tiramisu::o_asin:
            case tiramisu::o_acos:
            case tiramisu::o_atan:
            case tiramisu::o_sinh:
            case tiramisu::o_cosh:
            case tiramisu::o_tanh:
            case tiramisu::o_asinh:
            case tiramisu::o_acosh:
            case tiramisu::o_atanh:
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
                ERROR("Unsupported tiramisu expression passed to access_is_affine().", 1);
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
            ERROR("Currently only Add, Sub, Minus, and Mul operations for accesses are supported for now.", true);
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

    if ((exp.get_expr_type() == tiramisu::e_op) && ((exp.get_op_type() == tiramisu::o_access) ||
                                                    (exp.get_op_type() == tiramisu::o_lin_index) ||
                                                    (exp.get_op_type() == tiramisu::o_buffer) ||
                                                    (exp.get_op_type() == tiramisu::o_address_of)))
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
            case tiramisu::o_sinh:
            case tiramisu::o_cosh:
            case tiramisu::o_tanh:
            case tiramisu::o_asinh:
            case tiramisu::o_acosh:
            case tiramisu::o_atanh:
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
            case tiramisu::o_memcpy:
                // They do not have any access.
                break;
            default:
                ERROR("Extracting access function from an unsupported tiramisu expression.", 1);
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
        exp.get_expr_type() == tiramisu::e_var ||
        exp.get_expr_type() == tiramisu::e_sync)
    {
        output_expr = exp;
    }
    else if ((exp.get_expr_type() == tiramisu::e_op) && ((exp.get_op_type() == tiramisu::o_access) ||
                                                         (exp.get_op_type() == tiramisu::o_address_of) ||
                                                         (exp.get_op_type() == tiramisu::o_lin_index) ||
                                                         (exp.get_op_type() == tiramisu::o_buffer)))
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
            case tiramisu::o_sinh:
            case tiramisu::o_cosh:
            case tiramisu::o_tanh:
            case tiramisu::o_asinh:
            case tiramisu::o_acosh:
            case tiramisu::o_atanh:
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
            case tiramisu::o_memcpy:
                output_expr = exp;
                break;
            default:
                ERROR("Unsupported tiramisu expression passed to traverse_expr_and_replace_non_affine_accesses().",
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
    if (comp->is_wait()) {
        // Need to swap the access map of the operation we wait on
        tiramisu::computation *waitee = func->get_computation_by_name(rhs.get_name())[0];
        isl_map *orig = isl_map_copy(waitee->get_access_relation());
        waitee->set_access(waitee->wait_access_map);
        generator::traverse_expr_and_extract_accesses(func, comp, rhs, accesses, return_buffer_accesses);
        waitee->set_access(orig);
    } else {
        generator::traverse_expr_and_extract_accesses(func, comp, rhs, accesses, return_buffer_accesses);
    }

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(3);
}

tiramisu::expr tiramisu_expr_from_isl_ast_expr(isl_ast_expr *isl_expr)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Input expression: ", isl_ast_expr_to_C_str(isl_expr)));
    DEBUG_NEWLINE(10);

    tiramisu::expr result;

    if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_int)
    {
        isl_val *init_val = isl_ast_expr_get_val(isl_expr);
        result = value_cast(tiramisu::global::get_loop_iterator_data_type(), isl_val_get_num_si(init_val));
        isl_val_free(init_val);
    }
    else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_id)
    {
        isl_id *identifier = isl_ast_expr_get_id(isl_expr);
        std::string name_str(isl_id_get_name(identifier));
        isl_id_free(identifier);
        result = tiramisu::var(tiramisu::global::get_loop_iterator_data_type(), name_str);
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
                ERROR("isl_ast_op_and_then operator found in the AST. This operator is not well supported.",
                                0);
                break;
            case isl_ast_op_or:
                result = tiramisu::expr(tiramisu::o_logical_or, op0, op1);
                break;
            case isl_ast_op_or_else:
                result = tiramisu::expr(tiramisu::o_logical_or, op0, op1);
                ERROR("isl_ast_op_or_then operator found in the AST. This operator is not well supported.",
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
                result = tiramisu::expr(tiramisu::o_cast, tiramisu::global::get_loop_iterator_data_type(), result);
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
                ERROR("Translating an unsupported ISL expression into a Tiramisu expression.", 1);
        }
    }
    else
    {
        tiramisu::str_dump("Transforming the following expression",
                           (const char *)isl_ast_expr_to_C_str(isl_expr));
        tiramisu::str_dump("\n");
        ERROR("Translating an unsupported ISL expression into a Tiramisu expression.", 1);
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
        std::map<std::string, isl_ast_expr *>::iterator it;
        it = iterators_map.find(exp.get_name());
        if (it != iterators_map.end())
            output_expr = tiramisu::expr(o_cast, global::get_loop_iterator_data_type(),
                                         tiramisu_expr_from_isl_ast_expr(iterators_map[exp.get_name()]));
        else
            output_expr = exp;
    }
    else if ((exp.get_expr_type() == tiramisu::e_op) && ((exp.get_op_type() == tiramisu::o_access) ||
                                                         (exp.get_op_type() == tiramisu::o_address_of)))
    {
        DEBUG(10, tiramisu::str_dump("Replacing the occurrences of original iterators in an o_access or o_address_of."));

        output_expr = exp.apply_to_operands([&iterators_map](const tiramisu::expr &exp){
            return replace_original_indices_with_transformed_indices(exp, iterators_map);
        });

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
            case tiramisu::o_sinh:
            case tiramisu::o_cosh:
            case tiramisu::o_tanh:
            case tiramisu::o_asinh:
            case tiramisu::o_acosh:
            case tiramisu::o_atanh:
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
                ERROR("Unsupported tiramisu expression passed to replace_original_indices_with_transformed_indices().", 1);
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
    identity = isl_map_apply_domain(identity, schedule);

    DEBUG(3, tiramisu::str_dump("Creating an isl_ast_index_expression for the access :",
                                isl_map_to_str(identity)));
    isl_ast_expr *idx_expr = create_isl_ast_index_expression(build, identity, comp->get_level_to_drop());
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
    std::vector<tiramisu::computation *> comp_vec = generator::get_computation_by_node(func, node);
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

        // This requires some type of wait on an asynchronous or non-blocking operation.
        isl_map *req_access = nullptr;
        if (comp->wait_access_map) {
            // This has a special LHS access into the request buffer
            isl_map *acc_copy = comp->get_access_relation() ? isl_map_copy(comp->get_access_relation()) : nullptr;
            comp->set_access(comp->wait_access_map);
            req_access = comp->get_access_relation_adapted_to_time_processor_domain();
            if (acc_copy) {
                comp->set_access(acc_copy);
            }
        }

        if (req_access) {
            comp->wait_index_expr = create_isl_ast_index_expression(build, req_access, comp->get_level_to_drop());
            isl_map_free(req_access);
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
                // For each access in accesses (i.e. for each access in the computation),
                // compute the corresponding isl_ast expression.
                for (size_t i = 0; i < accesses.size(); ++i)
                {
                    if (accesses[i] != NULL)
                    {
                        DEBUG(3, tiramisu::str_dump("Creating an isl_ast_index_expression for the access (isl_map *):",
                                                    isl_map_to_str(accesses[i])));
                        isl_ast_expr *idx_expr = create_isl_ast_index_expression(build, accesses[i], comp->get_level_to_drop());
                        DEBUG(3, tiramisu::str_dump("The created isl_ast_expr expression for the index expression is :", isl_ast_expr_to_str(idx_expr)));
                        index_expressions.push_back(idx_expr);
                        isl_map_free(accesses[i]);
                    }
                    else
                    {
                        // If this is not a let stmt and it is supposed to have accesses to other computations,
                        // it should have an access function.
                        if ((!comp->is_let_stmt()) && (!comp->is_library_call()))
                        {
                            tiramisu::str_dump("This is computation " + comp->get_name() +"\n");
                            // TODO better error message
                            ERROR("An access function should be provided for computation " + comp->get_name() + "'s #" + std::to_string(i) + " access before generating code.", true);
                        }
                    }
                }
            }

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

template <typename T, int N>
Halide::Expr halide_expr_from_isl_ast_expr_temp(isl_ast_expr *isl_expr)
{
    Halide::Expr result;

    if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_int)
    {
        isl_val *init_val = isl_ast_expr_get_val(isl_expr);
        result = Halide::Expr(static_cast<T>(isl_val_get_num_si(init_val)));
        isl_val_free(init_val);
    }
    else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_id)
    {
        isl_id *identifier = isl_ast_expr_get_id(isl_expr);
        std::string name_str(isl_id_get_name(identifier));
        isl_id_free(identifier);
        result = Halide::Internal::Variable::make(halide_type_from_tiramisu_type(global::get_loop_iterator_data_type()), name_str);
    }
    else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_op)
    {
        Halide::Expr op0, op1, op2;

        isl_ast_expr *expr0 = isl_ast_expr_get_op_arg(isl_expr, 0);
        op0 = halide_expr_from_isl_ast_expr_temp<T, N>(expr0);
        isl_ast_expr_free(expr0);

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 1)
        {
            isl_ast_expr *expr1 = isl_ast_expr_get_op_arg(isl_expr, 1);
            op1 = halide_expr_from_isl_ast_expr_temp<T, N>(expr1);
            isl_ast_expr_free(expr1);
        }

        if (isl_ast_expr_get_op_n_arg(isl_expr) > 2)
        {
            isl_ast_expr *expr2 = isl_ast_expr_get_op_arg(isl_expr, 2);
            op2 = halide_expr_from_isl_ast_expr_temp<T, N>(expr2);
            isl_ast_expr_free(expr2);
        }

        switch (isl_ast_expr_get_op_type(isl_expr))
        {
            case isl_ast_op_and:
                result = Halide::Internal::And::make(op0, op1);
                break;
            case isl_ast_op_and_then:
                result = Halide::Internal::And::make(op0, op1);
                ERROR("isl_ast_op_and_then operator found in the AST. This operator is not well supported.",
                                0);
                break;
            case isl_ast_op_or:
                result = Halide::Internal::Or::make(op0, op1);
                break;
            case isl_ast_op_or_else:
                result = Halide::Internal::Or::make(op0, op1);
                ERROR("isl_ast_op_or_then operator found in the AST. This operator is not well supported.",
                                0);
                break;
            case isl_ast_op_max:
                result = Halide::Internal::Max::make2(op0, op1, true);
                break;
            case isl_ast_op_min:
                result = Halide::Internal::Min::make2(op0, op1, true);
                break;
            case isl_ast_op_minus:
                result = Halide::Internal::Sub::make(Halide::cast(op0.type(), Halide::Expr(0)), op0, true);
                break;
            case isl_ast_op_add:
                result = Halide::Internal::Add::make(op0, op1, true);
                break;
            case isl_ast_op_sub:
                result = Halide::Internal::Sub::make(op0, op1, true);
                break;
            case isl_ast_op_mul:
                result = Halide::Internal::Mul::make(op0, op1, true);
                break;
            case isl_ast_op_div:
                result = Halide::Internal::Div::make(op0, op1, true);
                break;
            case isl_ast_op_fdiv_q:
            case isl_ast_op_pdiv_q:
                result = Halide::Internal::Div::make(op0, op1, true);
                result = Halide::Internal::Cast::make(Halide::Int(N), Halide::floor(result));
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
                result = Halide::Internal::LE::make(op0, op1, true);
                break;
            case isl_ast_op_lt:
                result = Halide::Internal::LT::make(op0, op1, true);
                break;
            case isl_ast_op_ge:
                result = Halide::Internal::GE::make(op0, op1, true);
                break;
            case isl_ast_op_gt:
                result = Halide::Internal::GT::make(op0, op1, true);
                break;
            case isl_ast_op_eq:
                result = Halide::Internal::EQ::make(op0, op1, true);
                break;
            default:
                tiramisu::str_dump("Transforming the following expression",
                                   (const char *)isl_ast_expr_to_C_str(isl_expr));
                tiramisu::str_dump("\n");
                ERROR("Translating an unsupported ISL expression in a Halide expression.", 1);
        }
    }
    else
    {
        tiramisu::str_dump("Transforming the following expression",
                           (const char *)isl_ast_expr_to_C_str(isl_expr));
        tiramisu::str_dump("\n");
        ERROR("Translating an unsupported ISL expression in a Halide expression.", 1);
    }

    return result;
}

Halide::Expr halide_expr_from_isl_ast_expr(isl_ast_expr *isl_expr)
{
    if (global::get_loop_iterator_data_type() == p_int32)
        return halide_expr_from_isl_ast_expr_temp<int32_t, 32>(isl_expr);
    else
        return halide_expr_from_isl_ast_expr_temp<int64_t, 64>(isl_expr);
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

Halide::Internal::Stmt tiramisu::generator::make_halide_block(const Halide::Internal::Stmt &first,
                                                              const Halide::Internal::Stmt &second)
{
    if (first->node_type == Halide::Internal::IRNodeType::LetStmt)
    {
        DEBUG(3, tiramisu::str_dump("The Halide block is a let statement"));
        auto * let_stmt = first.as<Halide::Internal::LetStmt>();
        return Halide::Internal::LetStmt::make(let_stmt->name, let_stmt->value,
                                               generator::make_halide_block(let_stmt->body, second));
    }
    else
    {
        return Halide::Internal::Block::make(first, second);
    }
}

Halide::Internal::Stmt
tiramisu::generator::halide_stmt_from_isl_node(const tiramisu::function &fct, isl_ast_node *node, int level,
                                               std::vector<std::pair<std::string, std::string>> &tagged_stmts,
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

            auto op_type = o_none;

            if (isl_ast_node_get_type(child) == isl_ast_node_user)
                op_type = get_computation_annotated_in_a_node(child)->get_expr().get_op_type();

            if ((isl_ast_node_get_type(child) == isl_ast_node_user) &&
                (op_type == o_allocate || op_type == o_free || op_type == o_memcpy))
            {
                tiramisu::computation *comp = get_computation_annotated_in_a_node(child);
                if (op_type == tiramisu::o_allocate)
                {
                    DEBUG(3, tiramisu::str_dump("Adding a computation to vector of allocate stmts (for later construction)"));
                    allocate_stmts_vector.push_back(comp);
                }
                else if (op_type == tiramisu::o_free)
                {
                    auto * buffer = (comp->get_access_relation() != nullptr) ? fct.get_buffers().at(get_buffer_name(comp)) : nullptr;
                    block = generator::make_buffer_free(buffer);
                }
                else
                {
                    auto const &e = comp->get_expr();
                    auto buffer_1 = fct.get_buffers().at(e.get_operand(0).get_name());
                    auto buffer_2 = fct.get_buffers().at(e.get_operand(1).get_name());
                    buffer * gpu_b, * host_b;
                    bool to_host = false, from_host = false;
                    if (buffer_1->location != cuda_ast::memory_location::host && buffer_2->location == cuda_ast::memory_location::host) {
                        gpu_b = buffer_1;
                        host_b = buffer_2;
                        to_host = true;
                    }
                    else if (buffer_1->location == cuda_ast::memory_location::host && buffer_2->location != cuda_ast::memory_location::host) {
                        gpu_b = buffer_2;
                        host_b = buffer_1;
                        from_host = true;
                    }
                    assert(from_host || to_host);
                    // Tiramisu buffer is from outermost to innermost, whereas Halide buffer is from innermost
                    // to outermost; thus, we need to reverse the order
                    // TODO: refactor
                    int stride = 1;
                    Halide::Expr stride_expr = Halide::Expr(1);

                    if (host_b->has_constant_extents())
                    {
                        DEBUG(10, tiramisu::str_dump("Buffer has constant extents."));
                        for (size_t i = 0; i < host_b->get_dim_sizes().size(); i++)
                        {
                            int dim_idx = host_b->get_dim_sizes().size() - i - 1;
                            stride *= (int)host_b->get_dim_sizes()[dim_idx].get_int_val();
                        }
                        stride_expr = stride;
                    }
                    else
                    {
                        DEBUG(10, tiramisu::str_dump("Buffer has non-constant extents."));
                        std::vector<isl_ast_expr *> empty_index_expr;
                        for (int i = 0; i < host_b->get_dim_sizes().size(); i++)
                        {
                            int dim_idx = host_b->get_dim_sizes().size() - i - 1;
                            stride_expr = stride_expr * generator::halide_expr_from_tiramisu_expr(&fct, empty_index_expr, host_b->get_dim_sizes()[dim_idx]);
                        }
                    }
                    auto h_type = halide_type_from_tiramisu_type(host_b->get_elements_type());
                    auto size = Halide::cast(Halide::type_of<uint64_t >(), stride_expr * h_type.bytes());
                    Halide::Internal::Parameter param =
                            Halide::Internal::Parameter{h_type,
                                                        true,
                                                        (int)(host_b->get_dim_sizes().size()),
                                                        host_b->get_name()};

                    Halide::Expr loaded_symbol = Halide::Internal::Variable::make(Halide::type_of<struct halide_buffer_t *>(),
                                                                           host_b->get_name() + ".buffer");
                    Halide::Expr buffer_address = Halide::Internal::Call::make(Halide::Handle(1, h_type.handle_type),
                                                          "tiramisu_address_of_" +
                                                          str_from_tiramisu_type_primitive(host_b->get_elements_type()),
                                                          {loaded_symbol, Halide::Expr(0)},
                                                          Halide::Internal::Call::Extern);

                    auto gpu_buffer = (gpu_b->location == cuda_ast::memory_location::constant)
                                      ? Halide::Internal::Call::make(Halide::type_of<void *>(), gpu_b->get_name() + "_get_symbol", {}, Halide::Internal::Call::Extern)
                                      : Halide::Internal::Variable::make(Halide::type_of<void *>(), gpu_b->get_name());
                    auto host_result_buffer = Halide::Internal::Variable::make(Halide::type_of<struct halide_buffer_t *>(),
                                                                               host_b->get_name() + ".buffer");
                    if (to_host)
                        block = Halide::Internal::Evaluate::make(
                                Halide::Internal::Call::make(Halide::Int(32), "tiramisu_cuda_memcpy_to_host",
                                                             {buffer_address, gpu_buffer, size}, Halide::Internal::Call::Extern)
                        );
                    else if (from_host)
                        block = Halide::Internal::Evaluate::make(
                                Halide::Internal::Call::make(Halide::Int(32), (gpu_b->location == cuda_ast::memory_location::constant) ? "tiramisu_cuda_memcpy_to_symbol" : "tiramisu_cuda_memcpy_to_device",
                                                             {gpu_buffer, buffer_address, size}, Halide::Internal::Call::Extern)
                        );
                }
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
                DEBUG(3, tiramisu::str_dump("Block undefined."));
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
                DEBUG(3, tiramisu::str_dump("Block defined."));
                if (result.defined())
                {
                    // result = Halide::Internal::Block::make(block, result);
                    result = generator::make_halide_block(block, result);
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
                    tiramisu::expr dim_sz = replace_original_indices_with_transformed_indices(sz, comp->get_iterators_map());
                    halide_dim_sizes.push_back(generator::halide_expr_from_tiramisu_expr(NULL, ie, dim_sz, comp));
                }

                if (comp->get_expr().get_op_type() == tiramisu::o_allocate)
                {
//                    result = Halide::Internal::Allocate::make(
//                           buf->get_name(),
//                           halide_type_from_tiramisu_type(buf->get_elements_type()),
//                           halide_dim_sizes, Halide::Internal::const_true(), result);
                    result = make_buffer_alloc(buf, halide_dim_sizes, result);


                    buf->mark_as_allocated();

                    for (const auto &l_stmt : comp->get_associated_let_stmts())
                    {
                        DEBUG(3, tiramisu::str_dump("Generating the following let statement."));
                        DEBUG(3, tiramisu::str_dump("Name : " + l_stmt.first));
                        DEBUG(3, tiramisu::str_dump("Expression of the let statement: "));

                        l_stmt.second.dump(false);

                        std::vector<isl_ast_expr *> ie = {}; // Dummy variable.
                        tiramisu::expr tiramisu_let = replace_original_indices_with_transformed_indices(l_stmt.second, comp->get_iterators_map());
                        Halide::Expr let_expr = halide_expr_from_tiramisu_expr(comp->get_function(), ie, tiramisu_let, comp);
                        result = Halide::Internal::LetStmt::make(
                                l_stmt.first,
                                let_expr,
                                result);
                        DEBUG(10, tiramisu::str_dump("Generated let stmt:"));
                        DEBUG_NO_NEWLINE(10, std::cout << result);
                    }
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

        auto it_kernel = fct.iterator_to_kernel_map.find(node);

        if (it_kernel != fct.iterator_to_kernel_map.end())
        {
            auto k = it_kernel->second;
            std::vector<Halide::Expr> args;
            std::vector<isl_ast_expr *> ie;
            for (auto &id: k->get_arguments())
            {
                args.push_back(
                        Halide::Internal::Variable::make(halide_type_from_tiramisu_type(id->get_type()), id->get_name())
                );
            }


            result = Halide::Internal::Evaluate::make(
                    Halide::Internal::Call::make(
                            halide_type_from_tiramisu_type(cuda_ast::kernel::wrapper_return_type),
                            k->get_wrapper_name(), args, Halide::Internal::Call::Extern));
        } else {

            isl_ast_expr *init = isl_ast_node_for_get_init(node);
            isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
            isl_ast_expr *inc = isl_ast_node_for_get_inc(node);

            isl_val *inc_val = isl_ast_expr_get_val(inc);
            if (!isl_val_is_one(inc_val)) {
                ERROR("The increment in one of the loops is not +1."
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
            if (isl_ast_expr_get_op_type(cond) == isl_ast_op_lt) {
                cond_upper_bound_isl_format = isl_ast_expr_get_op_arg(cond, 1);
            } else if (isl_ast_expr_get_op_type(cond) == isl_ast_op_le) {
                // Create an expression of "1".
                isl_val *one = isl_val_one(isl_ast_node_get_ctx(node));
                // Add 1 to the ISL ast upper bound to transform it into a strinct bound.
                cond_upper_bound_isl_format = isl_ast_expr_add(
                        isl_ast_expr_get_op_arg(cond, 1),
                        isl_ast_expr_from_val(one));
            } else {
                ERROR("The for loop upper bound is not an isl_est_expr of type le or lt", 1);
            }
            assert(cond_upper_bound_isl_format != NULL);
            DEBUG(3, tiramisu::str_dump("Creating for loop init expression."));

            Halide::Expr init_expr = halide_expr_from_isl_ast_expr(init);
            if (init_expr.type() !=
                halide_type_from_tiramisu_type(global::get_loop_iterator_data_type())) {
                init_expr =
                        Halide::Internal::Cast::make(
                                halide_type_from_tiramisu_type(global::get_loop_iterator_data_type()),
                                init_expr);
            }
            DEBUG(3, tiramisu::str_dump("init expression: ");
                    std::cout << init_expr);
            Halide::Expr cond_upper_bound_halide_format =
                    simplify(halide_expr_from_isl_ast_expr(cond_upper_bound_isl_format));
            if (cond_upper_bound_halide_format.type() !=
                halide_type_from_tiramisu_type(global::get_loop_iterator_data_type())) {
                cond_upper_bound_halide_format =
                        Halide::Internal::Cast::make(
                                halide_type_from_tiramisu_type(global::get_loop_iterator_data_type()),
                                cond_upper_bound_halide_format);
            }
            DEBUG(3, tiramisu::str_dump("Upper bound expression: ");
                    std::cout << cond_upper_bound_halide_format);
            Halide::Internal::Stmt halide_body =
                    tiramisu::generator::halide_stmt_from_isl_node(fct, body, level + 1, tagged_stmts, false);
            Halide::Internal::ForType fortype = Halide::Internal::ForType::Serial;
            Halide::DeviceAPI dev_api = Halide::DeviceAPI::Host;

            // Change the type from Serial to parallel or vector if the
            // current level was marked as such.
            size_t tt = 0;
            bool convert_to_conditional = false;
            while (tt < tagged_stmts.size()) {
                if (tagged_stmts[tt].first != "") {
                    if (tagged_stmts[tt].second == "parallelize" &&
                        fct.should_parallelize(tagged_stmts[tt].first, level)) {
                        fortype = Halide::Internal::ForType::Parallel;
                        // Since this statement is treated, remove it from the list of
                        // tagged statements so that it does not get treated again later.
                        tagged_stmts[tt].first = "";
                        // As soon as we find one tagged statement that actually useful we exit
                        break;
                    } else if (tagged_stmts[tt].second == "vectorize" &&
                               fct.should_vectorize(tagged_stmts[tt].first, level)) {
                        DEBUG(3, tiramisu::str_dump("Trying to vectorize at level "
                                                    + std::to_string(level) + ", tagged stmt is " +
                                                    tagged_stmts[tt].first));

                        int vector_length = fct.get_vector_length(tagged_stmts[tt].first, level);

                        for (auto vd: fct.vector_dimensions) {
                            DEBUG(3, "stmt = " + std::get<0>(vd) + ", level = " +
                                     std::to_string(std::get<1>(vd)) + ", length = " +
                                     std::to_string(std::get<2>(vd)));
                        }

                        DEBUG(3, tiramisu::str_dump("Tagged statements (before removing this tagged stmt):"));
                        size_t tttt = 0;
                        while (tttt < tagged_stmts.size()) {
                            DEBUG(3, tiramisu::str_dump(
                                    "Tagged stmt: " + std::to_string(tttt) + ": " + tagged_stmts[tttt].first +
                                    " with tag " + tagged_stmts[tttt].second));
                            tttt++;
                        }

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
                        tagged_stmts[tt].first = "";

                        DEBUG(3, tiramisu::str_dump("Tagged statements:"));
                        tttt = 0;
                        while (tttt < tagged_stmts.size()) {
                            DEBUG(3, tiramisu::str_dump(
                                    "Tagged stmt: " + std::to_string(tttt) + ": " + tagged_stmts[tttt].first +
                                    " with tag " + tagged_stmts[tttt].second));
                            tttt++;
                        }
                        break;
                    } else if (tagged_stmts[tt].second == "map_to_gpu_thread" &&
                               fct.should_map_to_gpu_thread(tagged_stmts[tt].first, level)) {
                        fortype = Halide::Internal::ForType::GPUThread;
                        dev_api = Halide::DeviceAPI::OpenCL;
                        std::string gpu_iter = fct.get_gpu_thread_iterator(tagged_stmts[tt].first, level);
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
                        tagged_stmts[tt].first = "";
                        break;
                    } else if (tagged_stmts[tt].second == "map_to_gpu_block" &&
                               fct.should_map_to_gpu_block(tagged_stmts[tt].first, level)) {
                        assert(false);
                        fortype = Halide::Internal::ForType::GPUBlock;
                        dev_api = Halide::DeviceAPI::OpenCL;
                        std::string gpu_iter = fct.get_gpu_block_iterator(tagged_stmts[tt].first, level);
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
                        tagged_stmts[tt].first = "";
                        break;
                    } else if (tagged_stmts[tt].second == "unroll" &&
                               fct.should_unroll(tagged_stmts[tt].first, level)) {
                        DEBUG(3, tiramisu::str_dump("Trying to unroll at level ");
                                tiramisu::str_dump(std::to_string(level)));

                        const Halide::Internal::IntImm *extent =
                                cond_upper_bound_halide_format.as<Halide::Internal::IntImm>();
                        if (extent) {
                            fortype = Halide::Internal::ForType::Unrolled;
                            DEBUG(3, tiramisu::str_dump("Loop unrolled"));
                        } else {
                            DEBUG(3, tiramisu::str_dump("Loop not unrolled (extent is non constant)"));
                            DEBUG(3, std::cout << cond_upper_bound_halide_format << std::endl);
                        }

                        // Since this statement is treated, remove it from the list of
                        // tagged statements so that it does not get treated again later.
                        tagged_stmts[tt].first = "";
                        break;
                    } else if (tagged_stmts[tt].second == "distribute" &&
                               fct.should_distribute(tagged_stmts[tt].first, level)) {
                        // Change this loop into an if statement instead
                        convert_to_conditional = true;
                        tagged_stmts[tt].first = "";
                        break;
                    }
                }
                tt++;
            }

            DEBUG(10, tiramisu::str_dump("The full list of tagged statements is now:"));
            for (const auto &ts: tagged_stmts) DEBUG(10, tiramisu::str_dump(ts.first + " with tag " + ts.second));
            DEBUG(10, tiramisu::str_dump(""));

            if (convert_to_conditional) {
                DEBUG(3, tiramisu::str_dump("Converting for loop into a rank conditional."));
                Halide::Expr rank_var =
                        Halide::Internal::Variable::make(
                                halide_type_from_tiramisu_type(global::get_loop_iterator_data_type()), "rank");
                Halide::Expr condition = rank_var >= init_expr;
                condition = condition && (rank_var < cond_upper_bound_halide_format);
                Halide::Internal::Stmt else_s;
                // We need a reference still to this iterator name, so set it equal to the rank
                halide_body = Halide::Internal::LetStmt::make(iterator_str, rank_var, halide_body);
                result = Halide::Internal::IfThenElse::make(condition, halide_body, else_s);
            } else {
                DEBUG(3, tiramisu::str_dump("Creating the for loop."));
                result = Halide::Internal::For::make(iterator_str, init_expr,
                                                     cond_upper_bound_halide_format - init_expr,
                                                     fortype, dev_api, halide_body);
                DEBUG(3, tiramisu::str_dump("For loop created."));
                DEBUG(10, std::cout << result);
            }

            isl_ast_expr_free(init);
            isl_ast_expr_free(cond);
            isl_ast_expr_free(inc);
            isl_ast_node_free(body);
            isl_ast_expr_free(cond_upper_bound_isl_format);
        }
        isl_ast_expr_free(iter);
        free(cstr);
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_user)
    {
        DEBUG(3, tiramisu::str_dump("Generating code for user node"));

        if ((isl_ast_node_get_type(node) == isl_ast_node_user) &&
            ((get_computation_annotated_in_a_node(node)->get_expr().get_op_type() == tiramisu::o_allocate) ||
             (get_computation_annotated_in_a_node(node)->get_expr().get_op_type() == tiramisu::o_free)))
        {
            if (get_computation_annotated_in_a_node(node)->get_expr().get_op_type() == tiramisu::o_allocate)
            {
                ERROR("Allocate node should not appear as a user ISL AST node. It should only appear with block construction (because of its scope).", true);
            }
            else
            {
                tiramisu::computation *comp = get_computation_annotated_in_a_node(node);
                auto * buffer = (comp->get_access_relation() != nullptr) ? fct.get_buffers().at(get_buffer_name(comp)) : nullptr;
                result = generator::make_buffer_free(buffer);
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
                    tagged_stmts.push_back(std::pair<std::string, std::string>(computation_name, "parallelize"));
                if (fct.should_vectorize(computation_name, l))
                    tagged_stmts.push_back(std::pair<std::string, std::string>(computation_name, "vectorize"));
                if (fct.should_map_to_gpu_block(computation_name, l))
                    tagged_stmts.push_back(std::pair<std::string, std::string>(computation_name, "map_to_gpu_block"));
                if (fct.should_map_to_gpu_thread(computation_name, l))
                    tagged_stmts.push_back(std::pair<std::string, std::string>(computation_name, "map_to_gpu_thread"));
                if (fct.should_unroll(computation_name, l))
                    tagged_stmts.push_back(std::pair<std::string, std::string>(computation_name, "unroll"));
                if (fct.should_distribute(computation_name, l))
                    tagged_stmts.push_back(std::pair<std::string, std::string>(computation_name, "distribute"));

                DEBUG(10, tiramisu::str_dump("The full list of tagged statements is now"));
                for (const auto &ts: tagged_stmts)
                DEBUG(10, tiramisu::str_dump(ts.first + " with tag " + ts.second));
            }

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
                Halide::Expr let_expr = halide_expr_from_tiramisu_expr(comp->get_function(), ie, tiramisu_let, comp);
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
                Halide::Expr predicate = halide_expr_from_tiramisu_expr(comp->get_function(), ie, tiramisu_predicate, comp);
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
                    tiramisu::generator::halide_stmt_from_isl_node(fct, if_stmt, level, tagged_stmts, false);

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
                    else_s = tiramisu::generator::halide_stmt_from_isl_node(fct, else_stmt, level, tagged_stmts, false);
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

    Halide::Internal::set_always_upcast();

    // This vector is used in generate_Halide_stmt_from_isl_node to figure
    // out what are the statements that have already been visited in the
    // AST tree.
    std::vector<std::pair<std::string, std::string>> generated_stmts;
    Halide::Internal::Stmt stmt;

    // Generate the statement that represents the whole function
    stmt = tiramisu::generator::halide_stmt_from_isl_node(*this, this->get_isl_ast(), 0, generated_stmts, false);

    DEBUG(3, tiramisu::str_dump("The following Halide statement was generated:\n"); std::cout << stmt << std::endl);

    Halide::Internal::Stmt freestmts;
    for (const auto &b : this->get_buffers())
    {
        tiramisu::buffer *buf = b.second;
        if (buf->get_argument_type() == tiramisu::a_temporary && buf->get_auto_allocate() == true && buf->location == cuda_ast::memory_location::global)
        {
            auto free = generator::make_buffer_free(buf);
            if (freestmts.defined())
                freestmts = Halide::Internal::Block::make(free, freestmts);
            else
                freestmts = free;
        }
    }

    if (freestmts.defined())
        stmt = Halide::Internal::Block::make(stmt, freestmts);

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
                const auto sz = buf->get_dim_sizes()[i];
                std::vector<isl_ast_expr *> ie = {};
                halide_dim_sizes.push_back(generator::halide_expr_from_tiramisu_expr(this, ie, sz));
            }
            stmt = generator::make_buffer_alloc(buf, halide_dim_sizes, stmt);
//            stmt = Halide::Internal::Allocate::make(
//                       buf->get_name(),
//                       halide_type_from_tiramisu_type(buf->get_elements_type()),
//                       halide_dim_sizes, Halide::Internal::const_true(), stmt);

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
                generator::halide_expr_from_tiramisu_expr(this, ie, param.get_expr()),
                stmt);
    }

    if (this->_needs_rank_call) {
        // add a call to MPI rank to the beginning of the function
        Halide::Expr mpi_rank_var =
                Halide::Internal::Variable::make(halide_type_from_tiramisu_type(tiramisu::p_int32), "rank");
        Halide::Expr mpi_rank = Halide::cast(halide_type_from_tiramisu_type(global::get_loop_iterator_data_type()),
                                             Halide::Internal::Call::make(Halide::Int(32), "tiramisu_MPI_Comm_rank",
                                                                          {this->rank_offset},
                                                                          Halide::Internal::Call::Extern));
        stmt = Halide::Internal::LetStmt::make("rank", mpi_rank, stmt);
    }

    // Add producer tag
    stmt = Halide::Internal::ProducerConsumer::make_produce("", stmt);

    this->halide_stmt = stmt;

    DEBUG(3, tiramisu::str_dump("\n\nGenerated Halide stmt before lowering:"));
    DEBUG(3, std::cout << stmt);

    DEBUG_INDENT(-4);
}

Halide::Internal::Stmt generator::make_buffer_alloc(buffer *b, const std::vector<Halide::Expr> &extents,
                                                    Halide::Internal::Stmt &stmt) {
    using cuda_ast::memory_location;
    auto h_type = halide_type_from_tiramisu_type(b->get_elements_type());
    if (b->location == memory_location::host)
    {
        return Halide::Internal::Allocate::make(
                b->get_name(),
                h_type,
                extents, Halide::Internal::const_true(), stmt);
    }
    else if (b->location == memory_location::global)
    {
        Halide::Expr size = extents[0];
        for (int  i = 1; i < extents.size(); i++)
        {
            size = size * extents[i];
        }
        return Halide::Internal::LetStmt::make(
                b->get_name(),
                Halide::Internal::Call::make(Halide::type_of<void *>(), "tiramisu_cuda_malloc",
                                             {Halide::cast(Halide::UInt(64), size * h_type.bytes())}, Halide::Internal::Call::Extern),
                stmt
        );

    }

    return stmt;

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


static inline Halide::Expr empty_index()
{
    if (global::get_loop_iterator_data_type() == p_int32)
        return Halide::Expr(static_cast<int32_t >(0));
    else
        return Halide::Expr(static_cast<int64_t >(0));
}

Halide::Expr generator::linearize_access(int dims, const halide_dimension_t *shape, isl_ast_expr *index_expr)
{
    assert(isl_ast_expr_get_op_n_arg(index_expr) > 1);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // ISL dimension is ordered from outermost to innermost.

    Halide::Expr index {empty_index()};
    for (int i = dims; i >= 1; --i)
    {
        isl_ast_expr *operand = isl_ast_expr_get_op_arg(index_expr, i);
        Halide::Expr operand_h = halide_expr_from_isl_ast_expr(operand);
        index += operand_h * Halide::Expr(shape[dims - i].stride);
        isl_ast_expr_free(operand);
    }

    DEBUG_INDENT(-4);

    return index;
}

Halide::Expr generator::linearize_access(int dims, const halide_dimension_t *shape, std::vector<tiramisu::expr> index_expr)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(index_expr.size() > 0);

    // ISL dimension is ordered from outermost to innermost.

    Halide::Expr index {empty_index()};
    for (int i = dims; i >= 1; --i)
    {
        std::vector<isl_ast_expr *> ie = {};
        Halide::Expr operand_h = generator::halide_expr_from_tiramisu_expr(NULL, ie, index_expr[i-1]);
        index += operand_h * Halide::Expr(shape[dims - i].stride);
    }

    DEBUG_INDENT(-4);

    return index;
}

Halide::Expr generator::linearize_access(int dims, std::vector<Halide::Expr> &strides, std::vector<tiramisu::expr> index_expr)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(index_expr.size() > 0);

    // ISL dimension is ordered from outermost to innermost.

    Halide::Expr index {empty_index()};
    for (int i = dims; i >= 1; --i)
    {
        std::vector<isl_ast_expr *> ie = {};
        Halide::Expr operand_h = generator::halide_expr_from_tiramisu_expr(NULL, ie, index_expr[i-1]);
        index += operand_h * strides[dims - i];
    }

    DEBUG_INDENT(-4);

    return index;
}

Halide::Expr generator::linearize_access(int dims, std::vector<Halide::Expr> &strides, isl_ast_expr *index_expr)
{
    assert(isl_ast_expr_get_op_n_arg(index_expr) > 1);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // ISL dimension is ordered from outermost to innermost.

    Halide::Expr index {empty_index()};
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

tiramisu::expr generator::linearize_access(int dims, std::vector<tiramisu::expr> &strides, std::vector<tiramisu::expr> index_expr)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(!index_expr.empty());

    // ISL dimension is ordered from outermost to innermost.

    tiramisu::expr index = value_cast(global::get_loop_iterator_data_type(), 0);
    auto stride = strides.rbegin();
    auto i_expr = index_expr.begin();
    for (;
            stride != strides.rend() && i_expr != strides.end();
            stride ++, i_expr ++)
    {
        index = index + (*stride) * (*i_expr);
    }

    DEBUG_INDENT(-4);

    return index;
}

tiramisu::expr generator::linearize_access(int dims, std::vector<tiramisu::expr> &strides, isl_ast_expr *index_expr)
{
    assert(isl_ast_expr_get_op_n_arg(index_expr) > 1);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // ISL dimension is ordered from outermost to innermost.

    tiramisu::expr index = value_cast(global::get_loop_iterator_data_type(), 0);
    for (int i = dims; i >= 1; --i)
    {
        isl_ast_expr *operand = isl_ast_expr_get_op_arg(index_expr, i);
        tiramisu::expr operand_h = tiramisu_expr_from_isl_ast_expr(operand);
        index = index + operand_h * strides[dims - i];
        isl_ast_expr_free(operand);
    }

    DEBUG_INDENT(-4);

    return index;
}

std::string generator::get_buffer_name(const tiramisu::computation * comp)
{
    isl_map *access = comp->get_access_relation_adapted_to_time_processor_domain();
    isl_space *space = isl_map_get_space(access);
    const char *buffer_name = isl_space_get_tuple_name(space, isl_dim_out);
    std::cerr << comp->get_name() << std::endl;
    isl_map_dump(access);
    assert(buffer_name != nullptr);
    return std::string{buffer_name};
}

tiramisu::expr generator::comp_to_buffer(tiramisu::computation *comp, std::vector<isl_ast_expr *> &index_expr,
                                         const tiramisu::expr *expr)
{
    auto buffer_name = generator::get_buffer_name(comp);

    DEBUG(3, tiramisu::str_dump("Buffer name extracted from the access relation: ", buffer_name.c_str()));

    isl_map *access = comp->get_access_relation_adapted_to_time_processor_domain();
    isl_space *space = isl_map_get_space(access);

    // Get the number of dimensions of the ISL map representing
    // the access.
    int access_dims = isl_space_dim(space, isl_dim_out);

    // Fetch the actual buffer.
    const auto &buffer_entry = comp->get_function()->get_buffers().find(buffer_name);
    assert(buffer_entry != comp->get_function()->get_buffers().end());

    const auto &tiramisu_buffer = buffer_entry->second;
    DEBUG(3, tiramisu::str_dump("A Tiramisu buffer that corresponds to the buffer indicated in the access relation was found."));

    DEBUG(10, tiramisu_buffer->dump(true));

    auto dim_sizes = tiramisu_buffer->get_dim_sizes();

    std::vector<tiramisu::expr> strides;
    tiramisu::expr stride = value_cast(global::get_loop_iterator_data_type(), 1);

    std::vector<isl_ast_expr *> empty_index_expr;
    for (auto dim = dim_sizes.rbegin(); dim != dim_sizes.rend(); dim++)
    {
        strides.push_back(stride);
        stride = stride * tiramisu::expr(o_cast, global::get_loop_iterator_data_type(), replace_original_indices_with_transformed_indices(*dim, comp->get_iterators_map()));
    }
    tiramisu::expr index;

    // The number of dimensions in the Halide buffer should be equal to
    // the number of dimensions of the access function.
    assert(dim_sizes.size() == access_dims);
    if (index_expr.empty())
    {
        assert(expr != nullptr);
        DEBUG(10, tiramisu::str_dump("index_expr is empty. Retrieving access indices directly from the tiramisu access expression without scheduling."));

        for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++)
        {
            // Actually any access that does not require
            // scheduling is supported but currently we only
            // accept literal constants as anything else was not
            // needed til now.
            assert(expr->get_access()[i].is_constant() && "Only constant accesses are supported.");
        }

        index = tiramisu::generator::linearize_access((int) dim_sizes.size(), strides, expr->get_access());
    }
    else
    {
        assert(index_expr[0] != nullptr);
        DEBUG(3, tiramisu::str_dump("Linearizing access of the LHS index expression."));

        index = tiramisu::generator::linearize_access((int) dim_sizes.size(), strides, index_expr[0]);

        DEBUG(3, tiramisu::str_dump("After linearization: ");
                std::cout << index.to_str() << std::endl);

        DEBUG(3, tiramisu::str_dump("Index expressions of this statement are:"));
        print_isl_ast_expr_vector(index_expr);

        DEBUG(3, tiramisu::str_dump(
                "Erasing the first index expression from the vector of index expressions (the first index has just been linearized)."));
        index_expr.erase(index_expr.begin());
    }

    return tiramisu::expr{o_access, buffer_name, {index}, tiramisu_buffer->get_elements_type()};
}

std::pair<expr, expr> computation::create_tiramisu_assignment(std::vector<isl_ast_expr *> &index_expr)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Generating stmt for assignment."));

    tiramisu::expr lhs, rhs;

    // TODO handle let statements
    if (this->is_let_stmt())
    {
        DEBUG(3, tiramisu::str_dump("This is a let statement."));
        DEBUG_NO_NEWLINE(10, tiramisu::str_dump("The expression associated with the let statement: ");
                this->expression.dump(false));
        DEBUG_NEWLINE(10);

        // Assuming this computation is not the original computation, but a
        // definition that was added to the original computation. We need to
        // retrieve the original computation.
        auto *root = (tiramisu::constant *)
                this->get_root_of_definition_tree();

        if (root->get_computation_with_whom_this_is_computed() != NULL)
        {

            rhs = generator::replace_accesses(this->get_function(),
                                              this->get_index_expr(),
                                              replace_original_indices_with_transformed_indices(this->expression,
                                                                                                root->get_computation_with_whom_this_is_computed()->get_iterators_map()));
        }
        else
        {

            rhs = generator::replace_accesses(this->get_function(),
                                              this->get_index_expr(),
                                              this->expression);

        }


        if (this->get_data_type() != rhs.get_data_type())
        {
            rhs = cast(this->get_data_type(), rhs);
        }

        const std::string &let_stmt_name = this->get_name();

        lhs = var(this->get_data_type(), let_stmt_name);
    }
    else
    {

        DEBUG(3, tiramisu::str_dump("This is not a let statement."));


        lhs = generator::comp_to_buffer(this, index_expr);

        // Replace the RHS expression to the transformed expressions.
        // We do not need to transform the indices of expression (this->index_expr), because in Tiramisu we assume
        // that an access can only appear when accessing a computation. And that case should be handled in the following transformation
        // so no need to transform this->index_expr separately.
        rhs = generator::replace_accesses(
                this->get_function(), index_expr,
                replace_original_indices_with_transformed_indices(this->expression, this->get_iterators_map()));

    }
    DEBUG(3, std::cout << "LHS: " << lhs.to_str());
    DEBUG(3, std::cout << "RHS: " << rhs.to_str());

    DEBUG_INDENT(-4);

    return std::make_pair(lhs, rhs);

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

        // Assuming this computation is not the original computation, but a
        // definition that was added to the original computation. We need to
        // retrieve the original computation.
        tiramisu::constant *root = (tiramisu::constant *)
                this->get_root_of_definition_tree();

        Halide::Expr result;
        if (root->get_computation_with_whom_this_is_computed() != NULL)
        {
            DEBUG(10, tiramisu::str_dump("1."));

            result = generator::halide_expr_from_tiramisu_expr(this->get_function(),
                                                               this->get_index_expr(),
                                                               replace_original_indices_with_transformed_indices(this->expression,
                                                                                                                 root->get_computation_with_whom_this_is_computed()->get_iterators_map()), this);
            DEBUG(10, tiramisu::str_dump("2."));
        }
        else
        {
            DEBUG(10, tiramisu::str_dump("3."));

            result = generator::halide_expr_from_tiramisu_expr(this->get_function(),
                                                               this->get_index_expr(),
                                                               this->expression, this);

            DEBUG(10, tiramisu::str_dump("4."));
        }

        DEBUG(10, tiramisu::str_dump("The expression translated to a Halide expression: "); std::cout << result << std::endl);

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
    else {
        DEBUG(3, tiramisu::str_dump("This is not a let statement."));
        if (this->is_send() || this->is_recv()) {
          // This is the iterator, but it is still in the user's form. Transform it.
          this->library_call_args[1] = replace_original_indices_with_transformed_indices(this->library_call_args[1],
                                                                                           this->get_iterators_map());
        }
        // The majority of code generation for computations will fall into this first if statement as they are not library calls. This is the original code
        // Some library calls take the usual lhs as an actual argument however, so we may need to compute it anyway for some library calls 
        if (!this->is_library_call() || this->lhs_argument_idx != -1) { // This has an LHS to compute.
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
            const auto &buffer_entry = this->fct->get_buffers().find(buffer_name);
            assert(buffer_entry != this->get_function()->get_buffers().end());

            const auto &tiramisu_buffer = buffer_entry->second;
            DEBUG(3, tiramisu::str_dump(
                    "A Tiramisu buffer that corresponds to the buffer indicated in the access relation was found."));

            DEBUG(10, tiramisu_buffer->dump(true));

            Halide::Type type = halide_type_from_tiramisu_type(this->get_data_type());
            int buf_dims = tiramisu_buffer->get_dim_sizes().size();

            // Tiramisu buffer is from outermost to innermost, whereas Halide buffer is
            // from innermost to outermost; thus, we need to reverse the order
            halide_dimension_t *shape = new halide_dimension_t[tiramisu_buffer->get_dim_sizes().size()];
            int stride = 1;
            std::vector<Halide::Expr> strides_vector;

            if (tiramisu_buffer->has_constant_extents()) {
                for (int i = 0; i < buf_dims; i++) {
                    shape[i].min = 0;
                    int dim_idx = tiramisu_buffer->get_dim_sizes().size() - i - 1;
                    shape[i].extent = (int) tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
                    shape[i].stride = stride;
                    stride *= (int) tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
                }
            } else {
                std::vector<isl_ast_expr *> empty_index_expr;
                Halide::Expr stride_expr = Halide::Expr(1);
                for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++) {
                    int dim_idx = tiramisu_buffer->get_dim_sizes().size() - i - 1;
                    strides_vector.push_back(stride_expr);
                    stride_expr = stride_expr *
                                  generator::halide_expr_from_tiramisu_expr(this->get_function(), empty_index_expr,
                                                                            replace_original_indices_with_transformed_indices(
                                                                                    tiramisu_buffer->get_dim_sizes()[dim_idx],
                                                                                    this->get_iterators_map()), this);
                }
            }

            // The number of dimensions in the Halide buffer should be equal to
            // the number of dimensions of the access function.
            assert(buf_dims == access_dims);
            assert(this->index_expr[0] != NULL);
            DEBUG(3, tiramisu::str_dump("Linearizing access of the LHS index expression."));


            Halide::Expr index;
            if (tiramisu_buffer->has_constant_extents())
                index = tiramisu::generator::linearize_access(buf_dims, shape, this->index_expr[0]);
            else
                index = tiramisu::generator::linearize_access(buf_dims, strides_vector, this->index_expr[0]);

            DEBUG(3, tiramisu::str_dump("After linearization: ");
                    std::cout << index << std::endl);

            DEBUG(3, tiramisu::str_dump(
                    "Index expressions of this statement are (the first is the LHS and the others are the RHS) :"));
            print_isl_ast_expr_vector(this->index_expr);

            DEBUG(3, tiramisu::str_dump(
                    "Erasing the LHS index expression from the vector of index expressions (the LHS index has just been linearized)."));
            this->index_expr.erase(this->index_expr.begin());

            Halide::Internal::Parameter param;
            std::vector<Halide::Expr> halide_call_args;
            halide_call_args.resize(this->library_call_args.size());
            if (this->lhs_access_type == tiramisu::o_access) { // The majority of computations have this access type for the left hand side
                if (tiramisu_buffer->get_argument_type() == tiramisu::a_output) {
                    if (tiramisu_buffer->has_constant_extents()) {
                        Halide::Buffer<> buffer =
                                Halide::Buffer<>(
                                        halide_type_from_tiramisu_type(tiramisu_buffer->get_elements_type()),
                                        NULL,
                                        tiramisu_buffer->get_dim_sizes().size(),
                                        shape,
                                        tiramisu_buffer->get_name());
                        param = Halide::Internal::Parameter(buffer.type(), true, buffer.dimensions(), buffer.name());
                        param.set_buffer(buffer);
                        DEBUG(3, tiramisu::str_dump(
                                "Halide buffer object created.  This object will be passed to the Halide function that creates an assignment to a buffer."));
                    } else {
                        param = Halide::Internal::Parameter(
                                halide_type_from_tiramisu_type(tiramisu_buffer->get_elements_type()),
                                true,
                                tiramisu_buffer->get_dim_sizes().size(),
                                tiramisu_buffer->get_name());
                        std::vector<isl_ast_expr *> empty_index_expr;
                        Halide::Expr stride_expr = Halide::Expr(1);
                        for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++) {
                            int dim_idx = tiramisu_buffer->get_dim_sizes().size() - i - 1;
                            param.set_min_constraint(i, Halide::Expr(0));
                            param.set_extent_constraint(i,
                                                        generator::halide_expr_from_tiramisu_expr(this->get_function(),
                                                                                                  empty_index_expr,
                                                                                                  tiramisu_buffer->get_dim_sizes()[dim_idx], this));
                            param.set_stride_constraint(i, stride_expr);
                            stride_expr = stride_expr *
                                          generator::halide_expr_from_tiramisu_expr(this->get_function(),
                                                                                    empty_index_expr,
                                                                                    tiramisu_buffer->get_dim_sizes()[dim_idx], this);
                        }
                    }
                }
                DEBUG(3, tiramisu::str_dump(
                        "Calling the Halide::Internal::Store::make function which creates the store statement."));
                DEBUG(3, tiramisu::str_dump(
                        "The RHS index expressions are first transformed to Halide expressions then passed to the make function."));

                // Replace the RHS expression to the transformed expressions.
                // We do not need to transform the indices of expression (this->index_expr), because in Tiramisu we assume
                // that an access can only appear when accessing a computation. And that case should be handled in the following transformation
                // so no need to transform this->index_expr separately.
                tiramisu::expr tiramisu_rhs = replace_original_indices_with_transformed_indices(this->expression,
                                                                                                this->get_iterators_map());

                this->stmt = Halide::Internal::Store::make(
                        buffer_name,
                        generator::halide_expr_from_tiramisu_expr(this->get_function(), this->index_expr, tiramisu_rhs, this),
                        index, param, Halide::Internal::const_true(type.lanes()));

                DEBUG(3, tiramisu::str_dump("Halide::Internal::Store::make statement created."));
            } else if (this->is_library_call()) {
              // We need to make sure to process all of the other arguments for this library call
                for (int i = 0; i < this->library_call_args.size(); i++) {
                    if (i != this->rhs_argument_idx && i != this->lhs_argument_idx &&
                        i != this->wait_argument_idx) {
                        std::vector<isl_ast_expr *> dummy;
                        if (this->library_call_args[i].defined) {
                            halide_call_args[i] = generator::halide_expr_from_tiramisu_expr(this->fct, dummy,
                                                                                            this->library_call_args[i], this
                            );
                        }
                    }
                }
                if (this->lhs_argument_idx != -1) {
                    // The LHS is a parameter of the library call. We need to take the address of buffer at lhs_index as we assume that all 
                    // library calls requiring the LHS buffer also take the index into that buffer as either a separate argument (o_lin_index)
                    // or as an address_of into the buffer
                    Halide::Expr result;
                    Halide::Expr result2;
                    if (this->lhs_access_type == tiramisu::o_lin_index) { // pass in the index directly
                        result = index;
                    } else { // pass in LHS index and buffer into address_of extern function
                        result = Halide::Internal::Variable::make(Halide::type_of<struct halide_buffer_t *>(),
                                                                  tiramisu_buffer->get_name() + ".buffer");
                        result = Halide::Internal::Call::make(Halide::Handle(1, type.handle_type),
                                                              "tiramisu_address_of_" +
                                                              str_from_tiramisu_type_primitive(tiramisu_buffer->get_elements_type()),
                                                              {result, index},
                                                              Halide::Internal::Call::Extern);
                    }
                    halide_call_args[lhs_argument_idx] = result;
                } // else we don't care about the LHS 
                if (this->rhs_argument_idx != -1) { // THis library call also requires a RHS buffer and index, so process that
                    if (this->get_expr().get_op_type() == tiramisu::o_buffer) {
                      // TODO(Jess): Can we remove this assumption?? It will probably require adding yet another special index type (such as rhs_argument_index_index)
                        // In this case, we make a (correct for now) assumption that the last call arg gets the linear index and the rhs_argument_idx gets the buffer
                        expr old = this->get_expr(); 
                        expr mod_rhs(tiramisu::o_address,
                                     this->get_expr().get_name());
                        this->expression = mod_rhs;
                        assert(this->get_expr().get_op_type() != o_buffer);
                        halide_call_args[rhs_argument_idx] = // the buffer
                                generator::halide_expr_from_tiramisu_expr(this->fct, this->get_index_expr(),
                                                                          mod_rhs, this);

                        expr mod_rhs2(tiramisu::o_lin_index, old.get_name(), old.get_access(), old.get_data_type());
                        this->expression = mod_rhs2;
                        if (this->library_call_name == "tiramisu_cudad_memcpy_async_d2h" ||
                            this->library_call_name == "tiramisu_cudad_memcpy_d2h") {
                            halide_call_args[halide_call_args.size() - 1] = // just the index
                                    generator::halide_expr_from_tiramisu_expr(this->fct, this->get_index_expr(),
                                                                              this->get_expr(), this) *
                                    halide_type_from_tiramisu_type(this->get_data_type()).bytes();
                        }
                        this->set_expression(old);

                    } else {
                        halide_call_args[rhs_argument_idx] =
                                generator::halide_expr_from_tiramisu_expr(this->fct, this->get_index_expr(),
                                                                          this->get_expr(), this);
                    }
                }
            } else {
                assert(false && "Unsupported LHS operation type.");
            }
            // Defines writing into the wait buffer when a transfer is initiated (for nonblocking operations)
            if (this->wait_argument_idx != -1) {
                ERROR("Nonblocking not currently supported", 0);
                assert((this->is_recv() || this->is_send_recv()) && "This should be a recv or one-sided operation.");
                assert(this->wait_access_map && "A wait access map must be provided.");
                // We treat this like another LHS access, so we'll recompute the LHS access using the req access map.
                // First, find the request buffer.
                const auto &wait_buffer_entry = this->fct->get_buffers().find(
                        isl_map_get_tuple_name(this->wait_access_map, isl_dim_out));
                assert(wait_buffer_entry != this->fct->get_buffers().end());
                const auto &wait_tiramisu_buffer = wait_buffer_entry->second;
                // Now, compute the index into the buffer
                halide_dimension_t *wait_shape = new halide_dimension_t[wait_tiramisu_buffer->get_dim_sizes().size()];
                int wait_stride = 1;
                int wait_buf_dims = wait_tiramisu_buffer->get_dim_sizes().size();
                std::vector<Halide::Expr> wait_strides_vector;
                if (wait_tiramisu_buffer->has_constant_extents()) {
                    for (int i = 0; i < wait_buf_dims; i++) {
                        wait_shape[i].min = 0;
                        int dim_idx = wait_tiramisu_buffer->get_dim_sizes().size() - i - 1;
                        wait_shape[i].extent = (int) wait_tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
                        wait_shape[i].stride = wait_stride;
                        wait_stride *= (int) wait_tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
                    }
                } else {
                    std::vector<isl_ast_expr *> empty_index_expr;
                    Halide::Expr stride_expr = Halide::Expr(1);
                    for (int i = 0; i < wait_tiramisu_buffer->get_dim_sizes().size(); i++) {
                        int dim_idx = wait_tiramisu_buffer->get_dim_sizes().size() - i - 1;
                        wait_strides_vector.push_back(stride_expr);
                        stride_expr = stride_expr * generator::halide_expr_from_tiramisu_expr(fct, empty_index_expr,
                                                                                              wait_tiramisu_buffer->get_dim_sizes()[dim_idx], this);
                    }
                }

                assert(this->wait_index_expr != NULL);
                Halide::Expr wait_index;
                if (wait_tiramisu_buffer->has_constant_extents()) {
                    wait_index = tiramisu::generator::linearize_access(wait_buf_dims, wait_shape,
                                                                       this->wait_index_expr);
                } else {
                    wait_index = tiramisu::generator::linearize_access(wait_tiramisu_buffer->get_dim_sizes().size(),
                                                                       wait_strides_vector, this->wait_index_expr);
                }
                // Finally, index into the buffer
                Halide::Expr result = Halide::Internal::Variable::make(Halide::type_of<struct halide_buffer_t *>(),
                                                          wait_tiramisu_buffer->get_name() + ".buffer");
                result = Halide::Internal::Call::make(Halide::Handle(1, type.handle_type),
                                                      "tiramisu_address_of_" +
                                                      str_from_tiramisu_type_primitive(wait_tiramisu_buffer->get_elements_type()),
                                                      {result, wait_index},
                                                      Halide::Internal::Call::Extern);
                // We now have an index into the request buffer so that we can write to it with the operation,
                // which is either a send or a receive
                halide_call_args[wait_argument_idx] = result;
            }
            if (this->is_library_call()) {
                // Now, create the library call and evaluate it. This becomes the Halide stmt.
                this->stmt = Halide::Internal::Evaluate::make(make_comm_call(Halide::Bool(), this->library_call_name,
                                                                             halide_call_args));
            }
            delete[] shape;
        } else { // No LHS to compute. Only valid for library calls. Regular computations do require a LHS access.
            assert(this->is_library_call() &&
                   "Only library calls and let statements are allowed to not have a LHS access function!");
            std::vector<Halide::Expr> halide_call_args;
            halide_call_args.resize(this->library_call_args.size());
            for (int i = 0; i < this->library_call_args.size(); i++) {
                if (i != this->rhs_argument_idx && i != this->lhs_argument_idx && i != this->wait_argument_idx) {
                    std::vector<isl_ast_expr *> dummy;
                    halide_call_args[i] = generator::halide_expr_from_tiramisu_expr(this->get_function(), dummy,
                                                                                    this->library_call_args[i], this);
                }
            }
            // Process the RHS
            if (this->rhs_argument_idx != -1) {
                halide_call_args[rhs_argument_idx] = generator::halide_expr_from_tiramisu_expr(this->get_function(),
                                                                                               this->get_index_expr(),
                                                                                               this->get_expr(), this);
            }
            if (this->wait_argument_idx != -1) {
                ERROR("Nonblocking not currently supported", 0);
                assert(this->is_send() && "This should be a send operation.");
                assert(this->wait_access_map && "A request access map must be provided.");
                // We treat this like another LHS access, so we'll recompute the LHS access using the req access map.
                // First, find the request buffer.
                const auto &req_buffer_entry = this->fct->get_buffers().find(
                        isl_map_get_tuple_name(this->wait_access_map, isl_dim_out));
                assert(req_buffer_entry != this->fct->get_buffers().end());
                const auto &req_tiramisu_buffer = req_buffer_entry->second;
                // Now, compute the index into the buffer
                halide_dimension_t *req_shape = new halide_dimension_t[req_tiramisu_buffer->get_dim_sizes().size()];
                int req_stride = 1;
                int req_buf_dims = req_tiramisu_buffer->get_dim_sizes().size();
                if (req_tiramisu_buffer->has_constant_extents()) {
                    for (int i = 0; i < req_buf_dims; i++) {
                        req_shape[i].min = 0;
                        int dim_idx = req_tiramisu_buffer->get_dim_sizes().size() - i - 1;
                        req_shape[i].extent = (int) req_tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
                        req_shape[i].stride = req_stride;
                        req_stride *= (int) req_tiramisu_buffer->get_dim_sizes()[dim_idx].get_int_val();
                    }
                }

                assert(this->wait_index_expr != NULL);
                Halide::Expr req_index = tiramisu::generator::linearize_access(req_buf_dims, req_shape,
                                                                               this->wait_index_expr);
                // Finally, index into the buffer
                Halide::Type req_type = halide_type_from_tiramisu_type(p_wait_ptr);
                Halide::Expr result = Halide::Internal::Variable::make(Halide::type_of<struct halide_buffer_t *>(),
                                                                       req_tiramisu_buffer->get_name() + ".buffer");
                result = Halide::Internal::Call::make(Halide::Handle(1, req_type.handle_type),
                                                      "tiramisu_address_of_" +
                                                      str_from_tiramisu_type_primitive(req_tiramisu_buffer->get_elements_type()),
                                                      {result, req_index},
                                                      Halide::Internal::Call::Extern);
                // We now have an index into the request buffer so that we can write to it with the operation,
                // which is either a send or a receive
                halide_call_args[wait_argument_idx] = result;
            }
            // Create the library call (assumed to be a communication call for right now)
            this->stmt = Halide::Internal::Evaluate::make(make_comm_call(Halide::Bool(), this->library_call_name,
                                                                         halide_call_args));

        }
    }

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("End of create_halide_stmt. Generated statement is: ");
            std::cout << this->stmt);

    DEBUG_INDENT(-4);
}
tiramisu::expr generator::replace_accesses(const tiramisu::function *fct, std::vector<isl_ast_expr *> &index_expr,
                                           const tiramisu::expr &tiramisu_expr){
    tiramisu::expr result;

    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    DEBUG(10, tiramisu::str_dump("Input Tiramisu expression: "); tiramisu_expr.dump(false));
    if (fct != nullptr)
    {
        DEBUG(10, tiramisu::str_dump("The input function is " + fct->get_name()));
    }
    else
    {
        DEBUG(10, tiramisu::str_dump("The input function is NULL."));
    }
    if (index_expr.empty())
    {
        DEBUG(10, tiramisu::str_dump("The input index_expr is empty."));
    }
    else
    {
        DEBUG(10, tiramisu::str_dump("The input index_expr is not empty."));
    }

    if (tiramisu_expr.get_expr_type() == tiramisu::e_op) {
        auto op_type = tiramisu_expr.get_op_type();

        expr modified_expr = tiramisu_expr.apply_to_operands([&fct, &index_expr](const expr & e){
            return generator::replace_accesses(fct, index_expr, e);
        });

        if (op_type == o_access || op_type == o_address) {
            DEBUG(10, tiramisu::str_dump("op type: o_access or o_address"));

            const char *access_comp_name = nullptr;

            if (op_type == tiramisu::o_access) {
                access_comp_name = modified_expr.get_name().c_str();
            } else {
                access_comp_name = modified_expr.get_operand(0).get_name().c_str();
            }

            assert(access_comp_name != nullptr);

            DEBUG(10, tiramisu::str_dump("Computation being accessed: ");
                    tiramisu::str_dump(access_comp_name));

            // Since we modify the names of update computations but do not modify the
            // expressions.  When accessing the expressions we find the old names, so
            // we need to look for the new names instead of the old names.
            // We do this instead of actually changing the expressions, because changing
            // the expressions will make the semantics of the printed program ambiguous,
            // since we do not have any way to distinguish between which update is the
            // consumer is consuming exactly.
            std::vector<tiramisu::computation *> computations_vector
                    = fct->get_computation_by_name(access_comp_name);
            if (computations_vector.empty()) {
                // Search for update computations.
                computations_vector
                        = fct->get_computation_by_name("_" + std::string(access_comp_name) + "_update_0");
                assert((!computations_vector.empty()) && "Computation not found.");
            }

            // We assume that computations that have the same name write all to the same buffer
            // but may have different access relations.
            tiramisu::computation *access_comp = computations_vector[0];
            assert((access_comp != nullptr) && "Accessed computation is NULL.");
            if (op_type == tiramisu::o_access) {
                result = generator::comp_to_buffer(access_comp, index_expr, &modified_expr);

            } else {
                // It's an o_address
                std::string buffer_name = generator::get_buffer_name(access_comp);
                // TODO what to do in this case?
            }
        }
        else
        {
            result = modified_expr;
        }
    } else
    {
        result = tiramisu_expr;
    }

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(10);

    return result;
}

Halide::Expr generator::halide_expr_from_tiramisu_expr(const tiramisu::function *fct,
                                                       std::vector<isl_ast_expr *> &index_expr,
                                                       const tiramisu::expr &tiramisu_expr, tiramisu::computation *comp)
{
    Halide::Expr result;

    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    DEBUG(10, tiramisu::str_dump("Input Tiramisu expression: "); tiramisu_expr.dump(false));
    if (fct != NULL)
    {
        DEBUG(10, tiramisu::str_dump("The input function is " + fct->get_name()));
    }
    else
    {
        DEBUG(10, tiramisu::str_dump("The input function is NULL."));
    }
    if (index_expr.size() > 0)
    {
        DEBUG(10, tiramisu::str_dump("The input index_expr is not empty."));
    }
    else
    {
        DEBUG(10, tiramisu::str_dump("The input index_expr is empty."));
    }

    if (tiramisu_expr.get_expr_type() == tiramisu::e_val)
    {
        DEBUG(10, tiramisu::str_dump("tiramisu expression of type tiramisu::e_val"));
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

        DEBUG(10, tiramisu::str_dump("tiramisu expression of type tiramisu::e_op"));

        if (tiramisu_expr.get_n_arg() > 0)
        {
            tiramisu::expr expr0 = tiramisu_expr.get_operand(0);
            op0 = generator::halide_expr_from_tiramisu_expr(fct, index_expr, expr0, comp);
        }

        if (tiramisu_expr.get_n_arg() > 1)
        {
            tiramisu::expr expr1 = tiramisu_expr.get_operand(1);
            op1 = generator::halide_expr_from_tiramisu_expr(fct, index_expr, expr1, comp);
        }

        if (tiramisu_expr.get_n_arg() > 2)
        {
            tiramisu::expr expr2 = tiramisu_expr.get_operand(2);
            op2 = generator::halide_expr_from_tiramisu_expr(fct, index_expr, expr2, comp);
        }

        switch (tiramisu_expr.get_op_type())
        {
            case tiramisu::o_logical_and:
                result = Halide::Internal::And::make(op0, op1);
                DEBUG(10, tiramisu::str_dump("op type: o_logical_and"));
                break;
            case tiramisu::o_logical_or:
                result = Halide::Internal::Or::make(op0, op1);
                DEBUG(10, tiramisu::str_dump("op type: o_logical_or"));
                break;
            case tiramisu::o_max:
                result = Halide::Internal::Max::make2(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_max"));
                break;
            case tiramisu::o_min:
                result = Halide::Internal::Min::make2(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_min"));
                break;
            case tiramisu::o_minus:
                result = Halide::Internal::Sub::make(Halide::cast(op0.type(), Halide::Expr(0)), op0, true);
                DEBUG(10, tiramisu::str_dump("op type: o_minus"));
                break;
            case tiramisu::o_add:
                result = Halide::Internal::Add::make(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_add"));
                break;
            case tiramisu::o_sub:
                result = Halide::Internal::Sub::make(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_sub"));
                break;
            case tiramisu::o_mul:
                result = Halide::Internal::Mul::make(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_mul"));
                break;
            case tiramisu::o_div:
                result = Halide::Internal::Div::make(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_div"));
                break;
            case tiramisu::o_mod:
                result = Halide::Internal::Mod::make(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_mod"));
                break;
            case tiramisu::o_select:
                result = Halide::Internal::Select::make(op0, op1, op2);
                DEBUG(10, tiramisu::str_dump("op type: o_select"));
                break;
            case tiramisu::o_lerp:
                result = Halide::lerp(op0, op1, op2);
                DEBUG(10, tiramisu::str_dump("op type: lerp"));
                break;
            case tiramisu::o_cond:
                ERROR("Code generation for o_cond is not supported yet.", true);
                break;
            case tiramisu::o_le:
                result = Halide::Internal::LE::make(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_le"));
                break;
            case tiramisu::o_lt:
                result = Halide::Internal::LT::make(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_lt"));
                break;
            case tiramisu::o_ge:
                result = Halide::Internal::GE::make(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_ge"));
                break;
            case tiramisu::o_gt:
                result = Halide::Internal::GT::make(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_gt"));
                break;
            case tiramisu::o_logical_not:
                result = Halide::Internal::Not::make(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_not"));
                break;
            case tiramisu::o_eq:
                result = Halide::Internal::EQ::make(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_eq"));
                break;
            case tiramisu::o_ne:
                result = Halide::Internal::NE::make(op0, op1, true);
                DEBUG(10, tiramisu::str_dump("op type: o_ne"));
                break;
            case tiramisu::o_type:
                result = halide_expr_from_tiramisu_type(tiramisu_expr.get_data_type());
                break;
            case tiramisu::o_access:
            case tiramisu::o_lin_index:
            case tiramisu::o_address:
            case tiramisu::o_address_of:
            {
                DEBUG(10, tiramisu::str_dump("op type: o_access or o_address"));

                const char *access_comp_name = NULL;

                if (tiramisu_expr.get_op_type() == tiramisu::o_access ||
                    tiramisu_expr.get_op_type() == tiramisu::o_lin_index ||
                    tiramisu_expr.get_op_type() == tiramisu::o_address_of)
                {
                    access_comp_name = tiramisu_expr.get_name().c_str();
                }
                else if (tiramisu_expr.get_op_type() == tiramisu::o_address)
                {
                    access_comp_name = tiramisu_expr.get_operand(0).get_name().c_str();
                }
                else
                {
                    ERROR("Unsupported operation.", true);
                }

                assert(access_comp_name != NULL);

                DEBUG(10, tiramisu::str_dump("Computation being accessed: "); tiramisu::str_dump(access_comp_name));

                // Since we modify the names of update computations but do not modify the
                // expressions.  When accessing the expressions we find the old names, so
                // we need to look for the new names instead of the old names.
                // We do this instead of actually changing the expressions, because changing
                // the expressions will make the semantics of the printed program ambiguous,
                // since we do not have any way to distinguish between which update is the
                // consumer is consuming exactly.
                std::vector<tiramisu::computation *> computations_vector
                        = fct->get_computation_by_name(access_comp_name);
                if (computations_vector.size() == 0)
                {
                    // Search for update computations.
                    computations_vector
                            = fct->get_computation_by_name("_" + std::string(access_comp_name) + "_update_0");
                    assert((computations_vector.size() > 0) && "Computation not found.");
                }

                // We assume that computations that have the same name write all to the same buffer
                // but may have different access relations.
                tiramisu::computation *access_comp = computations_vector[0];
                assert((access_comp != NULL) && "Accessed computation is NULL.");
                if (comp && comp->is_wait()) {
                    // swap
                    // use operations_vector[0] instead of access_comp because we need it to be non-const
                    isl_map *orig = computations_vector[0]->get_access_relation();
                    computations_vector[0]->set_access(computations_vector[0]->wait_access_map);
                    computations_vector[0]->wait_access_map = orig;
                }
                isl_map *acc = access_comp->get_access_relation_adapted_to_time_processor_domain();
                if (comp && comp->is_wait()) {
                    // swap back
                    isl_map *orig = computations_vector[0]->get_access_relation();
                    computations_vector[0]->set_access(computations_vector[0]->wait_access_map);
                    computations_vector[0]->wait_access_map = orig;
                }
                const char *buffer_name = isl_space_get_tuple_name(
                        isl_map_get_space(acc),
                        isl_dim_out);
                assert(buffer_name != NULL);
                DEBUG(10, tiramisu::str_dump("Name of the associated buffer: "); tiramisu::str_dump(buffer_name));

                const auto &buffer_entry = fct->get_buffers().find(buffer_name);
                assert(buffer_entry != fct->get_buffers().end());

                const auto &tiramisu_buffer = buffer_entry->second;

                Halide::Type type = halide_type_from_tiramisu_type(tiramisu_buffer->get_elements_type());

                // Tiramisu buffer is from outermost to innermost, whereas Halide buffer is from innermost
                // to outermost; thus, we need to reverse the order
                halide_dimension_t *shape = new halide_dimension_t[tiramisu_buffer->get_dim_sizes().size()];
                int stride = 1;
                std::vector<Halide::Expr> strides_vector;

                if (tiramisu_buffer->has_constant_extents())
                {
                    DEBUG(10, tiramisu::str_dump("Buffer has constant extents."));
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
                    DEBUG(10, tiramisu::str_dump("Buffer has non-constant extents."));
                    std::vector<isl_ast_expr *> empty_index_expr;
                    Halide::Expr stride_expr = Halide::Expr(1);
                    for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++)
                    {
                        int dim_idx = tiramisu_buffer->get_dim_sizes().size() - i - 1;
                        strides_vector.push_back(stride_expr);
                        stride_expr = stride_expr * generator::halide_expr_from_tiramisu_expr(fct, empty_index_expr, tiramisu_buffer->get_dim_sizes()[dim_idx], comp);
                    }
                }
                DEBUG(10, tiramisu::str_dump("Buffer strides have been computed."));

                if (tiramisu_expr.get_op_type() == tiramisu::o_access ||
                    tiramisu_expr.get_op_type() == tiramisu::o_address_of ||
                    tiramisu_expr.get_op_type() == tiramisu::o_lin_index)
                {
                    Halide::Expr index;

                    // If index_expr is empty, and since tiramisu_expr is
                    // an access expression, this means that index_expr was not
                    // computed using the statement generator because this
                    // expression is not an expression that is associated with
                    // a computation. It is rather an expression used by
                    // a computation (for example, as the size of a buffer
                    // dimension). So in this case, we retrieve the indices directly
                    // from tiramisu_expr.
                    // The possible problem in this case, is that the indices
                    // in tiramisu_expr cannot be adapted to the schedule if
                    // these indices are i, j, .... This means that these
                    // indices have to be constant value only. So we check for this.
                    if (index_expr.size() == 0)
                    {
                        DEBUG(10, tiramisu::str_dump("index_expr is empty. Retrieving access indices directly from the tiramisu access expression without scheduling."));

                        for (int i = 0; i < tiramisu_buffer->get_dim_sizes().size(); i++)
                        {
                            // Actually any access that does not require
                            // scheduling is supported but currently we only
                            // accept literal constants as anything else was not
                            // needed til now.
                            assert(tiramisu_expr.get_access()[i].is_constant() && "Only constant accesses are supported.");
                        }

                        if (tiramisu_buffer->has_constant_extents())
                            index = tiramisu::generator::linearize_access(tiramisu_buffer->get_dim_sizes().size(), shape, tiramisu_expr.get_access());
                        else
                            index = tiramisu::generator::linearize_access(tiramisu_buffer->get_dim_sizes().size(), strides_vector, tiramisu_expr.get_access());
                    }
                    else
                    {
                        DEBUG(10, tiramisu::str_dump("index_expr is NOT empty. Retrieving access indices from index_expr (i.e., retrieving indices adapted to the schedule)."));
                        if (tiramisu_buffer->has_constant_extents())
                            index = tiramisu::generator::linearize_access(tiramisu_buffer->get_dim_sizes().size(), shape, index_expr[0]);
                        else
                            index = tiramisu::generator::linearize_access(tiramisu_buffer->get_dim_sizes().size(), strides_vector, index_expr[0]);

                        index_expr.erase(index_expr.begin());
                    }
                    if (tiramisu_expr.get_op_type() == tiramisu::o_lin_index) {
                        result = index;
                    }
                    else if (tiramisu_buffer->get_argument_type() == tiramisu::a_input)
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
                        if (tiramisu_expr.get_op_type() != tiramisu::o_address_of) {
                            result = Halide::Internal::Load::make(
                                    type, tiramisu_buffer->get_name(), index, Halide::Buffer<>(),
                                    param, Halide::Internal::const_true(type.lanes()));
                        } else {
                            result = Halide::Internal::Variable::make(Halide::type_of<struct halide_buffer_t *>(),
                                                                                   tiramisu_buffer->get_name() + ".buffer");
                            result = Halide::Internal::Call::make(Halide::Handle(1, type.handle_type),
                                                                  "tiramisu_address_of_" +
                                                                  str_from_tiramisu_type_primitive(tiramisu_buffer->get_elements_type()),
                                                                  {result, index},
                                                                  Halide::Internal::Call::Extern);
                        }
                    }
                    else
                    {
                        if (tiramisu_expr.get_op_type() != tiramisu ::o_address_of) {
                            result = Halide::Internal::Load::make(
                                    type, tiramisu_buffer->get_name(), index, Halide::Buffer<>(),
                                    Halide::Internal::Parameter(), Halide::Internal::const_true(type.lanes()));
                        } else {
                            result = Halide::Internal::Variable::make(Halide::type_of<struct halide_buffer_t *>(),
                                                                      tiramisu_buffer->get_name() + ".buffer");
                            result = Halide::Internal::Call::make(Halide::Handle(1, type.handle_type),
                                                                  "tiramisu_address_of_" +
                                                                  str_from_tiramisu_type_primitive(tiramisu_buffer->get_elements_type()),
                                                                  {result, index},
                                                                  Halide::Internal::Call::Extern);
                        }
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
                DEBUG(10, tiramisu::str_dump("op type: o_right_shift"));
                break;
            case tiramisu::o_left_shift:
                result = op0 << op1;
                DEBUG(10, tiramisu::str_dump("op type: o_left_shift"));
                break;
            case tiramisu::o_floor:
                result = Halide::floor(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_floor"));
                break;
            case tiramisu::o_cast:
                result = Halide::cast(halide_type_from_tiramisu_type(tiramisu_expr.get_data_type()), op0);
                DEBUG(10, tiramisu::str_dump("op type: o_cast"));
                break;
            case tiramisu::o_sin:
                result = Halide::sin(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_sin"));
                break;
            case tiramisu::o_cos:
                result = Halide::cos(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_cos"));
                break;
            case tiramisu::o_tan:
                result = Halide::tan(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_tan"));
                break;
            case tiramisu::o_asin:
                result = Halide::asin(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_asin"));
                break;
            case tiramisu::o_acos:
                result = Halide::acos(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_acos"));
                break;
            case tiramisu::o_atan:
                result = Halide::atan(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_atan"));
                break;
            case tiramisu::o_sinh:
                result = Halide::sinh(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_sinh"));
                break;
            case tiramisu::o_cosh:
                result = Halide::cosh(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_cosh"));
                break;
            case tiramisu::o_tanh:
                result = Halide::tanh(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_tanh"));
                break;
            case tiramisu::o_asinh:
                result = Halide::asinh(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_asinh"));
                break;
            case tiramisu::o_acosh:
                result = Halide::acosh(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_acosh"));
                break;
            case tiramisu::o_atanh:
                result = Halide::atanh(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_atanh"));
                break;
            case tiramisu::o_abs:
                result = Halide::abs(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_abs"));
                break;
            case tiramisu::o_sqrt:
                result = Halide::sqrt(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_sqrt"));
                break;
            case tiramisu::o_expo:
                result = Halide::exp(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_expo"));
                break;
            case tiramisu::o_log:
                result = Halide::log(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_log"));
                break;
            case tiramisu::o_ceil:
                result = Halide::ceil(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_ceil"));
                break;
            case tiramisu::o_round:
                result = Halide::round(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_round"));
                break;
            case tiramisu::o_trunc:
                result = Halide::trunc(op0);
                DEBUG(10, tiramisu::str_dump("op type: o_trunc"));
                break;
            case tiramisu::o_call:
            {
                std::vector<Halide::Expr> vec;
                for (const auto &e : tiramisu_expr.get_arguments())
                {
                    Halide::Expr he = generator::halide_expr_from_tiramisu_expr(fct, index_expr, e, comp);
                    vec.push_back(he);
                }
                result = Halide::Internal::Call::make(halide_type_from_tiramisu_type(tiramisu_expr.get_data_type()),
                                                      tiramisu_expr.get_name(),
                                                      vec,
                                                      Halide::Internal::Call::CallType::Extern);
                DEBUG(10, tiramisu::str_dump("op type: o_call"));
                break;
            }
            case tiramisu::o_allocate:
            case tiramisu::o_free:
                ERROR("An expression of type o_allocate or o_free "
                                        "should not be passed to this function", true);
                break;
            default:
                ERROR("Translating an unsupported ISL expression into a Halide expression.", 1);
        }
    }
    else if (tiramisu_expr.get_expr_type() == tiramisu::e_var)
    {
        DEBUG(3, tiramisu::str_dump("Generating a variable access expression."));
        DEBUG(3, tiramisu::str_dump("Expression is a variable of type: " + tiramisu::str_from_tiramisu_type_primitive(tiramisu_expr.get_data_type())));
        result = Halide::Internal::Variable::make(
                halide_type_from_tiramisu_type(tiramisu_expr.get_data_type()),
                tiramisu_expr.get_name());
    }
    else
    {
        tiramisu::str_dump("tiramisu type of expr: ",
                           str_from_tiramisu_type_expr(tiramisu_expr.get_expr_type()).c_str());
        ERROR("\nTranslating an unsupported ISL expression in a Halide expression.", 1);
    }

    if (result.defined())
    {
        DEBUG(10, tiramisu::str_dump("Generated stmt: "); std::cout << result);
    }

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(10);

    return result;
}

void function::gen_halide_obj(const std::string &obj_file_name, Halide::Target::OS os,
                              Halide::Target::Arch arch, int bits) const
{
    // TODO(tiramisu): For GPU schedule, we need to set the features, e.g.
    // Halide::Target::OpenCL, etc.
    // Note: "make test" fails on Travis machines when AVX2 is used.
    //       Disable travis tests in .travis.yml if you switch to AVX2.
    std::vector<Halide::Target::Feature> features =
            {
                    Halide::Target::AVX,
                    Halide::Target::SSE41,
                    // Halide::Target::AVX2,
                    Halide::Target::LargeBuffers
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

    if (nvcc_compiler) {
        nvcc_compiler->compile(obj_file_name);
    }
}

void tiramisu::generator::update_producer_expr_name(tiramisu::computation *comp, std::string name_to_replace,
                                                    std::string replace_with) {
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::generator::_update_producer_expr_name(comp->expression, name_to_replace, replace_with);

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(3);
}

void tiramisu::generator::_update_producer_expr_name(tiramisu::expr &current_exp, std::string name_to_replace,
                                                     std::string replace_with) {

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if ((current_exp.get_expr_type() == tiramisu::e_op) && ((current_exp.get_op_type() == tiramisu::o_access) ||
                                                            (current_exp.get_op_type() == tiramisu::o_lin_index) ||
                                                            (current_exp.get_op_type() == tiramisu::o_address_of))) {
        // Shouldn't be any sub-expressions to iterate through with an access.
        if (current_exp.get_name() == name_to_replace) {
            current_exp.set_name(replace_with);
        }

    } else if (current_exp.get_expr_type() == tiramisu::e_op) {
        DEBUG(3, tiramisu::str_dump("Extracting access from e_op."));

        switch (current_exp.get_op_type())
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
            case tiramisu::o_sinh:
            case tiramisu::o_cosh:
            case tiramisu::o_tanh:
            case tiramisu::o_asinh:
            case tiramisu::o_acosh:
            case tiramisu::o_atanh:
            case tiramisu::o_abs:
            case tiramisu::o_sqrt:
            case tiramisu::o_expo:
            case tiramisu::o_log:
            case tiramisu::o_ceil:
            case tiramisu::o_round:
            case tiramisu::o_trunc:
            case tiramisu::o_address:
            {
                tiramisu::expr &exp0 = current_exp.op[0];
                generator::_update_producer_expr_name(exp0, name_to_replace, replace_with);
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
                tiramisu::expr &exp0 = current_exp.op[0];
                tiramisu::expr &exp1 = current_exp.op[1];
                generator::_update_producer_expr_name(exp0, name_to_replace, replace_with);
                generator::_update_producer_expr_name(exp1, name_to_replace, replace_with);
                break;
            }
            case tiramisu::o_select:
            case tiramisu::o_cond:
            {
                tiramisu::expr &exp0 = current_exp.op[0];
                tiramisu::expr &exp1 = current_exp.op[1];
                tiramisu::expr &exp2 = current_exp.op[2];
                generator::_update_producer_expr_name(exp0, name_to_replace, replace_with);
                generator::_update_producer_expr_name(exp1, name_to_replace, replace_with);
                generator::_update_producer_expr_name(exp2, name_to_replace, replace_with);
                break;
            }
            case tiramisu::o_call:
            {
                for (auto &e : current_exp.argument_vector) {
                    generator::_update_producer_expr_name(e, name_to_replace, replace_with);
                }
                break;
            }
            case tiramisu::o_allocate:
            case tiramisu::o_free:
            case tiramisu::o_type:
                // They do not have any access.
                break;
            default:
                ERROR("Replacing expression name for an unsupported tiramisu expression.", 1);
        }
    }

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(3);
}

Halide::Expr make_comm_call(Halide::Type type, std::string func_name, std::vector<Halide::Expr> args) {
    return Halide::Internal::Call::make(type, func_name, args, Halide::Internal::Call::CallType::Extern);
}

Halide::Expr halide_expr_from_tiramisu_type(tiramisu::primitive_t ptype) {
    switch (ptype) {
        case p_uint8: return Halide::Expr((uint8_t)0);
        case p_uint16: return Halide::Expr((uint16_t)0);
        case p_uint32: return Halide::Expr((uint32_t)0);
        case p_uint64: return Halide::Expr((uint64_t)0);
        case p_int8: return Halide::Expr((int8_t)0);
        case p_int16: return Halide::Expr((int16_t)0);
        case p_int32: return Halide::Expr((int32_t)0);
        case p_int64: return Halide::Expr((int64_t)0);
        case p_float32: return Halide::Expr((float)0);
        case p_float64: return Halide::Expr((double)0);
        default: { assert(false && "Bad type specified"); return Halide::Expr(); }
    }
}

Halide::Internal::Stmt generator::make_buffer_free(buffer * b) {
    assert(b != nullptr);
    if (b->location == cuda_ast::memory_location::global)
    {
        return Halide::Internal::Evaluate::make(
                Halide::Internal::Call::make(Halide::Int(32), "tiramisu_cuda_free",
                                             {Halide::Internal::Variable::make(Halide::type_of<void *>(), b->get_name())}, Halide::Internal::Call::Extern)
        );
    } else {
        return Halide::Internal::Free::make(b->get_name());
    }
}

}
