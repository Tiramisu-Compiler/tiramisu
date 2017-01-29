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

#include <coli/debug.h>
#include <coli/core.h>
#include <coli/type.h>
#include <coli/expr.h>

#include <string>

namespace coli
{

Halide::Argument::Kind halide_argtype_from_coli_argtype(coli::argument_t type);
Halide::Expr linearize_access(Halide::Internal::BufferPtr *buffer, isl_ast_expr *index_expr);

computation *function::get_computation_by_name(std::string name) const
{
    assert(name.size() > 0);

    DEBUG(10, coli::str_dump ("Searching computation " + name));

    coli::computation *res_comp = NULL;

    for (const auto &comp : this->get_computations())
    {
        if (name == comp->get_name())
        {
            res_comp = comp;
        }
    }

    if (res_comp == NULL)
    {
        DEBUG(10, coli::str_dump ("Computation not found."));
    }
    else
    {
        DEBUG(10, coli::str_dump ("Computation found."));
    }

    return res_comp;
}

/**
  * Get the computation associated with a node.
  */
coli::computation *get_computation_by_node(coli::function *fct, isl_ast_node *node)
{
    isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
    isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
    isl_id *id = isl_ast_expr_get_id(arg);
    isl_ast_expr_free(arg);
    std::string computation_name(isl_id_get_name(id));
    isl_id_free(id);
    coli::computation *comp = fct->get_computation_by_name(computation_name);

    assert((comp != NULL) && "Computation not found for this node.");

    return comp;
}

isl_map* create_map_from_domain_and_range (isl_set* domain, isl_set* range)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, coli::str_dump ("Domain:", isl_set_to_str(domain)));
    DEBUG(3, coli::str_dump ("Range:", isl_set_to_str(range)));
    // Extracting the spaces and aligning them
    isl_space* sp1 = isl_set_get_space (domain);
    isl_space* sp2 = isl_set_get_space (range);
    sp1 = isl_space_align_params (sp1, isl_space_copy (sp2));
    sp2 = isl_space_align_params (sp2, isl_space_copy (sp1));
    // Create the space access_domain -> sched_range.
    isl_space* sp = isl_space_map_from_domain_and_range (
            isl_space_copy (sp1), isl_space_copy (sp2));
    isl_map* adapter = isl_map_universe (sp);
    DEBUG(3, coli::str_dump ("Transformation map:", isl_map_to_str (adapter)));
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
    DEBUG(3, coli::str_dump(
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
    DEBUG(3, coli::str_dump("Schedule:", isl_map_to_str(schedule)));

    isl_map* map = isl_map_reverse(isl_map_copy(schedule));
    DEBUG(3, coli::str_dump("Schedule reversed:", isl_map_to_str(map)));

    isl_pw_multi_aff* iterator_map = isl_pw_multi_aff_from_map(map);
    DEBUG_NO_NEWLINE(3, coli::str_dump("The iterator map of an AST leaf (after scheduling):");
                        isl_pw_multi_aff_dump(iterator_map));
    DEBUG(3, coli::str_dump("Access:", isl_map_to_str(access)));
    isl_pw_multi_aff* index_aff = isl_pw_multi_aff_from_map(isl_map_copy(access));
    DEBUG_NO_NEWLINE(3, coli::str_dump("isl_pw_multi_aff_from_map(access):");
                        isl_pw_multi_aff_dump(index_aff));
    iterator_map = isl_pw_multi_aff_pullback_pw_multi_aff(index_aff, iterator_map);
    DEBUG_NO_NEWLINE(3, coli::str_dump("isl_pw_multi_aff_pullback_pw_multi_aff(index_aff,iterator_map):");
                        isl_pw_multi_aff_dump(iterator_map));
    isl_ast_expr* index_expr = isl_ast_build_access_from_pw_multi_aff(
                                    build,
                                    isl_pw_multi_aff_copy(iterator_map));
    DEBUG(3, coli::str_dump("isl_ast_build_access_from_pw_multi_aff(build, iterator_map):",
                            (const char * ) isl_ast_expr_to_C_str(index_expr)));

    DEBUG_INDENT(-4);

    return index_expr;
}

/**
 * Traverse the coli expression and extract accesses.
 */
void traverse_expr_and_extract_accesses(coli::function *fct,
                                        coli::computation *comp,
                                        const coli::expr &exp,
                                        std::vector<isl_map *> &accesses)
{
    assert(fct != NULL);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if ((exp.get_expr_type() == coli::e_op) && (exp.get_op_type() == coli::o_access))
    {
        DEBUG(3, coli::str_dump("Extracting access from o_access."));

        // Create the access map for this access node.
        coli::expr id = exp.get_operand(0);

        // Get the corresponding computation
        coli::computation *comp2 = fct->get_computation_by_name(id.get_name());
        DEBUG(3, coli::str_dump("The computation corresponding to the access: "
                                + comp2->get_name()));

        isl_map *access_function = isl_map_copy(comp2->get_access());

        DEBUG(3, coli::str_dump("The original access function of this computation (before transforming its domain into time-space) : ",
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

        DEBUG(3, coli::str_dump("Transformation map before adding constraints:",
                                isl_map_to_str(identity)));

        //TODO: make the following a recursive function that translates the access
        // into proper constraints.

        // The dimension_number is a counter that indicates to which dimension
        // is the access associated.
        int dimension_number = 0;
        for (const auto &access: exp.get_access())
        {
            isl_local_space *ls = isl_local_space_from_space(
                                        isl_map_get_space(
                                                isl_map_copy(identity)));
            isl_constraint *cst = isl_constraint_alloc_equality(
                                        isl_local_space_copy(ls));

            DEBUG(3, coli::str_dump(
                    "Assigning 1 to the coefficient of output dimension " +
                    std::to_string(dimension_number)));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out,
                                                    dimension_number, 1);

            if (access.get_expr_type() == coli::e_val)
            {
                cst = isl_constraint_set_constant_si(cst, (-1)*access.get_int_val());
                DEBUG(3, coli::str_dump(
                 "Assigning (-1)*access.get_int_val() to the cst dimension "));
            }
            else if (access.get_expr_type() == coli::e_id)
            {
                DEBUG(3, coli::str_dump("Looking for a dimension named ");
                         coli::str_dump(access.get_name());
                         coli::str_dump(" in the domain of ", isl_map_to_str(identity)));

                int dim0 = isl_space_find_dim_by_name(
                                isl_map_get_space(identity),
                                isl_dim_in,
                                access.get_name().c_str());
                if(dim0 >= 0)
                {
                    cst = isl_constraint_set_coefficient_si(cst, isl_dim_in,
                                                        dim0, -1);
                    DEBUG(3, coli::str_dump(
                         "Dimension found. Assigning -1 to the input coefficient of dimension " +
                          std::to_string(dim0)));
                }
                else
                {
                    DEBUG(3, coli::str_dump(
                            "Dimension not found.  Adding dimension as a parameter."));

                    identity = isl_map_add_dims(identity, isl_dim_param, 1);
                    int pos = isl_map_dim(identity, isl_dim_param);
                    isl_id *param_id = isl_id_alloc(fct->get_ctx(),
                                                    access.get_name().c_str(),
                                                    NULL);
                    identity = isl_map_set_dim_id(identity, isl_dim_param,
                                    pos-1, param_id);

                    ls = isl_local_space_from_space(
                             isl_map_get_space(
                                 isl_map_copy(identity)));
                     cst = isl_constraint_alloc_equality(
                               isl_local_space_copy(ls));

                     dim0 = isl_space_find_dim_by_name(
                                isl_map_get_space(identity),
                                isl_dim_param,
                                access.get_name().c_str());
                     assert((dim0 >= 0) && "Dimension not found");
                     cst = isl_constraint_set_coefficient_si(cst, isl_dim_param,
                                dim0, -1);
                     cst = isl_constraint_set_coefficient_si(cst, isl_dim_out,
                               dimension_number, 1);
                     DEBUG(3, coli::str_dump("After adding a parameter:",
                                                   isl_map_to_str(identity)));
                }
            }
            else if (access.get_expr_type() == coli::e_op)
            {
                if (access.get_op_type() == coli::o_add)
                {
                    coli::expr op0 = access.get_operand(0);
                    coli::expr op1 = access.get_operand(1);

                    if (op0.get_expr_type() == coli::e_id)
                    {
                        int dim0 = isl_space_find_dim_by_name(
                                        isl_map_get_space(identity),
                                        isl_dim_in,
                                        op0.get_name().c_str());
                        assert((dim0 >= 0) && "Dimension not found");
                        cst = isl_constraint_set_coefficient_si(cst, isl_dim_in,
                                                                dim0, -1);
                        DEBUG(3, coli::str_dump(
                         "Assigning -1 to the input coefficient of dimension " +
                         std::to_string(dim0)));
                    }
                    if (op1.get_expr_type() == coli::e_id)
                    {
                        int dim0 = isl_space_find_dim_by_name(
                                        isl_map_get_space(identity),
                                        isl_dim_in,
                                        op1.get_name().c_str());
                        assert((dim0 >= 0) && "Dimension not found");
                        cst = isl_constraint_set_coefficient_si(cst, isl_dim_in,
                                                                dim0, -1);
                        DEBUG(3, coli::str_dump(
                         "Assigning -1 to the input coefficient of dimension " +
                         std::to_string(dim0)));
                    }
                    if (op0.get_expr_type() == coli::e_val)
                    {
                        cst = isl_constraint_set_constant_si(cst, (-1)*op0.get_int_val());
                        DEBUG(3, coli::str_dump(
                         "Assigning (-1)*op0.get_int_val() to the cst dimension "));
                    }
                    if (op1.get_expr_type() == coli::e_val)
                    {
                        cst = isl_constraint_set_constant_si(cst, (-1)*op1.get_int_val());
                        DEBUG(3, coli::str_dump(
                         "Assigning (-1)*op1.get_int_val() to the cst dimension "));
                    }
                }
                else if (access.get_op_type() == coli::o_sub)
                {
                    coli::expr op0 = access.get_operand(0);
                    coli::expr op1 = access.get_operand(1);

                    if (op0.get_expr_type() == coli::e_id)
                    {
                        int dim0 = isl_space_find_dim_by_name(
                                        isl_map_get_space(identity),
                                        isl_dim_in,
                                        op0.get_name().c_str());
                         DEBUG(3, coli::str_dump("Searching for " + op0.get_name() + " in the range of ");
                                  coli::str_dump(isl_space_to_str(isl_map_get_space(identity))));
                        assert((dim0 >= 0) && "Dimension not found");
                        cst = isl_constraint_set_coefficient_si(cst, isl_dim_in,
                                                                dim0, -1);
                        DEBUG(3, coli::str_dump(
                         "Assigning -1 to the input coefficient of dimension " +
                         std::to_string(dim0)));
                    }
                    if (op1.get_expr_type() == coli::e_id)
                    {
                        int dim0 = isl_space_find_dim_by_name(
                                        isl_map_get_space(identity),
                                        isl_dim_in,
                                        op1.get_name().c_str());
                        assert((dim0 >= 0) && "Dimension not found");
                        cst = isl_constraint_set_coefficient_si(cst, isl_dim_in,
                                                                dim0, -1);
                        DEBUG(3, coli::str_dump(
                         "Assigning -1 to the input coefficient of dimension " +
                         std::to_string(dim0)));
                    }
                    if (op0.get_expr_type() == coli::e_val)
                    {
                        cst = isl_constraint_set_constant_si(cst, op0.get_int_val());
                        DEBUG(3, coli::str_dump(
                         "Assigning (-1)*op0.get_int_val() to the cst dimension "));
                    }
                    if (op1.get_expr_type() == coli::e_val)
                    {
                        cst = isl_constraint_set_constant_si(cst, op1.get_int_val());
                        DEBUG(3, coli::str_dump(
                         "Assigning (-1)*op1.get_int_val() to the cst dimension "));
                    }
                }
                else
                {
                    coli::error("Currently only Add and Sub operations for accesses are supported." , true);
                }
            }
            dimension_number++;
            identity = isl_map_add_constraint(identity, cst);
            DEBUG(3, coli::str_dump("After adding a constraint:",
                                          isl_map_to_str(identity)));
        }

        DEBUG(3, coli::str_dump("Access function:",
                                isl_map_to_str(access_function)));
        DEBUG(3, coli::str_dump("Transformation function after adding constraints:",
                                isl_map_to_str(identity)));

        access_function = isl_map_apply_range(isl_map_copy(identity),access_function);
        DEBUG(3, coli::str_dump(
                "Applying access function on the range of transformation function:",
                isl_map_to_str(access_function)));

        if (global::is_auto_data_mapping_set() == true)
        {
            DEBUG(3, coli::str_dump("Apply the schedule on the domain of the access function. Access functions:", isl_map_to_str(access_function)));
            DEBUG(3, coli::str_dump("Schedule:", isl_map_to_str(comp->get_schedule())));
            access_function = isl_map_apply_domain(access_function,isl_map_copy(comp->get_schedule()));
            DEBUG(3, coli::str_dump("Result: ", isl_map_to_str(access_function)));
        }
        accesses.push_back(access_function);
    }
    else if (exp.get_expr_type() == coli::e_op)
    {
            DEBUG(3, coli::str_dump("Extracting access from e_op."));

            switch(exp.get_op_type())
            {
                case coli::o_logical_and:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_logical_or:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_max:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_min:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_minus:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    break;
                case coli::o_add:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_sub:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_mul:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_div:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_mod:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_cond:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(2), accesses);
                    break;
                case coli::o_le:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_lt:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_ge:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_gt:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_not:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    break;
                case coli::o_eq:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_ne:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_right_shift:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_left_shift:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(1), accesses);
                    break;
                case coli::o_floor:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    break;
                case coli::o_cast:
                    traverse_expr_and_extract_accesses(fct, comp, exp.get_operand(0), accesses);
                    break;
                default:
                    coli::error("Extracting access function from an unsupported coli expression.", 1);
            }
        }

    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(3);
}

/**
 * Compute the accesses of the RHS of the computation
 * \p comp and store them in the accesses vector.
 */
void get_rhs_accesses(coli::function *func, coli::computation *comp, std::vector<isl_map *> &accesses)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    const coli::expr &rhs = comp->get_expr();
    traverse_expr_and_extract_accesses(func, comp, rhs, accesses);

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

    coli::function *func = (coli::function *) user;

    // Find the name of the computation associated to this AST leaf node.
    coli::computation *comp = get_computation_by_node(func, node);
    assert((comp != NULL) && "Computation not found!");;

    DEBUG(3, coli::str_dump("Computation:", comp->get_name().c_str()));

    // Get the accesses of the computation.  The first access is the access
    // for the LHS.  The following accesses are for the RHS.
    std::vector<isl_map *> accesses;
    isl_map *access = comp->get_transformed_access();
    accesses.push_back(access);
    // Add the accesses of the RHS to the accesses vector
    get_rhs_accesses(func, comp, accesses);

    // For each access in accesses (i.e. for each access in the computation),
    // compute the corresponding isl_ast expression.
    for (auto &access: accesses)
    {
        if (access != NULL)
        {
            DEBUG(3, coli::str_dump("Creating an isl_ast_index_expression for the access (isl_map *):", isl_map_to_str(access)));
            comp->get_index_expr().push_back(create_isl_ast_index_expression(build, access));
        }
        else
        {
            if (!comp->is_let_stmt()) // If this is not let stmt,
                                      // it should have an access function.
                coli::error("An access function should be provided before generating code.", true);
        }
    }

    for (const auto &i_expr : comp->get_index_expr())
    {
        DEBUG(3, coli::str_dump("Generated Index expression:", (const char *)
                                isl_ast_expr_to_C_str(i_expr)));
    }
    DEBUG(3, coli::str_dump("\n\n"));
    DEBUG_INDENT(-4);
    DEBUG_FCT_NAME(3);

    return node;
}

void print_isl_ast_expr_vector(
          const std::vector<isl_ast_expr*>& index_expr_cp)
{
    DEBUG(3, coli::str_dump ("List of index expressions."));
    for (auto& i_expr : index_expr_cp)
        DEBUG_NO_NEWLINE(3, coli::str_dump (" ", (const char * ) isl_ast_expr_to_C_str (i_expr)));
    DEBUG(3, coli::str_dump (" "));
}

Halide::Expr halide_expr_from_coli_expr(coli::computation *comp,
                                        std::vector<isl_ast_expr *> &index_expr,
                                        const coli::expr &coli_expr)
{
    Halide::Expr result;

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if (coli_expr.get_expr_type() == coli::e_val)
    {
        DEBUG(3, coli::str_dump("coli expression of type coli::e_val"));
        if (coli_expr.get_data_type() == coli::p_uint8)
            result = Halide::Expr(coli_expr.get_uint8_value());
        else if (coli_expr.get_data_type() == coli::p_int8)
            result = Halide::Expr(coli_expr.get_int8_value());
        else if (coli_expr.get_data_type() == coli::p_uint16)
            result = Halide::Expr(coli_expr.get_uint16_value());
        else if (coli_expr.get_data_type() == coli::p_int16)
            result = Halide::Expr(coli_expr.get_int16_value());
        else if (coli_expr.get_data_type() == coli::p_uint32)
            result = Halide::Expr(coli_expr.get_uint32_value());
        else if (coli_expr.get_data_type() == coli::p_int32)
            result = Halide::Expr(coli_expr.get_int32_value());
        else if (coli_expr.get_data_type() == coli::p_uint64)
            result = Halide::Expr(coli_expr.get_uint64_value());
        else if (coli_expr.get_data_type() == coli::p_int64)
            result = Halide::Expr(coli_expr.get_int64_value());
        else if (coli_expr.get_data_type() == coli::p_float32)
            result = Halide::Expr(coli_expr.get_float32_value());
        else if (coli_expr.get_data_type() == coli::p_float64)
            result = Halide::Expr(coli_expr.get_float64_value());
    }
    else if (coli_expr.get_expr_type() == coli::e_op)
    {
        Halide::Expr op0, op1, op2;

        DEBUG(3, coli::str_dump("coli expression of type coli::e_op"));

        op0 = halide_expr_from_coli_expr(comp, index_expr, coli_expr.get_operand(0));

        if (coli_expr.get_n_arg() > 1)
            op1 = halide_expr_from_coli_expr(comp, index_expr, coli_expr.get_operand(1));

        if (coli_expr.get_n_arg() > 2)
            op2 = halide_expr_from_coli_expr(comp, index_expr, coli_expr.get_operand(2));

        switch(coli_expr.get_op_type())
        {
            case coli::o_logical_and:
                result = Halide::Internal::And::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_logical_and"));
                break;
            case coli::o_logical_or:
                result = Halide::Internal::Or::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_logical_or"));
                break;
            case coli::o_max:
                result = Halide::Internal::Max::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_max"));
                break;
            case coli::o_min:
                result = Halide::Internal::Min::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_min"));
                break;
            case coli::o_minus:
                result = Halide::Internal::Sub::make(Halide::Expr(0), op0);
                DEBUG(3, coli::str_dump("op type: o_minus"));
                break;
            case coli::o_add:
                result = Halide::Internal::Add::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_add"));
                break;
            case coli::o_sub:
                result = Halide::Internal::Sub::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_sub"));
                break;
            case coli::o_mul:
                result = Halide::Internal::Mul::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_mul"));
                break;
            case coli::o_div:
                result = Halide::Internal::Div::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_div"));
                break;
            case coli::o_mod:
                result = Halide::Internal::Mod::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_mod"));
                break;
            case coli::o_cond:
                result = Halide::Internal::Select::make(op0, op1, op2);
                DEBUG(3, coli::str_dump("op type: o_cond"));
                break;
            case coli::o_le:
                result = Halide::Internal::LE::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_le"));
                break;
            case coli::o_lt:
                result = Halide::Internal::LT::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_lt"));
                break;
            case coli::o_ge:
                result = Halide::Internal::GE::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_ge"));
                break;
            case coli::o_gt:
                result = Halide::Internal::GT::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_gt"));
                break;
            case coli::o_not:
                result = Halide::Internal::Not::make(op0);
                DEBUG(3, coli::str_dump("op type: o_not"));
                break;
            case coli::o_eq:
                result = Halide::Internal::EQ::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_eq"));
                break;
            case coli::o_ne:
                result = Halide::Internal::NE::make(op0, op1);
                DEBUG(3, coli::str_dump("op type: o_ne"));
                break;
            case coli::o_access:
            {
                DEBUG(3, coli::str_dump("op type: o_access"));
                const char *access_comp_name = coli_expr.get_operand(0).get_name().c_str();
                DEBUG(3, coli::str_dump("Computation being accessed: ");coli::str_dump(access_comp_name));
                coli::computation *access_comp = comp->get_function()->get_computation_by_name(access_comp_name);
                const char *buffer_name = isl_space_get_tuple_name(
                                            isl_map_get_space(access_comp->get_transformed_access()), isl_dim_out);
                DEBUG(3, coli::str_dump("Name of the associated buffer: ");coli::str_dump(buffer_name));
                assert(buffer_name != NULL);

                auto buffer_entry = comp->get_function()->get_buffers_list().find(buffer_name);
                assert(buffer_entry != comp->get_function()->get_buffers_list().end());

                auto coli_buffer = buffer_entry->second;

                // COLi buffer is from outermost to innermost, whereas Halide buffer is from innermost
                // to outermost; thus, we need to reverse the order
                halide_dimension_t shape[coli_buffer->get_dim_sizes().size()];
                int stride = 1;

                for (int i = 0; i < coli_buffer->get_dim_sizes().size(); i++) {
                    shape[i].min = 0;
                    shape[i].extent = (int) coli_buffer->get_dim_sizes()[coli_buffer->get_dim_sizes().size()- i - 1].get_int_val();
                    shape[i].stride = stride;
                    stride *= (int) coli_buffer->get_dim_sizes()[coli_buffer->get_dim_sizes().size()- i - 1].get_int_val();
                }

                Halide::Internal::BufferPtr *buffer =
                    new Halide::Internal::BufferPtr(
                            Halide::Image<>(halide_type_from_coli_type(coli_buffer->get_type()),
                                            coli_buffer->get_data(),
                                            coli_buffer->get_dim_sizes().size(),
                                            shape),
                            coli_buffer->get_name());

                print_isl_ast_expr_vector(index_expr);

                Halide::Expr index = coli::linearize_access(buffer, index_expr[0]);
                index_expr.erase(index_expr.begin());

                Halide::Internal::Parameter param(
                    buffer->type(), true, buffer->dimensions(), buffer->name());
                param.set_buffer(*buffer);

                result = Halide::Internal::Load::make(
                            halide_type_from_coli_type(coli_buffer->get_type()),
                                                       coli_buffer->get_name(),
                                                       index, *buffer, param);
                }
                break;
            case coli::o_right_shift:
                result = op0 >> op1;
                DEBUG(3, coli::str_dump("op type: o_right_shift"));
                break;
            case coli::o_left_shift:
                result = op0 << op1;
                DEBUG(3, coli::str_dump("op type: o_left_shift"));
                break;
            case coli::o_floor:
                result = Halide::floor(op0);
                DEBUG(3, coli::str_dump("op type: o_floor"));
                break;
            case coli::o_cast:
                result = Halide::cast(halide_type_from_coli_type(coli_expr.get_data_type()), op0);
                DEBUG(3, coli::str_dump("op type: o_cast"));
                break;
            default:
                coli::error("Translating an unsupported ISL expression into a Halide expression.", 1);
        }
    }
    else if (coli_expr.get_expr_type() == coli::e_var)
        result = Halide::Internal::Variable::make(
            halide_type_from_coli_type(coli_expr.get_data_type()),
            coli_expr.get_name());
    else if (coli_expr.get_expr_type() != coli::e_id)// Do not signal an error for expressions of type coli::e_id
    {
        coli::str_dump("coli type of expr: ", str_from_coli_type_expr(coli_expr.get_expr_type()).c_str());
        coli::error("\nTranslating an unsupported ISL expression in a Halide expression.", 1);
    }

    if (result.defined() == true)
        DEBUG(10, coli::str_dump("Generated stmt: "); std::cout << result);

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
                coli::error("isl_ast_op_and_then operator found in the AST. This operator is not well supported.", 0);
                break;
            case isl_ast_op_or:
                result = Halide::Internal::Or::make(op0, op1);
                break;
            case isl_ast_op_or_else:
                result = Halide::Internal::Or::make(op0, op1);
                coli::error("isl_ast_op_or_then operator found in the AST. This operator is not well supported.", 0);
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
                coli::str_dump("Transforming the following expression",
                        (const char *) isl_ast_expr_to_C_str(isl_expr));
                coli::str_dump("\n");
                coli::error("Translating an unsupported ISL expression in a Halide expression.", 1);
        }
    }
    else
    {
        coli::str_dump("Transforming the following expression",
                (const char *) isl_ast_expr_to_C_str(isl_expr));
        coli::str_dump("\n");
        coli::error("Translating an unsupported ISL expression in a Halide expression.", 1);
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
    coli::function fct, isl_ast_node *node,
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
        DEBUG(3, coli::str_dump("Generating code for a block"));

        isl_ast_node_list *list = isl_ast_node_block_get_children(node);
        isl_ast_node *child1;

        for (i=isl_ast_node_list_n_ast_node(list)-1; i>=0; i--)
        {
            child1 = isl_ast_node_list_get_ast_node(list, i);

            DEBUG(3, coli::str_dump("Generating block."));

            Halide::Internal::Stmt *block1 =
                coli::halide_stmt_from_isl_node(fct, child1, level, tagged_stmts);

            DEBUG_NO_NEWLINE(10, coli::str_dump("Generated block: "); std::cout << *block1);

            if (block1->defined() == false) // Probably block1 is a let stmt.
            {
                if (let_stmts_vector.empty() == false) // i.e. non-consumed let statements
                {
                    if (result->defined() == true) // if some stmts have already been created
                                                   // in this loop we can generate letStmt
                    {
                        for (auto l_stmt: let_stmts_vector)
                        {
                            DEBUG(3, coli::str_dump("Generating the following let statement."));
                            DEBUG(3, coli::str_dump("Name : " + l_stmt.first));
                            DEBUG(3, coli::str_dump("Expression of the let statement: ");
                                     std::cout << l_stmt.second);

                            *result = Halide::Internal::LetStmt::make(
                                                l_stmt.first,
                                                l_stmt.second,
                                                *result);

                            DEBUG(10, coli::str_dump("Generated let stmt:"));
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
        DEBUG(3, coli::str_dump("Generating code for Halide::For"));

        isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
        std::string iterator_str = std::string(isl_ast_expr_to_C_str(iter));

        isl_ast_expr *init = isl_ast_node_for_get_init(node);
        isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
        isl_ast_expr *inc  = isl_ast_node_for_get_inc(node);

        if (!isl_val_is_one(isl_ast_expr_get_val(inc)))
            coli::error("The increment in one of the loops is not +1."
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
            coli::error("The for loop upper bound is not an isl_est_expr of type le or lt" ,1);

        assert(cond_upper_bound_isl_format != NULL);
        DEBUG(3, coli::str_dump("Creating for loop init expression."));

		Halide::Expr init_expr = halide_expr_from_isl_ast_expr(init);
        if (init_expr.type() != halide_type_from_coli_type(global::get_loop_iterator_default_data_type()))
            init_expr = Halide::Internal::Cast::make(halide_type_from_coli_type(global::get_loop_iterator_default_data_type()), init_expr);
        DEBUG(3, coli::str_dump("init expression: "); std::cout << init_expr);
		Halide::Expr cond_upper_bound_halide_format =
		        halide_expr_from_isl_ast_expr(cond_upper_bound_isl_format);
		cond_upper_bound_halide_format = simplify(cond_upper_bound_halide_format);
		if (cond_upper_bound_halide_format.type() != halide_type_from_coli_type(global::get_loop_iterator_default_data_type()))
		    cond_upper_bound_halide_format =
		        Halide::Internal::Cast::make(halide_type_from_coli_type(global::get_loop_iterator_default_data_type()), cond_upper_bound_halide_format);
        DEBUG(3, coli::str_dump("Upper bound expression: "); std::cout << cond_upper_bound_halide_format);
        Halide::Internal::Stmt *halide_body = coli::halide_stmt_from_isl_node(fct, body, level+1, tagged_stmts);
        Halide::Internal::ForType fortype = Halide::Internal::ForType::Serial;
        Halide::DeviceAPI dev_api = Halide::DeviceAPI::Host;

        // Change the type from Serial to parallel or vector if the
        // current level was marked as such.
        for (int tt = 0; tt < tagged_stmts.size(); tt++)
        {
            if (fct.should_parallelize(tagged_stmts[tt], level))
            {
                fortype = Halide::Internal::ForType::Parallel;
                //tagged_stmts.erase(tagged_stmts.begin() + tt);
            }
            else if (fct.should_vectorize(tagged_stmts[tt], level))
            {
                DEBUG(3, coli::str_dump("Trying to vectorize at level "); coli::str_dump(std::to_string(level)));

                const Halide::Internal::IntImm *extent = cond_upper_bound_halide_format.as<Halide::Internal::IntImm>();
                if (extent) {
                    fortype = Halide::Internal::ForType::Vectorized;
                    //tagged_stmts.erase(tagged_stmts.begin() + tt);
                    DEBUG(3, coli::str_dump("Loop vectorized"));
                }
                else
                {
                    DEBUG(3, coli::str_dump("Loop not vectorized (extent is non constant)"));
		    // Currently we can only print Halide expressions using
		    // "std::cout << ".
                    DEBUG(3, std::cout << cond_upper_bound_halide_format << std::endl);
                }
            }
            else if (fct.should_map_to_gpu(tagged_stmts[tt], level))
            {
                    fortype = Halide::Internal::ForType::Parallel;
                    dev_api = Halide::DeviceAPI::OpenCL;
                    std::string gpu_iter = fct.get_gpu_iterator(
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
                        DEBUG(3, coli::str_dump("Loop over " + gpu_iter +
                             " created.\n"));
                        //tagged_stmts.erase(tagged_stmts.begin() + tt);
            }
        }

        DEBUG(3, coli::str_dump("Creating the for loop."));
        *result = Halide::Internal::For::make(iterator_str, init_expr, cond_upper_bound_halide_format, fortype,
                dev_api, *halide_body);
        DEBUG(3, coli::str_dump("For loop created."));
        DEBUG(10, std::cout<< *result);
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_user)
    {
        DEBUG(3, coli::str_dump("Generating code for user node"));

        isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
        isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
        isl_id *id = isl_ast_expr_get_id(arg);
        isl_ast_expr_free(arg);
        std::string computation_name(isl_id_get_name(id));
        DEBUG(3, coli::str_dump("Computation name: ");coli::str_dump(computation_name));
        isl_id_free(id);

        // Check if any loop around this statement should be
        // parallelized, vectorized or mapped to GPU.
        for (int l = 0; l < level; l++)
        {
            if (fct.should_parallelize(computation_name, l) ||
                fct.should_vectorize(computation_name, l) ||
                fct.should_map_to_gpu(computation_name, l))
            tagged_stmts.push_back(computation_name);
        }

        coli::computation *comp = fct.get_computation_by_name(computation_name);
        DEBUG(10, comp->dump());

        comp->create_halide_stmt();

        *result = comp->get_halide_stmt();
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_if)
    {
        DEBUG(3, coli::str_dump("Generating code for conditional"));

        isl_ast_expr *cond = isl_ast_node_if_get_cond(node);
        isl_ast_node *if_stmt = isl_ast_node_if_get_then(node);
        isl_ast_node *else_stmt = isl_ast_node_if_get_else(node);

        Halide::Expr c = halide_expr_from_isl_ast_expr(cond);

        DEBUG(3, coli::str_dump("Condition: "); std::cout << c);

        Halide::Internal::Stmt *if_s =
                coli::halide_stmt_from_isl_node(fct, if_stmt,
                                                level, tagged_stmts);

        Halide::Internal::Stmt else_s;

        if (else_stmt != NULL)
        {
            Halide::Internal::Stmt else_s =
                 *coli::halide_stmt_from_isl_node(fct, else_stmt,
                                                  level, tagged_stmts);
        }
        else
            DEBUG(3, coli::str_dump("Else statement is NULL."));

        *result = Halide::Internal::IfThenElse::make(
                    c,
                    *if_s,
                    else_s);
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
        const coli::buffer *buf = b.second;
        // Allocate only arrays that are not passed to the function as arguments.
        if (buf->is_argument() == false)
        {
            *stmt = Halide::Internal::Block::make(Halide::Internal::Free::make(buf->get_name()), *stmt);
        }
    }*/

    // Generate the statement that represents the whole function
    stmt = coli::halide_stmt_from_isl_node(*this, this->get_isl_ast(), 0, generated_stmts);

    // Allocate buffers that are not passed as an argument to the function
    for (const auto &b : this->get_buffers_list())
    {
        const coli::buffer *buf = b.second;
        // Allocate only arrays that are not passed to the function as arguments.
        if (buf->get_argument_type() == coli::a_temporary)
        {
            std::vector<Halide::Expr> halide_dim_sizes;
            // Create a vector indicating the size that should be allocated.
            // COLi buffer is defined from outermost to innermost, whereas Halide is from
            // innermost to outermost; thus, we need to reverse the order.
            for (int i = buf->get_dim_sizes().size() - 1; i >= 0; --i)
            {
                // TODO: if the size of an array is a computation access
                // this is not supported yet. Mainly because in the code below
                // we pass NULL pointers for parameters that are necessary
                // in case we are computing the halide expression from a coli expression
                // that represents a computation access.
                const auto &sz = buf->get_dim_sizes()[i];
                std::vector<isl_ast_expr *> ie = {};
                halide_dim_sizes.push_back(halide_expr_from_coli_expr(NULL, ie, sz));
            }
            *stmt = Halide::Internal::Allocate::make(
                        buf->get_name(),
                        halide_type_from_coli_type(buf->get_type()),
                        halide_dim_sizes, Halide::Internal::const_true(), *stmt);
        }
    }

    // Generate the invariants of the function.
    for (const auto &param : this->get_invariants())
    {
        std::vector<isl_ast_expr *> ie = {};
        *stmt = Halide::Internal::LetStmt::make(
                    param.get_name(),
                    halide_expr_from_coli_expr(NULL, ie, param.get_expr()),
                    *stmt);
    }

    this->halide_stmt = stmt;

    DEBUG(3, coli::str_dump("\n\nGenerated Halide stmt before lowering:"));
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
Halide::Expr linearize_access(Halide::Internal::BufferPtr *buffer,
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
void computation::create_halide_stmt()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, coli::str_dump("Generating stmt for assignment."));

    if (this->is_let_stmt())
    {
        DEBUG(3, coli::str_dump("This is a let statement."));
        DEBUG(10, coli::str_dump("The expression associated with the let statement."));
        DEBUG(10, this->expression.dump(true));

        Halide::Expr result = halide_expr_from_coli_expr(this,
                                                         this->get_index_expr(),
                                                         this->expression);

        Halide::Type l_type = halide_type_from_coli_type(this->get_data_type());

        if (l_type != result.type())
            result = Halide::Internal::Cast::make(l_type, result);

        std::string let_stmt_name = this->get_name();
        int pos = let_stmt_name.find(LET_STMT_PREFIX);
        // if LET_STMT_PREFIX is found and is in the beginning of
        // let_stmt_name.
        if ((pos != std::string::npos) && (pos == 0))
            let_stmt_name = let_stmt_name.erase(0, std::strlen(LET_STMT_PREFIX));
        else
            coli::error(LET_STMT_PREFIX " not found in let statement name.", true);

        let_stmts_vector.push_back(std::pair<std::string, Halide::Expr>(
                                            let_stmt_name,
                                            result));
        DEBUG(10, coli::str_dump("A let statement was added to the vector of let statements."));

    }
    else
    {
        DEBUG(3, coli::str_dump("This is not a let statement."));

        const char *buffer_name = isl_space_get_tuple_name(
                              isl_map_get_space(this->get_transformed_access()), isl_dim_out);
        assert(buffer_name != NULL);

        isl_map *access = this->get_transformed_access();
        isl_space *space = isl_map_get_space(access);
        // Get the number of dimensions of the ISL map representing
        // the access.
        int access_dims = isl_space_dim(space, isl_dim_out);

        // Fetch the actual buffer.
        auto buffer_entry = this->function->get_buffers_list().find(buffer_name);
        assert(buffer_entry != this->function->get_buffers_list().end());

        auto coli_buffer = buffer_entry->second;

        DEBUG(10, coli_buffer->dump(true));

        // COLi buffer is from outermost to innermost, whereas Halide buffer is
        // from innermost to outermost; thus, we need to reverse the order
        halide_dimension_t shape[coli_buffer->get_dim_sizes().size()];
        int stride = 1;
        for (int i = 0; i < coli_buffer->get_dim_sizes().size(); i++) {
            shape[i].min = 0;
            shape[i].extent = (int) coli_buffer->get_dim_sizes()[coli_buffer->get_dim_sizes().size() - i - 1].get_int_val();
            shape[i].stride = stride;
            stride *= (int) coli_buffer->get_dim_sizes()[coli_buffer->get_dim_sizes().size() - i - 1].get_int_val();
        }

        Halide::Internal::BufferPtr *buffer =
              new Halide::Internal::BufferPtr(
                              Halide::Image<>(halide_type_from_coli_type(coli_buffer->get_type()),
                                                              coli_buffer->get_data(),
                                                              coli_buffer->get_dim_sizes().size(),
                                                              shape),
                              coli_buffer->get_name());

        int buf_dims = buffer->dimensions();

        // The number of dimensions in the Halide buffer should be equal to
        // the number of dimensions of the access function.
        assert(buf_dims == access_dims);

        auto index_expr = this->index_expr[0];
        assert(index_expr != NULL);

        Halide::Expr index = coli::linearize_access(buffer, index_expr);

        Halide::Internal::Parameter param(
              buffer->type(), true, buffer->dimensions(), buffer->name());
        param.set_buffer(*buffer);

        std::vector<isl_ast_expr *> index_expr_cp = this->index_expr;
        index_expr_cp.erase(index_expr_cp.begin());

        print_isl_ast_expr_vector(index_expr_cp);

        this->stmt = Halide::Internal::Store::make (
                        buffer_name,
                        halide_expr_from_coli_expr(this, index_expr_cp,
                                                   this->expression), index, param);
    }

    DEBUG_NO_NEWLINE(3, coli::str_dump("End of create_halide_stmt. Generated statement is: ");
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
            halide_argtype_from_coli_argtype(buf->get_argument_type()),
            halide_type_from_coli_type(buf->get_type()),
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
