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

Halide::Argument::Kind coli_argtype_to_halide_argtype(coli::type::argument type);
Halide::Expr linearize_access(Halide::Internal::BufferPtr *buffer, isl_ast_expr *index_expr);

computation *function::get_computation_by_name(std::string name) const
{
	coli::computation *res_comp = NULL;

	for (const auto &comp : this->get_computations())
	{
		if (name == comp->get_name())
		{
			res_comp = comp;
		}
	}

	assert((res_comp != NULL) && "Computation not found");
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


isl_ast_expr* create_isl_ast_index_expression(isl_ast_build* build,
		isl_map* access)
{
	DEBUG_FCT_NAME(3);
	DEBUG_INDENT(4);

	isl_map *schedule = isl_map_from_union_map(isl_ast_build_get_schedule(build));
	DEBUG(3, coli::str_dump("Schedule:", isl_map_to_str(schedule)));
	schedule = isl_map_set_tuple_name(schedule, isl_dim_in, isl_map_get_tuple_name(access, isl_dim_in));
	DEBUG(3, coli::str_dump("After renaming the Schedule:", isl_map_to_str(schedule)));
	isl_map* map = isl_map_reverse(isl_map_copy(schedule));
	DEBUG(3, coli::str_dump("Schedule reversed:", isl_map_to_str(map)));
	isl_pw_multi_aff* iterator_map = isl_pw_multi_aff_from_map(map);
	DEBUG_NO_NEWLINE(3, coli::str_dump("The iterator map of an AST leaf (after scheduling):"));
	DEBUG_NO_NEWLINE(3, isl_pw_multi_aff_dump(iterator_map));
	DEBUG(3, coli::str_dump("Access:", isl_map_to_str(access)));
	isl_pw_multi_aff* index_aff = isl_pw_multi_aff_from_map(isl_map_copy(access));
	DEBUG_NO_NEWLINE(3, coli::str_dump("isl_pw_multi_aff_from_map(access):"));
	DEBUG_NO_NEWLINE(3, isl_pw_multi_aff_dump(index_aff));
	iterator_map = isl_pw_multi_aff_pullback_pw_multi_aff(index_aff, iterator_map);
	DEBUG_NO_NEWLINE(3, coli::str_dump("isl_pw_multi_aff_pullback_pw_multi_aff(index_aff,iterator_map):"));
	DEBUG_NO_NEWLINE(3, isl_pw_multi_aff_dump(iterator_map));
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
void traverse_expr_and_extract_accesses(coli::function *fct, const coli::expr *exp, std::vector<isl_map *> &accesses)
{
	assert(exp != NULL);
	assert(fct != NULL);

	DEBUG_FCT_NAME(3);
	DEBUG_INDENT(4);

	if ((exp->get_expr_type() == coli::type::expr::op) && (exp->get_op_type() == coli::type::op::access))
	{
		// Create the access map for this access node.
		coli::expr *id = exp->get_operand(0);

		// Get the corresponding computation
		coli::computation *comp = fct->get_computation_by_name(id->get_id_name());

		isl_map *access_function = isl_map_copy(comp->get_access());

		isl_set *domain = isl_set_universe(
								isl_space_domain(
									isl_map_get_space(
											isl_map_copy(access_function))));
		isl_map *identity = isl_map_identity(
								isl_space_map_from_set(
										isl_set_get_space(domain)));

		identity = isl_map_universe(isl_map_get_space(identity));


		int dim = 0;
		for (const auto &access: exp->get_access())
		{
			isl_local_space *ls = isl_local_space_from_space(
										isl_map_get_space(
												isl_map_copy(identity)));
			isl_constraint *cst = isl_constraint_alloc_equality(
										isl_local_space_copy(ls));

			cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, dim, 1);

			if (access->get_expr_type() == coli::type::expr::val)
			{
				if (access->get_data_type() == coli::type::primitive::int32)
				{
					cst = isl_constraint_set_constant_si(cst, (-1)*access->get_int32_value());
				}
				else
					coli::error("Access values can only be of type coli::type::primitive::int32" , true);
			}
			else if (access->get_expr_type() == coli::type::expr::id)
			{
				int dim0 = isl_space_find_dim_by_name(
								isl_map_get_space(access_function),
								isl_dim_in,
								access->get_id_name().c_str());
				cst = isl_constraint_set_coefficient_si(cst, isl_dim_in,
														dim0, -1);
			}
			else if (access->get_expr_type() == coli::type::expr::op)
			{
				if (access->get_op_type() == coli::type::op::add)
				{
					coli::expr *op0 = access->get_operand(0);
					coli::expr *op1 = access->get_operand(1);

					assert(op0 != NULL);
					assert(op1 != NULL);

					if (op0->get_expr_type() == coli::type::expr::id)
					{
						int dim0 = isl_space_find_dim_by_name(
										isl_map_get_space(access_function),
				                		isl_dim_in,
										op0->get_id_name().c_str());
						cst = isl_constraint_set_coefficient_si(cst, isl_dim_in,
																dim0, -1);
					}
					if (op1->get_expr_type() == coli::type::expr::id)
					{
						int dim0 = isl_space_find_dim_by_name(
										isl_map_get_space(access_function),
										isl_dim_in,
										op1->get_id_name().c_str());
						cst = isl_constraint_set_coefficient_si(cst, isl_dim_in,
																dim0, -1);
					}
					if (op0->get_expr_type() == coli::type::expr::val)
					{
						if (op0->get_data_type() == coli::type::primitive::int32)
						{
							cst = isl_constraint_set_constant_si(cst, (-1)*op0->get_int32_value());
						}
						else
							coli::error("Access values can only be of type coli::type::primitive::int32" , true);
					}
					if (op1->get_expr_type() == coli::type::expr::val)
					{
						if (op1->get_data_type() == coli::type::primitive::int32)
						{
							cst = isl_constraint_set_constant_si(cst, (-1)*op1->get_int32_value());
						}
						else
							coli::error("Access values can only be of type coli::type::primitive::int32" , true);
					}
					DEBUG(3, coli::str_dump("\n"));

				}
				else if (access->get_op_type() == coli::type::op::sub)
				{
					coli::expr *op0 = access->get_operand(0);
					coli::expr *op1 = access->get_operand(1);

					assert(op0 != NULL);
					assert(op1 != NULL);

					if (op0->get_expr_type() == coli::type::expr::id)
					{
						int dim0 = isl_space_find_dim_by_name(
										isl_map_get_space(access_function),
				                		isl_dim_in,
										op0->get_id_name().c_str());
						cst = isl_constraint_set_coefficient_si(cst, isl_dim_in,
																dim0, -1);
					}
					if (op1->get_expr_type() == coli::type::expr::id)
					{
						int dim0 = isl_space_find_dim_by_name(
										isl_map_get_space(access_function),
										isl_dim_in,
										op1->get_id_name().c_str());
						cst = isl_constraint_set_coefficient_si(cst, isl_dim_in,
																dim0, -1);
					}
					if (op0->get_expr_type() == coli::type::expr::val)
					{
						if (op0->get_data_type() == coli::type::primitive::int32)
						{
							cst = isl_constraint_set_constant_si(cst, op0->get_int32_value());
						}
						else
							coli::error("Access values can only be of type coli::type::primitive::int32" , true);
					}
					if (op1->get_expr_type() == coli::type::expr::val)
					{
						if (op1->get_data_type() == coli::type::primitive::int32)
						{
							cst = isl_constraint_set_constant_si(cst, op1->get_int32_value());
						}
						else
							coli::error("Access values can only be of type coli::type::primitive::int32" , true);
					}
					DEBUG(3, coli::str_dump("\n"));
				}
				else
					coli::error("Currently only Add and Sub operations for accesses are supported." , true);
			}

			dim++;

			identity = isl_map_add_constraint(identity, cst);
		}
		access_function = isl_map_apply_domain(access_function, isl_map_copy(identity));
		DEBUG(3, coli::str_dump("Updated access function:", isl_map_to_str(access_function)));
		accesses.push_back(access_function);
	}
	else if (exp->get_expr_type() == coli::type::expr::op)
	{
			coli::expr *expr0 = NULL, *expr1 = NULL, *expr2 = NULL;
			expr0 = exp->get_operand(0);

			if (exp->get_n_arg() > 1)
				expr1 = exp->get_operand(1);

			if (exp->get_n_arg() > 2)
				expr2 = exp->get_operand(2);

			switch(exp->get_op_type())
			{
				case coli::type::op::logical_and:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::logical_or:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::max:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::min:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::minus:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					break;
				case coli::type::op::add:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::sub:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::mul:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::div:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::mod:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::cond:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					traverse_expr_and_extract_accesses(fct, expr2, accesses);
					break;
				case coli::type::op::le:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::lt:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::ge:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::gt:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				case coli::type::op::eq:
					traverse_expr_and_extract_accesses(fct, expr0, accesses);
					traverse_expr_and_extract_accesses(fct, expr1, accesses);
					break;
				default:
					coli::error("Extracting access function from an unsupported coli expression.", 1);
			}
		}

	DEBUG_INDENT(-4);
}

/**
 * Compute the accesses of the RHS of the computation
 * \p comp and store them in the accesses vector.
 */
void get_rhs_accesses(coli::function *func, coli::computation *comp, std::vector<isl_map *> &accesses)
{
	const coli::expr *rhs = comp->get_expr();
	traverse_expr_and_extract_accesses(func, rhs, accesses);
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
	accesses.push_back(comp->get_access());
	// Add the accesses of the RHS to the accesses vector
	get_rhs_accesses(func, comp, accesses);

	// For each access in accesses (i.e. for each access in the computation),
	// compute the corresponding isl_ast expression.
	for (const auto &access: accesses)
	{
		assert((access != NULL) && "An access function should be provided before generating code.");;

		DEBUG(3, coli::str_dump("Access (isl_map *):", isl_map_to_str(access)));
		DEBUG(3, coli::str_dump("\n"));

		// Compute the isl_ast index expression for the LHS
		comp->get_index_expr().push_back(create_isl_ast_index_expression(build, access));
	}

	for (const auto &i_expr : comp->get_index_expr())
	{
		DEBUG(3, coli::str_dump("Generated Index expression:", (const char *)
								isl_ast_expr_to_C_str(i_expr)));
	}
	DEBUG(3, coli::str_dump("\n\n"));
	DEBUG_INDENT(-4);

	return node;
}

Halide::Expr create_halide_expr_from_coli_expr(coli::computation *comp, std::vector<isl_ast_expr *> &index_expr, const coli::expr *coli_expr)
{
	Halide::Expr result;

	DEBUG_FCT_NAME(3);
	DEBUG_INDENT(4);

	if (coli_expr->get_expr_type() == coli::type::expr::val)
	{
		DEBUG(3, coli::str_dump("coli expression of type coli::type::expr::val"));
		if (coli_expr->get_data_type() == coli::type::primitive::uint8)
			result = Halide::Expr(coli_expr->get_uint8_value());
		else if (coli_expr->get_data_type() == coli::type::primitive::int8)
			result = Halide::Expr(coli_expr->get_int8_value());
		else if (coli_expr->get_data_type() == coli::type::primitive::uint32)
			result = Halide::Expr(coli_expr->get_uint32_value());
		else if (coli_expr->get_data_type() == coli::type::primitive::int32)
			result = Halide::Expr(coli_expr->get_int32_value());
		else if (coli_expr->get_data_type() == coli::type::primitive::uint64)
			result = Halide::Expr(coli_expr->get_uint64_value());
		else if (coli_expr->get_data_type() == coli::type::primitive::int64)
			result = Halide::Expr(coli_expr->get_int64_value());
	}
	else if (coli_expr->get_expr_type() == coli::type::expr::op)
	{
		Halide::Expr op0, op1, op2;

		DEBUG(3, coli::str_dump("coli expression of type coli::type::expr::op"));

		op0 = create_halide_expr_from_coli_expr(comp, index_expr, coli_expr->get_operand(0));

		if (coli_expr->get_n_arg() > 1)
			op1 = create_halide_expr_from_coli_expr(comp, index_expr, coli_expr->get_operand(1));

		if (coli_expr->get_n_arg() > 2)
			op2 = create_halide_expr_from_coli_expr(comp, index_expr, coli_expr->get_operand(2));

		switch(coli_expr->get_op_type())
		{
			case coli::type::op::logical_and:
				result = Halide::Internal::And::make(op0, op1);
				break;
			case coli::type::op::logical_or:
				result = Halide::Internal::Or::make(op0, op1);
				break;
			case coli::type::op::max:
				result = Halide::Internal::Max::make(op0, op1);
				break;
			case coli::type::op::min:
				result = Halide::Internal::Min::make(op0, op1);
				break;
			case coli::type::op::minus:
				result = Halide::Internal::Sub::make(Halide::Expr(0), op0);
				break;
			case coli::type::op::add:
				result = Halide::Internal::Add::make(op0, op1);
				break;
			case coli::type::op::sub:
				result = Halide::Internal::Sub::make(op0, op1);
				break;
			case coli::type::op::mul:
				result = Halide::Internal::Mul::make(op0, op1);
				break;
			case coli::type::op::div:
				result = Halide::Internal::Div::make(op0, op1);
				break;
			case coli::type::op::mod:
				result = Halide::Internal::Mod::make(op0, op1);
				break;
			case coli::type::op::cond:
				result = Halide::Internal::Select::make(op0, op1, op2);
				break;
			case coli::type::op::le:
				result = Halide::Internal::LE::make(op0, op1);
				break;
			case coli::type::op::lt:
				result = Halide::Internal::LT::make(op0, op1);
				break;
			case coli::type::op::ge:
				result = Halide::Internal::GE::make(op0, op1);
				break;
			case coli::type::op::gt:
				result = Halide::Internal::GT::make(op0, op1);
				break;
			case coli::type::op::eq:
				result = Halide::Internal::EQ::make(op0, op1);
				break;
			case coli::type::op::access:
			{
				const char *comp_name = coli_expr->get_operand(0)->get_id_name().c_str();
				coli::computation *rhs_comp = comp->get_function()->get_computation_by_name(comp_name);
				const char *buffer_name = isl_space_get_tuple_name(
											isl_map_get_space(rhs_comp->get_access()), isl_dim_out);
				assert(buffer_name != NULL);

				auto buffer_entry = comp->get_function()->get_buffers_list().find(buffer_name);
				assert(buffer_entry != comp->get_function()->get_buffers_list().end());

				auto coli_buffer = buffer_entry->second;

				halide_dimension_t shape[coli_buffer->get_dim_sizes().size()];
				int stride = 1;
				for (int i = 0; i < coli_buffer->get_dim_sizes().size(); i++) {
		           	shape[i].min = 0;
		           	shape[i].extent = coli_buffer->get_dim_sizes()[i];
		           	shape[i].stride = stride;
		           	stride *= coli_buffer->get_dim_sizes()[i];
			   	}

			   	Halide::Internal::BufferPtr *buffer =
				   	new Halide::Internal::BufferPtr(
		   					Halide::Image<>(coli_type_to_halide_type(coli_buffer->get_type()),
		   									coli_buffer->get_data(),
											coli_buffer->get_dim_sizes().size(),
											shape),
							coli_buffer->get_name());

				Halide::Expr index = coli::linearize_access(buffer,index_expr[0]);
				index_expr.erase(index_expr.begin());

				Halide::Internal::Parameter param(
					buffer->type(), true, buffer->dimensions(), buffer->name());
				param.set_buffer(*buffer);

				result = Halide::Internal::Load::make(
							coli_type_to_halide_type(coli_buffer->get_type()),
													 coli_buffer->get_name(),
													 index, *buffer, param);
				}
				break;
			default:
				coli::error("Translating an unsupported ISL expression into a Halide expression.", 1);
		}
	}
	else if (coli_expr->get_expr_type() != coli::type::expr::id) // Do not signal an error for expressions of type coli::type::expr::id
	{
		coli::str_dump("coli type of expr: ", coli_type_expr_to_str(coli_expr->get_expr_type()).c_str());
		coli::error("\nTranslating an unsupported ISL expression in a Halide expression.", 1);
	}

	DEBUG_INDENT(-4);

	return result;
}

Halide::Expr create_halide_expr_from_isl_ast_expr(isl_ast_expr *isl_expr)
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

		op0 = create_halide_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(isl_expr, 0));

		if (isl_ast_expr_get_op_n_arg(isl_expr) > 1)
			op1 = create_halide_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(isl_expr, 1));

		if (isl_ast_expr_get_op_n_arg(isl_expr) > 2)
			op2 = create_halide_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(isl_expr, 2));

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
				result = Halide::Internal::Cast::make(Halide::Int(32), Halide::floor(op0));
				break;
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

/**
  * Generate a Halide statement from an ISL ast node object in the ISL ast
  * tree.
  * Level represents the level of the node in the schedule.  0 means root.
  */
Halide::Internal::Stmt *generate_Halide_stmt_from_isl_node(
	coli::function fct, isl_ast_node *node,
	int level, std::vector<std::string> &generated_stmts)
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
		isl_ast_node *child, *child2;

		if (isl_ast_node_list_n_ast_node(list) >= 2)
		{
			child = isl_ast_node_list_get_ast_node(list, 0);
			child2 = isl_ast_node_list_get_ast_node(list, 1);

			*result = Halide::Internal::Block::make(
						*coli::generate_Halide_stmt_from_isl_node(fct, child, level+1, generated_stmts),
						*coli::generate_Halide_stmt_from_isl_node(fct, child2, level+1, generated_stmts));

			for (i = 2; i < isl_ast_node_list_n_ast_node(list); i++)
			{
				child = isl_ast_node_list_get_ast_node(list, i);
				*result = Halide::Internal::Block::make(*result, *coli::generate_Halide_stmt_from_isl_node(fct, child, level+1, generated_stmts));
			}
		}
		else
			// The above code expects the isl ast block to have at least two statemenets so that the
			// Halide::Internal::Block::make works (because that function expects its two inputs to be define).
			coli::error("Expecting the block to have at least 2 statements but it does not.", true);
	}
	else if (isl_ast_node_get_type(node) == isl_ast_node_for)
	{
		DEBUG(3, coli::str_dump("Generating code for Halide::For"));

		isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
		char *iterator_str = isl_ast_expr_to_C_str(iter);

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
		Halide::Expr init_expr = create_halide_expr_from_isl_ast_expr(init);
		Halide::Expr cond_upper_bound_halide_format =  create_halide_expr_from_isl_ast_expr(cond_upper_bound_isl_format);
		Halide::Internal::Stmt *halide_body = coli::generate_Halide_stmt_from_isl_node(fct, body, level+1, generated_stmts);
		Halide::Internal::ForType fortype = Halide::Internal::ForType::Serial;

		// Change the type from Serial to parallel or vector if the
		// current level was marked as such.
		for (const auto &generated_stmt: generated_stmts)
		{
			if (fct.should_parallelize(generated_stmt, level))
			{
				fortype = Halide::Internal::ForType::Parallel;
			}
			else if (fct.should_vectorize(generated_stmt, level))
			{
				fortype = Halide::Internal::ForType::Vectorized;
			}
		}

		*result = Halide::Internal::For::make(iterator_str, init_expr, cond_upper_bound_halide_format, fortype,
				Halide::DeviceAPI::Host, *halide_body);
	}
	else if (isl_ast_node_get_type(node) == isl_ast_node_user)
	{
		DEBUG(3, coli::str_dump("Generating code for user node"));

		isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
		isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, 0);
		isl_id *id = isl_ast_expr_get_id(arg);
		isl_ast_expr_free(arg);
		std::string computation_name(isl_id_get_name(id));
		isl_id_free(id);
		generated_stmts.push_back(computation_name);

		coli::computation *comp = fct.get_computation_by_name(computation_name);

		comp->create_halide_assignement();

		*result = comp->get_halide_stmt();
	}
	else if (isl_ast_node_get_type(node) == isl_ast_node_if)
	{
		DEBUG(3, coli::str_dump("Generating code for conditional"));

		isl_ast_expr *cond = isl_ast_node_if_get_cond(node);
		isl_ast_node *if_stmt = isl_ast_node_if_get_then(node);
		isl_ast_node *else_stmt = isl_ast_node_if_get_else(node);

		*result = Halide::Internal::IfThenElse::make(
					create_halide_expr_from_isl_ast_expr(cond),
					*coli::generate_Halide_stmt_from_isl_node(fct, if_stmt,
						level+1, generated_stmts),
					*coli::generate_Halide_stmt_from_isl_node(fct, else_stmt,
						level+1, generated_stmts));
	}

	DEBUG_INDENT(-4);

	return result;
}

void function::gen_halide_stmt()
{
	// This vector is used in generate_Halide_stmt_from_isl_node to figure
	// out what are the statements that have already been visited in the
	// AST tree.
	std::vector<std::string> generated_stmts;
	Halide::Internal::Stmt *stmt = new Halide::Internal::Stmt();

	// Generate code to free buffers that are not passed as an argument to the function
	// TODO: Add Free().
	/*	for (auto b: this->get_buffers_list())
	{
		coli::buffer *buf = b.second;
		// Allocate only arrays that are not passed to the function as arguments.
		if (buf->is_argument() == false)
			*stmt = Halide::Internal::Block::make(Halide::Internal::Free::make(buf->get_name()), *stmt);
	}*/

	// Generate the statement that represents the whole function
	stmt = coli::generate_Halide_stmt_from_isl_node(*this, this->get_isl_ast(), 0, generated_stmts);

	// Allocate buffers that are not passed as an argument to the function
	for (const auto &b : this->get_buffers_list())
	{
		coli::buffer *buf = b.second;
		// Allocate only arrays that are not passed to the function as arguments.
		if (buf->get_argument_type() == coli::type::argument::temporary)
		{
			std::vector<Halide::Expr> halide_dim_sizes;
			// Create a vector indicating the size that should be allocated.
			for (const auto &sz: buf->get_dim_sizes())
			{
				halide_dim_sizes.push_back(Halide::Expr((uint32_t) sz));
			}
			*stmt = Halide::Internal::Allocate::make(
						buf->get_name(),
						coli_type_to_halide_type(buf->get_type()),
						halide_dim_sizes, Halide::Internal::const_true(), *stmt);
		}
	}

	// Generate the invariants of the function.
	for (const auto &param : this->get_invariants())
	{
		std::vector<isl_ast_expr *> ie = {};
		*stmt = Halide::Internal::LetStmt::make(
					param.get_name(),
				 	create_halide_expr_from_coli_expr(NULL, ie, param.get_expr()),
				 	*stmt);
	}

	this->halide_stmt = stmt;
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

	int buf_dims = buffer->dimensions();

	// Get the rightmost access index: in A[i][j], this will return j
	isl_ast_expr *operand = isl_ast_expr_get_op_arg(index_expr, buf_dims);
	Halide::Expr index = create_halide_expr_from_isl_ast_expr(operand);

	Halide::Expr extents;

	if (buf_dims > 1)
		extents = Halide::Expr(buffer->extent(buf_dims - 1));

	for (int i = buf_dims - 1; i >= 1; i--)
	{
		operand = isl_ast_expr_get_op_arg(index_expr, i);
		Halide::Expr operand_h = create_halide_expr_from_isl_ast_expr(operand);
		Halide::Expr mul = Halide::Internal::Mul::make(operand_h, extents);

		index = Halide::Internal::Add::make(index, mul);

		extents = Halide::Internal::Mul::make(extents, Halide::Expr(buffer->extent(i - 1)));
	}

	return index;
}

/*
 * Create a Halide assign statement from a computation.
 * The statement will assign the computations to a memory buffer based on the
 * access function provided in access.
 */
void computation::create_halide_assignement()
{
	assert(this->access != NULL);

	const char *buffer_name = isl_space_get_tuple_name(
				isl_map_get_space(this->access), isl_dim_out);
	assert(buffer_name != NULL);

	isl_map *access = this->access;
	isl_space *space = isl_map_get_space(access);
	// Get the number of dimensions of the ISL map representing
	// the access.
	int access_dims = isl_space_dim(space, isl_dim_out);

   	// Fetch the actual buffer.
   	auto buffer_entry = this->function->get_buffers_list().find(buffer_name);
   	assert(buffer_entry != this->function->get_buffers_list().end());

   	auto coli_buffer = buffer_entry->second;

   	halide_dimension_t shape[coli_buffer->get_dim_sizes().size()];
	int stride = 1;
	for (int i = 0; i < coli_buffer->get_dim_sizes().size(); i++) {
       	shape[i].min = 0;
       	shape[i].extent = coli_buffer->get_dim_sizes()[i];
       	shape[i].stride = stride;
       	stride *= coli_buffer->get_dim_sizes()[i];
   	}

	Halide::Internal::BufferPtr *buffer =
		new Halide::Internal::BufferPtr(
				Halide::Image<>(coli_type_to_halide_type(coli_buffer->get_type()),
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
	this->stmt = Halide::Internal::Store::make(buffer_name, create_halide_expr_from_coli_expr(this, index_expr_cp, this->expression), index, param);
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
			coli_argtype_to_halide_argtype(buf->get_argument_type()),
			coli_type_to_halide_type(buf->get_type()),
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
