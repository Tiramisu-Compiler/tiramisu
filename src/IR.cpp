#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <DebugIR.h>
#include <IR.h>

/* Schedule the iteration space.  */
isl_union_map *create_schedule_map(isl_ctx *ctx, std::string map)
{
	isl_union_map *schedule_map = isl_union_map_read_from_str(ctx,
			map.c_str());

	return schedule_map;
}

isl_ast_node *stmt_halide_code_generator(isl_ast_node *node, isl_ast_build *build, void *user)
{
	Halide::Internal::Stmt *s = new Halide::Internal::Stmt();
	*s = Halide::Internal::AssertStmt::make (Halide::Expr(0), Halide::Expr(1));
	user = (void *) s;

	return node;
}

Halide::Expr create_halide_expr_from_isl_ast_expr(isl_ast_expr *isl_expr)
{
	Halide::Expr result;

	if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_int)
	{
		isl_val *init_val = isl_ast_expr_get_val(isl_expr);
		result = Halide::Expr((int64_t)isl_val_get_num_si(init_val));
	}
	else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_id)
	{
		isl_id *identifier = isl_ast_expr_get_id(isl_expr);
		std::string name_str(isl_id_get_name(identifier));
		result = Halide::Internal::Variable::make(Halide::Int(64), name_str);
	}
	else if (isl_ast_expr_get_type(isl_expr) == isl_ast_expr_op)
	{
		Halide::Expr op0, op1;

		int nb_args = isl_ast_expr_get_op_n_arg(isl_expr);
		op0 = create_halide_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(isl_expr, 0));

		if (isl_ast_expr_get_op_n_arg(isl_expr) > 1)
			op1 = create_halide_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(isl_expr, 1));

		switch(isl_ast_expr_get_op_type(isl_expr))
		{
			case isl_ast_op_min:
				result = Halide::Internal::Min::make(op0, op1);
				break;
			case isl_ast_op_max:
				result = Halide::Internal::Max::make(op0, op1);
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
			case isl_ast_op_and:
				result = Halide::Internal::And::make(op0, op1);
				break;
			case isl_ast_op_or:
				result = Halide::Internal::Or::make(op0, op1);
				break;
			case isl_ast_op_minus:
				result = Halide::Internal::Sub::make(Halide::Expr(0), op0);
				break;
			default:
				Error("Translating an unsupported ISL expression in a Halide expression.", 1);
		}
	}
	else
		Error("Translating an unsupported ISL expression in a Halide expression.", 1);

	return result;
}

isl_ast_node *for_halide_code_generator_after_for(isl_ast_node *node, isl_ast_build *build, void *user)
{
	Halide::Internal::Stmt *s = (Halide::Internal::Stmt *) user;

	isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
	char *iterator_str = isl_ast_expr_to_str(iter);

	isl_ast_expr *init = isl_ast_node_for_get_init(node);
	isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
	isl_ast_expr *cond_upper_bound_isl_format;
	if (isl_ast_expr_get_op_type(cond) == isl_ast_op_le || isl_ast_expr_get_op_type(cond) == isl_ast_op_lt)
		cond_upper_bound_isl_format = isl_ast_expr_get_op_arg(cond, 1);
	else
		Error("The for loop upper bound is not an isl_est_expr of type le or lt" ,1);

        Halide::Expr init_expr = create_halide_expr_from_isl_ast_expr(init);
	Halide::Expr cond_upper_bound_halide_format =  create_halide_expr_from_isl_ast_expr(cond_upper_bound_isl_format);
	*s = Halide::Internal::For::make(iterator_str, init_expr, cond_upper_bound_halide_format, Halide::Internal::ForType::Serial,
			Halide::DeviceAPI::Host, *s);

	return node;
}

isl_schedule *create_schedule_tree(isl_ctx *ctx,
		   isl_union_set *udom,
		   isl_union_map *sched_map)
{
	isl_union_set *scheduled_domain = isl_union_set_apply(udom, sched_map);
	IF_DEBUG2(str_dump("[ir.c] Scheduled domain: "));
	IF_DEBUG2(isl_union_set_dump(scheduled_domain));

	isl_schedule *sched_tree = isl_schedule_from_domain(scheduled_domain);

	return sched_tree;
}

/* Schedule the iteration space.  */
isl_union_set *create_time_space(__isl_take isl_union_set *set, __isl_take isl_union_map *umap)
{
	return isl_union_set_apply(set, umap);
}

isl_ast_node *generate_code(isl_ctx *ctx,
		   isl_schedule *sched_tree)
{
	isl_ast_build *ast = isl_ast_build_alloc(ctx);
 	isl_ast_node *program = isl_ast_build_node_from_schedule(ast, sched_tree);
	isl_ast_build_free(ast);

	return program;
}

// Computation related methods
void Computation::dump()
{
	IF_DEBUG(isl_union_set_dump(this->iter_space)); IF_DEBUG(str_dump("\n"));
}
