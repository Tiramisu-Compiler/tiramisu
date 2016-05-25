#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <DebugIR.h>
#include <IR.h>

/* Schedule the iteration space.  */
isl_union_map *create_schedule_map(isl_ctx *ctx, isl_union_set *set,
		   std::string map)
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

isl_ast_node *for_halide_code_generator_after_for(isl_ast_node *node, isl_ast_build *build, void *user)
{
	Halide::Internal::Stmt *s = (Halide::Internal::Stmt *) user;

	isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
	char *iterator_str = isl_ast_expr_to_str(iter);
	Halide::Var x(iterator_str);

	isl_ast_expr *init = isl_ast_node_for_get_init(node);
	isl_val *init_val = isl_ast_expr_get_val(init);
	Halide::Expr init_expr = Halide::Expr((uint64_t)isl_val_get_num_si(init_val));

	isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
	isl_ast_expr *cond_upper_bound_isl_format = isl_ast_expr_get_op_arg(cond, 1);
	Halide::Expr cond_upper_bound_halide_format = Halide::Expr((uint64_t)isl_val_get_num_si(isl_ast_expr_get_val(cond_upper_bound_isl_format)));
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

	IF_DEBUG2(str_dump("[ir.c] Schedule tree: "));
	IF_DEBUG2(isl_schedule_dump(sched_tree));

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
