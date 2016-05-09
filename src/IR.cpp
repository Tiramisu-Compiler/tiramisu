#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <DebugIR.h>

/* Schedule the iteration space.  */
isl_union_map *create_schedule_map(isl_ctx *ctx,
		   int tile)
{
	// Identity schedule for 1D iteration space
	std::string map = "{S0[i,j]->S0[i,j]}";

	// Tiling
	// std::string map = "{S0[i,j]->S0[i/32, j/32, i, j]}";

	// One point into multiple points
	// std::string map = "{S0[i,j]->S0[0, i, j, k]: 0<i<10000 and 0<j<10000 and 0<k<5}";

	isl_union_map *schedule_map = isl_union_map_read_from_str(ctx, map.c_str());

	IF_DEBUG2(str_dump("[ir.c] Schedule map: "));
	IF_DEBUG2(str_dump(map));
	IF_DEBUG2(isl_union_map_dump(schedule_map));

	return schedule_map;
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

isl_ast_node *generate_code(isl_ctx *ctx,
		   isl_schedule *sched_tree)
{
	isl_ast_build *ast = isl_ast_build_alloc(ctx);
 	isl_ast_node *program = isl_ast_build_node_from_schedule(ast, sched_tree);
	isl_ast_build_free(ast);

	return program;
}
