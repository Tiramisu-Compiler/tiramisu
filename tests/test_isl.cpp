#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <DebugIR.h>
#include <IR.h>
#include <String.h>
#include <Halide.h>


int main(int argc, char **argv)
{
	isl_ctx *ctx = isl_ctx_alloc();
	IRProgram pgm("program");
	IRFunction fct("function");
	pgm.add_function(&fct);

	Halide::Internal::Stmt s = Halide::Internal::AssertStmt::make(Halide::Expr(0), Halide::Expr(3));
	Computation computation0(ctx, s, "{S0[i,j]: 1<=i<=1000 and 0<=j<=1000}", &fct);
	isl_union_map *schedule_map = create_schedule_map(ctx,"{S0[i,j]->S0[i1,j1,i,j]: i1=floor(i/32) and j1=floor(j/32) and 1<=i<=1000 and 0<=j<=1000}");
	isl_union_set *time_space = create_time_space(isl_union_set_copy(isl_union_set_from_set(computation0.iter_space)), isl_union_map_copy(schedule_map));


	pgm.dump_ISIR();
	IF_DEBUG(str_dump("Schedule:\n")); IF_DEBUG(isl_union_map_dump(schedule_map)); IF_DEBUG(str_dump("\n\n"));
	IF_DEBUG(str_dump("\n\nTime Space IR:\n")); IF_DEBUG(isl_union_set_dump(time_space)); IF_DEBUG(str_dump("\n\n"));


	Halide::Argument buffer_arg("buf", Halide::Argument::OutputBuffer, Halide::Int(32), 3);
    	std::vector<Halide::Argument> args(1);
    	args[0] = buffer_arg;
	Halide::Module::Module m("", Halide::get_host_target());
	m.append(Halide::Internal::LoweredFunc("test1", args, computation0.stmt, Halide::Internal::LoweredFunc::External));


	isl_ast_build *ast_build = isl_ast_build_alloc(ctx);
	isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
	ast_build = isl_ast_build_set_after_each_for(ast_build, &for_halide_code_generator_after_for, &computation0.stmt);
	ast_build = isl_ast_build_set_at_each_domain(ast_build, &stmt_halide_code_generator, &computation0.stmt);
	isl_ast_node *program = isl_ast_build_node_from_schedule_map(ast_build, schedule_map);
	isl_ast_build_free(ast_build);

	isl_ast_node_dump_c_code(ctx, program);
	halide_IR_dump(computation0.stmt);

	// pgm.dump();

	return 0;
}
