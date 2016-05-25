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

	Computation computation0(ctx, "{S0[i,j]: 0<=i<=1000 and 0<=j<=1000}");

	IF_DEBUG(str_dump("\nIteration Space IR:\n"));
	IF_DEBUG(isl_union_set_dump(computation0.iter_space)); IF_DEBUG(str_dump("\n"));

	isl_union_map *schedule_map = create_schedule_map(ctx, "{S0[i,j]->S0[i+100,j+100]}");

	IF_DEBUG(str_dump("Schedule:\n"));
	IF_DEBUG(isl_union_map_dump(schedule_map)); IF_DEBUG(str_dump("\n\n"));

	isl_schedule *schedule_tree = create_schedule_tree(ctx, isl_union_set_copy(computation0.iter_space), isl_union_map_copy(schedule_map));
	isl_union_set *time_space = create_time_space(isl_union_set_copy(computation0.iter_space), isl_union_map_copy(schedule_map));

	IF_DEBUG(str_dump("\n\nTime Space IR:\n"));
	IF_DEBUG(isl_union_set_dump(time_space)); IF_DEBUG(str_dump("\n\n"));

	Halide::Argument buffer_arg("buf", Halide::Argument::OutputBuffer, Halide::Int(32), 3);
    	std::vector<Halide::Argument> args(1);
    	args[0] = buffer_arg;

	Halide::Internal::IRPrinter pr(std::cout);
	Halide::Internal::Stmt s = Halide::Internal::AssertStmt::make (Halide::Expr(0), Halide::Expr(1));

	Halide::Module::Module m("", Halide::get_host_target());
	m.append(Halide::Internal::LoweredFunc("test1", args, s, Halide::Internal::LoweredFunc::External));

	isl_ast_build *ast_build = isl_ast_build_alloc(ctx);
	ast_build = isl_ast_build_set_after_each_for(ast_build, &for_halide_code_generator_after_for, &s);
	ast_build = isl_ast_build_set_at_each_domain(ast_build, &stmt_halide_code_generator, &s);
	isl_ast_node *program = isl_ast_build_node_from_schedule(ast_build, schedule_tree);
	isl_ast_build_free(ast_build);

	isl_printer *p;
        p = isl_printer_to_str(ctx);
	isl_ast_print_options *options = isl_ast_print_options_alloc(ctx);
	p = isl_ast_node_print(program, p, options);

	IF_DEBUG(str_dump("\n\n"));
	IF_DEBUG(str_dump("\nGenerated Halide Low Level IR:\n"));
    	pr.print(s);
	IF_DEBUG(str_dump("\n\n"));

	return 0;
}
