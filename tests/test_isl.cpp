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
	IRProgram pgm("program0");
	IRFunction fct("function0", &pgm);

	Halide::Internal::Stmt s0 = Halide::Internal::AssertStmt::make(Halide::Expr(0), Halide::Expr(3));
	Halide::Internal::Stmt s1 = Halide::Internal::AssertStmt::make(Halide::Expr(0), Halide::Expr(5));

	Computation computation0(ctx, s0, "{S0[i,j]: 0<=i<=1000 and 0<=j<=1000}", &fct);
	Computation computation1(ctx, s1, "{S1[i,j]: 0<=i<=1023 and 0<=j<=1023}", &fct);

/*	computation0.Split(0, 32);
	computation0.Split(2, 32);
	computation0.Interchange(1, 2);
	*/

	computation0.Tile(0,1,32,32);
	computation1.Schedule("{S1[i,j]->[1,i1,j1,i2,j3,j4]: i1=floor(i/32) and j1=floor(j/32) and i2=i and j3=floor(j/4) and j4=j%4 and 0<=i<=1023 and 0<=j<=1023}");
	pgm.tag_parallel_dimension("S0", 1);
	pgm.tag_vector_dimension("S1", 5);
	isl_union_map *schedule_map = pgm.get_schedule_map();

	isl_union_set *time_space = create_time_space(isl_union_set_copy(pgm.get_iteration_spaces()), isl_union_map_copy(schedule_map));


	isl_ast_build *ast_build = isl_ast_build_alloc(ctx);
	isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
	ast_build = isl_ast_build_set_after_each_for(ast_build, &for_halide_code_generator_after_for, NULL);
	ast_build = isl_ast_build_set_at_each_domain(ast_build, &stmt_halide_code_generator, NULL);
	isl_ast_node *program = isl_ast_build_node_from_schedule_map(ast_build, isl_union_map_copy(schedule_map));
	isl_ast_build_free(ast_build);


//	isl_ast_node_dump_c_code(ctx, program);

	std::vector<std::string> generated_stmts;
	Halide::Internal::Stmt halide_pgm = generate_Halide_stmt_from_isl_node(pgm, program, 0, generated_stmts);


	pgm.dump_ISIR();
	pgm.dump_schedule();
	IF_DEBUG(str_dump("\n\nTime Space IR:\n")); IF_DEBUG(isl_union_set_dump(time_space)); IF_DEBUG(str_dump("\n\n"));
	halide_IR_dump(halide_pgm);

	Halide::Argument buffer_arg("buf", Halide::Argument::OutputBuffer, Halide::Int(32), 3);
    	std::vector<Halide::Argument> args(1);
    	args[0] = buffer_arg;
	Halide::Module::Module m("", Halide::get_host_target());
	m.append(Halide::Internal::LoweredFunc("test1", args, halide_pgm, Halide::Internal::LoweredFunc::External));

	return 0;
}
