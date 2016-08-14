#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <String.h>
#include <Halide.h>


int main(int argc, char **argv)
{
	isl_ctx *ctx = isl_ctx_alloc();
	coli::program pgm("program0");
	coli::function fct("function0", &pgm);

	// Declare the computations.  Each computation has: (1) a Halide expression,
	// (2) an isl set representing its iteration space and (3) is attached to a
	// function.
	coli::computation computation0(ctx, Halide::Expr((uint8_t) 3), "{S0[i,j]: 0<=i<=1000 and 0<=j<=1000}", &fct);
	coli::computation computation1(ctx, Halide::Expr((uint8_t) 5), "{S1[i,j]: 0<=i<=1023 and 0<=j<=1023}", &fct);
	coli::computation computation2(ctx, Halide::Expr((uint8_t) 7), "{S2[i,j]: 0<=i<=1023 and 0<=j<=1023}", &fct);

	// Create memory buffers, then map the computations to those buffers.
	Halide::Buffer buf(Halide::Int(8), 10, 10, 0, 0, NULL, "buf");
	fct.buffers_list.insert(std::pair<std::string, Halide::Buffer *>(buf.name(), &buf));
	Halide::Argument buffer_arg("buf", Halide::Argument::OutputBuffer, Halide::Int(8), 2);
	std::vector<Halide::Argument> args(1);
	args[0] = buffer_arg;

	computation0.SetAccess("{S0[i,j]->buf[i, j]}");
	computation1.SetAccess("{S1[i,j]->buf[0, 3]}");
	computation2.SetAccess("{S2[i,j]->buf[0, 0]}");


	// Set the schedule of each computation.
	computation0.Tile(0,1,32,32);
	computation1.Schedule("{S1[i,j]->[2,i1,j1,i2,j3,j4]: i1=floor(i/32) and j1=floor(j/32) and i2=i and j3=floor(j/4) and j4=j%4 and 0<=i<=1023 and 0<=j<=1023}");
	computation2.Split(0, 32);
	computation2.Split(2, 32);
	computation2.Interchange(1, 2);
	pgm.tag_parallel_dimension("S0", 1);
//	pgm.tag_vector_dimension("S1", 5);

	isl_union_map *schedule_map = pgm.get_schedule_map();

	// Create time space IR
	isl_union_set *time_space_representaion =
		coli::create_time_space_representation(isl_union_set_copy(pgm.get_iteration_spaces()), isl_union_map_copy(schedule_map));

	// Generate code
	isl_ast_build *ast_build = isl_ast_build_alloc(ctx);
	isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
	ast_build = isl_ast_build_set_after_each_for(ast_build, &coli::for_halide_code_generator_after_for, NULL);
	ast_build = isl_ast_build_set_at_each_domain(ast_build, &coli::stmt_halide_code_generator, NULL);
	isl_ast_node *program = isl_ast_build_node_from_schedule_map(ast_build, isl_union_map_copy(schedule_map));
	isl_ast_build_free(ast_build);

	if (DEBUG)
		coli::isl_ast_node_dump_c_code(ctx, program);

	std::vector<std::string> generated_stmts, iterators;
	Halide::Internal::Stmt halide_pgm = coli::generate_Halide_stmt_from_isl_node(pgm, program, 0, generated_stmts, iterators);

	// Dump IRs
	pgm.dump_ISIR();
	pgm.dump_schedule();
	IF_DEBUG(coli::str_dump("\n\nTime Space IR:\n")); IF_DEBUG(isl_union_set_dump(time_space_representaion)); IF_DEBUG(coli::str_dump("\n\n"));
	coli::halide_IR_dump(halide_pgm);


	Halide::Target target;
	target.os = Halide::Target::OSX;
	target.arch = Halide::Target::X86;
	target.bits = 64;
	std::vector<Halide::Target::Feature> x86_features;
	x86_features.push_back(Halide::Target::AVX);
	x86_features.push_back(Halide::Target::SSE41);
	target.set_features(x86_features);



	Halide::Module::Module m("test1", target);
	m.append(Halide::Internal::LoweredFunc("test1", args, halide_pgm, Halide::Internal::LoweredFunc::External));

	Halide::compile_module_to_object(m, "LLVM_generated_code.o");

	return 0;
}
