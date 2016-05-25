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

#define INDENTATION 4

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
                
int main(int argc, char **argv)
{
	isl_ctx *ctx = isl_ctx_alloc();
	std::string str0 = "{S0[i,j]: 0<=i<=1000 and 0<=j<=1000}";
	isl_union_set *set0 = isl_union_set_read_from_str(ctx, str0.c_str());

	IF_DEBUG(str_dump("\nIteration Space IR:\n"));
	IF_DEBUG(str_dump(str0.c_str())); IF_DEBUG(str_dump("\n\n"));

	isl_union_map *schedule_map = create_schedule_map(ctx, set0, 1);
	isl_schedule *schedule_tree = create_schedule_tree(ctx, set0, schedule_map);

	Halide::Argument buffer_arg("buf", Halide::Argument::OutputBuffer, Halide::Int(32), 3);
    	std::vector<Halide::Argument> args(1);
    	args[0] = buffer_arg;

	Halide::Internal::IRPrinter pr(std::cout);
	Halide::Internal::Stmt s = Halide::Internal::AssertStmt::make (Halide::Expr(0), Halide::Expr(1));

	Halide::Module::Module m("", Halide::get_host_target());
	m.append(Halide::Internal::LoweredFunc("test1", args, s, Halide::Internal::LoweredFunc::External));

	isl_printer *p;
        p = isl_printer_to_str(ctx);
	isl_ast_print_options *options = isl_ast_print_options_alloc(ctx);

	isl_ast_build *ast_build = isl_ast_build_alloc(ctx);
	ast_build = isl_ast_build_set_after_each_for(ast_build, &for_halide_code_generator_after_for, &s);
	ast_build = isl_ast_build_set_at_each_domain(ast_build, &stmt_halide_code_generator, &s);
	isl_ast_node *program = isl_ast_build_node_from_schedule(ast_build, schedule_tree);
	isl_ast_build_free(ast_build);

	p = isl_ast_node_print(program, p, options);

	IF_DEBUG(str_dump("\n\n"));
	IF_DEBUG(str_dump("\nGenerated Halide Low Level IR:\n"));
    	pr.print(s);
	IF_DEBUG(str_dump("\n\n"));

	return 0;
}
