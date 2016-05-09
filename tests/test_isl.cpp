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

//#define ISL_ONLY_CODE_GENERATOR
#define ISL_AND_HALIDE_CODE_GENERATOR

isl_printer *for_halide_code_generator(isl_printer *p, isl_ast_print_options *options, isl_ast_node *node, void *user)
{
	Halide::Internal::Stmt *s = (Halide::Internal::Stmt *) user;

	isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
	char *iterator_str = isl_ast_expr_to_str(iter);
	Halide::Var x(iterator_str);

	isl_ast_expr *init = isl_ast_node_for_get_init(node);
	isl_val *init_val = isl_ast_expr_get_val(init);
	Halide::Expr init_expr = Halide::Expr((uint64_t)isl_val_get_num_si(init_val));

	*s = Halide::Internal::For::make(iterator_str, init_expr, Halide::Expr(10), Halide::Internal::ForType::Serial,
		  		    Halide::DeviceAPI::Host, *s);

	isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
	
	isl_ast_node *loop_body = isl_ast_node_for_get_body(node);

	p = isl_ast_node_print(loop_body, p, options);

	return p;
}

isl_printer *for_isl_printer(isl_printer *p, isl_ast_print_options *options, isl_ast_node *node, void *user)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "for (");
	isl_ast_expr *iter = isl_ast_node_for_get_iterator(node);
	p = isl_printer_print_ast_expr(p, iter);
	p = isl_printer_print_str(p, ", ");
	isl_ast_expr *init = isl_ast_node_for_get_init(node);
	p = isl_printer_print_ast_expr(p, init);
	p = isl_printer_print_str(p, ", ");
	isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
	p = isl_printer_print_ast_expr(p, cond);
	p = isl_printer_print_str(p, ")");
	p = isl_printer_end_line(p);
	
	p = isl_printer_start_line(p);
	p = isl_printer_indent(p, INDENTATION);
	isl_ast_node *loop_body = isl_ast_node_for_get_body(node);
	p = isl_ast_node_print(loop_body, p, options);
	p = isl_printer_indent(p, -INDENTATION);
	p = isl_printer_end_line(p);

	return p;
}
                
int main(int argc, char **argv)
{
	isl_ctx *ctx = isl_ctx_alloc();
	std::string str0 = "{S0[i,j]: 0<i<100 and 0<j<100}";
	isl_union_set *set0 = isl_union_set_read_from_str(ctx, str0.c_str());

	isl_union_map *schedule_map = create_schedule_map(ctx, 1);
	isl_schedule *schedule_tree = create_schedule_tree(ctx, set0, schedule_map);
	isl_ast_node *program = generate_code(ctx, schedule_tree);

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

	IF_DEBUG(str_dump("\nIteration space:\n"));
	IF_DEBUG(str_dump(str0.c_str())); IF_DEBUG(str_dump("\n\n"));

#ifdef ISL_ONLY_CODE_GENERATOR
        options = isl_ast_print_options_set_print_for(options, &for_isl_printer, &s);
#endif

#ifdef ISL_AND_HALIDE_CODE_GENERATOR
        options = isl_ast_print_options_set_print_for(options, &for_halide_code_generator, &s);
#endif

	p = isl_ast_node_print(program, p, options);

#ifdef ISL_ONLY_CODE_GENERATOR
	char *program_str = isl_printer_get_str(p);
	IF_DEBUG(str_dump("Generated code:\n"));
	IF_DEBUG(str_dump(program_str)); IF_DEBUG(str_dump("\n"));
#endif

#ifdef ISL_AND_HALIDE_CODE_GENERATOR
	IF_DEBUG(str_dump("\nHalide Low Level IR:\n\n"));
    	pr.print(s);
#endif

	return 0;
}
