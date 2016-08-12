#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/ir.h>

#include <string>


void isl_ast_node_dump_c_code(isl_ctx *ctx, isl_ast_node *root_node)
{
	coli_str_dump("\n\n");
	coli_str_dump("\nC like code:\n");
	isl_printer *p;
	p = isl_printer_to_file(ctx, stdout);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = isl_printer_print_ast_node(p, root_node);
	isl_printer_free(p);
	coli_str_dump("\n\n");
}
