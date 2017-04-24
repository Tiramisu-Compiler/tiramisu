#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string>


void tiramisu::function::gen_c_code() const
{
    tiramisu::str_dump("\n\n");
    tiramisu::str_dump("\nC like code:\n");
    isl_printer *p;
    p = isl_printer_to_file(this->get_isl_ctx(), stdout);
    p = isl_printer_set_output_format(p, ISL_FORMAT_C);
    p = isl_printer_print_ast_node(p, this->get_isl_ast());
    isl_printer_free(p);
    tiramisu::str_dump("\n\n");
}
