#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <string.h>
#include <Halide.h>

using namespace coli;

void generate_function_1(std::string name, int size, int val0, int val1)
{
    coli::global::set_default_coli_options();
    coli::global::set_auto_data_mapping(false); // No automatic data mapping.

    coli::function function0(name);
    coli::computation S0("{S0[i]: 0<=i<10}", coli::expr((uint8_t) 4), true, p_uint8, &function0);
    coli::computation S1("{S1[i]: 0<=i<10}", coli::expr((uint8_t) 4), true, p_uint8, &function0);
    coli::computation S2("{S2[i]: 0<=i<10}", coli::expr((uint8_t) 4), true, p_uint8, &function0);

    coli::buffer buf0("buf0", 1, {size}, coli::p_uint8, NULL, a_output, &function0);
    S0.set_access("{S0[0,i,0]->buf0[i]}");
    S1.set_access("{S1[0,i,1]->buf0[i]}");
    S2.set_access("{S2[1,i,0]->buf0[i]}");

    S0.set_schedule("{S0[i]->S0[0,i,0]}");
    S1.set_schedule("{S1[i]->S0[0,i,1]}");
    S2.set_schedule("{S2[i]->S0[1,i,0]}");


    //S0.tag_parallel_dimension(0);

    function0.dump(true);

    function0.set_arguments({&buf0});
    function0.gen_time_processor_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.dump_halide_stmt();
    function0.gen_halide_obj("build/generated_fct_test_07.o");
}


int main(int argc, char **argv)
{
    generate_function_1("test_duplication", 10, 4, 0);

    return 0;
}
