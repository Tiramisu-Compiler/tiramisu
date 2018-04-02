#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string.h>
#include <Halide.h>

#include "wrapper_test_SEM.h"

using namespace tiramisu;

void generate_function(std::string name)
{
    tiramisu::global::set_default_tiramisu_options();
    tiramisu::global::set_loop_iterator_type(p_int64);

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    tiramisu::function function0(name);
    var i("i"), j("j"), k("k"), x("x"), a("a"), b("b"), l("l");
    constant cN("cN", (int64_t)N, p_int64, true, NULL, 0, &function0);
    computation matrix_D("[cN]->{matrix_D[i,j]: 0<=i<cN and 0<=j<cN}",
			 expr(), false, p_float64, &function0);
    computation tensor_G("[cN]->{tensor_G[a,b,i,j,k]: 0<=a<3 and 0<=b<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			 expr(), false, p_float64, &function0);
    computation tensor_u("[cN]->{tensor_u[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			 expr(), false, p_float64, &function0);

    computation tensor_w_0_init("[cN]->{tensor_w_0_init[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				expr(0.0), true, p_float64, &function0);
    computation tensor_w_1_init("[cN]->{tensor_w_1_init[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				expr(0.0), true, p_float64, &function0);
    computation tensor_w_2_init("[cN]->{tensor_w_2_init[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				expr(0.0), true, p_float64, &function0);
    computation tensor_w_0("[cN]->{tensor_w_0[i,j,k,l]: 0<=l<cN and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			   tensor_w_0_init(i,j,k) + matrix_D(k,l) * tensor_u(i,j,l), true, p_float64, &function0);
    computation tensor_w_1("[cN]->{tensor_w_1[i,j,k,l]: 0<=l<cN and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			   tensor_w_1_init(i,j,k) + matrix_D(j,l) * tensor_u(i,l,k), true, p_float64, &function0);
    computation tensor_w_2("[cN]->{tensor_w_2[i,j,k,l]: 0<=l<cN and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			   tensor_w_2_init(i,j,k) + matrix_D(i,l) * tensor_u(l,j,k), true, p_float64, &function0);
    computation tensor_w("[cN]->{tensor_w[a,i,j,k]: 0<=a<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			 expr(), false, p_float64, &function0);
    computation tensor_z_init("[cN]->{tensor_z_init[a,i,j,k]:  0<=a<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			      expr(0.0), true, p_float64, &function0);
    computation tensor_z("[cN]->{tensor_z[a,b,i,j,k]: 0<=a<3 and 0<=b<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			 tensor_z_init(a,i,j,k) + tensor_G(a,b,i,j,k) * tensor_w(b,i,j,k), true, p_float64, &function0);
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    tensor_w_0_init.before(tensor_w_1_init, computation::root);
    tensor_w_1_init.before(tensor_w_2_init, computation::root);
    tensor_w_2_init.before(tensor_w_0, computation::root);
    tensor_w_0.before(tensor_w_1, computation::root);
    tensor_w_1.before(tensor_w_2, computation::root);
    tensor_w_2.before(tensor_z_init, computation::root);
    tensor_z_init.before(tensor_z, computation::root);
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buff_D("buff_D", {N, N}, p_float64, a_input, &function0);
    tiramisu::buffer buff_G("buff_G", {3, 3, N, N, N}, p_float64, a_input, &function0);
    tiramisu::buffer buff_u("buff_u", {N, N, N}, p_float64, a_input, &function0);
    tiramisu::buffer buff_tensor_w("buff_tensor_w", {3, N, N, N}, p_float64, a_temporary, &function0);
    tiramisu::buffer buff_tensor_z("buff_tensor_z", {3, N, N, N}, p_float64, a_output, &function0);

    matrix_D.set_access("{matrix_D[i,j]->buff_D[i,j]}");
    tensor_G.set_access("{tensor_G[a,b,i,j,k]->buff_G[a,b,i,j,k]}");
    tensor_u.set_access("{tensor_u[i,j,k]->buff_u[i,j,k]}");
    tensor_w_0_init.set_access("{tensor_w_0_init[i,j,k]->buff_tensor_w[0,i,j,k]}");
    tensor_w_1_init.set_access("{tensor_w_1_init[i,j,k]->buff_tensor_w[1,i,j,k]}");
    tensor_w_2_init.set_access("{tensor_w_2_init[i,j,k]->buff_tensor_w[2,i,j,k]}");
    tensor_w_0.set_access("{tensor_w_0[i,j,k,l]->buff_tensor_w[0,i,j,k]}");
    tensor_w_1.set_access("{tensor_w_1[i,j,k,l]->buff_tensor_w[1,i,j,k]}");
    tensor_w_2.set_access("{tensor_w_2[i,j,k,l]->buff_tensor_w[2,i,j,k]}");
    tensor_w.set_access("{tensor_w[a,i,j,k]->buff_tensor_w[a,i,j,k]}");
    tensor_z_init.set_access("{tensor_z_init[a,i,j,k]->buff_tensor_z[a,i,j,k]}");
    tensor_z.set_access("{tensor_z[a,b,i,j,k]->buff_tensor_z[a,i,j,k]}");
    
    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&buff_D, &buff_G, &buff_u, &buff_tensor_z});
    function0.gen_time_space_domain();
    function0.gen_isl_ast();
    function0.gen_halide_stmt();
    function0.gen_halide_obj("generated_fct_test_" + std::string(TEST_NAME_STR) + ".o");
}

int main(int argc, char **argv)
{
  generate_function("tiramisu_generated_code");

    return 0;
}
