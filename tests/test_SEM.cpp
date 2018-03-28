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
    var i("i"), j("j"), k("k"), x("x"), a("a");
    constant cN("cN", (int64_t)N, p_int64, true, NULL, 0, &function0);
    computation matrix_D("[cN]->{matrix_D[i,j]: 0<=i<cN and 0<=j<cN}",
			 expr(), false, p_float32, &function0);
    computation tensor_G("[cN]->{tensor_G[x,y,i,j,k]: 0<=x<3 and 0<=y<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			 expr(), false, p_float32, &function0);
    computation tensor_u("[cN]->{tensor_u[k,i,j]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			 expr(), false, p_float32, &function0);
    // initialize reductions
    computation reduce_D_init_along_dim_0("[cN]->{reduce_D_init_along_dim_0[j]: 0<=j<cN}", // reduction down columns
				     expr(0.0f), true, p_float32, &function0);
    computation reduce_D_init_along_dim_1("[cN]->{reduce_D_init_along_dim_1[i]: 0<=i<cN}", // reduction across rows
					  expr(0.0f), true, p_float32, &function0);
    computation reduce_D_along_dim_0("[cN]->{reduce_D_along_dim_0[i,j]: 0<=i<cN and 0<=j<cN}", // reduction down columns
				     reduce_D_init_along_dim_0(j) + matrix_D(i,j), true, p_float32, &function0);
    computation reduce_D_along_dim_1("[cN]->{reduce_D_along_dim_1[i,j]: 0<=i<cN and 0<=j<cN}", // reduction across rows
				     reduce_D_init_along_dim_1(i) + matrix_D(i,j), true, p_float32, &function0);
    computation reduce_u_init_along_dim_0("[cN]->{reduce_u_init_along_dim_0[k,j]: 0<=j<cN and 0<=k<cN}",
					  expr(0.0f), true, p_float32, &function0); // reduction across rows
    computation reduce_u_init_along_dim_1("[cN]->{reduce_u_init_along_dim_1[i,k]: 0<=i<cN and 0<=k<cN}",
					  expr(0.0f), true, p_float32, &function0); // reduction down columns
    computation reduce_u_init_along_dim_2("[cN]->{reduce_u_init_along_dim_2[i,j]: 0<=i<cN and 0<=j<cN}",
					  expr(0.0f), true, p_float32, &function0); // reduction across depth
    computation reduce_u_along_dim_0("[cN]->{reduce_u_along_dim_0[k,i,j]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				     reduce_u_init_along_dim_0(k,j) + tensor_u(k,i,j), true, p_float32, &function0);
    computation reduce_u_along_dim_1("[cN]->{reduce_u_along_dim_1[k,i,j]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				     reduce_u_init_along_dim_1(i,k) + tensor_u(k,i,j), true, p_float32, &function0);
    computation reduce_u_along_dim_2("[cN]->{reduce_u_along_dim_2[k,i,j]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				     reduce_u_init_along_dim_2(i,j) + tensor_u(k,i,j), true, p_float32, &function0);
    computation tensor_w_0("[cN]->{tensor_w_0[k,i,j]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			   reduce_D_along_dim_1(k,0) * reduce_u_along_dim_2(0,i,j), true, p_float32, &function0);
    computation tensor_w_1("[cN]->{tensor_w_1[k,i,j]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
    			   reduce_D_along_dim_1(j,0) * reduce_u_along_dim_2(i,0,k), true, p_float32, &function0);
    computation tensor_w_2("[cN]->{tensor_w_2[k,i,j]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			   reduce_D_along_dim_1(i,0) * reduce_u_along_dim_2(i,j,0), true, p_float32, &function0);
    /*
    computation tensor_z_init("[CN]->{tensor_z_init[a,i,j,k]: 0<=a<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			      expr(0.0f), true, p_float32, &function0);
    computation tensor_z_b_0("[cN]->{tensor_z_b_0[a,i,j,k]: 0<=a<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			     tensor_z_init(a,i,j,k) + tensor_G(a,0,i,j,k) * tensor_w_0(i,j,k), true, p_float32, &function0);
    computation tensor_z_b_1("[cN]->{tensor_z_b_1[a,i,j,k]: 0<=a<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			     tensor_z_init(a,i,j,k) + tensor_G(a,1,i,j,k) * tensor_w_1(i,j,k), true, p_float32, &function0);
    computation tensor_z_b_2("[cN]->{tensor_z_b_2[a,i,j,k]: 0<=a<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			     tensor_z_init(a,i,j,k) + tensor_G(a,2,i,j,k) * tensor_w_2(i,j,k), true, p_float32, &function0);    
    computation reduce_z_0_init_along_dim_2("[cN]->{reduce_z_0_init_along_dim_2[i,j]: 0<=i<cN and 0<=j<cN}",
					    expr(0.0f), true, p_float32, &function0);
    computation reduce_z_1_init_along_dim_1("[cN]->{reduce_z_1_init_along_dim_1[i,k]: 0<=i<cN and 0<=k<cN}",
					    expr(0.0f), true, p_float32, &function0);
    computation reduce_z_2_init_along_dim_0("[cN]->{reduce_z_2_init_along_dim_0[j,k]: 0<=j<cN and 0<=k<cN}",
					    expr(0.0f), true, p_float32, &function0);
    computation reduce_z_0_along_dim_2("[cN]->{reduce_z_0_along_dim_2[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				       reduce_z_0_along_dim_2(i,j) + tensor_z_init(0, i, j, k), true, p_float32, &function0);
    computation reduce_z_1_along_dim_1("[cN]->{reduce_z_1_along_dim_1[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				       reduce_z_1_along_dim_1(i,k) + tensor_z_init(1, i, j, k), true, p_float32, &function0);
    computation reduce_z_2_along_dim_0("[cN]->{reduce_z_2_along_dim_0[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				       reduce_z_2_along_dim_0(j,k) + tensor_z_init(2, i, j, k), true, p_float32, &function0);
    computation u_update_init("[cN]->{u_update_init[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			      reduce_D_along_dim_0(0,k) * reduce_z_0_along_dim_2(i,j), true, p_float32, &function0);
    computation u_update_z_1("[cN]->{u_update_z_1[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			     u_update_init(i,j,k) + reduce_D_along_dim_0(0,j) * reduce_z_1_along_dim_1(i,k), true, p_float32, &function0);
    computation u_update_z_2("[cN]->{u_update_z_2[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			     u_update_init(i,j,k) + reduce_D_along_dim_0(0,i) * reduce_z_2_along_dim_0(j,k), true, p_float32, &function0);       
    */
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    reduce_D_init_along_dim_0.before(reduce_D_init_along_dim_1, computation::root);
    reduce_D_init_along_dim_1.before(reduce_D_along_dim_0, computation::root);
    reduce_D_along_dim_0.before(reduce_D_along_dim_1, computation::root);
    reduce_D_along_dim_1.before(reduce_u_init_along_dim_0, computation::root);
    reduce_u_init_along_dim_0.before(reduce_u_init_along_dim_1, computation::root);
    reduce_u_init_along_dim_1.before(reduce_u_init_along_dim_2, computation::root);
    reduce_u_init_along_dim_2.before(reduce_u_along_dim_0, computation::root);
    reduce_u_along_dim_0.before(reduce_u_along_dim_1, computation::root);
    reduce_u_along_dim_1.before(reduce_u_along_dim_2, computation::root);
    reduce_u_along_dim_2.before(tensor_w_0, computation::root);
    tensor_w_0.before(tensor_w_1, computation::root);    
    tensor_w_1.before(tensor_w_2, computation::root);
    /*tensor_w_2.before(tensor_z_init, computation::root);
    tensor_z_init.before(tensor_z_b_0, computation::root);
    tensor_z_b_0.before(tensor_z_b_1, computation::root);
    tensor_z_b_1.before(tensor_z_b_2, computation::root);
    tensor_z_b_2.before(reduce_z_0_init_along_dim_2, computation::root);
    reduce_z_0_init_along_dim_2.before(reduce_z_1_init_along_dim_1, computation::root);
    reduce_z_1_init_along_dim_1.before(reduce_z_2_init_along_dim_0, computation::root);
    reduce_z_2_init_along_dim_0.before(reduce_z_0_along_dim_2, computation::root);
    reduce_z_0_along_dim_2.before(reduce_z_1_along_dim_1, computation::root);
    reduce_z_1_along_dim_1.before(reduce_z_2_along_dim_0, computation::root);
    reduce_z_2_along_dim_0.before(u_update_init, computation::root);
    u_update_init.before(u_update_z_1, computation::root);
    u_update_z_1.before(u_update_z_2, computation::root);    
    */    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buff_D("buff_D", {N, N}, p_float32, a_input, &function0);
    tiramisu::buffer buff_G("buff_G", {3, 3, N, N, N}, p_float32, a_input, &function0);
    tiramisu::buffer buff_u("buff_u", {N, N, N}, p_float32, a_input, &function0);
    tiramisu::buffer buff_D_reduce_along_dim_0("buff_D_reduce_along_dim_0", {N}, p_float32, a_temporary, &function0);
    tiramisu::buffer buff_D_reduce_along_dim_1("buff_D_reduce_along_dim_1", {N}, p_float32, a_temporary, &function0);
    tiramisu::buffer buff_u_reduce_along_dim_0("buff_u_reduce_along_dim_0", {N, N}, p_float32, a_temporary, &function0);
    tiramisu::buffer buff_u_reduce_along_dim_1("buff_u_reduce_along_dim_1", {N, N}, p_float32, a_temporary, &function0);
    tiramisu::buffer buff_u_reduce_along_dim_2("buff_u_reduce_along_dim_2", {N, N}, p_float32, a_temporary, &function0);
    tiramisu::buffer buff_tensor_w("buff_tensor_w", {3, N, N, N}, p_float32, a_output, &function0);
    tiramisu::buffer buff_tensor_z("buff_tensor_z", {3, N, N, N}, p_float32, a_temporary, &function0);
    tiramisu::buffer buff_z_0_reduce_along_dim_2("buff_z_0_reduce_along_dim_2", {N,N}, p_float32, a_temporary, &function0);
    tiramisu::buffer buff_z_1_reduce_along_dim_1("buff_z_1_reduce_along_dim_1", {N,N}, p_float32, a_temporary, &function0);
    tiramisu::buffer buff_z_2_reduce_along_dim_0("buff_z_2_reduce_along_dim_0", {N,N}, p_float32, a_temporary, &function0);

    matrix_D.set_access("{matrix_D[i,j]->buff_D[i,j]}");
    tensor_G.set_access("{tensor_G[x,y,i,j,k]->buff_G[x,y,i,j,k]}");
    tensor_u.set_access("{tensor_u[k,i,j]->buff_u[k,i,j]}");
    reduce_D_init_along_dim_0.set_access("{reduce_D_init_along_dim_0[j]->buff_D_reduce_along_dim_0[j]}");
    reduce_D_init_along_dim_1.set_access("{reduce_D_init_along_dim_1[i]->buff_D_reduce_along_dim_1[i]}");
    reduce_D_along_dim_0.set_access("{reduce_D_along_dim_0[i,j]->buff_D_reduce_along_dim_0[j]}");
    reduce_D_along_dim_1.set_access("{reduce_D_along_dim_1[i,j]->buff_D_reduce_along_dim_1[i]}");
    reduce_u_init_along_dim_0.set_access("{reduce_u_init_along_dim_0[k,j]->buff_u_reduce_along_dim_0[k,j]}");
    reduce_u_init_along_dim_1.set_access("{reduce_u_init_along_dim_1[i,k]->buff_u_reduce_along_dim_1[i,k]}");
    reduce_u_init_along_dim_2.set_access("{reduce_u_init_along_dim_2[i,j]->buff_u_reduce_along_dim_2[i,j]}");
    reduce_u_along_dim_0.set_access("{reduce_u_along_dim_0[k,i,j]->buff_u_reduce_along_dim_0[k,j]}");
    reduce_u_along_dim_1.set_access("{reduce_u_along_dim_1[k,i,j]->buff_u_reduce_along_dim_1[i,k]}");
    reduce_u_along_dim_2.set_access("{reduce_u_along_dim_2[k,i,j]->buff_u_reduce_along_dim_2[i,j]}");
    tensor_w_0.set_access("{tensor_w_0[k,i,j]->buff_tensor_w[0,k,i,j]}");
    tensor_w_1.set_access("{tensor_w_1[k,i,j]->buff_tensor_w[1,k,i,j]}");
    tensor_w_2.set_access("{tensor_w_2[k,i,j]->buff_tensor_w[2,k,i,j]}");
    /*tensor_z_init.set_access("{tensor_z_init[a,i,j,k]->buff_tensor_z[a,i,j,k]}");
    tensor_z_b_0.set_access("{tensor_z_b_0[a,i,j,k]->buff_tensor_z[a,i,j,k]}");
    tensor_z_b_1.set_access("{tensor_z_b_1[a,i,j,k]->buff_tensor_z[a,i,j,k]}");
    tensor_z_b_2.set_access("{tensor_z_b_2[a,i,j,k]->buff_tensor_z[a,i,j,k]}");
    reduce_z_0_init_along_dim_2.set_access("{reduce_z_0_init_along_dim_2[i,j]->buff_z_0_reduce_along_dim_2[i,j]}");
    reduce_z_1_init_along_dim_1.set_access("{reduce_z_1_init_along_dim_1[i,k]->buff_z_1_reduce_along_dim_1[i,k]}");
    reduce_z_2_init_along_dim_0.set_access("{reduce_z_2_init_along_dim_0[j,k]->buff_z_2_reduce_along_dim_0[j,k]}");
    reduce_z_0_along_dim_2.set_access("{reduce_z_0_along_dim_2[i,j,k]->buff_z_0_reduce_along_dim_2[i,j]}");
    reduce_z_1_along_dim_1.set_access("{reduce_z_1_along_dim_1[i,j,k]->buff_z_0_reduce_along_dim_1[i,k]}");
    reduce_z_2_along_dim_0.set_access("{reduce_z_2_along_dim_0[i,j,k]->buff_z_0_reduce_along_dim_0[j,k]}");
    u_update_init.set_access("{u_update_init[i,j,k]->buff_u[i,j,k]}");
    u_update_z_1.set_access("{u_update_z_1[i,j,k]->buff_u[i,j,k]}");
    u_update_z_2.set_access("{u_update_z_2[i,j,k]->buff_u[i,j,k]}");*/

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&buff_D, &buff_G, &buff_u, &buff_tensor_w});
	  //	  &buff_u_reduce_along_dim_0,
	  //	  &buff_u_reduce_along_dim_1,
	  //	  &buff_u_reduce_along_dim_2});
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
