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
    var i("i"), j("j"), k("k"), x("x"), a("a"), b("b");
    constant cN("cN", (int64_t)N, p_int64, true, NULL, 0, &function0);
    computation matrix_D("[cN]->{matrix_D[i,j]: 0<=i<cN and 0<=j<cN}",
			 expr(), false, p_float64, &function0);
    computation tensor_G("[cN]->{tensor_G[a,b,i,j,k]: 0<=a<3 and 0<=b<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			 expr(), false, p_float64, &function0);
    computation tensor_u("[cN]->{tensor_u[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			 expr(), false, p_float64, &function0);
    // initialize reductions
    computation reduce_D_init_along_dim_i("[cN]->{reduce_D_init_along_dim_i[i]: 0<=i<cN}", // reduction down columns
					  expr(0.0f), true, p_float64, &function0);
    computation reduce_D_init_along_dim_j("[cN]->{reduce_D_init_along_dim_j[j]: 0<=j<cN}", // reduction across rows
					  expr(0.0f), true, p_float64, &function0);
    computation reduce_D_along_dim_i("[cN]->{reduce_D_along_dim_i[i,j]: 0<=i<cN and 0<=j<cN}", // reduction down columns
				     reduce_D_init_along_dim_i(i) + matrix_D(i,j), true, p_float64, &function0);
    computation reduce_D_along_dim_j("[cN]->{reduce_D_along_dim_j[i,j]: 0<=i<cN and 0<=j<cN}", // reduction across rows
				     reduce_D_init_along_dim_j(j) + matrix_D(i,j), true, p_float64, &function0);
    computation reduce_u_init_along_dim_i("[cN]->{reduce_u_init_along_dim_i[j,k]: 0<=j<cN and 0<=k<cN}",
					  expr(0.0f), true, p_float64, &function0);
    computation reduce_u_init_along_dim_j("[cN]->{reduce_u_init_along_dim_j[i,k]: 0<=i<cN and 0<=k<cN}",
					  expr(0.0f), true, p_float64, &function0);
    computation reduce_u_init_along_dim_k("[cN]->{reduce_u_init_along_dim_k[i,j]: 0<=i<cN and 0<=j<cN}",
					  expr(0.0f), true, p_float64, &function0);
    computation reduce_u_along_dim_i("[cN]->{reduce_u_along_dim_i[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				     reduce_u_init_along_dim_i(j,k) + tensor_u(i,j,k), true, p_float64, &function0);
    computation reduce_u_along_dim_j("[cN]->{reduce_u_along_dim_j[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				     reduce_u_init_along_dim_j(i,k) + tensor_u(i,j,k), true, p_float64, &function0);
    computation reduce_u_along_dim_k("[cN]->{reduce_u_along_dim_k[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				     reduce_u_init_along_dim_k(i,j) + tensor_u(i,j,k), true, p_float64, &function0);
    computation tensor_w_0("[cN]->{tensor_w_0[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			   reduce_D_init_along_dim_i(k)/* * reduce_u_along_dim_k(i,j,0)*/, true, p_float64, &function0);
    computation tensor_w_1("[cN]->{tensor_w_1[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			   reduce_D_along_dim_j(j,0)/* * reduce_u_along_dim_j(i,0,k)*/, true, p_float64, &function0);
    computation tensor_w_2("[cN]->{tensor_w_2[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			   reduce_D_along_dim_j(i,0)/* * reduce_u_along_dim_i(0,j,k)*/, true, p_float64, &function0);    
    computation tensor_z_init("[cN]->{tensor_z_init[a,i,j,k]: 0<=a<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			      expr(0.0f), true, p_float64, &function0);
    // wrapper for w
    computation tensor_w("[cN]->{tensor_w[a,i,j,k]: 0<=a<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			 expr(), false, p_float64, &function0);
    computation tensor_z("[cN]->{tensor_z[a,b,i,j,k]: 0<=a<3 and 0<=b<3 and 0<=i<cN and 0<=j<cN and 0<=k<cN}",
			 tensor_z_init(a,i,j,k) + tensor_G(a,b,i,j,k)*tensor_w(b,i,j,k),
			 true, p_float64, &function0);
    computation tensor_z_init_reduce_along_dim_i("[cN]->{tensor_z_init_reduce_along_dim_i[j,k]: 0<=j<cN and 0<=k<cN}",
						 expr(0.0f), true, p_float64, &function0);
    computation tensor_z_init_reduce_along_dim_j("[cN]->{tensor_z_init_reduce_along_dim_j[i,k]: 0<=i<cN and 0<=k<cN}",
						 expr(0.0f), true, p_float64, &function0);
    computation tensor_z_init_reduce_along_dim_k("[cN]->{tensor_z_init_reduce_along_dim_k[i,j]: 0<=i<cN and 0<=j<cN}",
						 expr(0.0f), true, p_float64, &function0);
    computation tensor_z_reduce_along_dim_i("[cN]->{tensor_z_reduce_along_dim_i[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
					    tensor_z_init_reduce_along_dim_i(j,k) + tensor_z(2,0,i,j,k), true, p_float64, &function0);
    computation tensor_z_reduce_along_dim_j("[cN]->{tensor_z_reduce_along_dim_j[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
					    tensor_z_init_reduce_along_dim_j(i,k) + tensor_z(1,0,i,j,k), true, p_float64, &function0);
    computation tensor_z_reduce_along_dim_k("[cN]->{tensor_z_reduce_along_dim_k[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
					    tensor_z_init_reduce_along_dim_k(i,j) + tensor_z(0,0,i,j,k), true, p_float64, &function0);
    computation tensor_u_update_init("[cN]->{tensor_u_update_init[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				  reduce_D_along_dim_i(0,k)*tensor_z_reduce_along_dim_k(i,j,0), true, p_float64, &function0);
    computation tensor_u_update_0("[cN]->{tensor_u_update_0[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				  reduce_D_along_dim_i(0,j)*tensor_z_reduce_along_dim_j(i,0,k) +
				  tensor_u_update_init(i,j,k), true, p_float64, &function0);
    computation tensor_u_update_1("[cN]->{tensor_u_update_1[i,j,k]: 0<=i<cN and 0<=j<cN and 0<=k<cN}",
				  reduce_D_along_dim_i(0,i)*tensor_z_reduce_along_dim_i(i,j,0) +
				  tensor_u_update_init(i,j,k), true, p_float64, &function0);
    
    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    reduce_D_init_along_dim_i.before(reduce_D_init_along_dim_j, computation::root);
    reduce_D_init_along_dim_j.before(reduce_D_along_dim_i, computation::root);
    reduce_D_along_dim_i.before(reduce_D_along_dim_j, computation::root);
    reduce_D_along_dim_j.before(reduce_u_init_along_dim_i, computation::root);
    reduce_u_init_along_dim_i.before(reduce_u_init_along_dim_j, computation::root);
    reduce_u_init_along_dim_j.before(reduce_u_init_along_dim_k, computation::root);
    reduce_u_init_along_dim_k.before(reduce_u_along_dim_i, computation::root);
    reduce_u_along_dim_i.before(reduce_u_along_dim_j, computation::root);
    reduce_u_along_dim_j.before(reduce_u_along_dim_k, computation::root);
    reduce_u_along_dim_k.before(tensor_w_0, computation::root);
    tensor_w_0.before(tensor_w_1, computation::root);    
    tensor_w_1.before(tensor_w_2, computation::root);
    tensor_w_2.before(tensor_z_init, computation::root);
    tensor_z_init.before(tensor_z, computation::root);
    tensor_z.before(tensor_z_init_reduce_along_dim_i, computation::root);
    tensor_z_init_reduce_along_dim_i.before(tensor_z_init_reduce_along_dim_j, computation::root);
    tensor_z_init_reduce_along_dim_j.before(tensor_z_init_reduce_along_dim_k, computation::root);
    tensor_z_init_reduce_along_dim_k.before(tensor_z_reduce_along_dim_i, computation::root);
    tensor_z_reduce_along_dim_i.before(tensor_z_reduce_along_dim_j, computation::root);
    tensor_z_reduce_along_dim_j.before(tensor_z_reduce_along_dim_k, computation::root);
    tensor_z_reduce_along_dim_k.before(tensor_u_update_init, computation::root);
    tensor_u_update_init.before(tensor_u_update_0, computation::root);
    tensor_u_update_0.before(tensor_u_update_1, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    tiramisu::buffer buff_D("buff_D", {N, N}, p_float64, a_input, &function0);
    tiramisu::buffer buff_G("buff_G", {3, 3, N, N, N}, p_float64, a_input, &function0);
    tiramisu::buffer buff_u("buff_u", {N, N, N}, p_float64, a_output, &function0);
    tiramisu::buffer buff_D_reduce_along_dim_i("buff_D_reduce_along_dim_i", {N}, p_float64, a_temporary, &function0);
    tiramisu::buffer buff_D_reduce_along_dim_j("buff_D_reduce_along_dim_j", {N}, p_float64, a_temporary, &function0);
    tiramisu::buffer buff_u_reduce_along_dim_i("buff_u_reduce_along_dim_i", {N, N}, p_float64, a_temporary, &function0);
    tiramisu::buffer buff_u_reduce_along_dim_j("buff_u_reduce_along_dim_j", {N, N}, p_float64, a_temporary, &function0);
    tiramisu::buffer buff_u_reduce_along_dim_k("buff_u_reduce_along_dim_k", {N, N}, p_float64, a_temporary, &function0);
    tiramisu::buffer buff_tensor_w("buff_tensor_w", {3, N, N, N}, p_float64, a_output, &function0);
    tiramisu::buffer buff_tensor_z("buff_tensor_z", {3, N, N, N}, p_float64, a_temporary, &function0);
    tiramisu::buffer buff_z_0_reduce_along_dim_k("buff_z_0_reduce_along_dim_k", {N,N}, p_float64, a_temporary, &function0);
    tiramisu::buffer buff_z_1_reduce_along_dim_j("buff_z_1_reduce_along_dim_j", {N,N}, p_float64, a_temporary, &function0);
    tiramisu::buffer buff_z_2_reduce_along_dim_i("buff_z_2_reduce_along_dim_i", {N,N}, p_float64, a_temporary, &function0);

    matrix_D.set_access("{matrix_D[i,j]->buff_D[i,j]}");
    tensor_G.set_access("{tensor_G[a,b,i,j,k]->buff_G[a,b,i,j,k]}");
    tensor_u.set_access("{tensor_u[i,j,k]->buff_u[i,j,k]}");
    reduce_D_init_along_dim_i.set_access("{reduce_D_init_along_dim_i[i]->buff_D_reduce_along_dim_i[i]}");
    reduce_D_init_along_dim_j.set_access("{reduce_D_init_along_dim_j[j]->buff_D_reduce_along_dim_j[j]}");
    reduce_D_along_dim_i.set_access("{reduce_D_along_dim_i[i,j]->buff_D_reduce_along_dim_i[i]}");
    reduce_D_along_dim_j.set_access("{reduce_D_along_dim_j[i,j]->buff_D_reduce_along_dim_j[j]}");
    reduce_u_init_along_dim_i.set_access("{reduce_u_init_along_dim_i[k,j]->buff_u_reduce_along_dim_i[k,j]}");
    reduce_u_init_along_dim_j.set_access("{reduce_u_init_along_dim_j[i,k]->buff_u_reduce_along_dim_j[i,k]}");
    reduce_u_init_along_dim_k.set_access("{reduce_u_init_along_dim_k[i,j]->buff_u_reduce_along_dim_k[i,j]}");
    reduce_u_along_dim_i.set_access("{reduce_u_along_dim_i[i,j,k]->buff_u_reduce_along_dim_i[j,k]}");
    reduce_u_along_dim_j.set_access("{reduce_u_along_dim_j[i,j,k]->buff_u_reduce_along_dim_j[i,k]}");
    reduce_u_along_dim_k.set_access("{reduce_u_along_dim_k[i,j,k]->buff_u_reduce_along_dim_k[i,j]}");
    tensor_w_0.set_access("{tensor_w_0[i,j,k]->buff_tensor_w[0,i,j,k]}");
    tensor_w_1.set_access("{tensor_w_1[i,j,k]->buff_tensor_w[1,i,j,k]}");
    tensor_w_2.set_access("{tensor_w_2[i,j,k]->buff_tensor_w[2,i,j,k]}");
    tensor_w.set_access("{tensor_w[a,i,j,k]->buff_tensor_w[a,i,j,k]}");
    tensor_z_init.set_access("{tensor_z_init[a,i,j,k]->buff_tensor_z[a,i,j,k]}");
    tensor_z.set_access("{tensor_z[a,b,i,j,k]->buff_tensor_z[a,i,j,k]}");
    tensor_z_init_reduce_along_dim_i.set_access("{tensor_z_init_reduce_along_dim_i[j,k]->buff_z_2_reduce_along_dim_i[j,k]}");
    tensor_z_init_reduce_along_dim_j.set_access("{tensor_z_init_reduce_along_dim_j[i,k]->buff_z_1_reduce_along_dim_j[i,k]}");
    tensor_z_init_reduce_along_dim_k.set_access("{tensor_z_init_reduce_along_dim_k[i,j]->buff_z_0_reduce_along_dim_k[i,j]}");
    tensor_z_reduce_along_dim_i.set_access("{tensor_z_reduce_along_dim_i[i,j,k]->buff_z_2_reduce_along_dim_i[j,k]}");
    tensor_z_reduce_along_dim_j.set_access("{tensor_z_reduce_along_dim_j[i,j,k]->buff_z_1_reduce_along_dim_j[i,k]}");
    tensor_z_reduce_along_dim_k.set_access("{tensor_z_reduce_along_dim_k[i,j,k]->buff_z_0_reduce_along_dim_k[i,j]}");
    tensor_u_update_init.set_access("{tensor_u_update_init[i,j,k]->buff_u[i,j,k]}");
    tensor_u_update_0.set_access("{tensor_u_update_0[i,j,k]->buff_u[i,j,k]}");
    tensor_u_update_1.set_access("{tensor_u_update_1[i,j,k]->buff_u[i,j,k]}");
    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    function0.set_arguments({&buff_D, &buff_G, &buff_u, &buff_tensor_w});
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
