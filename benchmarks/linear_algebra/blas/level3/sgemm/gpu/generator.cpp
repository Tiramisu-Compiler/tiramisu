#include <tiramisu/tiramisu.h>

#include "configuration.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init(); // Set default tiramisu options.

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    function matmul("matmul");

    // Declare constants
    constant p0("N", expr((int32_t) N), p_int32, true, NULL, 0, &matmul);
    constant p1("M", expr((int32_t) M), p_int32, true, NULL, 0, &matmul);
    constant p2("K", expr((int32_t) K), p_int32, true, NULL, 0, &matmul);
    constant p3("B", expr((int32_t) BLOCK), p_int32, true, NULL, 0, &matmul);

    // Declare loop iterators
    var i("i"), j("j"), k("k"), i0("i0"), i1("i1"), j0("j0"), j1("j1"), k0("k0"), k1("k1");

    // Declare cpu buffers.
    buffer b_A("b_A", {expr(M), expr(K)}, DATA_PTYPE, a_input, &matmul);
    buffer b_B("b_B", {expr(K), expr(N)}, DATA_PTYPE, a_input, &matmul);
    buffer b_C("b_C", {expr(M), expr(N)}, DATA_PTYPE, a_output, &matmul);
    // Declare gpu buffers.
    buffer b_A_gpu("b_A_gpu", {M, K}, DATA_PTYPE, a_temporary, &matmul);
    buffer b_B_gpu("b_B_gpu", {K, N}, DATA_PTYPE, a_temporary, &matmul);
    buffer b_C_gpu("b_C_gpu", {M, N}, DATA_PTYPE, a_temporary, &matmul);
    buffer b_A_gpu_tile("b_A_gpu_tile", {expr(BLOCK), expr(BLOCK)}, DATA_PTYPE, a_temporary, &matmul);
    buffer b_B_gpu_tile("b_B_gpu_tile", {expr(BLOCK), expr(BLOCK)}, DATA_PTYPE, a_temporary, &matmul);
    buffer b_acc("b_acc", {1}, DATA_PTYPE, a_temporary, &matmul);
    b_A_gpu.tag_gpu_global();
    b_B_gpu.tag_gpu_global();
    b_C_gpu.tag_gpu_global();
    b_A_gpu_tile.tag_gpu_shared();
    b_B_gpu_tile.tag_gpu_shared();
    b_acc.tag_gpu_register();

    // Declare wrappers for input buffers
    computation c_A("[M,N,K]->{c_A[i,j,k0]: 0<=i<M and 0<=j<K and 0<=k0<(K-1)/" BLOCK_STR " + 1}", expr(), false, DATA_PTYPE, &matmul);
    computation c_B("[M,N,K]->{c_B[i,j,k0]: 0<=i<K and 0<=j<N and 0<=k0<(K-1)/" BLOCK_STR " + 1}", expr(), false, DATA_PTYPE, &matmul);
    // Declare wrappers for shared arrays
    computation c_A_tile("[M,K]->{c_A_tile[i,j]: 0<=i<M and 0<=j<K}", expr(), false, DATA_PTYPE, &matmul);
    computation c_A_tile_dec("[M,N]->{c_A_tile_dec[i,j]: 0<=i<M and 0<=j<N}", allocate(b_A_gpu_tile), true, p_none, &matmul);
    computation c_A_tile_init("[M,N,K]->{c_A_tile_init[i,j,k0]: 0<=i<M and 0<=j<N and 0<=k0<(K - 1)/" BLOCK_STR " + 1}", c_A(i, j, k0), true, DATA_PTYPE, &matmul);
    computation c_B_tile("[K,N]->{c_B_tile[i,j]: 0<=i<K and 0<=j<N}", expr(), false, DATA_PTYPE, &matmul);
    computation c_B_tile_dec("[M,N]->{c_B_tile_dec[i,j]: 0<=i<M and 0<=j<N}", allocate(b_B_gpu_tile), true, p_none, &matmul);
    computation c_B_tile_init("[M,N,K]->{c_B_tile_init[i,j,k0]: 0<=i<M and 0<=j<N and 0<=k0<(K - 1)/" BLOCK_STR " + 1}", c_B(i, j, k0), true, DATA_PTYPE, &matmul);
    // Declare synchronizer computations
    computation sync1("[M,N,K]->{sync1[i,j,k0]: 0<=i<M and 0<=j<N and 0<=k0<(K-1)/" BLOCK_STR " + 1}", tiramisu::sync(), true, p_none, &matmul);
    computation sync2("[M,N,K]->{sync2[i,j,k0]: 0<=i<M and 0<=j<N and 0<=k0<(K-1)/" BLOCK_STR " + 1}", tiramisu::sync(), true, p_none, &matmul);
    // Declare wrapper computation for accumulator
    computation c_acc_dec("[M,N]->{c_acc_dec[i,j]: 0<=i<M and 0<=j<N}", allocate(b_acc), true, p_none, &matmul);
    computation c_acc_init("[M,N]->{c_acc_init[i,j]: 0<=i<M and 0<=j<N}", (DATA_TYPE) 0, true, DATA_PTYPE, &matmul);
    computation c_acc("[M,N,K]->{c_acc[i,j,k]: 0<=i<M and 0<=j<N and 0<=k<K}", expr(), true, DATA_PTYPE, &matmul);
    c_acc.set_expression(c_acc(i, j, k - 1) + c_A_tile(i, k) * c_B_tile(k, j));
    // Declare wrapper for the output buffer
    computation c_C("[M,N]->{c_C[i,j]: 0<=i<M and 0<=j<N}", expr(), true, DATA_PTYPE, &matmul);
    c_C.set_expression(c_acc(i, j, 0) * alpha + c_C(i, j) * beta);
    // Declare host-gpu transfer computations.
    computation copy_A_to_device("{copy_A_to_device[0]}", memcpy(b_A, b_A_gpu), true, p_none, &matmul);
    computation copy_B_to_device("{copy_B_to_device[0]}", memcpy(b_B, b_B_gpu), true, p_none, &matmul);
    computation copy_C_to_device("{copy_C_to_device[0]}", memcpy(b_C, b_C_gpu), true, p_none, &matmul);
    computation copy_C_to_host("{copy_C_to_host[0]}", memcpy(b_C_gpu, b_C), true, p_none, &matmul);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Scheduling commands

    c_acc.split(k, BLOCK, k0, k1);
    c_C.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_A_tile_dec.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_A_tile_init.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_B_tile_dec.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_B_tile_init.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    sync1.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    sync2.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_acc_dec.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_acc_init.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_acc.gpu_tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);

    copy_B_to_device.after(copy_A_to_device, computation::root);
    copy_C_to_device.after(copy_B_to_device, computation::root);
    c_A_tile_dec.after(copy_C_to_device, computation::root);
    c_B_tile_dec.after(c_A_tile_dec, j1);
    c_acc_dec.after(c_B_tile_dec, j1);
    c_acc_init.after(c_acc_dec, j1);
    c_A_tile_init.after(c_acc_init, j1);
    c_B_tile_init.after(c_A_tile_init, k0);
    sync1.after(c_B_tile_init, k0);
    c_acc.after(sync1, k0);
    sync2.after(c_acc, k0);
    c_C.after(sync2, j1);
    copy_C_to_host.after(c_C, computation::root);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    c_A.set_access("{c_A[i,j,k0] -> b_A_gpu[i,k0 * " BLOCK_STR " + j % " BLOCK_STR "]}");
    c_B.set_access("{c_B[i,j,k0] -> b_B_gpu[k0 * " BLOCK_STR " + i % " BLOCK_STR ",j]}");
    c_A_tile.set_access("{c_A_tile[i, j] -> b_A_gpu_tile[i % " BLOCK_STR ", j % " BLOCK_STR "]}");
    c_A_tile_init.set_access("{c_A_tile_init[i, j, k0] -> b_A_gpu_tile[i % " BLOCK_STR ", j % " BLOCK_STR "]}");
    // Note the transpose
    c_B_tile.set_access("{c_B_tile[i, j] -> b_B_gpu_tile[j % " BLOCK_STR ", i % " BLOCK_STR "]}");
    c_B_tile_init.set_access("{c_B_tile_init[i, j, k0] -> b_B_gpu_tile[j % " BLOCK_STR ", i % " BLOCK_STR "]}");

    c_acc_init.set_access("{c_acc_init[i, j] -> b_acc[0]}");
    c_acc.set_access("{c_acc[i, j, k] -> b_acc[0]}");

    c_C.store_in(&b_C_gpu);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files. Last argument triggers cuda compilation.
    matmul.codegen({&b_A, &b_B, &b_C}, "fct.o", true);
    
    // Dump the generated Halide statement (just for debugging).
    matmul.dump_halide_stmt();

    return 0;
}
