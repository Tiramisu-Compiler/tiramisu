#include <tiramisu/tiramisu.h>

#define SIZE0 1024
#define BLOCK 32
using namespace tiramisu;

int main(int argc, char **argv)
{
    // Set default tiramisu options.
    tiramisu::init();

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    /*
     * Declare a function matmul.
     */
    function matmul("matmul");

    constant p0("N", expr((int32_t) SIZE0), p_int32, true, NULL, 0, &matmul);
    constant p1("B", expr((int32_t) BLOCK), p_int32, true, NULL, 0, &matmul);

    // Declare loop iterators
    var i("i"), j("j"), k("k"), i0("i0"), i1("i1"), j0("j0"), j1("j1"), k0("k0"), k1("k1");

    // Declare cpu buffers.
    buffer b_A("b_A", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input, &matmul);
    buffer b_B("b_B", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_input, &matmul);
    buffer b_C("b_C", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_output, &matmul);
    // Declare gpu buffers.
    buffer b_A_gpu("b_A_gpu", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_temporary, &matmul);
    buffer b_B_gpu("b_B_gpu", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_temporary, &matmul);
    buffer b_C_gpu("b_C_gpu", {expr(SIZE0), expr(SIZE0)}, p_uint8, a_temporary, &matmul);
    buffer b_A_gpu_tile("b_A_gpu_tile", {expr(BLOCK), expr(BLOCK)}, p_uint8, a_temporary, &matmul);
    buffer b_B_gpu_tile("b_B_gpu_tile", {expr(BLOCK), expr(BLOCK)}, p_uint8, a_temporary, &matmul);
    b_A_gpu.tag_gpu_global();
    b_B_gpu.tag_gpu_global();
    b_C_gpu.tag_gpu_global();
    b_A_gpu_tile.tag_gpu_shared();
    b_B_gpu_tile.tag_gpu_shared();

    // Declare computations that represents the input buffers (b_A and b_B)
    computation c_A("[N]->{c_A[i,j,k0]: 0<=i<N and 0<=j<N and 0<=k0<(N-1)/32 + 1}", expr(), false, p_uint8, &matmul);
    computation c_B("[N]->{c_B[i,j,k0]: 0<=i<N and 0<=j<N and 0<=k0<(N-1)/32 + 1}", expr(), false, p_uint8, &matmul);

    computation c_A_tile("[N]->{c_A_tile[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint8, &matmul);
    computation c_A_tile_dec("[N]->{c_A_tile_dec[i,j]: 0<=i<N and 0<=j<N}", expr(o_allocate, b_A_gpu_tile.get_name()), true, p_none, &matmul);
    computation c_A_tile_init("[N]->{c_A_tile_init[i,j,k0]: 0<=i<N and 0<=j<N and 0<=k0<(N - 1)/32 + 1}", c_A(i, j, k0), true, p_uint8, &matmul);
    computation c_B_tile("[N]->{c_B_tile[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint8, &matmul);
    computation c_B_tile_dec("[N]->{c_B_tile_dec[i,j]: 0<=i<N and 0<=j<N}", expr(o_allocate, b_B_gpu_tile.get_name()), true, p_none, &matmul);
    computation c_B_tile_init("[N]->{c_B_tile_init[i,j,k0]: 0<=i<N and 0<=j<N and 0<=k0<(N - 1)/32 + 1}", c_B(i, j, k0), true, p_uint8, &matmul);
    computation sync1("[N]->{sync1[i,j,k0]: 0<=i<N and 0<=j<N and 0<=k0<(N-1)/32 + 1}", tiramisu::sync(), true, p_none, &matmul);
    computation sync2("[N]->{sync2[i,j,k0]: 0<=i<N and 0<=j<N and 0<=k0<(N-1)/32 + 1}", tiramisu::sync(), true, p_none, &matmul);

    // Declare a computation to initialize the reduction c[i,j]
    computation C_init("[N]->{C_init[i,j,-1]: 0<=i<N and 0<=j<N}", expr((uint8_t) 0), true, p_uint8, &matmul);
    
    // Declare the reduction operation.
    computation c_C("[N]->{c_C[i,j,k]: 0<=i<N and 0<=j<N and 0<=k<N}", expr(), true, p_uint8, &matmul);
    // Note that the previous computation has an empty expression (because we can only use c_C in an expression after its declaration)
    c_C.set_expression(c_C(i, j, k - 1) + c_A_tile(i, j) * c_B_tile(k, j));

    // Declare host-gpu transfer computations.
    computation copy_A_to_device("{copy_A_to_device[0]}", memcpy(b_A, b_A_gpu), true, p_none, &matmul);
    computation copy_B_to_device("{copy_B_to_device[0]}", memcpy(b_B, b_B_gpu), true, p_none, &matmul);
    computation copy_C_to_device("{copy_C_to_device[0]}", memcpy(b_C, b_C_gpu), true, p_none, &matmul);
    computation copy_C_to_host("{copy_C_to_host[0]}", memcpy(b_C_gpu, b_C), true, p_none, &matmul);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Scheduling commands

    C_init.gpu_tile(i, j, BLOCK, BLOCK);
    c_C.split(k, BLOCK, k0, k1);
    c_C.tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_C.tag_gpu_level(i0, j0, i1, j1);
    c_A_tile_dec.tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_A_tile_dec.tag_gpu_level(i0, j0, i1, j1);
    c_A_tile_init.tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_A_tile_init.tag_gpu_level(i0, j0, i1, j1);
    c_B_tile_dec.tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_B_tile_dec.tag_gpu_level(i0, j0, i1, j1);
    c_B_tile_init.tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    c_B_tile_init.tag_gpu_level(i0, j0, i1, j1);
    sync1.tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);
    sync2.tile(i, j, BLOCK, BLOCK, i0, j0, i1, j1);


    copy_B_to_device.after(copy_A_to_device, computation::root);
    copy_C_to_device.after(copy_B_to_device, computation::root);
    C_init.after(copy_C_to_device, computation::root);
    c_A_tile_dec.after(C_init, computation::root);
    c_B_tile_dec.after(c_A_tile_dec, j1);
    c_A_tile_init.after(c_B_tile_dec, j1);
    c_B_tile_init.after(c_A_tile_init, k0);
    /// c_B_tile.after(c_A_tile, computation::root);
    sync1.after(c_B_tile_init, k0);
    c_C.after(sync1, k0);
    sync2.after(c_C, k0);
    copy_C_to_host.after(sync2, computation::root);


    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    // Map the computations to a buffer.
    c_A.set_access("{c_A[i,j,k0] -> b_A_gpu[i,k0*32+j%32]}");
    c_B.set_access("{c_B[i,j,k0] -> b_B_gpu[k0*32+i%32,j]}");
    c_A_tile.set_access("{c_A_tile[i, j] -> b_A_gpu_tile[i % 32, j % 32]}");
    c_A_tile_init.set_access("{c_A_tile_init[i, j, k0] -> b_A_gpu_tile[i % 32, j % 32]}");
    c_B_tile.set_access("{c_B_tile[i, j] -> b_B_gpu_tile[i % 32, j % 32]}");
    c_B_tile_init.set_access("{c_B_tile_init[i, j, k0] -> b_B_gpu_tile[i % 32, j % 32]}");
    /// c_B_tile.store_in(&b_B_gpu_tile, {i, j});

    // Store C_init[i,j,k] in b_C[i,j]
    C_init.store_in(&b_C_gpu, {i,j});
    // Store c_C[i,j,k] in b_C[i,j]
    c_C.store_in(&b_C_gpu, {i,j});
    // Note that both of the computations C_init and c_C store their
    // results in the buffer b_C.

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files. Last argument triggers cuda compilation.
    matmul.codegen({&b_A, &b_B, &b_C}, "fct.o", true);
    
    // Dump the generated Halide statement (just for debugging).
    matmul.dump_halide_stmt();

    return 0;
}
