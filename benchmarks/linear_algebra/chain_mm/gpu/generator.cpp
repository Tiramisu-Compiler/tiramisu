#include <tiramisu/tiramisu.h>

#include "configuration.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("matmul");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------

    // Declare loop iterators
    var i("i", 0, N), j("j", 0, N), k("k", 0, N), l("l", 0, N);
    var i0("i0"), k0("k0"), l0("l0"), i1("i1"), k1("k1"), l1("l1");

    // declare cpu buffers.
    buffer b_A("b_A", {N, N}, DATA_PTYPE, a_input);
    buffer b_B("b_B", {N, N}, DATA_PTYPE, a_input);
    buffer b_C("b_C", {N, N}, DATA_PTYPE, a_input);
    buffer b_O("b_O", {N, N}, DATA_PTYPE, a_output);
    // Declare gpu buffers.
    buffer b_A_gpu("b_A_gpu", {N, N}, DATA_PTYPE, a_temporary);
    buffer b_B_gpu("b_B_gpu", {N, N}, DATA_PTYPE, a_temporary);
    buffer b_C_gpu("b_C_gpu", {N, N}, DATA_PTYPE, a_temporary);
    buffer b_O_gpu("b_O_gpu", {N, N}, DATA_PTYPE, a_temporary);
    // Temporary buffer to store AxB
    buffer b_T1_gpu("b_T1_gpu", {N, N}, DATA_PTYPE, a_temporary);
    b_A_gpu.tag_gpu_global();
    b_B_gpu.tag_gpu_global();
    b_C_gpu.tag_gpu_global();
    b_T1_gpu.tag_gpu_global();
    b_O_gpu.tag_gpu_global();

    // Declare wrappers
    input c_A({i, j}, DATA_PTYPE);
    input c_B({i, j}, DATA_PTYPE);
    input c_C({i, j}, DATA_PTYPE);

    computation c_T1_init({i, k}, (DATA_TYPE) 0);
    computation c_T1({i, j, k}, DATA_PTYPE);
    c_T1.set_expression(c_T1(i, j, k) + c_A(i, j) * c_B(j, k));
    // Declare wrapper for the output buffer
    computation c_O_init({i, l}, (DATA_TYPE) 0);
    computation c_O({i, k, l}, DATA_PTYPE);
    c_O.set_expression(c_O(i, k, l) + c_T1(i, 0, k) * c_C(k, l));
    // Declare host-gpu transfer computations.
    computation copy_A_to_device({}, memcpy(b_A, b_A_gpu));
    computation copy_B_to_device({}, memcpy(b_B, b_B_gpu));
    computation copy_C_to_device({}, memcpy(b_C, b_C_gpu));
    computation copy_O_to_host({}, memcpy(b_O_gpu, b_O));

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    c_T1.interchange(j, k);
    c_T1_init.gpu_tile(i, k, 16, 16, i0, k0, i1, k1);
    c_T1.gpu_tile(i, k, 16, 16, i0, k0, i1, k1);
    c_O.interchange(k, l);
    c_O_init.gpu_tile(i, l, 16, 16, i0, l0, i1, l1);
    c_O.gpu_tile(i, l, 16, 16, i0, l0, i1, l1);

    // Scheduling commands

    copy_A_to_device.then(copy_B_to_device, computation::root)
                    .then(copy_C_to_device, computation::root)
                    .then(c_T1_init, computation::root)
                    .then(c_T1, k1)
                    .then(c_O_init, computation::root)
                    .then(c_O, l1)
                    .then(copy_O_to_host, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    c_A.store_in(&b_A_gpu);
    c_B.store_in(&b_B_gpu);
    c_C.store_in(&b_C_gpu);
    c_T1_init.store_in(&b_T1_gpu);
    c_T1.store_in(&b_T1_gpu, {i, k});
    c_O_init.store_in(&b_O_gpu);
    c_O.store_in(&b_O_gpu, {i, l});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files. Last argument triggers cuda compilation.
    tiramisu::codegen({&b_A, &b_B, &b_C, &b_O}, "fct.o", true);
    
    return 0;
}
