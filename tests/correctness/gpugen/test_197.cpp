#include <tiramisu/tiramisu.h>
#include <tiramisu/block.h>
#include "test_197_defs.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init("test_197");

    // generated code:
    // {
    //     for (int t = 0; t < T; ++t)
    //     {
    //         allocate A_gpu
    //         copy from A (host) to A_gpu (device)
    //         initialize A_gpu to 1
    //         copy from A_gpu (device) to A (host)
    //         deallocate A_gpu
    //     }
    //     for (int t = 0; t < T; ++t)
    //     {
    //         allocate B_gpu
    //         copy from B (host) to B_gpu (device)
    //         initialize B_gpu to 2
    //         copy from B_gpu (device) to B (host)
    //         deallocate B_gpu
    //     }
    // }
    var t( "t", 0, T_size );
    var A_iter1( "A_iter1", 0, A_size );
    var A_iter2( "A_iter2", 0, A_size );

    var B_iter1( "B_iter1", 0, B_size );
    var B_iter2( "B_iter2", 0, B_size );

    input A( "A", { t, A_iter1, A_iter2 }, p_int32 );
    input B( "B", { t, B_iter1, B_iter2 }, p_int32 );

    buffer b_A_gpu( "b_A_gpu", { T_size, A_size, A_size }, p_int32, a_temporary );
    b_A_gpu.tag_gpu_global();

    buffer b_B_gpu( "b_B_gpu", { T_size, B_size, B_size }, p_int32, a_temporary );
    b_B_gpu.tag_gpu_global();

    computation init_A( "init_A", { t, A_iter1, A_iter2 }, expr( 1 ) );
    init_A.store_in( &b_A_gpu, { t, A_iter1, A_iter2 } );
    init_A.tag_gpu_level( A_iter1, A_iter2 );

    computation init_B( "init_B", { t, B_iter1, B_iter2 }, expr( 2 ) );
    init_B.store_in( &b_B_gpu, { t, B_iter1, B_iter2 } );
    init_B.tag_gpu_level( B_iter1, B_iter2 );

    computation copy_A_host_to_device( {t}, memcpy( *A.get_buffer(), b_A_gpu ) );
    computation copy_B_host_to_device( {t}, memcpy( *B.get_buffer(), b_B_gpu ) );

    computation copy_A_device_to_host( {t}, memcpy( b_A_gpu, *A.get_buffer() ) );
    computation copy_B_device_to_host( {t}, memcpy( b_B_gpu, *B.get_buffer() ) );

    tiramisu::computation *allocate_A = b_A_gpu.allocate_at( copy_A_host_to_device, t );
    tiramisu::computation *deallocate_A = b_A_gpu.deallocate_at( copy_A_device_to_host, t );

    tiramisu::computation *allocate_B = b_B_gpu.allocate_at( copy_B_host_to_device, t );
    tiramisu::computation *deallocate_B = b_B_gpu.deallocate_at( copy_B_device_to_host, t );

    allocate_A->then( copy_A_host_to_device, t )
                .then( init_A, t )
                .then( copy_A_device_to_host, t )
                .then( *deallocate_A, t )
                .then( *allocate_B, computation::root )
                .then( copy_B_host_to_device, t )
                .then( init_B, t )
                .then( copy_B_device_to_host, t )
                .then( *deallocate_B, t );

    tiramisu::codegen({ A.get_buffer(), B.get_buffer() }, "generated_fct_test_197.o", true);
    return 0;
}
