#include "Halide.h"
#include "configure.h"

#define SCHEDULE_CPU 1

using namespace Halide;

int main(int argc, char **argv)
{
    ImageParam            input{Float(32), 4, "input"};
    ImageParam            filter{Float(32), 4, "filter"};
    ImageParam            bias{Float(32), 1, "bias"};
    ImageParam            filter2{Float(32), 4, "filter2"};
    ImageParam            bias2{Float(32), 1, "bias2"};

    Var x("x"), y("y"), z("z"), n("n");

    Func f_conv("conv"), f_conv2("conv2");
    RDom r(0, K, 0, K, 0, FIn);
    RDom r2(0, K, 0, K, 0, FIn);

    f_conv(x, y, z, n) = bias(z);
    f_conv(x, y, z, n) += filter(r.x, r.y, r.z, z) * input(x + r.x, y + r.y, r.z, n);
    f_conv2(x, y, z, n) = bias2(z);
    f_conv2(x, y, z, n) += filter2(r2.x, r2.y, r2.z, z) * f_conv(x + r2.x, y + r2.y, r2.z, n);

    /* THE SCHEDULE */
    if (SCHEDULE_CPU)
    {
	if (LARGE_DATA_SET)
	{
	    // Blocking spatially with vectorization
	    Var z_t("z_t"), y_t("y_t"), par("par");
	    int vec_len = 128;
	    int o_block_size = 8;
	    int y_block = 32;
	    f_conv.compute_root();
	    f_conv.fuse(z, n, par).parallel(par);
	    f_conv.update().reorder(x, y, r.z);
	    f_conv.update().split(y, y, y_t, y_block);
	    f_conv.update().split(z, z, z_t, o_block_size);
	    f_conv.update().reorder(y_t, z_t, y, r.z, z);
	    f_conv.update().vectorize(x, vec_len);
	    f_conv.update().unroll(r.x);
	    f_conv.update().unroll(r.y);
	    f_conv.update().fuse(z, n, par).parallel(par);

	    f_conv2.compute_root();
	    f_conv2.parallel(n);
	    f_conv2.update().reorder(x, y, r2.z);
//	    f_conv2.update().split(y, y, y_t, y_block);
//	    f_conv2.update().split(z, z, z_t, o_block_size);
//	    f_conv2.update().reorder(y_t, z_t, y, r2.z, z);
//	    f_conv2.update().vectorize(x, vec_len);
	    f_conv2.update().unroll(r2.x);
	    f_conv2.update().unroll(r2.y);
	    f_conv2.update().fuse(z, n, par).parallel(par);
	}
	else if (MEDIUM_DATA_SET)
	{
	    // Blocking spatially with vectorization
	    Var z_t("z_t"), y_t("y_t"), par("par");
	    int vec_len = 32;
	    int o_block_size = 4;
	    int y_block = 32;
	    int y_block_2 = 32;

	    f_conv.compute_root();
	    f_conv.fuse(z, n, par).parallel(par);
	    f_conv.update().reorder(x, y, r.z);
	    f_conv.update().split(y, y, y_t, y_block);
	    f_conv.update().split(z, z, z_t, o_block_size);
	    f_conv.update().reorder(y_t, z_t, y, r.z, z);
	    f_conv.update().vectorize(x, vec_len);
	    f_conv.update().unroll(r.x);
	    f_conv.update().unroll(r.y);
	    f_conv.update().fuse(z, n, par).parallel(par);

	    f_conv2.compute_root();
	    f_conv2.parallel(n);
	    f_conv2.update().reorder(x, y, r2.z);
//	    f_conv2.update().split(y, y, y_t, y_block_2);
//	    f_conv2.update().split(z, z, z_t, o_block_size);
//	    f_conv2.update().reorder(y_t, z_t, y, r2.z, z);
//	    f_conv2.update().vectorize(x, vec_len);
	    f_conv2.update().unroll(r2.x);
	    f_conv2.update().unroll(r2.y);
	    f_conv2.update().fuse(z, n, par).parallel(par);

//	    c_conv.compute_at(conv2, x); 
	}
	else if (SMALL_DATA_SET)
	{
	    // Blocking spatially with vectorization
	    Var z_t("z_t"), y_t("y_t"), par("par");
	    int vec_len = 16;
	    int o_block_size = 8;
	    int y_block = 32;

	    f_conv.compute_root();
	    f_conv.parallel(n);
	    f_conv.update().reorder(x, y, r.z);
	    f_conv.update().split(y, y, y_t, y_block);
	    f_conv.update().split(z, z, z_t, o_block_size);
	    f_conv.update().reorder(y_t, z_t, y, r.z, z);
	    f_conv.update().vectorize(x, vec_len);
	    f_conv.update().unroll(r.x);
	    f_conv.update().unroll(r.y);
	    f_conv.update().fuse(z, n, par).parallel(par);

	    f_conv2.compute_root();
	    f_conv2.parallel(n);
	    f_conv2.update().reorder(x, y, r2.z);
//	    f_conv2.update().split(y, y, y_t, y_block);
//	    f_conv2.update().split(z, z, z_t, o_block_size);
//	    f_conv2.update().reorder(y_t, z_t, y, r2.z, z);
//	    f_conv2.update().vectorize(x, vec_len);
	    f_conv2.update().unroll(r2.x);
	    f_conv2.update().unroll(r2.y);
	    f_conv2.update().fuse(z, n, par).parallel(par);
	}
    }
 
    Halide::Target target = Halide::get_host_target();

    f_conv2.compile_to_object("generated_conv_2.o",
                             {input, filter, bias, filter2, bias2},
                             "conv_halide",
                             target);
    return 0;
}
