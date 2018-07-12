#include "Halide.h"
#include "configure.h"

#define SCHEDULE_CPU 1

using namespace Halide;

int main(int argc, char **argv)
{
    ImageParam            input{Float(32), 4, "input"};
    ImageParam            filter{Float(32), 4, "filter"};
    ImageParam            bias{Float(32), 1, "bias"};

    Var x("x"), y("y"), z("z"), n("n");

    Func f_conv("conv");
    RDom r(0, K, 0, K, 0, FIn);

    f_conv(x, y, z, n) = bias(z);
    f_conv(x, y, z, n) += filter(r.x, r.y, r.z, z) * input(x + r.x, y + r.y, r.z, n);

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
// 0, 1,   2,   3,   4,   5,     6,             7,            8,
// n, z,   y,   x, r_z, r_y,   r_x,
// n, z, r_z,   y,   x, r_y,   r_x,
// n, z, r_z,   y, y_t,   x,   r_y,            r_x,
// n, z, z_t, r_z,   y, y_t, x (vec), r_y (unroll), r_x (unroll)

	}
	else if (MEDIUM_DATA_SET)
	{
	    // Blocking spatially with vectorization
	    Var z_t("z_t"), y_t("y_t"), par("par");
	    int vec_len = 32;
	    int o_block_size = 4;
	    int y_block = 32;
	    f_conv.compute_root();
	    f_conv.parallel(n);
	    f_conv.update().reorder(x, y, r.z);
	    f_conv.update().split(z, z, z_t, o_block_size);
	    f_conv.update().split(y, y, y_t, y_block);
	    f_conv.update().reorder(y_t, z_t, y, r.z, z);
	    f_conv.update().vectorize(x, vec_len);
	    f_conv.update().unroll(r.x);
	    f_conv.update().unroll(r.y);
	    f_conv.update().parallel(n);
	}
	else if (SMALL_DATA_SET)
	{
	    // Blocking spatially with vectorization
	    Var z_t("z_t"), y_t("y_t"), par("par");
	    int vec_len = 32;
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
	}
    }
 
    Halide::Target target = Halide::get_host_target();

    f_conv.compile_to_object("generated_conv.o",
                             {input, filter, bias},
                             "conv_halide",
                             target);
    return 0;
}
