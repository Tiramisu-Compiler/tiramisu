#include "Halide.h"
#include "configure.h"



using namespace Halide;
int main(int argc, char **argv)
{ 
    ImageParam            input{Float(32), 4, "input"};
    ImageParam            filter{Float(32), 4, "filter"};
    ImageParam            bias{Float(32), 1, "bias"};

   /* THE ALGORITHM */

    Var x("x"), y("y"), z("z"), n("n");
    Func f_conv("convolution_layer");


    RDom r(0, K+1, 0, K+1, 0, FIn);




    //convolution
    f_conv(x, y, z, n) = bias(z);
    f_conv(x, y, z, n) += filter(r.x, r.y, r.z, z) * input(x + r.x, y + r.y, r.z, n);


    /* THE SCHEDULE */

     // Provide estimates on the input image
           input.dim(0).set_bounds_estimate(0, N+K);
            input.dim(1).set_bounds_estimate(0, N+K);
            input.dim(2).set_bounds_estimate(0, FIn);
            input.dim(3).set_bounds_estimate(0, BATCH_SIZE);

            filter.dim(0).set_bounds_estimate(0, K+1);
            filter.dim(1).set_bounds_estimate(0, K+1);
            filter.dim(2).set_bounds_estimate(0, FIn);
            filter.dim(3).set_bounds_estimate(0,FOut);

            bias.dim(0).set_bounds_estimate(0, FOut);
            
        
    // Provide estimates on the pipeline output
            f_conv.estimate(x, 0,  N)
                  .estimate(y, 0,  N)
                  .estimate(z, 0, FOut)
                  .estimate(n, 0, BATCH_SIZE);  

        Pipeline p(f_conv);
        Target target = Halide::get_host_target();
        p.auto_schedule(target);

    f_conv.compile_to_object("build/generated_fct_convolution_layer_ref.o", {input, filter, bias}, "convolution_layer_ref");
    f_conv.compile_to_lowered_stmt("build/generated_fct_convolution_layer_ref.txt", {input, filter, bias}, Text);

    return 0;
}