#include "Halide.h"
#include "configure.h"



using namespace Halide;
int main(int argc, char **argv)
{ 
    ImageParam            input{Float(32), 4, "input"};
    ImageParam            filter{Float(32), 4, "filter"};
    ImageParam            bias{Float(32), 1, "bias"};
    ImageParam            filter2{Float(32), 4, "filter2"};
    ImageParam            bias2{Float(32), 1, "bias2"};
   /* THE ALGORITHM */

    Var x("x"), y("y"), z("z"), n("n");
    Func f_conv("conv"), f_conv2("conv2");
    Func f_ReLU("ReLU"), f_ReLU2("ReLU2") ;
    //Func f_Maxpool("Maxpool");
    Func f_vgg("vgg");

    RDom r(0, K+1, 0, K+1, 0, FIn);
    RDom r2(0, K+1, 0, K+1, 0, FOut);



    // First conv computations
    f_conv(x, y, z, n) = bias(z);
    f_conv(x, y, z, n) += filter(r.x, r.y, r.z, z) * input(x + r.x, y + r.y, r.z, n);

    //first relu
     f_ReLU(x, y, z, n) = max(0, f_conv(x, y, z, n));

    // Second conv computations
     f_conv2(x, y, z, n) = bias2(z);
     f_conv2(x, y, z, n) += filter2(r2.x, r2.y, r2.z, z) * f_ReLU(x + r2.x, y + r2.y, r2.z, n);

    //second relu
    f_ReLU2(x, y, z, n) = max(0, f_conv2(x, y, z, n));
  
    //maxpooling computation : f_vgg←f_Maxpool(x,y,z,n) so here f_vgg represent Maxpool;
    f_vgg(x,y,z,n)= maximum(f_ReLU2(x + r2.x, y + r2.y ,z, n));

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
            
            filter2.dim(0).set_bounds_estimate(0, K+1);
            filter2.dim(1).set_bounds_estimate(0, K+1);
            filter2.dim(2).set_bounds_estimate(0, FOut);
            filter2.dim(3).set_bounds_estimate(0,FOut);

            bias2.dim(0).set_bounds_estimate(0, FOut);
    // Provide estimates on the pipeline output
            f_vgg.estimate(x, 0,  N-2*K)
                  .estimate(y, 0,  N-2*K)
                  .estimate(z, 0, FOut)
                  .estimate(n, 0, BATCH_SIZE);  

        Pipeline p(f_vgg);
        Target target = Halide::get_host_target();
        p.auto_schedule(target);

    f_vgg.compile_to_object("build/generated_fct_vgg_ref.o", {input, filter, bias, filter2, bias2}, "vgg_ref");
    f_vgg.compile_to_lowered_stmt("build/generated_fct_vgg_ref.txt", {input, filter, bias, filter2, bias2}, Text);

    return 0;
}