#include "Halide.h"
#include "wrapper_heat3d.h"

using namespace Halide;
int main(int argc, char **argv) {
ImageParam input(Float(32), 3, "input");

Buffer<float>  output(_X,_Y,_Z,_TIME+1);

Func heat3d("heat3d_ref");

Var x("x"), y("y"),z("z"),k("K"),x0,x1,y0,y1,z0,z1;

RDom st{1, input.dim(0).extent()-2, 1, input.dim(1).extent()-2, 1, input.dim(2).extent()-2, 1, _TIME, "reduction_domaine"};

heat3d(x,y,z,k) = input(x,y,z);

heat3d(st.x, st.y, st.z, st.w) =  heat3d(st.x, st.y, st.z,st.w-1) +_ALPHA *
      (heat3d(st.x+1, st.y, st.z,st.w-1)-_BETA*  heat3d(st.x, st.y, st.z,st.w-1) + heat3d(st.x-1, st.y, st.z,st.w-1)+
      heat3d(st.x, st.y+1, st.z,st.w-1) -_BETA*  heat3d(st.x, st.y, st.z,st.w-1)+ heat3d(st.x, st.y-1, st.z,st.w-1) +
      heat3d(st.x, st.y, st.z+1,st.w-1)-_BETA*  heat3d(st.x, st.y, st.z,st.w-1) + heat3d(st.x, st.y, st.z-1,st.w-1));

Halide::Target target = Halide::get_host_target();

heat3d.compile_to_object("build/generated_fct_heat3d_ref.o",
                           {input},
                           "heat3d_ref",
                           target);

return 0;
}
