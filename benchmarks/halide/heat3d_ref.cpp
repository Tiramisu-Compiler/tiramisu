#include "Halide.h"
#include "wrapper_heat3d.h"

using namespace Halide;
int main(int argc, char **argv) {
ImageParam input(Float(32), 3, "input");

Buffer<float>  output(_X,_Y,_Z,_TIME+1);

Func heat3d;

Var x, y,z,k,x0,x1,y0,y1,z0,z1;

RDom st{1, input.dim(0).extent()-2, 1, input.dim(1).extent()-2, 1, input.dim(2).extent()-2, 0, _TIME, "spacetime"};

heat3d(z,y,x,k) = input(z,y,x);

heat3d(st.x, st.y, st.z, st.w+1) =  heat3d(st.x, st.y, st.z,st.w) +_ALPHA *
      (heat3d(st.x+1, st.y, st.z,st.w)-_BETA*  heat3d(st.x, st.y, st.z,st.w) + heat3d(st.x-1, st.y, st.z,st.w)+
      heat3d(st.x, st.y+1, st.z,st.w) -_BETA*  heat3d(st.x, st.y, st.z,st.w)+ heat3d(st.x, st.y-1, st.z,st.w) +
      heat3d(st.x, st.y, st.z+1,st.w)-_BETA*  heat3d(st.x, st.y, st.z,st.w) + heat3d(st.x, st.y, st.z-1,st.w));

      heat3d.split(x,x0,x1,32);
      heat3d.split(y,y0,y1,32);
      heat3d.split(z,z0,z1,32);
      heat3d.reorder(x1,y1,z1,x0,y0,z0);
      // Parallelize the outermost loop level of Z
      heat3d.parallel(z0);

Halide::Target target = Halide::get_host_target();

heat3d.compile_to_object("build/generated_fct_heat3d_ref.o",
                           {input},
                           "heat3d_ref",
                           target);

return 0;
}
