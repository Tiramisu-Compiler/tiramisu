#include "Halide.h"

using namespace Halide;

int main(int argc, char* argv[]) {
  ImageParam in{Float(32), 3, "input"};
  Param<float> alpha;
  Param<float> beta;
  Param<int> t;

  Func heat3d{"heat3d"};
  Var x, y, z;
  Var k;

  RDom time{1, t, "t"};

  heat3d(x, y, z, k) = undef<float>();
  heat3d(x, y, z, 0) = alpha * in(x, y, z) +
                 beta * (in(x+1, y, z) + in(x-1, y, z)+
                         in(x, y+1, z) + in(x, y-1, z) +
                         in(x, y, z+1) + in(x, y, z-1));


  heat3d(x, y, z, time.x) = alpha * heat3d(x, y, z, time.x-1) +
                 beta * (heat3d(x+1, y, z, time.x-1) + heat3d(x-1, y, z, time.x-1)+
                         heat3d(x, y+1, z, time.x-1) + heat3d(x, y-1, z, time.x-1) +
                         heat3d(x, y, z+1, time.x-1) + heat3d(x, y, z-1, time.x-1));


  heat3d.parallel(z).vectorize(x, 8);

  heat3d.compile_to_lowered_stmt("heat3d.html", {in, alpha, beta, t}, HTML);


  return 0;
}
