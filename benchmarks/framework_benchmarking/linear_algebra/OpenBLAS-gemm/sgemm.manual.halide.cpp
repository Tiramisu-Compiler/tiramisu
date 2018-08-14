#include "Halide.h"

using namespace Halide;
//using Generator<GEMMGenerator<float>>::target;
//using Generator<GEMMGenerator<float>>::get_target;
//using Generator<GEMMGenerator<float>>::natural_vector_size;

int main(int argc, char **argv)
{
        Param<float>   a_ = {"a", 1.0};
        Param<float>   b_ = {"b", 1.0};
        ImageParam A_ = {type_of<float>(), 2, "A"};
        ImageParam B_ = {type_of<float>(), 2, "B"};
        ImageParam C_ = {type_of<float>(), 2, "C"};

        // Matrices are interpreted as column-major by default. The
        // transpose GeneratorParams are used to handle cases where
        // one or both is actually row major.
        const Expr num_rows = A_.width();
        const Expr num_cols = B_.height();
        const Expr sum_size = A_.height();

        const int vec = 16; //Generator<GEMMGenerator<float>>::natural_vector_size(a_.type());
        const int s = vec * 2;

        ImageParam A_in = A_;
        ImageParam B_in = B_;

        Var i, j, ii, ji, jii, iii, io, jo, t, ti[3], tj[3], k("k");
        Func result("result"), A("A"), Btmp("Btmp"), As("As"), Atmp("Atmp"), AB("AB"), prod;
        RDom rv(0, sum_size);

        Atmp(i, j) = BoundaryConditions::constant_exterior(A_in, cast<float>(0))(i, j);
        As(i, j, io) = Atmp(io*s + i, j);
        A(i, j) = As(i % s, j, i / s);

        prod(k, i, j) = A(i, k) * B_in(k, j);
        AB(i, j) += prod(rv, i, j);
        result(i, j) = (a_ * AB(i, j) + b_ * C_(i, j));

        result.tile(i, j, ti[1], tj[1], i, j, 2*s, 2*s, TailStrategy::GuardWithIf);
        result.tile(i, j, ii, ji, s, 4).tile(i, j, ti[0], tj[0], i, j, 1, s/4);
        result.fuse(tj[1], ti[1], t).parallel(t);
        result.rename(tj[0], t);
        result.bound(i, 0, num_rows).bound(j, 0, num_cols);

//        As.compute_root().split(j, jo, ji, s).reorder(i, ji, io, jo).unroll(i).vectorize(ji).parallel(jo, 4);
        As.compute_at(result, ti[0]).split(j, jo, ji, s).reorder(i, ji, io, jo).unroll(i).vectorize(ji);//.parallel(jo, 4);

        Atmp.compute_at(As, io).vectorize(i).unroll(j);

        AB.compute_at(result, i).bound_extent(j, 4).unroll(j).bound_extent(i, s).vectorize(i)
            			.update().reorder(i, j, rv).unroll(j).unroll(rv, 2).vectorize(i);

	A_.dim(0).set_min(0).dim(1).set_min(0);
        B_.dim(0).set_bounds(0, sum_size).dim(1).set_min(0);
        C_.dim(0).set_bounds(0, num_rows);
        C_.dim(1).set_bounds(0, num_cols);
        result.output_buffer().dim(0).set_bounds(0, num_rows).dim(1).set_bounds(0, num_cols);

	Halide::Target target = Halide::get_host_target();

	result.compile_to_object("generated_sgemm_halide.o",
                                 {a_, b_, A_, B_, C_},
                                 "sgemm_halide",
                                  target);
}
