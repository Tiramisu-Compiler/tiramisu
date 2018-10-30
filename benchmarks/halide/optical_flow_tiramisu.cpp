#include "tiramisu/tiramisu.h"

// This code is the Tiramisu implementation of the following Matlab code
// https://www.mathworks.com/examples/computer-vision/community/35625-lucas-kanade-method-example-2?s_tid=examples_p1_BOTH

using namespace tiramisu;

int main(int argc, char* argv[])
{
    // Declare the function name
    tiramisu::init("optical_flow_tiramisu");

    // Declare input sizes
    // TODO: "input" dimension sizes should be expressions not variables.
    input SIZES("SIZES", {var("S", 0, 2)}, p_int32);

    constant N0("N0", SIZES(0));
    constant N1("N1", SIZES(1));
    constant NC("NC", 20); // Number of corners
    constant w("w", 10); // Window size

    // Loop iterators
    var x("x", 0, N1), y("y", 0, N0), k("k", 0, NC);

    // input images
    input im1("im1", {y, x}, p_uint8);
    input im2("im2", {y, x}, p_uint8);
    // Corners 
    input C1("C1", {k}, p_int32);
    input C2("C2", {k}, p_int32);

    // First convolution (partial on x)
    // Ix_m = conv2(im1, [-1 1; -1 1])
    expr e1 = cast(p_uint8, (cast(p_float32,   im1(y + 1, x) + im1(y + 1, x + 1)
					     - im1(y,     x) - im1(y,     x + 1))/expr((float) 4)));
    computation Ix_m("Ix_m", {y, x}, e1);

    // Second convolution  (partial on y)
    // Iy_m = conv2(im1, [-1 -1; 1 1])
    expr e2 = cast(p_uint8, (cast(p_float32,   im1(y, x + 1) + im1(y + 1, x + 1)
					     - im1(y,     x) - im1(y + 1, x    ))/expr((float) 4)));
    computation Iy_m("Iy_m", {y, x}, e2);

    // Third convolution
    // It_m = conv2(im1, ones(2)) + conv2(im2, -ones(2));
    expr e3 = cast(p_uint8, (cast(p_float32,    im1(y,     x)  + im1(y,     x + 1)
					      + im1(y + 1, x)  + im1(y + 1, x + 1))/expr((float) 4)));
    expr e4 = cast(p_uint8, (cast(p_float32, (- im2(y,     x)) - im2(y,     x + 1)
					      - im2(y + 1, x)  - im2(y + 1, x + 1))/expr((float) 4)));
    computation It_m("It_m", {y, x}, e3 + e4);


    // Second part of the algorithm
    // Compute "u" and "v" for each corner "k"
    computation i({k}, C2(k));
    computation j({k}, C1(k));

    // Ix = Ix_m(i-w:i+w, j-w:j+w);
    // Iy = Iy_m(i-w:i+w, j-w:j+w);
    // A = [Ix Iy];
    // It = It_m(i-w:i+w, j-w:j+w);
    // b = -It
    var x1("x1", 0, 2*w);
    var y1("y1", 0, 2*w);
    computation        A("A",        {k, y1, x1},   Ix_m(i(0)+y1-w, j(0)+x1-w));  //TODO: use i(k) and j(k) instead of i(0) and j(0)
    computation A_right("A_right",   {k, y1, x1},   Iy_m(i(0)+y1-w, j(0)+x1-w));  //i(k), j(k)
    computation        b("b",        {k, y1, x1}, (-It_m(i(0)+y1-w, j(0)+x1-w))); //i(k), j(k)

    // Compute pinv(A):
    //	    tA = transpose(A)
    //	    mul1 = tA * A
    //	    X = inv(mul1)
    //	    pinv(A) = X * tA
    var x2("x2", 0, 4*w);
    var y2("y2", 0, 4*w);
    var l1("l1", 0, 2*w);
    computation tA("tA", {k, x2, y1}, A(k, y1, x2));

    computation mul1("mul1", {k, x2, y2}, expr((uint8_t) 0));
    computation mul1_update("mul1_update", {k, x2, y2, l1}, mul1(k, x2, y2) + tA(k, x2, l1) * A(k, l1, y2));

    // Compute the inverse of mul1 using LU decomposition.
    // We use the following reference implementation (lines 95 to 126)
    // https://github.com/Meinersbur/polybench/blob/master/polybench-code/linear-algebra/solvers/ludcmp/ludcmp.c
    //	    1)- Compute the LU decomposition of mul1: LU = mul1
    //	    2)- Use LU to compute X, the inverse of mul1 by solving the following
    //	    system:
    //		    LU*X=I
    //	    where I is the identity matrix.
    var i1("i1", 0, 4*w);
    var j1("j1", 0, i1);
    var l2("l2", 0, j1);
    
    // LU decomposition of A
    computation        w1("w1",        {k, i1, j1},     mul1(k, i1, j1));
    computation w1_update("w1_update", {k, i1, j1, l2},   w1(k, i1, j1) - mul1(k, i1, l2)*mul1(k, l2, j1));
    computation      temp("temp",      {k, i1, j1},       w1(k, i1, j1)/mul1(k, j1, j1));

    var j2("j2", i1, 4*w);
    var l3("l3",  0,  i1);

    computation        w2("w2",        {k, i1, j2},     temp(k, i1, j2));
    computation w2_update("w2_update", {k, i1, j2, l3}, w2(k, i1, j2) - temp(k, i1, l3)*temp(k, l3, j2));
    computation        LU("LU",        {k, i1, j2},     w2(k, i1, j2));

    // Finding the inverse of A.
    // The inverse will be stored in X.
    var r("r", 0, 4*w);
    var r2("r2", r, r+1);
    var r3("r3", r, r+1);
    computation     Y("Y", {k, r, i1}, p_uint8);
    computation     bp("bp", {k, r, i1}, expr((uint8_t) 0));
//    computation     bp_update("bp_update", {k, r2, r3}, expr((uint8_t) 1));
    computation     w3("w3", {k, r, i1}, bp(k, r, i1));
    computation     w3_update("w3_update", {k, r, i1, j1}, w3(k, r, j1) - LU(k, i1, j1)*Y(k, r, j1));
    Y.set_expression(w3(k, r, i1));

    var j3("j3", i1+1, 4*w);
    computation     X("X", {k, r, i1}, p_uint8);
    computation     w4("w4", {k, r, i1}, Y(k, r, 4*w-i1));
    computation     w4_update("w4_update", {k, r, i1, j3}, w4(k, r, 4*w-i1) - LU(k, 4*w-i1, j3)*X(k, r, j3));
    X.set_expression(w4(k, r, i1)/LU(k, 4*w-i1, 4*w-i1));

    // Computing pinv(A)=X*tA
    var j4("j4", 0, 4*w);
    computation    pinvA("pinvA", {k, i1, y1}, expr((uint8_t) 0));
    computation    pinvA_update("pinvA_update", {k, i1, y1, j4}, pinvA(k, i1, y1) + X(k, i1, j4)*tA(k, j4, y1));

    // Compute nu
    // i1= 4w, y1: 2w
    computation nu("nu", {k, i1, x1}, expr((uint8_t) 0));
    computation nu_update("nu_update", {k, i1, x1, y1}, nu(k, i1, x1) + pinvA(k, i1, y1)*b(k, y1, x1));

    // Results
    // u(k) = nu(0)
    // v(k) = nu(1)

    Ix_m.then(Iy_m, x)
	.then(It_m, x)
	.then(i, computation::root)
	.then(j, k)
	.then(A, y1)
	.then(A_right, y1)
	.then(b, y1)
	.then(tA, computation::root)
	.then(mul1, computation::root)
	.then(mul1_update, y2)
	.then(w1, computation::root)
	.then(w1_update, j1)
	.then(temp, j1)
	.then(w2, computation::root)
	.then(w2_update, j2)
	.then(LU, j2)
	.then(bp, computation::root)
//	.then(bp_update, r)
	.then(w3, r)
	.then(w3_update, i1)
	.then(Y, i1)
	.then(w4, computation::root)
	.then(w4_update, i1)
	.then(X, i1)
	.then(pinvA, computation::root)
	.then(pinvA_update, y1)
	.then(nu, computation::root)
	.then(nu_update, x1);

    // Buffer allocation and mapping computations to buffers
    buffer b_SIZES("b_SIZES", {2}, p_int32, a_input);
    buffer b_im1("b_im1", {N0, N1}, p_uint8, a_input);
    buffer b_im2("b_im2", {N0, N1}, p_uint8, a_input);
    buffer b_Ix_m("b_Ix_m", {N0, N1}, p_float32, a_output);
    buffer b_Iy_m("b_Iy_m", {N0, N1}, p_float32, a_output);
    buffer b_It_m("b_It_m", {N0, N1}, p_float32, a_output);
    buffer b_C1("b_C1", {NC}, p_int32, a_input);
    buffer b_C2("b_C2", {NC}, p_int32, a_input);
    buffer b_i("b_i", {1}, p_int32, a_temporary);
    buffer b_j("b_j", {1}, p_int32, a_temporary);
    buffer b_A("b_A", {2*w, 4*w}, p_float32, a_temporary);
    buffer b_b("b_b", {2*w, 2*w}, p_float32, a_temporary);
    buffer b_tA("b_tA", {4*w, 2*w}, p_float32, a_temporary);
    buffer b_mul("b_mul", {2*w, 2*w}, p_float32, a_temporary);
    buffer b_w1("b_w1", {1}, p_float32, a_temporary);
    buffer b_temp("b_temp", {2*w, 2*w}, p_float32, a_temporary);
    buffer b_w2("b_w2", {1}, p_float32, a_temporary);
    buffer b_LU("b_LU", {2*w, 2*w}, p_float32, a_temporary);
    buffer b_y("b_y", {2*w, 2*w}, p_float32, a_temporary);
    buffer b_bp("b_bp", {2*w, 2*w}, p_float32, a_temporary);
    buffer b_w3("b_w3", {1}, p_float32, a_temporary);
    buffer b_x("b_x", {2*w, 2*w}, p_float32, a_temporary);
    buffer b_w4("b_w4", {1}, p_float32, a_temporary);
    buffer b_pinvA("b_pinvA", {4*w, 2*w}, p_float32, a_temporary);
    buffer b_nu("b_nu", {4*w, 2*w}, p_float32, a_temporary);


    SIZES.store_in(&b_SIZES);
    im1.store_in(&b_im1);
    im2.store_in(&b_im2);
    Ix_m.store_in(&b_Ix_m);
    Iy_m.store_in(&b_Iy_m);
    It_m.store_in(&b_It_m);
    C1.store_in(&b_C1);
    C2.store_in(&b_C2);
    i.store_in(&b_i, {0});
    j.store_in(&b_j, {0});
    A.store_in(&b_A, {x1, y1});
    A_right.store_in(&b_A, {x1+2*10, y1});  //2*w
    b.store_in(&b_b, {x1, y1});
    tA.store_in(&b_tA, {x2, y1});
    mul1.store_in(&b_mul, {x2, y2});
    mul1_update.store_in(&b_mul, {x2, y2});
    w1.store_in(&b_w1, {0});
    w1_update.store_in(&b_w1, {0});
    temp.store_in(&b_temp, {i1, j1});
    w2.store_in(&b_w2, {0});
    w2_update.store_in(&b_w2, {0});
    LU.store_in(&b_LU, {i1, j2});
    Y.store_in(&b_y, {r, i1});
    bp.store_in(&b_bp, {r, i1});
//    bp_update.store_in(&b_bp, {r, i1});
    w3.store_in(&b_w3, {0});
    w3_update.store_in(&b_w3, {0});
    X.store_in(&b_x, {r, i1});
    w4.store_in(&b_w4, {0});
    w4_update.store_in(&b_w4, {0});
    pinvA.store_in(&b_pinvA, {i1, y1});
    pinvA_update.store_in(&b_pinvA, {i1, y1});
    nu.store_in(&b_nu, {i1, x1});
    nu_update.store_in(&b_nu, {i1, x1});

    tiramisu::codegen({&b_SIZES, &b_im1, &b_im2, &b_Ix_m, &b_Iy_m, &b_It_m, &b_C1, &b_C2}, "build/generated_fct_optical_flow.o");

    return 0;
}
