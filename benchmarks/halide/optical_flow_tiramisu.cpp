#include "tiramisu/tiramisu.h"
#include "wrapper_optical_flow.h"

// This code is the Tiramisu implementation of the following Matlab code
// https://www.mathworks.com/examples/computer-vision/community/35625-lucas-kanade-method-example-2?s_tid=examples_p1_BOTH

using namespace tiramisu;

expr conv2(computation& im1, var y, var x, std::vector<int> weights)
{
    expr e = cast(p_float32, weights[0]*cast(p_int32, im1(y,     x)) + weights[1]*cast(p_int32, im1(y,     x + 1)) +
			     weights[2]*cast(p_int32, im1(y + 1, x)) + weights[3]*cast(p_int32, im1(y + 1, x + 1)));

    return e;
}

int main(int argc, char* argv[])
{
    // Declare the function name
    tiramisu::init("optical_flow_tiramisu");

    // Declare input sizes
    Input SIZES("SIZES", {2}, p_int32);

    constant N0("N0", SIZES(0));
    constant N1("N1", SIZES(1));
    constant NC("NC", _NC); // Number of corners

    // Loop iterators
    var x("x", 0, N1-1), y("y", 0, N0-1), k("k", 0, NC);

    // input images
    Input im1("im1", {N0-1, N1-1}, p_uint8);
    Input im2("im2", {N0-1, N1-1}, p_uint8);

    // Corners 
    Input C1("C1", {NC}, p_int32);
    Input C2("C2", {NC}, p_int32);

    // First convolution (partial on x)
    std::vector<int> w1 = {1, -1,  1, -1};
    computation Ix_m("Ix_m", {y, x}, conv2(im1, y, x, w1));  // Ix_m(y, x) = conv2(im1, y, x, w1);

    // Second convolution (partial on y)
    std::vector<int> w2 = {1, 1,  -1, -1};
    computation Iy_m("Iy_m", {y, x}, conv2(im1, y, x, w2));

    // Third convolution
    std::vector<int> w3 = {1, 1, 1, 1};
    std::vector<int> w4 = {-1, -1, -1, -1};
    computation It_m("It_m", {y, x}, conv2(im1, y, x, w3) + conv2(im2, y, x, w4));


    // Second part of the algorithm
    // Compute "u" and "v" for each corner "k"
    computation i("i", {k}, C2(k));
    computation j("j", {k}, C1(k));

    // Ix = Ix_m(i-w:i+w, j-w:j+w);
    // Iy = Iy_m(i-w:i+w, j-w:j+w);
    // It = It_m(i-w:i+w, j-w:j+w);
    // Ix = Ix(:); % flatten the IX 2D array into a vector
    // Iy = Iy(:);
    // A = [Ix Iy];
    // b = -It(:);
    var xp("xp", 0, 2*w);
    var yp("yp", 0, 2*w);
    computation        A1("A1",        {k, yp, xp},   Ix_m(i(0)+yp-w, j(0)+xp-w));
    computation  A1_right("A1_right",  {k, yp, xp},   Iy_m(i(0)+yp-w, j(0)+xp-w));
    computation        b1("b1",        {k, yp, xp}, (-It_m(i(0)+yp-w, j(0)+xp-w)));

    // Reshape A1 to A
    var x1("x1", 0, 2);
    var y1("y1", 0, 4*w*w);
    view        A("A",        {k, y1, x1}, p_float32); // Use A to reshape A1 and A1_right
    view        b("b",        {k, y1}, p_float32);     // Use b to reshape b1

    // Compute pinv(A):
    // 1)    tA = transpose(A)
    // 2)    tAA = tA * A
    // 3)    X = inv(tAA)
    // 4)    pinv(A) = X * tA
 
    // 1) Computing tA = transpose(A)
    computation tA("tA", {k, x1, y1}, A(k, y1, x1));

    // 2) Computing tAA = tA * A
    var y2("y2", 0, 2);
    var l1("l1", 0, 4*w*w);
    computation tAA("tAA", {k, x1, y2}, expr((float) 0));
    computation tAA_update("tAA_update", {k, x1, y2, l1}, tAA(k, x1, y2) + (tA(k, x1, l1) * A(k, l1, y2)));

    // 3) Computing X = inv(tAA)
    computation determinant("determinant", {k}, tAA(k,0,0)*tAA(k,1,1) - tAA(k,0,1)*tAA(k,1,0));
    computation tAAp_00("tAAp_00", {k},  tAA(k,1,1)/determinant(k));
    computation tAAp_11("tAAp_11", {k},  tAA(k,0,0)/determinant(k));
    computation tAAp_01("tAAp_01", {k}, -tAA(k,0,1)/determinant(k));
    computation tAAp_10("tAAp_10", {k}, -tAA(k,1,0)/determinant(k));
    view X("X", {k, x1, y2}, p_float32);

    // 4) Computing pinv(A) = X*tA
    var l2("l2", 0, 2);
    computation    pinvA("pinvA", {k, x1, y1}, expr((float) 0));
    computation    pinvA_update("pinvA_update", {k, x1, y1, l2}, pinvA(k, x1, y1) + X(k, x1, l2)*tA(k, l2, y1));

    // Compute nu = pinv(A)*b
    computation nu("nu", {k, x1}, expr((float) 0));
    computation nu_update("nu_update", {k, x1, y1}, nu(k, x1) + pinvA(k, x1, y1)*b(k, y1));

    // Results
    computation u("u", {k}, nu(k, 0));
    computation v("v", {k}, nu(k, 1));

    // Schedule
    Ix_m.then(Iy_m, x)
	.then(It_m, x)
	.then(i, computation::root)
	.then(j, k)
	.then(A1, k)
	.then(A1_right, xp)
	.then(b1, xp)
	.then(tA, k)
	.then(tAA, k)
	.then(tAA_update, y2)
	.then(determinant, k)
	.then(tAAp_00, k)
	.then(tAAp_11, k)
	.then(tAAp_01, k)
	.then(tAAp_10, k)
	.then(X, k)
	.then(pinvA, k)
	.then(pinvA_update, y1)
	.then(nu, k)
	.then(nu_update, x1)
	.then(u, k)
	.then(v, k);

    int VEC = 32;

    Ix_m.parallelize(y);
    A1.parallelize(k);
    Ix_m.vectorize(x, VEC);
    Iy_m.vectorize(x, VEC);
    It_m.vectorize(x, VEC);
    A1.vectorize(xp, VEC);
    A1_right.vectorize(xp, VEC);
    b1.vectorize(xp, VEC);
    tA.vectorize(y1, VEC/2);
    pinvA.vectorize(y1, VEC);
    pinvA_update.vectorize(y1, VEC);

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
    buffer b_A("b_A", {4*w*w, 2}, p_float32, a_output);
    buffer b_b("b_b", {4*w*w}, p_float32, a_temporary);
    buffer b_tA("b_tA", {2, 4*w*w}, p_float32, a_output);
    buffer b_tAA("b_tAA", {2, 2}, p_float32, a_output);
    buffer b_determinant("b_determinant", {1}, p_float32, a_output);
    buffer b_X("b_X", {2, 2}, p_float32, a_output);
    buffer b_pinvA("b_pinvA", {2, 4*w*w}, p_float32, a_output);
    buffer b_nu("b_nu", {2}, p_float32, a_temporary);
    buffer b_u("b_u", {NC}, p_float32, a_output);
    buffer b_v("b_v", {NC}, p_float32, a_output);

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
    A1.store_in(&b_A, {xp+yp*2*w, 0});
    A1_right.store_in(&b_A, {xp+yp*2*w, 1});
    b1.store_in(&b_b, {xp+yp*2*w});
    A.store_in(&b_A, {y1, x1});
    b.store_in(&b_b, {y1});
    tA.store_in(&b_tA, {x1, y1});
    tAA.store_in(&b_tAA, {x1, y2});
    tAA_update.store_in(&b_tAA, {x1, y2});
    determinant.store_in(&b_determinant, {0});
    tAAp_00.store_in(&b_X, {0, 0});
    tAAp_01.store_in(&b_X, {0, 1});
    tAAp_10.store_in(&b_X, {1, 0});
    tAAp_11.store_in(&b_X, {1, 1});
    X.store_in(&b_X, {x1, y2});
    pinvA.store_in(&b_pinvA, {x1, y1});
    pinvA_update.store_in(&b_pinvA, {x1, y1});
    nu.store_in(&b_nu, {x1});
    nu_update.store_in(&b_nu, {x1});
    u.store_in(&b_u, {k});
    v.store_in(&b_v, {k});

    tiramisu::codegen({&b_SIZES, &b_im1, &b_im2, &b_Ix_m, &b_Iy_m, &b_It_m, &b_C1, &b_C2, &b_u, &b_v, &b_A, &b_pinvA, &b_determinant, &b_tAA, &b_tA, &b_X}, "build/generated_fct_optical_flow.o");

    global::get_implicit_function()->dump_halide_stmt();

    return 0;
}
