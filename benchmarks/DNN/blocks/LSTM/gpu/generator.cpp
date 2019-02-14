#include <tiramisu/tiramisu.h>

using namespace tiramisu;

#define BLOCK 16

int main(int argc, char **argv)
{
    // Biases, R weights, and W weights are merged to reduce the number of GEMMs
    tiramisu::init("lstm");

    // -------------------------------------------------------
    // Layer I
    // -------------------------------------------------------
    var p_i("p_i", 0, 4);
    input params({p_i}, p_int32);
    constant feature_size("feature_size", params(0));
    constant batch_size("batch_size", params(1));
    constant num_layers("num_layers", params(2));
    constant seq_length("seq_length", params(3));

    // Inner dimensions
    var i("i", 0, feature_size), j("j", 0, feature_size), k("k", 0, batch_size);
    var i_merged("i_m", 0, feature_size * 4);
    var i0("i0"), i1("i1"), k0("k0"), k1("k1");
    var j0("j0", 0, feature_size / BLOCK), j1("j1", 0, BLOCK);
    // Outer dimensions
    var l("l", 0, num_layers), m("m", 0, seq_length);

    input R("R", {l, i_merged, j}, p_float32);
    input W("W", {l, i_merged, j}, p_float32);
    input b("b", {l, i_merged}, p_float32);
    input x("x", {m, k, i}, p_float32);

    buffer buf_params("buf_params", {4}, p_int32, a_input);
    buffer buf_Weights("buf_Weights", {num_layers, 2, 4 * feature_size, feature_size}, p_float32, a_input);
    buffer buf_biases("buf_biases", {num_layers, 4 * feature_size}, p_float32, a_input);
    buffer buf_x("buf_x", {seq_length, batch_size, feature_size}, p_float32, a_input);
    buffer buf_y("buf_y", {seq_length, batch_size, feature_size}, p_float32, a_output);
    buffer buf_Weights_gpu("buf_Weights_gpu", {num_layers, 2, 4 * feature_size, feature_size}, p_float32, a_temporary);
    buffer buf_biases_gpu("buf_biases_gpu", {num_layers, 4 * feature_size}, p_float32, a_temporary);
    buffer buf_x_gpu("buf_x_gpu", {seq_length, batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_y_gpu("buf_y_gpu", {seq_length, batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_workspace("buf_workspace", {num_layers, batch_size, 4 * feature_size}, p_float32, a_temporary);
    buffer buf_h("buf_h", {seq_length + 1, num_layers + 1, batch_size, feature_size}, p_float32, a_temporary);
    buffer buf_c("buf_c", {seq_length + 1, num_layers, batch_size, feature_size}, p_float32, a_temporary);

    buf_Weights_gpu.tag_gpu_global();
    buf_biases_gpu.tag_gpu_global();
    buf_x_gpu.tag_gpu_global();
    buf_y_gpu.tag_gpu_global();
    buf_workspace.tag_gpu_global();
    buf_h.tag_gpu_global();
    buf_c.tag_gpu_global();

    buffer buf_matmul1_R_shr("b_m1_R_shr", {BLOCK, BLOCK}, p_float32, a_temporary);
    buffer buf_matmul1_R_prefreg("b_m1_R_prefreg", {1}, p_float32, a_temporary);  // Prefetch register
    buffer buf_matmul1_h_shr("b_m1_h_shr", {BLOCK, BLOCK + 1}, p_float32, a_temporary);
    buffer buf_matmul1_h_prefreg("b_m1_h_prefreg", {1}, p_float32, a_temporary);  // Prefetch register
    buffer buf_matmul1_acc("b_m1_acc", {1}, p_float32, a_temporary);
    buf_matmul1_R_shr.tag_gpu_shared();
    buf_matmul1_R_prefreg.tag_gpu_register();
    buf_matmul1_h_shr.tag_gpu_shared();
    buf_matmul1_h_prefreg.tag_gpu_register();
    buf_matmul1_acc.tag_gpu_register();

    buffer buf_matmul2_W_shr("b_m2_W_shr", {BLOCK, BLOCK}, p_float32, a_temporary);
    buffer buf_matmul2_h_shr("b_m2_h_shr", {BLOCK, BLOCK + 1}, p_float32, a_temporary);
    buffer buf_matmul2_acc("b_m2_acc", {1}, p_float32, a_temporary);
    buf_matmul2_W_shr.tag_gpu_shared();
    buf_matmul2_h_shr.tag_gpu_shared();
    buf_matmul2_acc.tag_gpu_register();

    // h(m, l) is the output of the block (m, l)
    // which takes h(m - 1, l) and h(m, l - 1) as inputs
    // initial hidden states are h(-1, l) and c(-1, l)
    // input x is copied to h(m, -1)
    computation h({m, l, k, i}, p_float32);
    computation c({m, l, k, i}, p_float32);
    computation h_init({l, k, i}, expr(float(0)));
    computation c_init({l, k, i}, expr(float(0)));
    computation h_copy_x({m, k, i}, x(m, k, i));
    computation matmul_base({m, l, k, i_merged}, b(l, i_merged));

    computation matmul1_R_shr_dec({m, l, k, i_merged}, allocate(buf_matmul1_R_shr));
    computation matmul1_R_prefreg_dec({m, l, k, i_merged}, allocate(buf_matmul1_R_prefreg));
    input matmul1_R_access({l, k, i_merged, j0}, p_float32); // Access workaround
    computation matmul1_R_shr_pref({m, l, k, i_merged}, matmul1_R_access(l, k, i_merged, 0));
    computation matmul1_R_glb_to_prefreg({m, l, k, i_merged, j0}, matmul1_R_access(l, k, i_merged, j0 + 1));
    computation matmul1_R_prefreg_to_shr({m, l, k, i_merged, j0}, matmul1_R_glb_to_prefreg(m, l, k, i_merged, j0));
    input matmul1_R_shr_access({i_merged, j}, p_float32);
    computation matmul1_h_shr_dec({m, l, k, i_merged}, allocate(buf_matmul1_h_shr));
    computation matmul1_h_prefreg_dec({m, l, k, i_merged}, allocate(buf_matmul1_h_prefreg));
    input matmul1_h_access({m, l, k, i_merged, j0}, p_float32);
    computation matmul1_h_shr_pref({m, l, k, i_merged}, matmul1_h_access(m, l, k, i_merged, 0));
    computation matmul1_h_glb_to_prefreg({m, l, k, i_merged, j0}, matmul1_h_access(m, l, k, i_merged, j0 + 1));
    computation matmul1_h_prefreg_to_shr({m, l, k, i_merged, j0}, matmul1_h_glb_to_prefreg(m, l, k, i_merged, j0));
    input matmul1_h_shr_access({k, j}, p_float32);
    computation matmul1_sync1({m, l, k, i_merged}, tiramisu::sync());
    computation matmul1_sync2({m, l, k, i_merged, j0}, tiramisu::sync());
    computation matmul1_acc_dec({m, l, k, i_merged}, allocate(buf_matmul1_acc));
    computation matmul1_acc_init({m, l, k, i_merged}, (float) 0);
    computation matmul1_acc({m, l, k, i_merged, j}, matmul1_acc_init(m, l, k, i_merged)
                                                    + matmul1_R_shr_access(i_merged, j) * matmul1_h_shr_access(k, j));
    computation matmul1({m, l, k, i_merged}, matmul1_acc(m, l, k, i_merged, 0) + matmul_base(m, l, k, i_merged));

    computation matmul2_W_shr_dec({m, l, k, i_merged}, allocate(buf_matmul2_W_shr));
    input matmul2_W_access({l, k, i_merged, j0}, p_float32);
    computation matmul2_W_shr_copy({m, l, k, i_merged, j0}, matmul2_W_access(l, k, i_merged, j0));
    input matmul2_W_shr_access({i_merged, j}, p_float32);
    computation matmul2_h_shr_dec({m, l, k, i_merged}, allocate(buf_matmul2_h_shr));
    input matmul2_h_access({m, l, k, i_merged, j0}, p_float32);
    computation matmul2_h_shr_copy({m, l, k, i_merged, j0}, matmul2_h_access(m, l, k, i_merged, j0));
    input matmul2_h_shr_access({k, j}, p_float32);
    computation matmul2_sync({m, l, k, i_merged, j0}, tiramisu::sync());
    computation matmul2_acc_dec({m, l, k, i_merged}, allocate(buf_matmul2_acc));
    computation matmul2_acc_init({m, l, k, i_merged}, (float) 0);
    computation matmul2_acc({m, l, k, i_merged, j}, matmul2_acc_init(m, l, k, i_merged)
                                                    + matmul2_W_shr_access(i_merged, j) * matmul2_h_shr_access(k, j));
    computation matmul2({m, l, k, i_merged}, matmul2_acc(m, l, k, i_merged, 0) + matmul1(m, l, k, i_merged));

    #define sigmoid(x) expr(float(1)) / (1 + expr(o_expo, -(x)))
    computation sig_i("sig_i", {m, l, k, i}, sigmoid(matmul_base(m, l, k, i)));
    computation tnh_z("tnh_z", {m, l, k, i}, expr(o_tanh, matmul_base(m, l, k, i + feature_size)));
    computation sig_o("sig_o", {m, l, k, i}, sigmoid(matmul_base(m, l, k, i + 2 * feature_size)));
    computation sig_f("sig_f", {m, l, k, i}, sigmoid(matmul_base(m, l, k, i + 3 * feature_size)));
    computation mul_iz("mul_iz", {m, l, k, i}, sig_i(m, l, k, i) * tnh_z(m, l, k, i));
    computation mul_fc("mul_fc", {m, l, k, i}, sig_f(m, l, k, i) * c(m - 1, l, k, i));
    c.set_expression(mul_iz(m, l, k, i) + mul_fc(m, l, k, i));
    computation tnh_c("tnh_c", {m, l, k, i}, expr(o_tanh, c(m, l, k, i)));
    h.set_expression(tnh_c(m, l, k, i) * sig_o(m, l, k, i));
    // Output is the last layer
    computation y({m, k, i}, h(m, num_layers - 1, k, i));
    // Copies
    computation copy_Weights_to_device({}, memcpy(buf_Weights, buf_Weights_gpu));
    computation copy_biases_to_device({}, memcpy(buf_biases, buf_biases_gpu));
    computation copy_x_to_device({}, memcpy(buf_x, buf_x_gpu));
    computation copy_y_to_host({}, memcpy(buf_y_gpu, buf_y));

    // Block dimension derivation causes a bug
    global::get_implicit_function()->add_context_constraints("[feature_size, batch_size]->{:feature_size % 16 = 0 and batch_size % 16 = 0}");

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------

    // Fuse initialization
    h_init.interchange(l, k);
    h_init.interchange(l, i);
    c_init.interchange(l, k);
    c_init.interchange(l, i);
    h_copy_x.interchange(m, k);
    h_copy_x.interchange(m, i);
    y.interchange(m, k);
    y.interchange(m, i);

    h_init.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    c_init.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    h_copy_x.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul_base.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);

    matmul1_R_shr_dec.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_R_prefreg_dec.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_R_shr_pref.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_R_glb_to_prefreg.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_R_glb_to_prefreg.add_predicate(j0 < (feature_size - 1) / BLOCK);
    matmul1_R_prefreg_to_shr.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_R_prefreg_to_shr.add_predicate(j0 < (feature_size - 1) / BLOCK);
    matmul1_h_shr_dec.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_h_prefreg_dec.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_h_shr_pref.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_h_glb_to_prefreg.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_h_glb_to_prefreg.add_predicate(j0 < (feature_size - 1) / BLOCK);
    matmul1_h_prefreg_to_shr.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_h_prefreg_to_shr.add_predicate(j0 < (feature_size - 1) / BLOCK);
    matmul1_sync1.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_sync2.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_acc_dec.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_acc_init.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_acc.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul1_acc.split(j, BLOCK, j0, j1);
    matmul1.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);

    matmul2_W_shr_dec.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul2_W_shr_copy.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul2_h_shr_dec.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul2_h_shr_copy.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul2_sync.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul2_acc_dec.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul2_acc_init.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul2_acc.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);
    matmul2_acc.split(j, BLOCK, j0, j1);
    matmul2.gpu_tile(k, i_merged, BLOCK, BLOCK, k0, i0, k1, i1);

    sig_i.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    tnh_z.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    sig_o.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    sig_f.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    mul_iz.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    mul_fc.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    c.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    tnh_c.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    h.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);
    y.gpu_tile(k, i, BLOCK, BLOCK, k0, i0, k1, i1);

    // Scheduling commands
    copy_Weights_to_device.then(copy_biases_to_device, computation::root)
        .then(copy_x_to_device, computation::root)
        .then(h_init, i1)
        .then(c_init, i1)
        .then(h_copy_x, i1)
        .then(matmul_base, computation::root)
        .then(matmul1_acc_dec, l)
        .then(matmul1_R_shr_dec, i1)
        .then(matmul1_R_prefreg_dec, i1)
        .then(matmul1_h_shr_dec, i1)
        .then(matmul1_h_prefreg_dec, i1)
        .then(matmul1_acc_init, i1)
        .then(matmul1_R_shr_pref, i1)
        .then(matmul1_h_shr_pref, i1)
        .then(matmul1_sync1, i1)
        .then(matmul1_R_glb_to_prefreg, i1)
        .then(matmul1_h_glb_to_prefreg, j0)
        .then(matmul1_acc, j0)
        .then(matmul1_sync2, j0)
        .then(matmul1_R_prefreg_to_shr, j0)
        .then(matmul1_h_prefreg_to_shr, j0)
        .then(matmul1, i1)
        .then(matmul2_acc_dec, l)
        .then(matmul2_W_shr_dec, i1)
        .then(matmul2_h_shr_dec, i1)
        .then(matmul2_acc_init, i1)
        .then(matmul2_W_shr_copy, i1)
        .then(matmul2_h_shr_copy, j0)
        .then(matmul2_sync, j0)
        .then(matmul2_acc, j0)
        .then(matmul2, i1)
        .then(sig_i, l)
        .then(tnh_z, i1)
        .then(sig_o, i1)
        .then(sig_f, i1)
        .then(mul_iz, i1)
        .then(mul_fc, i1)
        .then(c, i1)
        .then(tnh_c, i1)
        .then(h, i1)
        .then(y, computation::root)
        .then(copy_y_to_host, computation::root);

    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------

    params.store_in(&buf_params);
    // Weights and biases are packed
    R.store_in(&buf_Weights_gpu, {l, 0, i_merged, j});
    W.store_in(&buf_Weights_gpu, {l, 1, i_merged, j});
    b.store_in(&buf_biases_gpu, {l, i_merged});
    x.store_in(&buf_x_gpu);
    y.store_in(&buf_y_gpu);
    matmul_base.store_in(&buf_workspace, {l, k, i_merged});

    matmul1_R_access.store_in(&buf_Weights_gpu, {l, 0, i_merged, j0 * BLOCK + k % BLOCK});
    matmul1_R_shr_pref.store_in(&buf_matmul1_R_shr, {i_merged % BLOCK, k % BLOCK});
    matmul1_R_glb_to_prefreg.store_in(&buf_matmul1_R_prefreg, {0});
    matmul1_R_prefreg_to_shr.store_in(&buf_matmul1_R_shr, {i_merged % BLOCK, k % BLOCK});
    matmul1_R_shr_access.store_in(&buf_matmul1_R_shr, {i_merged % BLOCK, j % BLOCK});
    matmul1_h_access.store_in(&buf_h, {m - 1, l, k - k % BLOCK + i_merged % BLOCK, j0 * BLOCK + k % BLOCK});
    matmul1_h_shr_pref.store_in(&buf_matmul1_h_shr, {i_merged % BLOCK, k % BLOCK});
    matmul1_h_glb_to_prefreg.store_in(&buf_matmul1_h_prefreg, {0});
    matmul1_h_prefreg_to_shr.store_in(&buf_matmul1_h_shr, {i_merged % BLOCK, k % BLOCK});
    matmul1_h_shr_access.store_in(&buf_matmul1_h_shr, {k % BLOCK, j % BLOCK});
    matmul1_acc_init.store_in(&buf_matmul1_acc, {0});
    matmul1_acc.store_in(&buf_matmul1_acc, {0});
    matmul1.store_in(&buf_workspace, {l, k, i_merged});

    matmul2_W_access.store_in(&buf_Weights_gpu, {l, 1, i_merged, j0 * BLOCK + k % BLOCK});
    matmul2_W_shr_copy.store_in(&buf_matmul2_W_shr, {i_merged % BLOCK, k % BLOCK});
    matmul2_W_shr_access.store_in(&buf_matmul2_W_shr, {i_merged % BLOCK, j % BLOCK});
    matmul2_h_access.store_in(&buf_h, {m, l - 1, k - k % BLOCK + i_merged % BLOCK, j0 * BLOCK + k % BLOCK});
    matmul2_h_shr_copy.store_in(&buf_matmul2_h_shr, {i_merged % BLOCK, k % BLOCK});
    matmul2_h_shr_access.store_in(&buf_matmul2_h_shr, {k % BLOCK, j % BLOCK});
    matmul2_acc_init.store_in(&buf_matmul2_acc, {0});
    matmul2_acc.store_in(&buf_matmul2_acc, {0});
    matmul2.store_in(&buf_workspace, {l, k, i_merged});

    // Workaround for missing store_in feature
    sig_i.set_access("[feature_size] -> {sig_i[m, l, k, i] -> buf_workspace[l, k, i + 0 * feature_size]}");
    tnh_z.set_access("[feature_size] -> {tnh_z[m, l, k, i] -> buf_workspace[l, k, i + 1 * feature_size]}");
    sig_o.set_access("[feature_size] -> {sig_o[m, l, k, i] -> buf_workspace[l, k, i + 2 * feature_size]}");
    sig_f.set_access("[feature_size] -> {sig_f[m, l, k, i] -> buf_workspace[l, k, i + 3 * feature_size]}");
    mul_iz.set_access("[feature_size] -> {mul_iz[m, l, k, i] -> buf_workspace[l, k, i + 0 * feature_size]}");
    mul_fc.set_access("[feature_size] -> {mul_fc[m, l, k, i] -> buf_workspace[l, k, i + 3 * feature_size]}");
    tnh_c.set_access("[feature_size] -> {tnh_c[m, l, k, i] -> buf_workspace[l, k, i + 0 * feature_size]}");
    h.store_in(&buf_h, {m + 1, l + 1, k, i});
    c.store_in(&buf_c, {m + 1, l, k, i});
    h_init.store_in(&buf_h, {0, l + 1, k, i});
    c_init.store_in(&buf_c, {0, l, k, i});
    h_copy_x.store_in(&buf_h, {m + 1, 0, k, i});

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Generate object files.
    tiramisu::codegen({
            &buf_params,
            &buf_Weights,
            &buf_biases,
            &buf_x,
            &buf_y,
        }, "lstm.o", true);

    return 0;
}
