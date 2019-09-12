#include "qblocks_2pt_scalar.h"

#if USE_REFERENCE

void tiramisu_wrapper_make_local_single_double_block(double *t_B1_Blocal_r1_re, double *t_B1_Blocal_r1_im, double *t_B1_Bsingle_r1_re, double *t_B1_Bsingle_r1_im, double *t_B1_Bdouble_r1_re, double *t_B1_Bdouble_r1_im, const double *B1_prop_re, const double *B1_prop_im, const double *B2_prop_re, const double *B2_prop_im, const int *src_color_weights_r1, const int *src_spin_weights_r1, const double *src_weights_r1, const double *src_psi_B1_re, const double *src_psi_B1_im, const int Nc, const int Ns, const int Vsrc, const int Nw, const int Nq, const int Nsrc);

#else

////////////////////////////////////////////////////////////
#include <cstdlib>
#include <iostream>
#include <complex>

#include "Halide.h"
#include <tiramisu/utils.h>

#include "tiramisu_generated_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

void tiramisu_wrapper_make_local_single_double_block(double *t_B1_Blocal_r1_re, double *t_B1_Blocal_r1_im, double *t_B1_Bsingle_r1_re, double *t_B1_Bsingle_r1_im, double *t_B1_Bdouble_r1_re, double *t_B1_Bdouble_r1_im, const double *B1_prop_re, const double *B1_prop_im, const double *B2_prop_re, const double *B2_prop_im, const int *src_color_weights_r1, const int *src_spin_weights_r1, const double *src_weights_r1, const double *src_psi_B1_re, const double *src_psi_B1_im, const int Nc, const int Ns, const int Vsrc, const int Nw, const int Nq, const int Nsrc)
{
    // Blocal
    // B1_Blocal_r1_r: tiramisu real part of Blocal.
    // B1_Blocal_r1_i: tiramisu imaginary part of Blocal.
    Halide::Buffer<double> B1_Blocal_r1_r(t_B1_Blocal_r1_re, {Nsrc, Ns, Nc, Ns, Nc, Ns, Nc});
    Halide::Buffer<double> B1_Blocal_r1_i(t_B1_Blocal_r1_im, {Nsrc, Ns, Nc, Ns, Nc, Ns, Nc});

    // prop
    Halide::Buffer<double> B1_prop_r((double *)B1_prop_re, {Vsrc, Ns, Nc, Ns, Nc, Nq});
    Halide::Buffer<double> B1_prop_i((double *)B1_prop_im, {Vsrc, Ns, Nc, Ns, Nc, Nq});
    Halide::Buffer<double> B2_prop_r((double *)B2_prop_re, {Vsrc, Ns, Nc, Ns, Nc, Nq});
    Halide::Buffer<double> B2_prop_i((double *)B2_prop_im, {Vsrc, Ns, Nc, Ns, Nc, Nq});

    // psi
    Halide::Buffer<double> psi_r((double *)src_psi_B1_re, {Nsrc, Vsrc});
    Halide::Buffer<double> psi_i((double *)src_psi_B1_im, {Nsrc, Vsrc});

    Halide::Buffer<int> color_weights_t((int *)src_color_weights_r1, {Nq, Nw});
    Halide::Buffer<int> spin_weights_t((int *)src_spin_weights_r1, {Nq, Nw});
    Halide::Buffer<double> weights_t((double *)src_weights_r1, Nw);

    Halide::Buffer<double> B1_Bsingle_r1_r(t_B1_Bsingle_r1_re, {Nsrc, Ns, Nc, Ns, Nc, Ns, Nc});
    Halide::Buffer<double> B1_Bsingle_r1_i(t_B1_Bsingle_r1_im, {Nsrc, Ns, Nc, Ns, Nc, Ns, Nc});

    Halide::Buffer<double> B1_Bdouble_r1_r(t_B1_Bdouble_r1_re, {Nsrc, Ns, Nc, Ns, Nc, Ns, Nc});
    Halide::Buffer<double> B1_Bdouble_r1_i(t_B1_Bdouble_r1_im, {Nsrc, Ns, Nc, Ns, Nc, Ns, Nc});

    tiramisu_make_local_single_double_block(B1_Blocal_r1_r.raw_buffer(),
				    B1_Blocal_r1_i.raw_buffer(),
				    B1_prop_r.raw_buffer(),
				    B1_prop_i.raw_buffer(),
				    B2_prop_r.raw_buffer(),
				    B2_prop_i.raw_buffer(),
				    B1_Bsingle_r1_r.raw_buffer(),
				    B1_Bsingle_r1_i.raw_buffer(),
				    B1_Bdouble_r1_r.raw_buffer(),
				    B1_Bdouble_r1_i.raw_buffer());
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
