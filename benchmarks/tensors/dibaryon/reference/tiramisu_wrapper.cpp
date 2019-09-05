#include "qblocks_2pt_scalar.h"

#if USE_REFERENCE

void tiramisu_wrapper_make_local_single_double_block(double *t_B1_Blocal_r1_re, double *t_B1_Blocal_r1_im, double *t_B1_Bsingle_r1_re, double *t_B1_Bsingle_r1_im, double *t_B1_Bdouble_r1_re, double *t_B1_Bdouble_r1_im, const double *B1_prop_re, const double *B1_prop_im, const int *src_color_weights_r1, const int *src_spin_weights_r1, const double *src_weights_r1, const double *src_psi_B1_re, const double *src_psi_B1_im, const int Nc, const int Ns, const int Vsrc, const int Vsnk, const int Nt, const int Nw, const int Nq, const int Nsrc);

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

void tiramisu_wrapper_make_local_single_double_block(double *t_B1_Blocal_r1_re, double *t_B1_Blocal_r1_im, double *t_B1_Bsingle_r1_re, double *t_B1_Bsingle_r1_im, double *t_B1_Bdouble_r1_re, double *t_B1_Bdouble_r1_im, const double *B1_prop_re, const double *B1_prop_im, const int *src_color_weights_r1, const int *src_spin_weights_r1, const double *src_weights_r1, const double *src_psi_B1_re, const double *src_psi_B1_im, const int Nc, const int Ns, const int Vsrc, const int Vsnk, const int Nt, const int Nw, const int Nq, const int Nsrc)
{
    long mega = 1024*1024;

    std::cout << "Array sizes" << std::endl;
    std::cout << "Blocal & Prop:" <<  std::endl;
    std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Nt <<  std::endl;
    std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Nt*sizeof(std::complex<double>)/mega << " Mega bytes" << std::endl;
    std::cout << "Bsingle, Bdouble, Q, O & P:" <<  std::endl;
    std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Vsnk*Nt <<  std::endl;
    std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Vsnk*Nt*sizeof(std::complex<double>)/mega << " Mega bytes" <<  std::endl;
    std::cout << std::endl;

    // Blocal
    // B1_Blocal_r1_r: tiramisu real part of Blocal.
    // B1_Blocal_r1_i: tiramisu imaginary part of Blocal.
    Halide::Buffer<double> B1_Blocal_r1_r(t_B1_Blocal_r1_re, {Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B1_Blocal_r1_i(t_B1_Blocal_r1_im, {Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    // prop
    Halide::Buffer<double> B1_prop_r((double *)B1_prop_re, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, Nq});
    Halide::Buffer<double> B1_prop_i((double *)B1_prop_im, {Vsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, Nq});

    // psi
    Halide::Buffer<double> psi_r((double *)src_psi_B1_re, {Vsrc, Nsrc});
    Halide::Buffer<double> psi_i((double *)src_psi_B1_im, {Vsrc, Nsrc});

    Halide::Buffer<int> color_weights_t((int *)src_color_weights_r1, {Nq, Nw});
    Halide::Buffer<int> spin_weights_t((int *)src_spin_weights_r1, {Nq, Nw});
    Halide::Buffer<double> weights_t((double *)src_weights_r1, Nw);

    Halide::Buffer<double> B1_Bsingle_r1_r(t_B1_Bsingle_r1_re, {Vsrc, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B1_Bsingle_r1_i(t_B1_Bsingle_r1_im, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    Halide::Buffer<double> B1_Bdouble_r1_r(t_B1_Bdouble_r1_re, {Vsrc, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});
    Halide::Buffer<double> B1_Bdouble_r1_i(t_B1_Bdouble_r1_im, {Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt});

    tiramisu_make_local_single_double_block(B1_Blocal_r1_r.raw_buffer(),
				    B1_Blocal_r1_i.raw_buffer(),
				    B1_prop_r.raw_buffer(),
				    B1_prop_i.raw_buffer(),
				    psi_r.raw_buffer(),
				    psi_i.raw_buffer(),
				    B1_Bsingle_r1_r.raw_buffer(),
				    B1_Bsingle_r1_i.raw_buffer(),
				    B1_Bdouble_r1_r.raw_buffer(),
				    B1_Bdouble_r1_i.raw_buffer());
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
