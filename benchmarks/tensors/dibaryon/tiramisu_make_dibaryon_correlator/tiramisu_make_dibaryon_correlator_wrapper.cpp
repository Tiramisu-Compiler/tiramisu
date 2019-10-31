#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"

#include "tiramisu_make_dibaryon_correlator_wrapper.h"
#include "tiramisu_make_dibaryon_correlator_ref.cpp"

#define RUN_REFERENCE 1
#define RUN_CHECK 0
int nb_tests = 3;

int main(int, char **)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    long mega = 1024*1024;

    std::cout << "Array sizes" << std::endl;
    std::cout << "Blocal & Prop:" <<  std::endl;
    std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Nt <<  std::endl;
    std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Nt*sizeof(std::complex<double>)/mega << " Mega bytes" << std::endl;
    std::cout << "Bsingle, Bdouble, Q, O & P:" <<  std::endl;
    std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Vsnk*Nt <<  std::endl;
    std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Vsnk*Nt*sizeof(std::complex<double>)/mega << " Mega bytes" <<  std::endl;
    std::cout << std::endl;

    // C_r and C_i
    Halide::Buffer<double> C_r(Nt, Nsnk, Nsrc, "C_r");
    Halide::Buffer<double> C_i(Nt, Nsnk, Nsrc, "C_i");

    Halide::Buffer<double> ref_C_r(Nt, Nsnk, Nsrc, "C_r");
    Halide::Buffer<double> ref_C_i(Nt, Nsnk, Nsrc, "C_i");


    // Blocal
    // B1_Blocal_r1_r: tiramisu real part of Blocal.
    // B1_Blocal_r1_i: tiramisu imaginary part of Blocal.
    Halide::Buffer<double>        B1_Blocal_r1_r(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B1_Blocal_r1_r");
    Halide::Buffer<double>        B1_Blocal_r1_i(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B1_Blocal_r1_i");
    Halide::Buffer<double> B1_Bsingle_r1_r(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B1_Bsingle_r1_r");
    Halide::Buffer<double> B1_Bsingle_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B1_Bsingle_r1_i");
    Halide::Buffer<double> B1_Bdouble_r1_r(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B1_Bdouble_r1_r");
    Halide::Buffer<double> B1_Bdouble_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B1_Bdouble_r1_i");

    Halide::Buffer<double>        B2_Blocal_r1_r(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B2_Blocal_r1_r");
    Halide::Buffer<double>        B2_Blocal_r1_i(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B2_Blocal_r1_i");
    Halide::Buffer<double> B2_Bsingle_r1_r(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B2_Bsingle_r1_r");
    Halide::Buffer<double> B2_Bsingle_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B2_Bsingle_r1_i");
    Halide::Buffer<double> B2_Bdouble_r1_r(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B2_Bdouble_r1_r");
    Halide::Buffer<double> B2_Bdouble_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B2_Bdouble_r1_i");

    Halide::Buffer<double>        t_B1_Blocal_r1_r(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B1_Blocal_r1_r");
    Halide::Buffer<double>        t_B1_Blocal_r1_i(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B1_Blocal_r1_i");
    Halide::Buffer<double> t_B1_Bsingle_r1_r(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B1_Bsingle_r1_r");
    Halide::Buffer<double> t_B1_Bsingle_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B1_Bsingle_r1_i");
    Halide::Buffer<double> t_B1_Bdouble_r1_r(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B1_Bdouble_r1_r");
    Halide::Buffer<double> t_B1_Bdouble_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B1_Bdouble_r1_i");

    Halide::Buffer<double>        t_B2_Blocal_r1_r(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B2_Blocal_r1_r");
    Halide::Buffer<double>        t_B2_Blocal_r1_i(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B2_Blocal_r1_i");
    Halide::Buffer<double> t_B2_Bsingle_r1_r(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B2_Bsingle_r1_r");
    Halide::Buffer<double> t_B2_Bsingle_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B2_Bsingle_r1_i");
    Halide::Buffer<double> t_B2_Bdouble_r1_r(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B2_Bdouble_r1_r");
    Halide::Buffer<double> t_B2_Bdouble_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "t_B2_Bdouble_r1_i");

    for (int i = 0; i < Nt; i++)
      for (int j = 0; j < Nsnk; j++)
        for (int k = 0; k < Nsrc; k++)
        {
          C_r(i, j, k) = (double) 1;
	  C_i(i, j, k) = (double) 1;
	  ref_C_r(i, j, k) = (double) 1;
	  ref_C_i(i, j, k) = (double) 1;
        }

    for (int i0 = 0; i0 < Ns; i0++)
      for (int i1 = 0; i1 < Nc; i1++)
        for (int i2 = 0; i2 < Nsrc; i2++)
          for (int i3 = 0; i3 < Vsnk; i3++)
            for (int i4 = 0; i4 < Ns; i4++)
	      for (int i5 = 0; i5 < Nc; i5++)
	   	for (int i6 = 0; i6 < Ns; i6++)
		  for (int i7 = 0; i7 < Nc; i7++)
		    for (int t = 0; t < Nt; t++)
		    {
			B1_Blocal_r1_r(i0, i1, i2, i3, i4, i5, i6, i7, t) = (double) 1;
			B1_Blocal_r1_i(i0, i1, i2, i3, i4, i5, i6, i7, t) = (double) 1;
			B2_Blocal_r1_r(i0, i1, i2, i3, i4, i5, i6, i7, t) = (double) 1;
			B2_Blocal_r1_i(i0, i1, i2, i3, i4, i5, i6, i7, t) = (double) 1;
		    }

    for (int m=0; m<Nsrc; m++)
      for (int iCprime=0; iCprime<Nc; iCprime++)
         for (int iSprime=0; iSprime<Ns; iSprime++)
            for (int jCprime=0; jCprime<Nc; jCprime++)
              for (int jSprime=0; jSprime<Ns; jSprime++)
                 for (int kCprime=0; kCprime<Nc; kCprime++)
                    for (int kSprime=0; kSprime<Ns; kSprime++)
                      for (int x=0; x<Vsnk; x++)
                        for (int x2=0; x2<Vsnk; x2++)
                          for (int t=0; t<Nt; t++)
			  {
                             B1_Bsingle_r1_r(t, x2, x, kSprime, kCprime, jSprime, jCprime, iSprime, iCprime, m) = (double) 1;
			  }

    for (int m=0; m<Nsrc; m++)
      for (int iCprime=0; iCprime<Nc; iCprime++)
         for (int iSprime=0; iSprime<Ns; iSprime++)
            for (int jCprime=0; jCprime<Nc; jCprime++)
              for (int jSprime=0; jSprime<Ns; jSprime++)
                 for (int kCprime=0; kCprime<Nc; kCprime++)
                    for (int kSprime=0; kSprime<Ns; kSprime++)
                      for (int x=0; x<Vsnk; x++)
                        for (int x2=0; x2<Vsnk; x2++)
                          for (int t=0; t<Nt; t++)
			  {
                             t_B1_Bsingle_r1_r(x2, jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t) = (double) 1;
			  }

    int perms_array[36][6] = { {1,2,3,4,5,6}, {1, 4, 3, 2, 5, 6}, {1, 6, 3, 2, 5, 4}, {1, 2, 3, 6, 5, 4}, {1, 4, 3, 6, 5, 2}, {1, 6, 3, 4, 5, 2}, {3, 2, 1, 4, 5, 6}, {3, 4, 1, 2, 5, 6}, {3, 6, 1, 2, 5, 4}, {3, 2, 1, 6, 5, 4}, {3, 4, 1, 6, 5, 2}, {3, 6, 1, 4, 5, 2}, {5, 2, 1, 4, 3, 6}, {5, 4, 1, 2, 3, 6}, {5, 6, 1, 2, 3, 4}, {5, 2, 1, 6, 3, 4}, {5, 4, 1, 6, 3, 2}, {5, 6, 1, 4, 3, 2}, {1, 2, 5, 4, 3, 6}, {1, 4, 5, 2, 3, 6}, {1, 6, 5, 2, 3, 4}, {1, 2, 5, 6, 3, 4}, {1, 4, 5, 6, 3, 2}, {1, 6, 5, 4, 3, 2}, {3, 2, 5, 4, 1, 6}, {3, 4, 5, 2, 1, 6}, {3, 6, 5, 2, 1, 4}, {3, 2, 5, 6, 1, 4}, {3, 4, 5, 6, 1, 2}, {3, 6, 5, 4, 1, 2}, {5, 2, 3, 4, 1, 6}, {5, 4, 3, 2, 1, 6}, {5, 6, 3, 2, 1, 4}, {5, 2, 3, 6, 1, 4}, {5, 4, 3, 6, 1, 2}, {5, 6, 3, 4, 1, 2} };
   int sigs_array[36] = {1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1};
   int* perms = (int *) malloc(Nperms * 2*Nq * sizeof (int));
   int sigs[Nperms];
   int permnum = 0;
   for (int i = 0; i < 36; i++) {
      if (perms_array[i][0] > perms_array[i][2]) {
         continue;
      }
      else if (perms_array[i][3] > perms_array[i][5]) {
         continue;
      }
      else {
         for (int q = 0; q < 2*Nq; q++) {
            perms[index_2d(permnum,q ,2*Nq)] = perms_array[i][q];
         }
         sigs[permnum] = sigs_array[i];
         permnum += 1;
      }
    }

    double* overall_weight = (double*) malloc(1 * sizeof (double)); 
    int* snk_color_weights = (int*) malloc(2 * Nw2 * Nq * sizeof (int)); 
    int* snk_spin_weights = (int *) malloc(2 * Nw2 * Nq * sizeof (int));
    double* snk_weights = (double *) malloc(Nw2 * sizeof (double));
    double* snk_psi_re = (double *) malloc(Vsnk * Vsnk * Nsnk * sizeof (double));
    double* snk_psi_im = (double *) malloc(Vsnk * Vsnk * Nsnk * sizeof (double));

    overall_weight[0] = 1;

    for (int i = 0; i<2; i++)
	    for (int j = 0; j<Nw2; j++)
		    for (int k = 0; k<Nq; k++)
		    {
			snk_color_weights[i*Nw2*Nq + j*Nq + k] = 1;
			snk_spin_weights[i*Nw2*Nq + j*Nq + k] = 1;
			snk_weights[i*Nw2*Nq + j*Nq + k] = (double) 1;
		    }

    for (int i = 0; i<Vsnk; i++)
	for (int k = 0; k<Vsnk; k++)
	    for (int j = 0; j<Nsnk; j++)
		    {
			snk_psi_re[i*Nsnk*Vsnk + k*Nsnk + j] = (double) 1;
			snk_psi_im[i*Nsnk*Vsnk + k*Nsnk + j] = (double) 1;
		    }

    // Halide buffers
    Halide::Buffer<int> b_perms(perms, {2*Nq, Nperms});
    Halide::Buffer<int> b_sigs(sigs, Nperms);
    Halide::Buffer<double> b_overall_weight(overall_weight, 1);
    Halide::Buffer<int> b_snk_color_weights(snk_color_weights, {Nq, Nw2, 2});
    Halide::Buffer<int> b_snk_spin_weights(snk_spin_weights, {Nq, Nw2, 2});
    Halide::Buffer<double> b_snk_weights(snk_weights, Nw2);
    Halide::Buffer<double> b_snk_psi_re(snk_psi_re, {Nsnk, Vsnk, Vsnk});
    Halide::Buffer<double> b_snk_psi_im(snk_psi_im, {Nsnk, Vsnk, Vsnk});

#if RUN_REFERENCE
    std::cout << "Start reference C code." <<  std::endl;
    for (int i = 0; i < nb_tests; i++)
    {
	    std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	    auto start2 = std::chrono::high_resolution_clock::now();

	    make_dibaryon_correlator(
				    (double *) ref_C_r.raw_buffer()->host,
				    (double *) ref_C_i.raw_buffer()->host,
				    (const double *) B1_Blocal_r1_r.raw_buffer()->host,
				    (const double *) B1_Blocal_r1_i.raw_buffer()->host,
				    (const double *) B1_Bsingle_r1_r.raw_buffer()->host,
				    (const double *) B1_Bsingle_r1_i.raw_buffer()->host,
				    (const double *) B1_Bdouble_r1_r.raw_buffer()->host,
				    (const double *) B1_Bdouble_r1_i.raw_buffer()->host,

				    (const double *) B2_Blocal_r1_r.raw_buffer()->host,
				    (const double *) B2_Blocal_r1_i.raw_buffer()->host,
				    (const double *) B2_Bsingle_r1_r.raw_buffer()->host,
				    (const double *) B2_Bsingle_r1_i.raw_buffer()->host,
				    (const double *) B2_Bdouble_r1_r.raw_buffer()->host,
				    (const double *) B2_Bdouble_r1_i.raw_buffer()->host,

				    (const int *) b_perms.raw_buffer()->host,
				    (const int *) b_sigs.raw_buffer()->host,
				    (const double) *(b_overall_weight.raw_buffer()->host),
				    (const int *) b_snk_color_weights.raw_buffer()->host,
				    (const int *) b_snk_spin_weights.raw_buffer()->host,
				    (const double *) b_snk_weights.raw_buffer()->host,
				    (const double *) b_snk_psi_re.raw_buffer()->host,
				    (const double *) b_snk_psi_im.raw_buffer()->host);

	    auto end2 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	    duration_vector_2.push_back(duration2);
    }
    std::cout << "End reference C code." <<  std::endl;
#endif

    std::cout << "Start Tiramisu code." <<  std::endl;

    for (int i = 0; i < nb_tests; i++)
    {
	    std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	    auto start1 = std::chrono::high_resolution_clock::now();

	    tiramisu_make_dibaryon_correlator(
				    C_r.raw_buffer(),
				    C_i.raw_buffer(),
				    B1_Blocal_r1_r.raw_buffer(),
				    B1_Blocal_r1_i.raw_buffer(),
				    t_B1_Bsingle_r1_r.raw_buffer(), //TODO: all of the buffers passed to this function should be be prefixed with t_.
								    // They should also be initialized.
				    B1_Bsingle_r1_i.raw_buffer(),
				    B1_Bdouble_r1_r.raw_buffer(),
				    B1_Bdouble_r1_i.raw_buffer(),

				    B2_Blocal_r1_r.raw_buffer(),
				    B2_Blocal_r1_i.raw_buffer(),
				    B2_Bsingle_r1_r.raw_buffer(),
				    B2_Bsingle_r1_i.raw_buffer(),
				    B2_Bdouble_r1_r.raw_buffer(),
				    B2_Bdouble_r1_i.raw_buffer(),

				    b_perms.raw_buffer(),
				    b_sigs.raw_buffer(),
				    b_overall_weight.raw_buffer(),
				    b_snk_color_weights.raw_buffer(),
				    b_snk_spin_weights.raw_buffer(),
				    b_snk_weights.raw_buffer(),
				    b_snk_psi_re.raw_buffer(),
				    b_snk_psi_im.raw_buffer());

	    auto end1 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	    duration_vector_1.push_back(duration1);
    }
    std::cout << "End Tiramisu code." <<  std::endl;

    print_time("performance_CPU.csv", "dibaryon", {"Ref", "Tiramisu"}, {median(duration_vector_2), median(duration_vector_1)});
    std::cout << "\nSpeedup = " << median(duration_vector_2)/median(duration_vector_1) << std::endl;

#if RUN_CHECK
    // Compare outputs.
	for (int n=0; n<Nsrc; n++)
	  for (int iCprime=0; iCprime<Nc; iCprime++)
	    for (int iSprime=0; iSprime<Ns; iSprime++)
	       for (int jCprime=0; jCprime<Nc; jCprime++)
		  for (int jSprime=0; jSprime<Ns; jSprime++)
		     for (int kCprime=0; kCprime<Nc; kCprime++)
			for (int kSprime=0; kSprime<Ns; kSprime++)
			   for (int x=0; x<Vsnk; x++)
			      for (int t=0; t<Nt; t++)
				  if ((std::abs(B1_Blocal_r1[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][t].real() -
					        B1_Blocal_r1_r(jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t)) >= 0.01) ||
				      (std::abs(B1_Blocal_r1[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][t].imag() -
					        B1_Blocal_r1_i(jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t)) >= 0.01))
				  {
				      std::cout << "Error: different computed values for B1_Blocal_r1!" << std::endl;
				      exit(1);
				  }

	for (int n=0; n<Nsrc; n++)
	  for (int iCprime=0; iCprime<Nc; iCprime++)
	    for (int iSprime=0; iSprime<Ns; iSprime++)
	       for (int jCprime=0; jCprime<Nc; jCprime++)
		  for (int jSprime=0; jSprime<Ns; jSprime++)
		     for (int kCprime=0; kCprime<Nc; kCprime++)
			for (int kSprime=0; kSprime<Ns; kSprime++)
			   for (int x=0; x<Vsnk; x++)
			     for (int x2=0; x2<Vsnk; x2++)
				 for (int t=0; t<Nt; t++)
				 if ((std::abs(B1_Bsingle_r1[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].real() -
					       B1_Bsingle_r1_r(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t)) >= 0.01) ||
				     (std::abs(B1_Bsingle_r1[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].imag() -
					       B1_Bsingle_r1_i(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t)) >= 0.01))
				  {
				      std::cout << "Error: different computed values for B1_Bsingle_r1!" << std::endl;
				      exit(1);
				  }

    for (int n=0; n<Nsrc; n++)
      for (int iCprime=0; iCprime<Nc; iCprime++)
        for (int iSprime=0; iSprime<Ns; iSprime++)
           for (int jCprime=0; jCprime<Nc; jCprime++)
              for (int jSprime=0; jSprime<Ns; jSprime++)
                 for (int kCprime=0; kCprime<Nc; kCprime++)
                    for (int kSprime=0; kSprime<Ns; kSprime++)
                       for (int x=0; x<Vsnk; x++)
		         for (int x2=0; x2<Vsnk; x2++)
			     for (int t=0; t<Nt; t++)
                             if ((std::abs(B1_Bdouble_r1[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].real() -
					   B1_Bdouble_r1_r(x2, iSprime, iCprime, n, x, kSprime, kCprime, jSprime, jCprime, t)) >= 0.01) ||
				 (std::abs(B1_Bdouble_r1[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].imag() -
					   B1_Bdouble_r1_i(x2, iSprime, iCprime, n, x, kSprime, kCprime, jSprime, jCprime, t)) >= 0.01))
			      {
				  std::cout << "Error: different computed values for B1_Bdouble_r1!" << std::endl;
				  exit(1);
			      }

    for (int i = 0; i < Nt; i++)
      for (int j = 0; j < Nsnk; j++)
        for (int k = 0; k < Nsrc; k++)
           if ((std::abs(ref_C_r(i,j,k) - C_r(i,j,k)) >= 0.01) ||
	       (std::abs(ref_C_i(i,j,k) - C_i(i,j,k)) >= 0.01))
	    {
		  std::cout << "Error: different computed values for C_r or C_i!" << std::endl;
		  exit(1);
	    }

#endif

    std::cout << "\n\n\033[1;32mSuccess: computed values are equal!\033[0m\n\n" << std::endl;

    return 0;
}
