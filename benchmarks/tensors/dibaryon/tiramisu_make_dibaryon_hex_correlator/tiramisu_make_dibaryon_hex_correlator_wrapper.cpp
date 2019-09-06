#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"

#include "tiramisu_make_dibaryon_hex_correlator_wrapper.h"
#include "tiramisu_make_dibaryon_hex_correlator_ref.cpp"

#define RUN_REFERENCE 1
#define RUN_CHECK 1
int nb_tests = 3;

int main(int, char **)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    std::complex<double> (*B1_Blocal_r1) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt]);
    std::complex<double> (*B1_Blocal_r2) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt]);
    std::complex<double> (*B1_Bsingle_r1) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt]);
    std::complex<double> (*B1_Bsingle_r2) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt]);
    std::complex<double> (*B1_Bdouble_r1) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt]);
    std::complex<double> (*B1_Bdouble_r2) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt]);
    std::complex<double> (*B1_prop) [Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc] = new (std::complex<double> [Nq][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]);
    std::complex<double> (*B1_Q_r1) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]);
    std::complex<double> (*B1_Q_r2) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]);
    std::complex<double> (*B1_O_r1) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]);
    std::complex<double> (*B1_O_r2) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]);
    std::complex<double> (*B1_P_r1) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]);
    std::complex<double> (*B1_P_r2) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]);

    int color_weights[Nw][Nq];
    int spin_weights[Nw][Nq];
    double weights[Nw];
    std::complex<double> psi[Nsrc][Vsrc];

    long mega = 1024*1024;

    std::cout << "Array sizes" << std::endl;
    std::cout << "Blocal & Prop:" <<  std::endl;
    std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Lt <<  std::endl;
    std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Lt*sizeof(std::complex<double>)/mega << " Mega bytes" << std::endl;
    std::cout << "Bsingle, Bdouble, Q, O & P:" <<  std::endl;
    std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Vsnk*Lt <<  std::endl;
    std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Vsnk*Lt*sizeof(std::complex<double>)/mega << " Mega bytes" <<  std::endl;
    std::cout << std::endl;

    // Blocal
    // B1_Blocal_r1_r: tiramisu real part of Blocal.
    // B1_Blocal_r1_i: tiramisu imaginary part of Blocal.
    Halide::Buffer<double> B1_Blocal_r1_r(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "B1_Blocal_r1_r");
    Halide::Buffer<double> B1_Blocal_r1_i(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "B1_Blocal_r1_i");

    // prop
    Halide::Buffer<double> B1_prop_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq, "B1_prop_r");
    Halide::Buffer<double> B1_prop_i(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq, "B1_prop_i");

    // psi
    Halide::Buffer<double> psi_r(Vsrc, Nsrc, "psi_r");
    Halide::Buffer<double> psi_i(Vsrc, Nsrc, "psi_i");

    Halide::Buffer<int> color_weights_t(Nq, Nw, "color_weights_t");
    Halide::Buffer<int> spin_weights_t(Nq, Nw, "spin_weights_t");
    Halide::Buffer<double> weights_t(Nw, "weights_t");

    Halide::Buffer<double> B1_Bsingle_r1_r(Vsrc, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "B1_Bsingle_r1_r");
    Halide::Buffer<double> B1_Bsingle_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "B1_Bsingle_r1_i");

    Halide::Buffer<double> B1_Bdouble_r1_r(Vsrc, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "B1_Bdouble_r1_r");
    Halide::Buffer<double> B1_Bdouble_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "B1_Bdouble_r1_i");

    std::cout << "Start data initialization." <<  std::endl;

    // Initialization
   for (int wnum=0; wnum<Nw; wnum++)
   {
       weights[wnum] = src_weights_r1[wnum];
   }

   for (int n=0; n<Nsrc; n++)
     for (int y=0; y<Vsrc; y++)
     {
	double v1 = rand()%10;
	double v2 = rand()%10;
	std::complex<double> c(v1, v2);
	psi[n][y] = c;
	psi_r(y, n) = v1;
	psi_i(y, n) = v2;
     }

   for (int tri=0; tri<Nq; tri++)
       for (int iCprime=0; iCprime<Nc; iCprime++)
	  for (int iSprime=0; iSprime<Ns; iSprime++)
	     for (int jCprime=0; jCprime<Nc; jCprime++)
		for (int jSprime=0; jSprime<Ns; jSprime++)
                   for (int x=0; x<Vsnk; x++)
                      for (int t=0; t<Lt; t++)
		        for (int y=0; y<Vsrc; y++)
			{
			    double v1 = rand()%10;
			    double v2 = rand()%10;
			    std::complex<double> c(v1, v2);
			    B1_prop[tri][iCprime][iSprime][jCprime][jSprime][x][t][y] = c;
			    B1_prop_r(y, x, jSprime, jCprime, iSprime, iCprime, t, tri) = v1;
			    B1_prop_i(y, x, jSprime, jCprime, iSprime, iCprime, t, tri) = v2;
 		        }

   for (int wnum=0; wnum<Nw; wnum++)
	for (int tri=0; tri<Nq; tri++)
	{
		color_weights[wnum][tri] = src_color_weights_r1[wnum][tri];
		color_weights_t(tri, wnum) = src_color_weights_r1[wnum][tri];
		spin_weights[wnum][tri] = src_spin_weights_r1[wnum][tri];
		spin_weights_t(tri, wnum) = src_spin_weights_r1[wnum][tri];
	}

   std::cout << "End data initialization." <<  std::endl << std::endl;

#if RUN_REFERENCE
    std::cout << "Start reference C code." <<  std::endl;
    for (int i = 0; i < nb_tests; i++)
    {
	    std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	    auto start2 = std::chrono::high_resolution_clock::now();

	    make_local_block(B1_Blocal_r1, B1_prop, color_weights, spin_weights, weights, psi);
	    make_single_block(B1_Bsingle_r1, B1_prop, color_weights, spin_weights, weights, psi, B1_Q_r1);
	    make_double_block(B1_Bdouble_r1, B1_prop, color_weights, spin_weights, weights, psi, B1_O_r1, B1_P_r1);

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


	    tiramisu_make_dibaryon_hex_correlator(B1_Blocal_r1_r.raw_buffer(),
				    B1_Blocal_r1_i.raw_buffer(),
				    B1_prop_r.raw_buffer(),
				    B1_prop_i.raw_buffer(),
				    psi_r.raw_buffer(),
				    psi_i.raw_buffer(),
				    B1_Bsingle_r1_r.raw_buffer(),
				    B1_Bsingle_r1_i.raw_buffer(),
				    B1_Bdouble_r1_r.raw_buffer(),
				    B1_Bdouble_r1_i.raw_buffer());

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
			      for (int t=0; t<Lt; t++)
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
				 for (int t=0; t<Lt; t++)
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
			     for (int t=0; t<Lt; t++)
                             if ((std::abs(B1_Bdouble_r1[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].real() -
					   B1_Bdouble_r1_r(x2, iSprime, iCprime, n, x, kSprime, kCprime, jSprime, jCprime, t)) >= 0.01) ||
				 (std::abs(B1_Bdouble_r1[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].imag() -
					   B1_Bdouble_r1_i(x2, iSprime, iCprime, n, x, kSprime, kCprime, jSprime, jCprime, t)) >= 0.01))
			      {
				  std::cout << "Error: different computed values for B1_Bdouble_r1!" << std::endl;
				  exit(1);
			      }
#endif

    std::cout << "\n\n\033[1;32mSuccess: computed values are equal!\033[0m\n\n" << std::endl;

    return 0;
}
