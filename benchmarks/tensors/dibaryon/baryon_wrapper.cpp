#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"

#include "baryon_wrapper.h"
#include "baryon_ref.cpp"

#define RUN_REFERENCE 1
#define RUN_CHECK 1

int main(int, char **)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    std::complex<double> (*Blocal) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt]);
    std::complex<double> (*Bsingle) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt]);
    std::complex<double> (*Bdouble) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Vsnk][Lt]);
    std::complex<double> (*prop) [Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc] = new (std::complex<double> [Nq][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]);
    std::complex<double> (*Q) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]);
    std::complex<double> (*O) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]);
    std::complex<double> (*P) [Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc] = new (std::complex<double> [Nsrc][Nc][Ns][Nc][Ns][Nc][Ns][Vsnk][Lt][Vsrc]);
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
    // Blocal_r: tiramisu real part of Blocal.
    // Blocal_i: tiramisu imaginary part of Blocal.
    Halide::Buffer<double> Blocal_r(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "Blocal_r");
    Halide::Buffer<double> Blocal_i(Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "Blocal_i");

    // prop
    Halide::Buffer<double> prop_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq, "prop_r");
    Halide::Buffer<double> prop_i(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, Nq, "prop_i");

    // psi
    Halide::Buffer<double> psi_r(Vsrc, Nsrc, "psi_r");
    Halide::Buffer<double> psi_i(Vsrc, Nsrc, "psi_i");

    Halide::Buffer<int> color_weights_t(Nq, Nw, "color_weights_t");
    Halide::Buffer<int> spin_weights_t(Nq, Nw, "spin_weights_t");
    Halide::Buffer<double> weights_t(Nw, "weights_t");

    Halide::Buffer<double> Bsingle_r(Vsrc, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "Bsingle_r");
    Halide::Buffer<double> Bsingle_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "Bsingle_i");

    Halide::Buffer<double> Q_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "Q_r");
    Halide::Buffer<double> Q_i(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "Q_i");
    Halide::Buffer<double> O_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "O_r");
    Halide::Buffer<double> O_i(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "O_i");
    Halide::Buffer<double> P_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "P_r");
    Halide::Buffer<double> P_i(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "P_i");

    Halide::Buffer<double> Bdouble_r(Vsrc, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "Bdouble_r");
    Halide::Buffer<double> Bdouble_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Lt, "Bdouble_i");

    std::cout << "Start data initialization." <<  std::endl;

    // Initialization
   for (int wnum=0; wnum<Nw; wnum++)
   {
       weights[wnum] = test_weights[wnum];
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
			    prop[tri][iCprime][iSprime][jCprime][jSprime][x][t][y] = c;
			    prop_r(y, x, jSprime, jCprime, iSprime, iCprime, t, tri) = v1;
			    prop_i(y, x, jSprime, jCprime, iSprime, iCprime, t, tri) = v2;
 		        }

   for (int wnum=0; wnum<Nw; wnum++)
	for (int tri=0; tri<Nq; tri++)
	{
		color_weights[wnum][tri] = test_color_weights[wnum][tri];
		color_weights_t(tri, wnum) = test_color_weights[wnum][tri];
		spin_weights[wnum][tri] = test_spin_weights[wnum][tri];
		spin_weights_t(tri, wnum) = test_spin_weights[wnum][tri];
	}

   std::cout << "End data initialization." <<  std::endl << std::endl;

   int nb_tests = 3;

#if RUN_REFERENCE
    std::cout << "Start reference C code." <<  std::endl;
    for (int i = 0; i < nb_tests; i++)
    {
	    std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	    auto start2 = std::chrono::high_resolution_clock::now();

	    make_local_block(Blocal, prop, color_weights, spin_weights, weights, psi);
	    make_single_block(Bsingle, prop, color_weights, spin_weights, weights, psi, Q);
	    make_double_block(Bdouble, prop, color_weights, spin_weights, weights, psi, O, P);

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


	    tiramisu_generated_code(Blocal_r.raw_buffer(),
				    Blocal_i.raw_buffer(),
				    prop_r.raw_buffer(),
				    prop_i.raw_buffer(),
				    psi_r.raw_buffer(),
				    psi_i.raw_buffer(),
				    Bsingle_r.raw_buffer(),
				    Bsingle_i.raw_buffer(),
				    Bdouble_r.raw_buffer(),
				    Bdouble_i.raw_buffer());

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
				  if (std::abs(Blocal[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][t].real() -
						    Blocal_r(jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t)) >= 0.01)
				  {
				      std::cout << "Error: different computed values for Blocal! Ref = " << Blocal[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][t].real() << " - Tiramisu = " << Blocal_r(jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t) << std::endl;
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
				 if (std::abs(Bsingle[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].real() -
					     Bsingle_r(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t)) >= 0.01)
				  {
				      std::cout << "Error: different computed values for Bsingle! Ref = " << Bsingle[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].real() << " - Tiramisu = " << Bsingle_r(x2, jSprime, jCprime, n, x, kSprime, kCprime, iSprime, iCprime, t) << std::endl;
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
                             if (std::abs(Bdouble[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].real() -
					 Bdouble_r(x2, kSprime, kCprime, n, x, jSprime, jCprime, iSprime, iCprime, t)) >= 0.01)
			      {
				  std::cout << "Error: different computed values for Bdouble! Ref = " << Bdouble[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].real() << " - Tiramisu = " << Bdouble_r(x2, kSprime, kCprime, n, x, jSprime, jCprime, iSprime, iCprime, t) << std::endl;
				  exit(1);
			      }
#endif

    std::cout << "\n\n\033[1;32mSuccess: computed values are equal!\033[0m\n\n" << std::endl;

    return 0;
}
