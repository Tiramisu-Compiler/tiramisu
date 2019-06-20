#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"

#include "baryon_wrapper.h"
#include "baryon_ref.cpp"

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
    std::cout << "Blocal:" <<  std::endl;
    std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Lt <<  std::endl;
    std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Lt*sizeof(std::complex<double>)/mega << " Mega bytes" << std::endl;
    std::cout << "Bsingle & Bdouble:" <<  std::endl;
    std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Vsnk*Lt <<  std::endl;
    std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Vsnk*Lt*sizeof(std::complex<double>)/mega << " Mega bytes" <<  std::endl;
    std::cout << "Prop:" <<  std::endl;
    std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Lt <<  std::endl;
    std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Lt*sizeof(std::complex<double>)/mega <<  std::endl;
    std::cout << "Q, O & P:" <<  std::endl;
    std::cout << "	Max index size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Lt*Vsrc <<  std::endl;
    std::cout << "	Array size = " << Nsrc*Nc*Ns*Nc*Ns*Nc*Ns*Vsnk*Lt*Vsrc*sizeof(std::complex<double>)/mega << " Mega bytes" <<  std::endl;
    std::cout << std::endl;

    // Blocal
    // Blocal_r: tiramisu real part of Blocal.
    // Blocal_i: tiramisu imaginary part of Blocal.
    Halide::Buffer<double> Blocal_r(Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "Blocal_r");
    Halide::Buffer<double> Blocal_i(Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "Blocal_i");

    // prop
    Halide::Buffer<double> prop_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Nq, Lt, "prop_r");
    Halide::Buffer<double> prop_i(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Nq, Lt, "prop_i");

    // psi
    Halide::Buffer<double> psi_r(Vsrc, Nsrc, "psi_r");
    Halide::Buffer<double> psi_i(Vsrc, Nsrc, "psi_i");

    Halide::Buffer<int> color_weights_t(Nq, Nw, "color_weights_t");
    Halide::Buffer<int> spin_weights_t(Nq, Nw, "spin_weights_t");
    Halide::Buffer<double> weights_t(Nw, "weights_t");

    Halide::Buffer<double> Bsingle_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "Bsingle_r");
    Halide::Buffer<double> Bsingle_i(Vsnk, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "Bsingle_i");

    Halide::Buffer<double> Q_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "Q_r");
    Halide::Buffer<double> Q_i(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "Q_i");
    Halide::Buffer<double> O_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "O_r");
    Halide::Buffer<double> O_i(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "O_i");
    Halide::Buffer<double> P_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "P_r");
    Halide::Buffer<double> P_i(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "P_i");

    Halide::Buffer<double> Bdouble_r(Vsrc, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "Bdouble_r");
    Halide::Buffer<double> Bdouble_i(Vsnk, Vsnk, Ns, Nc, Ns, Nc, Ns, Nc, Nsrc, Lt, "Bdouble_i");

    std::cout << "Start data initialization." <<  std::endl;

    // Initialization
   for (int wnum=0; wnum<Nw; wnum++)
   {
       double v = rand()%10;
       weights[wnum] = v;
       weights_t(wnum) = v;
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
			    prop_r(y, x, jSprime, jCprime, iSprime, iCprime, tri, t) = v1;
			    prop_i(y, x, jSprime, jCprime, iSprime, iCprime, tri, t) = v2;
 		        }

   for (int wnum=0; wnum<Nw; wnum++)
	for (int tri=0; tri<Nq; tri++)
	{
		color_weights[wnum][tri] = 0; // tri
		color_weights_t(tri, wnum) = 0; //tri
		spin_weights[wnum][tri] = 0; //tri
		spin_weights_t(tri, wnum) = 0; //tri
	}

   std::cout << "End data initialization." <<  std::endl << std::endl;

   int nb_tests = 1;

   std::cout << "Start reference C code." <<  std::endl;

    for (int i = 0; i < nb_tests; i++)
    {
	    std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	    auto start2 = std::chrono::high_resolution_clock::now();

	    //make_local_block(Blocal, prop, color_weights, spin_weights, weights, psi);
	    make_single_block(Bsingle, prop, color_weights, spin_weights, weights, psi, Q);
	    //make_double_block(Bdouble, prop, color_weights, spin_weights, weights, psi, O, P);

	    auto end2 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	    duration_vector_2.push_back(duration2);
    }

    std::cout << "End reference C code." <<  std::endl;
    std::cout << "Start Tiramisu code." <<  std::endl;

    for (int i = 0; i < nb_tests; i++)
    {
	    std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	    auto start1 = std::chrono::high_resolution_clock::now();

	    tiramisu_generated_code(Blocal_r.raw_buffer(),
				    Blocal_i.raw_buffer(),
				    prop_r.raw_buffer(),
				    prop_i.raw_buffer(),
				    weights_t.raw_buffer(),
				    psi_r.raw_buffer(),
				    psi_i.raw_buffer(),
				    color_weights_t.raw_buffer(),
				    spin_weights_t.raw_buffer(),
				    Bsingle_r.raw_buffer(),
				    Bsingle_i.raw_buffer(),
				    Bdouble_r.raw_buffer(),
				    Bdouble_i.raw_buffer(),
				    O_r.raw_buffer(),
				    O_i.raw_buffer(),
				    P_r.raw_buffer(),
				    P_i.raw_buffer(),
				    Q_r.raw_buffer(),
				    Q_i.raw_buffer());

	    auto end1 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	    duration_vector_1.push_back(duration1);
    }
    std::cout << "Start Tiramisu code." <<  std::endl;

    print_time("performance_CPU.csv", "dibaryon", {"Ref", "Tiramisu"}, {median(duration_vector_2), median(duration_vector_1)});
    std::cout << "\nSpeedup = " << median(duration_vector_2)/median(duration_vector_1) << std::endl;

#if 0
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
						    Blocal_r(x, kSprime, kCprime, jSprime, jCprime, iSprime, iCprime, n, t)) >= 0.01)
				  {
				      std::cout << "Error: different computed values for Blocal! Ref = " << Blocal[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][t].real() << " - Tiramisu = " << Blocal_r(x, kSprime, kCprime, jSprime, jCprime, iSprime, iCprime, n, t) << std::endl;
				      exit(1);
				  }
#endif

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
					     Bsingle_r(x2, x, kSprime, kCprime, jSprime, jCprime, iSprime, iCprime, n, t)) >= 0.01)
				  {
				      std::cout << "Error: different computed values for Bsingle! Ref = " << Bsingle[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].real() << " - Tiramisu = " << Bsingle_r(x2, x, kSprime, kCprime, jSprime, jCprime, iSprime, iCprime, n, t) << std::endl;
				    std::cout << "Position: (" << t << ", " << n << ", " << iCprime << ", " << iSprime << ", " << jCprime << ", " << jSprime << ", " << kCprime << ", " << kSprime << ", " << x << ", " << x2 << ")" << std::endl;
				      exit(1);
				  }

#if 0
	for (int jCprime=0; jCprime<Nc; jCprime++)
	  for (int jSprime=0; jSprime<Ns; jSprime++)
	     for (int kCprime=0; kCprime<Nc; kCprime++)
		for (int kSprime=0; kSprime<Ns; kSprime++)
		   for (int jC=0; jC<Nc; jC++)
		      for (int jS=0; jS<Ns; jS++)
			 for (int x=0; x<Vsnk; x++)
			    for (int t=0; t<Lt; t++)
			       for (int y=0; y<Vsrc; y++)
				  for (int n=0; n<Nsrc; n++)
					  if (std::abs(O[n][jCprime][jSprime][kCprime][kSprime][jC][jS][x][t][y].real() -
							    O_r(y, x, jS, jC, kSprime, kCprime, jSprime, jCprime, n, t)) >= 0.01)
					  {
						    std::cout << "Error: different computed values for O! Ref = " << O[n][jCprime][jSprime][kCprime][kSprime][jC][jS][x][t][y].real()
							    << " - Tiramisu = " << O_r(y, x, jS, jC, kSprime, kCprime, jSprime, jCprime, n, t) << std::endl;
						    exit(1);
					  }

	for (int jCprime=0; jCprime<Nc; jCprime++)
	  for (int jSprime=0; jSprime<Ns; jSprime++)
	     for (int kCprime=0; kCprime<Nc; kCprime++)
		for (int kSprime=0; kSprime<Ns; kSprime++)
		   for (int jC=0; jC<Nc; jC++)
		      for (int jS=0; jS<Ns; jS++)
			 for (int x=0; x<Vsnk; x++)
			    for (int t=0; t<Lt; t++)
			       for (int y=0; y<Vsrc; y++)
				  for (int n=0; n<Nsrc; n++)
					  if (std::abs(P[n][jCprime][jSprime][kCprime][kSprime][jC][jS][x][t][y].real() -
							    P_r(y, x, jS, jC, kSprime, kCprime, jSprime, jCprime, n, t)) >= 0.01)
					  {
						    std::cout << "Error: different computed values for P! Ref = " << P[n][jCprime][jSprime][kCprime][kSprime][jC][jS][x][t][y].real()
							    << " - Tiramisu = " << P_r(y, x, jS, jC, kSprime, kCprime, jSprime, jCprime, n, t) << std::endl;
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
					 Bdouble_r(x2, x, kSprime, kCprime, jSprime, jCprime, iSprime, iCprime, n, t)) >= 0.01)
			      {
				  std::cout << "Error: different computed values for Bdouble! Ref = " << Bdouble[n][iCprime][iSprime][jCprime][jSprime][kCprime][kSprime][x][x2][t].real() << " - Tiramisu = " << Bdouble_r(x2, x, kSprime, kCprime, jSprime, jCprime, iSprime, iCprime, n, t) << std::endl;
				  std::cout << "Position: (" << t << ", " << n << ", " << iCprime << ", " << iSprime << ", " << jCprime << ", " << jSprime << ", " << kCprime << ", " << kSprime << ", " << x << ", " << x2 << ")" << std::endl;
				  exit(1);
			      }
#endif

    std::cout << "\n\n\033[1;32mSuccess: computed values are equal!\033[0m\n\n" << std::endl;

    return 0;
}
