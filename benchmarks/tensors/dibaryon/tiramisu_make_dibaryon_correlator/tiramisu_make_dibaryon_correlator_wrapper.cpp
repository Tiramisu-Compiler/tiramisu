#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"

#include "tiramisu_make_dibaryon_correlator_wrapper.h"
#include "tiramisu_make_dibaryon_correlator_ref.cpp"

#define RUN_REFERENCE 1
#define RUN_CHECK 1
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
    Halide::Buffer<double> B2_Bsingle_r1_r(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B1_Bsingle_r1_r");
    Halide::Buffer<double> B2_Bsingle_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B1_Bsingle_r1_i");
    Halide::Buffer<double> B2_Bdouble_r1_r(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B1_Bdouble_r1_r");
    Halide::Buffer<double> B2_Bdouble_r1_i(Vsnk, Ns, Nc, Nsrc, Vsnk, Ns, Nc, Ns, Nc, Nt, "B1_Bdouble_r1_i");

   double B1_Bsingle_r1_re[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc];
   double B1_Bdouble_r1_re[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc];
	double B1_Blocal_r1_re[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc];
   double B1_Bsingle_r1_im[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc];
   double B1_Bdouble_r1_im[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc];
	double B1_Blocal_r1_im[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc];

   double B2_Bsingle_r1_re[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc];
   double B2_Bdouble_r1_re[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc];
	double B2_Blocal_r1_re[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc];
   double B2_Bsingle_r1_im[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc];
   double B2_Bdouble_r1_im[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Vsnk * Nsrc];
	double B2_Blocal_r1_im[Lt * Nc * Ns * Nc * Ns * Vsnk * Nc * Ns * Nsrc];

    // Halide buffers
    Halide::Buffer<int> b_perms(2*Nq, Nperms, "perms");
    Halide::Buffer<int> b_sigs(Nperms, "sigs");
    Halide::Buffer<double> b_overall_weight(1, "overall_weight");
    Halide::Buffer<int> b_snk_color_weights(Nq, Nw2, 2, "snk_color_weights");
    Halide::Buffer<int> b_snk_spin_weights(Nq, Nw2, 2, "snk_spin_weights");
    Halide::Buffer<double> b_snk_weights(Nw2, "snk_weights");
    Halide::Buffer<double> b_snk_psi_re(Nsnk, Vsnk, Vsnk, "snk_psi_re");
    Halide::Buffer<double> b_snk_psi_im(Nsnk, Vsnk, Vsnk, "snk_psi_im");
    Halide::Buffer<double> buf_term_r(1, "buf_term_r");
    Halide::Buffer<double> buf_term_i(1, "buf_term_i");
    Halide::Buffer<double> buf_new_term_r(1, "buf_new_term_r");
    Halide::Buffer<double> buf_new_term_i(1, "buf_new_term_i");
    Halide::Buffer<int> buf_snk_1(2, "buf_snk_1");
    Halide::Buffer<int> buf_snk_1_b(2, "buf_snk_1_b");
    Halide::Buffer<int> buf_snk_1_nq(2, "buf_snk_1_nq");

    for (int i = 0; i < Nt; i++)
      for (int j = 0; j < Nsnk; j++)
        for (int k = 0; k < Nsrc; k++)
        {
          C_r(i, j, k) = (double) 1;
	  C_i(i, j, k) = (double) 1;
	  ref_C_r(i, j, k) = (double) 1;
	  ref_C_i(i, j, k) = (double) 1;
        }

   for (int m=0; m<Nsrc; m++)
      for (int iCprime=0; iCprime<Nc; iCprime++)
         for (int iSprime=0; iSprime<Ns; iSprime++)
            for (int jCprime=0; jCprime<Nc; jCprime++)
              for (int jSprime=0; jSprime<Ns; jSprime++)
                 for (int kCprime=0; kCprime<Nc; kCprime++)
                    for (int kSprime=0; kSprime<Ns; kSprime++)
                      for (int x=0; x<Vsnk; x++)
                          for (int t=0; t<Nt; t++)
			  {
	double v1 = 1.0;// rand()%10;
	double v2 = 1.0;// rand()%10;
	double v3 = 1.0;// rand()%10;
	double v4 = 1.0;// rand()%10;
                             B1_Blocal_r1_r(jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t) = v1;
                             B2_Blocal_r1_r(jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t) = v2;
                             B1_Blocal_r1_i(jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t) = v3;
                             B2_Blocal_r1_i(jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t) = v4;
                             B1_Blocal_r1_re[Blocal_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, m)] = v1;
                             B1_Blocal_r1_im[Blocal_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, m)] = v2;
                             B2_Blocal_r1_re[Blocal_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, m)] = v3;
                             B2_Blocal_r1_im[Blocal_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, m)] = v4;
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
	double v1 = 1.0;// rand()%10;
	double v2 = 1.0;// rand()%10;
	double v3 = 1.0;// rand()%10;
	double v4 = 1.0;// rand()%10;
	double v5 = 1.0;// rand()%10;
	double v6 = 1.0;// rand()%10;
	double v7 = 1.0;// rand()%10;
	double v8 = 1.0;// rand()%10;
                             B1_Bsingle_r1_r(x2, jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t) = v1;
                             B2_Bsingle_r1_r(x2, jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t) = v2;
                             B1_Bsingle_r1_i(x2, jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t) = v3;
                             B2_Bsingle_r1_i(x2, jSprime, jCprime, m, x, kSprime, kCprime, iSprime, iCprime, t) = v4;
                             B1_Bsingle_r1_re[Bdouble_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, x2, m)] = v1;
                             B1_Bsingle_r1_im[Bdouble_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, x2, m)] = v2;
                             B2_Bsingle_r1_re[Bdouble_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, x2, m)] = v3;
                             B2_Bsingle_r1_im[Bdouble_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, x2, m)] = v4;

                             B1_Bdouble_r1_r(x2, iSprime, iCprime, m, x, kSprime, kCprime, jSprime, jCprime, t) = v5;
                             B2_Bdouble_r1_r(x2, iSprime, iCprime, m, x, kSprime, kCprime, jSprime, jCprime, t) = v6;
                             B1_Bdouble_r1_i(x2, iSprime, iCprime, m, x, kSprime, kCprime, jSprime, jCprime, t) = v7;
                             B2_Bdouble_r1_i(x2, iSprime, iCprime, m, x, kSprime, kCprime, jSprime, jCprime, t) = v8;
                             B1_Bdouble_r1_re[Bdouble_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, x2, m)] = v5;
                             B1_Bdouble_r1_im[Bdouble_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, x2, m)] = v6;
                             B2_Bdouble_r1_re[Bdouble_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, x2, m)] = v7;
                             B2_Bdouble_r1_im[Bdouble_index(t, iCprime, iSprime, kCprime, kSprime, x, jCprime, jSprime, x2, m)] = v8;
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
            b_perms(q,permnum) = perms_array[i][q];
         }
         sigs[permnum] = sigs_array[i];
         b_sigs(permnum) = sigs_array[i];
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
    b_overall_weight(0) = 1;

    for (int i = 0; i<2; i++)
	    for (int j = 0; j<Nw2; j++)
		    for (int k = 0; k<Nq; k++)
		    {
	double v1 = rand()%10;
			snk_color_weights[i*Nw2*Nq + j*Nq + k] = k;
         b_snk_color_weights(k,j,i) = k;
			snk_spin_weights[i*Nw2*Nq + j*Nq + k] = i;
         b_snk_spin_weights(k,j,i) = i;
			snk_weights[j] = v1;
         b_snk_weights(j) = v1;
		    }

    for (int i = 0; i<Vsnk; i++)
	for (int k = 0; k<Vsnk; k++)
	    for (int j = 0; j<Nsnk; j++)
		    {
	double v1 = rand()%10;
	double v2 = rand()%10;
			snk_psi_re[i*Nsnk*Vsnk + k*Nsnk + j] = v1;
			snk_psi_im[i*Nsnk*Vsnk + k*Nsnk + j] = v2;
         b_snk_psi_re(i,k,j) = v1;
         b_snk_psi_im(i,k,j) = v2;
		    }


#if RUN_REFERENCE
    std::cout << "Start reference C code." <<  std::endl;
    for (int i = 0; i < nb_tests; i++)
    {
	    std::cout << "Run " << i << "/" << nb_tests <<  std::endl;
	    auto start2 = std::chrono::high_resolution_clock::now();

	    make_dibaryon_correlator(
				    (double *) ref_C_r.raw_buffer()->host,
				    (double *) ref_C_i.raw_buffer()->host,
				    B1_Blocal_r1_re,
				    B1_Blocal_r1_im,
				    B1_Bsingle_r1_re,
				    B1_Bsingle_r1_im,
				    B1_Bdouble_r1_re,
				    B1_Bdouble_r1_im,

				    B2_Blocal_r1_re,
				    B2_Blocal_r1_im,
				    B2_Bsingle_r1_re,
				    B2_Bsingle_r1_im,
				    B2_Bdouble_r1_re,
				    B2_Bdouble_r1_im,

                perms,
                sigs,
                overall_weight[0],
                snk_color_weights,
                snk_spin_weights,
                snk_weights,
                snk_psi_re,
                snk_psi_im);

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
				    B1_Bsingle_r1_r.raw_buffer(),
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
				    b_snk_psi_im.raw_buffer(),
                buf_term_r.raw_buffer(),
                buf_term_i.raw_buffer(),
                buf_new_term_r.raw_buffer(),
                buf_new_term_i.raw_buffer(),
                buf_snk_1.raw_buffer(),
                buf_snk_1_b.raw_buffer(),
                buf_snk_1_nq.raw_buffer());
       
       printf("buf_term = %4.1f + I( %4.1f ) \n", buf_term_r(0), buf_term_i(0));
       printf("buf_new_term = %4.1f + I( %4.1f ) \n", buf_new_term_r(0), buf_term_i(0));
       printf("buf_snk_1 = %d %d \n", buf_snk_1(0), buf_snk_1(1));
       printf("buf_snk_1_b = %d %d \n", buf_snk_1_b(0), buf_snk_1_b(1));
       printf("buf_snk_1_nq = %d %d \n", buf_snk_1_nq(0), buf_snk_1_nq(1));

	    auto end1 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	    duration_vector_1.push_back(duration1);
    }
    std::cout << "End Tiramisu code." <<  std::endl;

    print_time("performance_CPU.csv", "dibaryon", {"Ref", "Tiramisu"}, {median(duration_vector_2), median(duration_vector_1)});
    std::cout << "\nSpeedup = " << median(duration_vector_2)/median(duration_vector_1) << std::endl;

#if RUN_CHECK
    // Compare outputs.

    for (int i = 0; i < Nt; i++)
      for (int j = 0; j < Nsnk; j++)
        for (int k = 0; k < Nsrc; k++)
        {printf("%4.1f %4.1f \n", ref_C_r(i,j,k), C_r(i,j,k));
           if ((std::abs(ref_C_r(i,j,k) - C_r(i,j,k)) >= 0.01) ||
	       (std::abs(ref_C_i(i,j,k) - C_i(i,j,k)) >= 0.01))
	    {
		  std::cout << "Error: different computed values for C_r or C_i!" << std::endl;
		  exit(1);
	    }}

#endif

    std::cout << "\n\n\033[1;32mSuccess: computed values are equal!\033[0m\n\n" << std::endl;

    return 0;
}
