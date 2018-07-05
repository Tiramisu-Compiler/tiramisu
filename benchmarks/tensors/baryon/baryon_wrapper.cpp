#include "Halide.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"

#include "baryon_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif


void ref(Halide::Buffer<float> &Res2, Halide::Buffer<float> &S, Halide::Buffer<float> &wp)
{
  int c1 = 0, c2 = 0, c3 = 0, t = 0, a1 = 0, a2 = 0, a3 = 0, xp0 = 0, b0 = 0, b1 = 0, b2 = 0;

  Res2(0) = 0;
  Halide::Buffer<float> Res0(1, "Res0");
  Halide::Buffer<float> Res1(1, "Res1");

  for (int x0 = 0; x0 < BARYON_N; x0++)
    for (int x1 = 0; x1 < BARYON_N; x1++)
      for (int x2 = 0; x2 < BARYON_N; x2++)
       {
         Res0(0) = S(c1, x0, x1, x2, t, a1, xp0) * S(c2, x0, x1, x2, t, a2, xp0) * S(c3, x0, x1, x2, t, a3, xp0)
                  +S(c2, x0, x1, x2, t, a1, xp0) * S(c3, x0, x1, x2, t, a2, xp0) * S(c1, x0, x1, x2, t, a3, xp0)
                  +S(c3, x0, x1, x2, t, a1, xp0) * S(c1, x0, x1, x2, t, a2, xp0) * S(c2, x0, x1, x2, t, a3, xp0)
                  -S(c2, x0, x1, x2, t, a1, xp0) * S(c1, x0, x1, x2, t, a2, xp0) * S(c3, x0, x1, x2, t, a3, xp0)
                  -S(c3, x0, x1, x2, t, a1, xp0) * S(c2, x0, x1, x2, t, a2, xp0) * S(c1, x0, x1, x2, t, a3, xp0)
                  -S(c1, x0, x1, x2, t, a1, xp0) * S(c3, x0, x1, x2, t, a2, xp0) * S(c2, x0, x1, x2, t, a3, xp0);

         Res1(0) = 0;
         for (int k = 1; k <= BARYON_N; k++)
           Res1(0) = Res1(0) + wp(c1, c2, c3, b0, b1, b2, k) * Res0(0);

         Res2(0) = Res2(0) + Res1(0);
       }
}

int main(int, char **)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<float> buf_res2(1, "buf_res2");
    Halide::Buffer<float> buf_res2_ref(1, "buf_res2_ref");
    Halide::Buffer<float> buf_S(BARYON_P, BARYON_N, BARYON_N, BARYON_N, BARYON_P, BARYON_P,  BARYON_P, "buf_S");
    Halide::Buffer<float> buf_wp(BARYON_P, BARYON_P, BARYON_P, BARYON_P, BARYON_P, BARYON_P, BARYON_N, "buf_wp");

    init_buffer(buf_S, (float)1);
    init_buffer(buf_wp, (float)1);

    for (int i = 0; i < NB_TESTS; i++)
    {
    	    init_buffer(buf_res2_ref, (float)0);
	    auto start2 = std::chrono::high_resolution_clock::now();
	    ref(buf_res2_ref, buf_S, buf_wp);
	    auto end2 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
	    duration_vector_2.push_back(duration2);
    }

    for (int i = 0; i < NB_TESTS; i++)
    {
	    init_buffer(buf_res2, (float)0);
	    auto start1 = std::chrono::high_resolution_clock::now();
	    tiramisu_generated_code(buf_res2.raw_buffer(), buf_S.raw_buffer(), buf_wp.raw_buffer());
	    auto end1 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	    duration_vector_1.push_back(duration1);
    }

    compare_buffers("benchmark_" + std::string(TEST_NAME_STR), buf_res2, buf_res2_ref);

    print_time("performance_CPU.csv", "baryon",
               {"Ref", "Tiramisu"},
               {median(duration_vector_2), median(duration_vector_1)});

    return 0;
}
