#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include "benchmarks.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

/*
 * Code extracted from Baryon official repo.
 
   for(int s4(0);s4<Ns;s4++) //sink spin index of the 3rd quark 
      for(QuarkIndex i3;i3.NotEnd();++i3)
	for(QuarkIndex i2;i2.NotEnd();++i2)
	  for(QuarkIndex i1;i1.NotEnd();++i1)
          {
		cc=0.0;
		for(int s(0);s<Ns;s++) //contract the diquark on the sink
		  	cc += colorContract(peekSpin(f1[i1.s][i1.c],s ),
					    peekSpin(f2[i2.s][i2.c],s ),
					    peekSpin(f3[i3.s][i3.c],s4));

		foo = phases.sft(cc); 

		for(int sink_mom_num(0); sink_mom_num < num_mom; sink_mom_num++) 
		    for(int t = 0; t < length; ++t){
		        int t_eff = (t - t0 + length) % length;
		        qqq[sink_mom_num][t_eff](i1,i2,i3,s4) =  (bc_spec < 0 && (t_eff+t0) >= length) ? -foo[sink_mom_num][t]:foo[sink_mom_num][t] ;
		  }
	  }
 */

/**
 * Res2 = 0
   For i3 in 0 to BARYON_N
     For i2 in 0 to BARYON_N
       For i1 in 0 to BARYON_N
       {
          Res0 = S(c1, i3, i2, i1, t, a1, x’0)*S(c2, i3, i2, i1, t, a2, x’0)*S(c3, i3, i2, i1, t, a3, x’0)
                +S(c2, i3, i2, i1, t, a1, x’0)*S(c3, i3, i2, i1, t, a2, x’0)*S(c1, i3, i2, i1, t, a3, x’0)
                +S(c3, i3, i2, i1, t, a1, x’0)*S(c1, i3, i2, i1, t, a2, x’0)*S(c2, i3, i2, i1, t, a3, x’0)
                -S(c2, i3, i2, i1, t, a1, x’0)*S(c1, i3, i2, i1, t, a2, x’0)*S(c3, i3, i2, i1, t, a3, x’0)
                -S(c3, i3, i2, i1, t, a1, x’0)*S(c2, i3, i2, i1, t, a2, x’0)*S(c1, i3, i2, i1, t, a3, x’0)
                -S(c1, i3, i2, i1, t, a1, x’0)*S(c3, i3, i2, i1, t, a2, x’0)*S(c2, i3, i2, i1, t, a3, x’0)

         Res1 = 0
         For k = 1 to N(B(b0, b1, b2))
           Res1 += w’(c1, c2, c3, b0, b1, b2, k) * Res0;

         Res2 += exp(i(i3*px+i2*py+i1*pz)) * Res1;
       }
 */

void ref(float Res2[1], float S[BARYON_P][BARYON_P][BARYON_P][BARYON_N][BARYON_N][BARYON_N][BARYON_P1],
			float wp[BARYON_N][BARYON_P][BARYON_P][BARYON_P][BARYON_P1][BARYON_P1][BARYON_P1])
{
  const int c1 = 0, c2 = 1, c3 = 2, t = 0, a1 = 0, a2 = 0, a3 = 0, xp0 = 0, b0 = 0, b1 = 0, b2 = 0;

  Res2[0] = 0;
  float Res0;
  float Res1;

  for (int i1 = 0; i1 < BARYON_N; i1++)
    for (int i2 = 0; i2 < BARYON_N; i2++)
      for (int i3 = 0; i3 < BARYON_N; i3++)
       {
         Res0    =  S[xp0][a1][t][i1][i2][i3][c1] * S[xp0][a2][t][i1][i2][i3][c2] * S[xp0][a3][t][i1][i2][i3][c3]
                  + S[xp0][a1][t][i1][i2][i3][c2] * S[xp0][a2][t][i1][i2][i3][c3] * S[xp0][a3][t][i1][i2][i3][c1]
                  + S[xp0][a1][t][i1][i2][i3][c3] * S[xp0][a2][t][i1][i2][i3][c1] * S[xp0][a3][t][i1][i2][i3][c2]
                  - S[xp0][a1][t][i1][i2][i3][c2] * S[xp0][a2][t][i1][i2][i3][c1] * S[xp0][a3][t][i1][i2][i3][c3]
                  - S[xp0][a1][t][i1][i2][i3][c3] * S[xp0][a2][t][i1][i2][i3][c2] * S[xp0][a3][t][i1][i2][i3][c1]
                  - S[xp0][a1][t][i1][i2][i3][c1] * S[xp0][a2][t][i1][i2][i3][c3] * S[xp0][a3][t][i1][i2][i3][c2];

         Res1 = 0;
         for (int k = 1; k <= BARYON_N; k++)
           Res1 = Res1 + wp[k][b2][b1][b0][c3][c2][c1] * Res0;

         Res2[0] = Res2[0] + Res1;
       }
}

void init_buffers(float S[BARYON_P][BARYON_P][BARYON_P][BARYON_N][BARYON_N][BARYON_N][BARYON_P1],
			float wp[BARYON_N][BARYON_P][BARYON_P][BARYON_P][BARYON_P1][BARYON_P1][BARYON_P1], float val)
{
  for (int xp0 = 0; xp0 < BARYON_P; xp0++)
    for (int a3 = 0; a3 < BARYON_P; a3++)
      for (int t = 0; t < BARYON_P; t++)
        for (int i1 = 0; i1 < BARYON_N; i1++)
          for (int i2 = 0; i2 < BARYON_N; i2++)
            for (int i3 = 0; i3 < BARYON_N; i3++)
	      for (int c = 0; c < BARYON_P; c++)
	      {
		S[xp0][a3][t][i1][i2][i3][c] = val;
	      }

  for (int k = 0; k < BARYON_N; k++)
    for (int b2 = 0; b2 < BARYON_P; b2++)
      for (int b1 = 0; b1 < BARYON_P; b1++)
        for (int b0 = 0; b0 < BARYON_P; b0++)
          for (int c3 = 0; c3 < BARYON_N; c3++)
            for (int c2 = 0; c2 < BARYON_P; c2++)
	      for (int c1 = 0; c1 < BARYON_P; c1++)
	      {
                wp[k][b2][b1][b0][c3][c2][c1] = val;
	      }
}
