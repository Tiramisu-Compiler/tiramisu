// This file is generated from test alphabets program by code generator in alphaz
// To compile this code, use -lm option for math library.

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <omp.h>
#include <immintrin.h>
#include <malloc.h>


// Common Macros
#define max(x, y)   ((x)>(y) ? (x) : (y))
#define MAX(x, y)	((x)>(y) ? (x) : (y))
#define min(x, y)   ((x)>(y) ? (y) : (x))
#define MIN(x, y)	((x)>(y) ? (y) : (x))
#define CEILD(n,d)  (int)ceil(((float)(n))/((float)(d)))
#define ceild(n,d)  (int)ceil(((float)(n))/((float)(d)))
#define FLOORD(n,d) (int)floor(((float)(n))/((float)(d)))
#define floord(n,d) (int)floor(((float)(n))/((float)(d)))
#define CDIV(x,y)    CEILD((x),(y))
#define div(x,y)    CDIV((x),(y))
#define FDIV(x,y)    FLOORD((x),(y))
#define LB_SHIFT(b,s)  ((int)ceild(b,s) * s)
#define MOD(i,j)   ((i)%(j))
#define mallocCheck(v,s,d) if ((v) == NULL) { printf("Failed to allocate memory for %s : size=%lu\n", "sizeof(d)*(s)", sizeof(d)*(s)); exit(-1); }
// Reduction Operators
#define RADD(x,y)    ((x)+=(y))
#define RMUL(x,y)    ((x)*=(y))
#define RMAX(x,y)    ((x)=MAX((x),(y)))
#define RMIN(x,y)    ((x)=MIN((x),(y)))

// Common functions for min and max
//functions for integer max
inline int __max_int(int x, int y){
	return ((x)>(y) ? (x) : (y));
}

inline short __max_short(short x, short y){
	return ((x)>(y) ? (x) : (y));
}

inline long __max_long(long x, long y){
	return ((x)>(y) ? (x) : (y));
}

inline unsigned int __max_unsigned_int(unsigned int x, unsigned int y){
	return ((x)>(y) ? (x) : (y));
}

inline unsigned short __max_unsigned_short(unsigned short x, unsigned short y){
	return ((x)>(y) ? (x) : (y));
}

//function for float max
inline float __max_float(float x, float y){
	return ((x)>(y) ? (x) : (y));
}

//function for integer min
inline int __min_int(int x, int y){
	return ((x)>(y) ? (y) : (x));
}

inline short __min_short(short x, short y){
	return ((x)>(y) ? (y) : (x));
}

inline long __min_long(long x, long y){
	return ((x)>(y) ? (y) : (x));
}

inline unsigned int __min_unsigned_int(unsigned int x, unsigned int y){
	return ((x)>(y) ? (y) : (x));
}

inline unsigned short __min_unsigned_short(unsigned short x, unsigned short y){
	return ((x)>(y) ? (y) : (x));
}

inline unsigned long __min_unsigned_long(unsigned long x, unsigned long y){
	return ((x)>(y) ? (y) : (x));
}

inline float __min_float(float x, float y){
	return ((x)>(y) ? (y) : (x));
}







//Memory Macros
#define A(i,j) A[(i) * (Q) + j]
#define B(i,j) B[(i) * (R) + j]
#define Cout(i,j) Cout[(i) * (R) + j]
#define Acc(i,j) Acc[(i) * (R) + j]

void gemm(long P, long Q, long R, long ts1_l1, long ts2_l1, long ts3_l1, float* alpha, float* beta, float* A, float* B, float* Cout){
	///Parameter checking
	if (!((P >= 2 && Q >= 2 && R >= 2 && ts1_l1 > 0 && ts2_l1 > 0 && ts3_l1 > 0))) {
		printf("The value of parameters are not valid.\n");
		exit(-1);
	}
	//Memory Allocation
	float* Acc = (float*)malloc(sizeof(float)*((P) * (R)));
 	mallocCheck(Acc, ((P) * (R)), float);
 	#define S1(i,j,k) Acc(i,k) = (Acc(i,k))+((A(i,j))*(B(j,k)))
	#define S2(i,j,k) Acc(i,k) = (A(i,j))*(B(j,k))
 	#define S0(i,j,i2) Cout(i,i2) = ((*alpha)*(Acc(i,i2)))+((*beta)*(Cout(i,i2)))
	{
		//Domain
		//{i,j,k|P>=2 && R>=2 && i>=0 && P>=i+1 && k>=0 && R>=k+1 && Q>=j+1 && j>=1 && Q>=2}
		//{i,j,k|j==0 && P>=2 && Q>=2 && R>=2 && i>=0 && P>=i+1 && k>=0 && R>=k+1}
		//{i,j,i2|j==Q-1 && i>=0 && P>=i+1 && Q>=2 && R>=i2+1 && P>=2 && i2>=0 && R>=2}
		int ti1_l1,ti2_l1,ti3_l1,start_l1_d0,end_l1_d0,time_l1_d0,c1,c2,c3;
		if ((Q >= 3)) {
			{
				{
					start_l1_d0 = INT_MAX;
					end_l1_d0 = INT_MIN;
					ti1_l1 = (ceild((-ts1_l1+1),(ts1_l1))) * (ts1_l1);
					ti2_l1 = (ceild((-ts2_l1+1),(ts2_l1))) * (ts2_l1);
					ti3_l1 = (ceild((-ts3_l1+1),(ts3_l1))) * (ts3_l1);
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti3_l1 = R-1;
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti2_l1 = Q-1;
					ti3_l1 = (ceild((-ts3_l1+1),(ts3_l1))) * (ts3_l1);
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti3_l1 = R-1;
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti1_l1 = P-1;
					ti2_l1 = (ceild((-ts2_l1+1),(ts2_l1))) * (ts2_l1);
					ti3_l1 = (ceild((-ts3_l1+1),(ts3_l1))) * (ts3_l1);
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti3_l1 = R-1;
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti2_l1 = Q-1;
					ti3_l1 = (ceild((-ts3_l1+1),(ts3_l1))) * (ts3_l1);
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti3_l1 = R-1;
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					for(time_l1_d0=start_l1_d0;time_l1_d0 <= end_l1_d0;time_l1_d0+=1)
					 {
					 	#pragma omp parallel for private(c1,c2,c3,ti1_l1,ti2_l1,ti3_l1) schedule(static ,1)
					 	for(ti1_l1=(ceild((-ts1_l1+1),(ts1_l1))) * (ts1_l1);ti1_l1 <= P-1;ti1_l1+=ts1_l1)
					 	 {
					 	 	for(ti2_l1=(ceild((-ts2_l1+1),(ts2_l1))) * (ts2_l1);ti2_l1 <= Q-1;ti2_l1+=ts2_l1)
					 	 	 {
					 	 	 	ti3_l1 = (time_l1_d0 + ((ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1)) * (-1)) * (ts3_l1);
					 	 	 	if (((ceild((-ts3_l1+1),(ts3_l1))) * (ts3_l1) <= ti3_l1 && ti3_l1 <= R-1)) {
					 	 	 		//guard that isolates selected statements for generic point loops
					 	 	 		if ((0 < ti2_l1 && ti2_l1+ts2_l1 < Q)) {
					 	 	 			//full-tile guard
					 	 	 			if ((0 <= ti1_l1 && ti1_l1+ts1_l1 <= P && 0 <= ti2_l1-1 && ti2_l1+ts2_l1 <= Q-1 && 0 <= ti3_l1 && ti3_l1+ts3_l1 <= R)) {
					 	 	 				for(c1=ti1_l1;c1 <= ti1_l1+ts1_l1-1;c1+=1)
					 	 	 				 {
					 	 	 				 	for(c2=ti2_l1;c2 <= ti2_l1+ts2_l1-1;c2+=1)
					 	 	 				 	 {
													//#pragma ivdep
													//#pragma vector always
					 	 	 				 	 	for(c3=ti3_l1;c3 <= ti3_l1+ts3_l1-1;c3+=1)
					 	 	 				 	 	 {
					 	 	 				 	 	 	S1((c1),(c2),(c3));
					 	 	 				 	 	 }
					 	 	 				 	 }
					 	 	 				 }
					 	 	 			} else {
					 	 	 				for(c1=max(ti1_l1,0);c1 <= min(ti1_l1+ts1_l1-1,P-1);c1+=1)
					 	 	 				 {
					 	 	 				 	for(c2=max(ti2_l1,1);c2 <= min(ti2_l1+ts2_l1-1,Q-2);c2+=1)
					 	 	 				 	 {
													//#pragma ivdep
													//#pragma vector always
					 	 	 				 	 	for(c3=max(ti3_l1,0);c3 <= min(ti3_l1+ts3_l1-1,R-1);c3+=1)
					 	 	 				 	 	 {
					 	 	 				 	 	 	S1((c1),(c2),(c3));
					 	 	 				 	 	 }
					 	 	 				 	 }
					 	 	 				 }
					 	 	 			}
					 	 	 		} else {
					 	 	 			{
					 	 	 				for(c1=max(ti1_l1,0);c1 <= min(ti1_l1+ts1_l1-1,P-1);c1+=1)
					 	 	 				 {
					 	 	 				 	for(c2=max(ti2_l1,0);c2 <= min(ti2_l1+ts2_l1-1,0);c2+=1)
					 	 	 				 	 {
					 	 	 				 	 	for(c3=max(ti3_l1,0);c3 <= min(ti3_l1+ts3_l1-1,R-1);c3+=1)
					 	 	 				 	 	 {
					 	 	 				 	 	 	S2((c1),(0),(c3));
					 	 	 				 	 	 }
					 	 	 				 	 }
					 	 	 				 	for(c2=max(ti2_l1,1);c2 <= min(ti2_l1+ts2_l1-1,Q-2);c2+=1)
					 	 	 				 	 {
													//#pragma ivdep
													//#pragma vector always
					 	 	 				 	 	for(c3=max(ti3_l1,0);c3 <= min(ti3_l1+ts3_l1-1,R-1);c3+=1)
					 	 	 				 	 	 {
					 	 	 				 	 	 	S1((c1),(c2),(c3));
					 	 	 				 	 	 }
					 	 	 				 	 }
					 	 	 				 	for(c2=max(ti2_l1,Q-1);c2 <= min(ti2_l1+ts2_l1-1,Q-1);c2+=1)
					 	 	 				 	 {
					 	 	 				 	 	for(c3=max(ti3_l1,0);c3 <= min(ti3_l1+ts3_l1-1,R-1);c3+=1)
					 	 	 				 	 	 {
					 	 	 				 	 	 	S1((c1),(Q-1),(c3));
					 	 	 				 	 	 	S0((c1),(Q-1),(c3));
					 	 	 				 	 	 }
					 	 	 				 	 }
					 	 	 				 }
					 	 	 			}
					 	 	 		}
					 	 	 	}
					 	 	 }
					 	 }
					 }
				}
			}
		}
		if (Q == 2) {
			{
				{
					start_l1_d0 = INT_MAX;
					end_l1_d0 = INT_MIN;
					ti1_l1 = (ceild((-ts1_l1+1),(ts1_l1))) * (ts1_l1);
					ti2_l1 = (ceild((-ts2_l1+1),(ts2_l1))) * (ts2_l1);
					ti3_l1 = (ceild((-ts3_l1+1),(ts3_l1))) * (ts3_l1);
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti3_l1 = R-1;
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti2_l1 = 1;
					ti3_l1 = (ceild((-ts3_l1+1),(ts3_l1))) * (ts3_l1);
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti3_l1 = R-1;
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti1_l1 = P-1;
					ti2_l1 = (ceild((-ts2_l1+1),(ts2_l1))) * (ts2_l1);
					ti3_l1 = (ceild((-ts3_l1+1),(ts3_l1))) * (ts3_l1);
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti3_l1 = R-1;
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti2_l1 = 1;
					ti3_l1 = (ceild((-ts3_l1+1),(ts3_l1))) * (ts3_l1);
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					ti3_l1 = R-1;
					start_l1_d0 = min(start_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					end_l1_d0 = max(end_l1_d0,(ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1) + (ti3_l1)/(ts3_l1));
					for(time_l1_d0=start_l1_d0;time_l1_d0 <= end_l1_d0;time_l1_d0+=1)
					 {
					 	#pragma omp parallel for private(c1,c2,c3,ti1_l1,ti2_l1,ti3_l1) schedule(static ,1)
					 	for(ti1_l1=(ceild((-ts1_l1+1),(ts1_l1))) * (ts1_l1);ti1_l1 <= P-1;ti1_l1+=ts1_l1)
					 	 {
					 	 	for(ti2_l1=(ceild((-ts2_l1+1),(ts2_l1))) * (ts2_l1);ti2_l1 <= 1;ti2_l1+=ts2_l1)
					 	 	 {
					 	 	 	ti3_l1 = (time_l1_d0 + ((ti1_l1)/(ts1_l1) + (ti2_l1)/(ts2_l1)) * (-1)) * (ts3_l1);
					 	 	 	if (((ceild((-ts3_l1+1),(ts3_l1))) * (ts3_l1) <= ti3_l1 && ti3_l1 <= R-1)) {
					 	 	 		//guard that isolates selected statements for generic point loops
					 	 	 		if (0 < ti2_l1) {
					 	 	 			//full-tile guard
					 	 	 			if (0 <= ti1_l1 && ti1_l1+ts1_l1 <= P && 0 <= ti2_l1-1 && ti2_l1+ts2_l1 <= 2 && 0 <= ti3_l1 && ti3_l1+ts3_l1 <= R) {
					 	 	 				for(c1=ti1_l1;c1 <= ti1_l1+ts1_l1-1;c1+=1)
					 	 	 				 {
					 	 	 				 	for(c2=ti2_l1;c2 <= ti2_l1+ts2_l1-1;c2+=1)
					 	 	 				 	 {
					 	 	 				 	 	for(c3=ti3_l1;c3 <= ti3_l1+ts3_l1-1;c3+=1)
					 	 	 				 	 	 {
					 	 	 				 	 	 	S1((c1),(1),(c3));
					 	 	 				 	 	 	S0((c1),(1),(c3));
					 	 	 				 	 	 }
					 	 	 				 	 }
					 	 	 				 }
					 	 	 			} else {
					 	 	 				for(c1=max(ti1_l1,0);c1 <= min(ti1_l1+ts1_l1-1,P-1);c1+=1)
					 	 	 				 {
					 	 	 				 	for(c2=max(ti2_l1,1);c2 <= min(ti2_l1+ts2_l1-1,1);c2+=1)
					 	 	 				 	 {
					 	 	 				 	 	for(c3=max(ti3_l1,0);c3 <= min(ti3_l1+ts3_l1-1,R-1);c3+=1)
					 	 	 				 	 	 {
					 	 	 				 	 	 	S1((c1),(1),(c3));
					 	 	 				 	 	 	S0((c1),(1),(c3));
					 	 	 				 	 	 }
					 	 	 				 	 }
					 	 	 				 }
					 	 	 			}
					 	 	 		} else {
					 	 	 			{
					 	 	 				for(c1=max(ti1_l1,0);c1 <= min(ti1_l1+ts1_l1-1,P-1);c1+=1)
					 	 	 				 {
					 	 	 				 	for(c2=max(ti2_l1,0);c2 <= min(ti2_l1+ts2_l1-1,0);c2+=1)
					 	 	 				 	 {
					 	 	 				 	 	for(c3=max(ti3_l1,0);c3 <= min(ti3_l1+ts3_l1-1,R-1);c3+=1)
					 	 	 				 	 	 {
					 	 	 				 	 	 	S2((c1),(0),(c3));
					 	 	 				 	 	 }
					 	 	 				 	 }
					 	 	 				 	for(c2=max(ti2_l1,1);c2 <= min(ti2_l1+ts2_l1-1,1);c2+=1)
					 	 	 				 	 {
					 	 	 				 	 	for(c3=max(ti3_l1,0);c3 <= min(ti3_l1+ts3_l1-1,R-1);c3+=1)
					 	 	 				 	 	 {
					 	 	 				 	 	 	S1((c1),(1),(c3));
					 	 	 				 	 	 	S0((c1),(1),(c3));
					 	 	 				 	 	 }
					 	 	 				 	 }
					 	 	 				 }
					 	 	 			}
					 	 	 		}
					 	 	 	}
					 	 	 }
					 	 }
					 }
				}
			}
		}
	}
	#undef S1
	#undef S2
	#undef S0
	
	//Memory Free
	free(Acc);
}

//Memory Macros
#undef A
#undef B
#undef Cout
#undef Acc


//Common Macro undefs
#undef max
#undef MAX
#undef min
#undef MIN
#undef CEILD
#undef ceild
#undef FLOORD
#undef floord
#undef CDIV
#undef FDIV
#undef LB_SHIFT
#undef MOD
#undef RADD
#undef RMUL
#undef RMAX
#undef RMIN
