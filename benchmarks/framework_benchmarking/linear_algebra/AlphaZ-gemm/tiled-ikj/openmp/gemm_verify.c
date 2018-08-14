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


// Common Macros
#define max(x, y)   ((x)>(y) ? (x) : (y))
#define MAX(x, y)	((x)>(y) ? (x) : (y))
#define min(x, y)   ((x)>(y) ? (y) : (x))
#define MIN(x, y)	((x)>(y) ? (y) : (x))
#define CEILD(n,d)  (int)ceil(((double)(n))/((double)(d)))
#define ceild(n,d)  (int)ceil(((double)(n))/((double)(d)))
#define FLOORD(n,d) (int)floor(((double)(n))/((double)(d)))
#define floord(n,d) (int)floor(((double)(n))/((double)(d)))
#define CDIV(x,y)    CEILD((x),(y))
#define div(x,y)    CDIV((x),(y))
#define FDIV(x,y)    FLOORD((x),(y))
#define LB_SHIFT(b,s)  ((int)ceild(b,s) * s)
#define MOD(i,j)   ((i)%(j))
#define mallocCheck(v,s,d) if ((v) == NULL) { printf("Failed to allocate memory for %s : size=%lu\n", "sizeof(d)*(s)", sizeof(d)*(s)); exit(-1); }

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

//function for double max
inline double __max_double(double x, double y){
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

inline double __min_double(double x, double y){
	return ((x)>(y) ? (y) : (x));
}




///Global Variables
static double alpha;
static double beta;
static double* A;
static double* B;
static double* Cin;
static double* _serCout;
static double* Cout;
static char* _flag_Cout;
static char* _flag__serCout;


//Local Function Declarations
double eval_Cout(long, long, long, int, int);
double eval__serCout(long, long, long, int, int, int);

//Memory Macros
#define A(i,j) A[(i) * (Q) + j]
#define B(i,j) B[(i) * (R) + j]
#define Cin(i,j) Cin[(i) * (R) + j]
#define _serCout(i,j,k) _serCout[(i) * ((R) * (Q)) + (j) * (Q) + k]
#define Cout(i,j) Cout[(i) * (R) + j]
#define _flag_Cout(i,j) _flag_Cout[(i) * (R) + j]
#define _flag__serCout(i,j,k) _flag__serCout[(i) * ((R) * (Q)) + (j) * (Q) + k]

void gemm_verify(long P, long Q, long R, long ts1_l1, long ts2_l1, long ts3_l1, double* _local_alpha, double* _local_beta, double* _local_A, double* _local_B, double* _local_Cin, double* _local_Cout){
	///Parameter checking
	if (!((P >= 2 && Q >= 2 && R >= 2))) {
		printf("The value of parameters are not valid.\n");
		exit(-1);
	}
	//Copy to global
	
	alpha = *_local_alpha;
	
	beta = *_local_beta;
	A = _local_A;
	B = _local_B;
	Cin = _local_Cin;
	Cout = _local_Cout;
	
	//Memory Allocation
	int mz1, mz2, mz3;
	
	_serCout = (double*)malloc(sizeof(double)*((P) * (R) * (Q)));
	mallocCheck(_serCout, ((P) * (R) * (Q)), double);
	
	_flag_Cout = (char*)malloc(sizeof(char)*((P) * (R)));
	mallocCheck(_flag_Cout, ((P) * (R)), char);
	memset(_flag_Cout, 'N', ((P) * (R)));
	
	_flag__serCout = (char*)malloc(sizeof(char)*((P) * (R) * (Q)));
	mallocCheck(_flag__serCout, ((P) * (R) * (Q)), char);
	memset(_flag__serCout, 'N', ((P) * (R) * (Q)));
	#define S0(i,j) eval_Cout(P,Q,R,i,j)
	{
		//Domain
		//{i,j|i>=0 && P>=i+1 && j>=0 && R>=j+1 && P>=2 && Q>=2 && R>=2}
		int c1,c2;
		for(c1=0;c1 <= P-1;c1+=1)
		 {
		 	for(c2=0;c2 <= R-1;c2+=1)
		 	 {
		 	 	S0((c1),(c2));
		 	 }
		 }
	}
	#undef S0
	
	//Memory Free
	free(_serCout);
	free(_flag_Cout);
	free(_flag__serCout);
}
double eval_Cout(long P, long Q, long R, int i, int j){
	if ( _flag_Cout(i,j) == 'N' ) {
		_flag_Cout(i,j) = 'I';
	//Body for Cout
		Cout(i,j) = ((alpha)*(eval__serCout(P,Q,R,i,j,Q-1)))+((beta)*(Cin(i,j)));
		_flag_Cout(i,j) = 'F';
	} else if ( _flag_Cout(i,j) == 'I' ) {
		printf("There is a self dependence on Cout at (%d,%d) \n",i,j);
		exit(-1);
	}
	return Cout(i,j);
}
double eval__serCout(long P, long Q, long R, int i, int j, int k){
	if ( _flag__serCout(i,j,k) == 'N' ) {
		_flag__serCout(i,j,k) = 'I';
	//Body for _serCout
		_serCout(i,j,k) = (((k >= 1))?(eval__serCout(P,Q,R,i,j,k-1))+((A(i,k))*(B(k,j))):((A(i,k))*(B(k,j))));
		_flag__serCout(i,j,k) = 'F';
	} else if ( _flag__serCout(i,j,k) == 'I' ) {
		printf("There is a self dependence on _serCout at (%d,%d,%d) \n",i,j,k);
		exit(-1);
	}
	return _serCout(i,j,k);
}

//Memory Macros
#undef A
#undef B
#undef Cin
#undef _serCout
#undef Cout
#undef _flag_Cout
#undef _flag__serCout


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
