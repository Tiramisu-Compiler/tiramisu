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
#include <time.h>
#include <sys/time.h>
#include <sys/errno.h>
#include <omp.h>


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
#define EPSILON 1.0E-9







//Memory Macros
#define A(i,j) A[(i) * (Q) + j]
#define B(i,j) B[(i) * (R) + j]
#define Cout(i,j) Cout[(i) * (R) + j]

#define Cout_verify(i,j) Cout_verify[(i) * (R) + j]
#define var_Cout(i,j) Cout(i,j)
#define var_Cout_verify(i,j) Cout_verify(i,j)

//function prototypes
void gemm(long, long, long, long, long, long, float*, float*, float*, float*, float*);
void gemm_verify(long, long, long, long, long, long, float*, float*, float*, float*, float*, float*);

//main
int main(int argc, char** argv) {
	//Check number of args
	if (argc <= 6) {
		printf("Number of argument is smaller than expected.\n");
		printf("Expecting P,Q,R,ts1_l1,ts2_l1,ts3_l1\n");
		exit(0);
	}
	
	char *end = 0;
	char *val = 0;
	//Read Parameters
	//Initialisation of P
	errno = 0;
	end = 0;
	val = argv[1];
	long P = strtol(val,&end,10);
	if ((errno == ERANGE && (P == LONG_MAX || P == LONG_MIN)) || (errno != 0 && P == 0)) {
		perror("strtol");
		exit(EXIT_FAILURE);
	}
	if (end == val) {
		fprintf(stderr, "No digits were found for P\n");
		exit(EXIT_FAILURE);
	}
	if (*end != '\0'){
		printf("For parameter P: Converted part: %ld, non-convertible part: %s\n", P, end);
		exit(EXIT_FAILURE);
	}
	
	//Initialisation of Q
	errno = 0;
	end = 0;
	val = argv[2];
	long Q = strtol(val,&end,10);
	if ((errno == ERANGE && (Q == LONG_MAX || Q == LONG_MIN)) || (errno != 0 && Q == 0)) {
		perror("strtol");
		exit(EXIT_FAILURE);
	}
	if (end == val) {
		fprintf(stderr, "No digits were found for Q\n");
		exit(EXIT_FAILURE);
	}
	if (*end != '\0'){
		printf("For parameter Q: Converted part: %ld, non-convertible part: %s\n", Q, end);
		exit(EXIT_FAILURE);
	}
	
	//Initialisation of R
	errno = 0;
	end = 0;
	val = argv[3];
	long R = strtol(val,&end,10);
	if ((errno == ERANGE && (R == LONG_MAX || R == LONG_MIN)) || (errno != 0 && R == 0)) {
		perror("strtol");
		exit(EXIT_FAILURE);
	}
	if (end == val) {
		fprintf(stderr, "No digits were found for R\n");
		exit(EXIT_FAILURE);
	}
	if (*end != '\0'){
		printf("For parameter R: Converted part: %ld, non-convertible part: %s\n", R, end);
		exit(EXIT_FAILURE);
	}
	
	//Initialisation of ts1_l1
	errno = 0;
	end = 0;
	val = argv[4];
	long ts1_l1 = strtol(val,&end,10);
	if ((errno == ERANGE && (ts1_l1 == LONG_MAX || ts1_l1 == LONG_MIN)) || (errno != 0 && ts1_l1 == 0)) {
		perror("strtol");
		exit(EXIT_FAILURE);
	}
	if (end == val) {
		fprintf(stderr, "No digits were found for ts1_l1\n");
		exit(EXIT_FAILURE);
	}
	if (*end != '\0'){
		printf("For parameter ts1_l1: Converted part: %ld, non-convertible part: %s\n", ts1_l1, end);
		exit(EXIT_FAILURE);
	}
	
	//Initialisation of ts2_l1
	errno = 0;
	end = 0;
	val = argv[5];
	long ts2_l1 = strtol(val,&end,10);
	if ((errno == ERANGE && (ts2_l1 == LONG_MAX || ts2_l1 == LONG_MIN)) || (errno != 0 && ts2_l1 == 0)) {
		perror("strtol");
		exit(EXIT_FAILURE);
	}
	if (end == val) {
		fprintf(stderr, "No digits were found for ts2_l1\n");
		exit(EXIT_FAILURE);
	}
	if (*end != '\0'){
		printf("For parameter ts2_l1: Converted part: %ld, non-convertible part: %s\n", ts2_l1, end);
		exit(EXIT_FAILURE);
	}
	
	//Initialisation of ts3_l1
	errno = 0;
	end = 0;
	val = argv[6];
	long ts3_l1 = strtol(val,&end,10);
	if ((errno == ERANGE && (ts3_l1 == LONG_MAX || ts3_l1 == LONG_MIN)) || (errno != 0 && ts3_l1 == 0)) {
		perror("strtol");
		exit(EXIT_FAILURE);
	}
	if (end == val) {
		fprintf(stderr, "No digits were found for ts3_l1\n");
		exit(EXIT_FAILURE);
	}
	if (*end != '\0'){
		printf("For parameter ts3_l1: Converted part: %ld, non-convertible part: %s\n", ts3_l1, end);
		exit(EXIT_FAILURE);
	}
	
	
	///Parameter checking
	if (!((P >= 2 && Q >= 2 && R >= 2 && ts1_l1 > 0 && ts2_l1 > 0 && ts3_l1 > 0))) {
		printf("The value of parameters are not valid.\n");
		exit(-1);
	}
	
	
	//Memory Allocation
	int mz1, mz2;
	float alpha;
	float beta;
	float* A = (float*)malloc(sizeof(float)*((P) * (Q)));
	mallocCheck(A, ((P) * (Q)), float);
	float* B = (float*)malloc(sizeof(float)*((Q) * (R)));
	mallocCheck(B, ((Q) * (R)), float);
	float* Cout = (float*)malloc(sizeof(float)*((P) * (R)));
	mallocCheck(Cout, ((P) * (R)), float);
	#ifdef VERIFY
		float* Cout_verify = (float*)malloc(sizeof(float)*((P) * (R)));
		mallocCheck(Cout_verify, ((P) * (R)), float);
	#endif

	//Initialization of rand
	srand((unsigned)time(NULL));
	 
	//Input Initialization
	{
		#if defined (RANDOM)
			#define S0() (alpha = rand()%50) 
		#elif defined (CHECKING) || defined (VERIFY)
			#ifdef NO_PROMPT
				#define S0() scanf("%f", &alpha)
			#else
				#define S0() printf("alpha="); scanf("%f", &alpha)
			#endif
		#else
			#define S0() (alpha = 1)   //Default value
		#endif
		
		
		S0();
		#undef S0
	}
	{
		#if defined (RANDOM)
			#define S0() (beta = rand()%50) 
		#elif defined (CHECKING) || defined (VERIFY)
			#ifdef NO_PROMPT
				#define S0() scanf("%f", &beta)
			#else
				#define S0() printf("beta="); scanf("%f", &beta)
			#endif
		#else
			#define S0() (beta = 1)   //Default value
		#endif
		
		
		S0();
		#undef S0
	}
	{
		#if defined (RANDOM)
			#define S0(i,j) (A(i,j) = rand()%50) 
		#elif defined (CHECKING) || defined (VERIFY)
			#ifdef NO_PROMPT
				#define S0(i,j) scanf("%f", &A(i,j))
			#else
				#define S0(i,j) printf("A(%ld,%ld)=",(long) i,(long) j); scanf("%f", &A(i,j))
			#endif
		#else
			#define S0(i,j) (A(i,j) = 1)   //Default value
		#endif
		
		
		int c1,c2;
		for(c1=0;c1 <= P-1;c1+=1)
		 {
		 	for(c2=0;c2 <= Q-1;c2+=1)
		 	 {
		 	 	S0((c1),(c2));
		 	 }
		 }
		#undef S0
	}
	{
		#if defined (RANDOM)
			#define S0(i,j) (B(i,j) = rand()%50) 
		#elif defined (CHECKING) || defined (VERIFY)
			#ifdef NO_PROMPT
				#define S0(i,j) scanf("%f", &B(i,j))
			#else
				#define S0(i,j) printf("B(%ld,%ld)=",(long) i,(long) j); scanf("%f", &B(i,j))
			#endif
		#else
			#define S0(i,j) (B(i,j) = 1)   //Default value
		#endif
		
		
		int c1,c2;
		for(c1=0;c1 <= Q-1;c1+=1)
		 {
		 	for(c2=0;c2 <= R-1;c2+=1)
		 	 {
		 	 	S0((c1),(c2));
		 	 }
		 }
		#undef S0
	}
	{
		#if defined (RANDOM)
			#if defined (VERIFY)
				#define S0(i,j) Cout(i,j) = rand()%50; Cout_verify(i,j)=Cout(i,j);
			#else
				#define S0(i,j) (Cout(i,j) = rand()%50)
			#endif
		#elif defined (CHECKING) || defined (VERIFY)
			#if defined (VERIFY)
				#ifdef NO_PROMPT
					#define S0(i,j) scanf("%f", &Cout(i,j)); Cout_verify(i,j)=Cout(i,j);
				#else
					#define S0(i,j) printf("Cout(%ld,%ld)=",(long) i,(long) j); scanf("%f", &Cout(i,j)); Cout_verify(i,j)=Cout(i,j);
				#endif
			#else
				#ifdef NO_PROMPT
					#define S0(i,j) scanf("%f", &Cout(i,j))
				#else
					#define S0(i,j) printf("Cout(%ld,%ld)=",(long) i,(long) j); scanf("%f", &Cout(i,j))
				#endif
			#endif
		#else
			#define S0(i,j) (Cout(i,j) = 1)   //Default value
		#endif
		
				
		int c1,c2;
		for(c1=0;c1 <= P-1;c1+=1)
		 {
		 	for(c2=0;c2 <= R-1;c2+=1)
		 	 {
		 	 	S0((c1),(c2));
		 	 }
		 }
		#undef S0
	}
	
	//Timing
	struct timeval time;
	double elapsed_time;
	
	//Call the main computation
	gettimeofday(&time, NULL);
	elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
	
	gemm(P, Q, R, ts1_l1, ts2_l1, ts3_l1, &alpha, &beta, A, B, Cout);

	gettimeofday(&time, NULL);
	elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time;

	// timing information
	printf("Execution time : %lf sec.\n", elapsed_time);
	
	#ifdef TIMING
		FILE * fp = fopen( "trace.dat","a+");
		if (fp == NULL) {
				printf("I couldn't open trace.dat for writing.\n");
				exit(EXIT_FAILURE);
		}
		fprintf(fp, "%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%lf\n",P,Q,R,ts1_l1,ts2_l1,ts3_l1,elapsed_time);
		fclose(fp);
	#endif
	
	//Verification Run
	#ifdef VERIFY
		#ifdef TIMING
			gettimeofday(&time, NULL);
			elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
		#endif
    	gemm_verify(P, Q, R, ts1_l1, ts2_l1, ts3_l1, &alpha, &beta, A, B, Cout_verify, Cout_verify);
    	#ifdef TIMING
    		gettimeofday(&time, NULL);
			elapsed_time = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000) - elapsed_time;
			
			FILE * fp_verify = fopen( "trace_verify.dat","a+");
			if (fp == NULL) {
					printf("I couldn't open trace_verify.dat for writing.\n");
					exit(EXIT_FAILURE);
			}
			fprintf(fp, "%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%lf\n",P,Q,R,ts1_l1,ts2_l1,ts3_l1,elapsed_time);
			fclose(fp_verify);
		#endif
	#endif
    	
	#ifdef CHECKING
    	//Print Outputs
		
		{
			#ifdef NO_PROMPT
				#define S0(i,j) printf("%0.2lf\n",var_Cout(i,j))
			#else
				#define S0(i,j) printf("Cout(%ld,%ld)=",(long) i,(long) j);printf("%0.2lf\n",var_Cout(i,j))
			#endif
			int c1,c2;
			for(c1=0;c1 <= P-1;c1+=1)
			 {
			 	for(c2=0;c2 <= R-1;c2+=1)
			 	 {
			 	 	S0((c1),(c2));
			 	 }
			 }
			#undef S0
		}
	#elif VERIFY
		//Compare outputs for verification
		{
			//Error Counter
			int _errors_ = 0;
			#define S0(i,j) if (fabs(1.0 - var_Cout_verify(i,j)/var_Cout(i,j)) > EPSILON) _errors_++;
			int c1,c2;
			for(c1=0;c1 <= P-1;c1+=1)
			 {
			 	for(c2=0;c2 <= R-1;c2+=1)
			 	 {
			 	 	S0((c1),(c2));
			 	 }
			 }
			#undef S0
			if(_errors_ == 0){
				printf("TEST for Cout PASSED\n");
			}else{
				printf("TEST for Cout FAILED. #Errors: %d\n", _errors_);
			}
		}
    #endif
    
	//Memory Free
	free(A);
	free(B);
	free(Cout);
	#ifdef VERIFY
		free(Cout_verify);
	#endif
	
	return EXIT_SUCCESS;
}

//Memory Macros
#undef A
#undef B
#undef Cout


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
#undef EPSILON
