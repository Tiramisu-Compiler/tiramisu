#include "Halide.h"
#include "wrapper_ger.h"
#include <tiramisu/utils.h>
#include <cstdlib>
#include <iostream>
#include <chrono>


using namespace std;
using namespace std::chrono;


#define MM 100
#define NN 200
#define alpha 3
int main(int, char **)
{
    Halide::Buffer<uint8_t> A_buf(NN, MM);
    Halide::Buffer<uint8_t> X_buf(MM);
    Halide::Buffer<uint8_t> Y_buf(NN);

    // Initialize matrix A with pseudorandom values:
    for (int i = 0; i < MM; i++) {
        for (int j = 0; j < NN; j++) {
            A_buf(j, i) = (i + 3) * (j + 1);
        }

    }
    // Initialize Vector X with pseudorandom values:
    for(int i=0 ; i<MM ;i++){
        X_buf(i) = (i + 1) ;

    }
    // Initialize Vector Y with pseudorandom values:
    for(int i=0 ; i<NN ;i++){ 
        Y_buf(i) = i ;

    }

    // Output buffer
    Halide::Buffer<uint8_t> C_buf(NN, MM);
    init_buffer(C_buf, (uint8_t)0);
    // TRAMISU CODE EXECUTION STARTS:
    auto start1 = std::chrono::high_resolution_clock::now();
    ger(A_buf.raw_buffer(), X_buf.raw_buffer(),Y_buf.raw_buffer(),C_buf.raw_buffer());
    auto end1 = std::chrono::high_resolution_clock::now();
    auto  duration1 =duration_cast<microseconds>(end1 - start1);
    // TRAMISU CODE EXECUTION ENDS.




    // REFERENCE Output buffer
    Halide::Buffer<uint8_t> C2_buf(NN, MM);
    init_buffer(C2_buf, (uint8_t)0);
    // REFERENCE C++ CODE EXECUTION STARTS:
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < MM; i++) {
        for (int j = 0; j < NN; j++) {

                C2_buf(j, i) = A_buf(j, i) +(X_buf(i)*Y_buf(j))*alpha;
         }
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto  duration2 =duration_cast<microseconds>(end2 - start2);
    // REFERENCE C++ CODE EXECUTION ENDS.


   //===== you can print MATRIX A =====
     //printf("\n MAT A  :");
     //print_buffer(A_buf);

   //===== you can print VECT X =====
     //printf("\n VECTEUR X  :");
     //print_buffer(X_buf);

   //===== you can print VECT Y =====
     //printf("\n VECTEUR Y  :");
     //print_buffer(Y_buf);

   //===== you can print MATRIX C =====
     //printf("\n SOL  :");
     //print_buffer(C_buf);

   //===== you can print MATRIX C_REF =====
     //printf("\n SOL_ref  :");
     //print_buffer(C2_buf);


   //===== printing REFERECE EXEC TIME: =====
    std::cout << "\n REF RESOLUTION TIME : " << duration2.count() << "microseconds";
   //===== printing TIRAMISU EXEC TIME: =====
    std::cout << "\n TIRAMISU RESOLUTION TIME : " << duration1.count() << "microseconds";
    printf("\n");

   //===== Verify if TIRAMISU output is correct: =====
    compare_buffers("ger", C_buf, C2_buf);

    printf("\n");

    return 0;
}
