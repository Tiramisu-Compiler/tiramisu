/**
  The goal of this tutorial is to implement, in Tiramisu, a code that is
  equivalent to the following

  for (int i = 0; i < 10; i++)
      buf0[i] = 3 + 4;

  Tiramisu is a code generator, therefore the goal of a Tiramisu program is to
  generate code that is supposed to be called from another program (the user program).

  A Tiramisu program is structures as follows:
	- It starts with a call to initialize the Tiramisu compiler.  The name of the
	function to be generated is specified during this initialization.
	- The user then declares the algorithm: Tiramisu expressions.
	- The user then specifies how the algorithm should be optimized
	using scheduling and data mapping commands.
	- The user then calls the codegen() function which generates code.
	This call compiles the Tiramisu program and generates an object file.
	The user can call the function compiled in the object file
  	from any place in his program.

  How to compile ? You can use the makefile to compile the tutorial or you can do it manually.
  To compile and run the tutorials using the makefile use:

  cd build/
  make run_developers_tutorial_01

  Detailed compilation process (without makefile) are explained below at the end of this tutorial.
 */


// Every Tiramisu program needs to include the header file tiramisu/tiramisu.h
// which defines classes for declaring and compiling Tiramisu expressions.
#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char **argv)
{
    // Initialize the tiramisu compiler and declare a function called "function0".
    // A function in tiramisu is the equivalent of a function in C.
    // It can have input and output arguments (buffers).  These arguments
    // are declared later in the tutorial.
    tiramisu::init("function0");

    // -------------------------------------------------------
    // Layer I: provide the algorithm.
    // -------------------------------------------------------

    // Declare an iterator. The range of this iterator is [0, 10[
    // i.e., 0<=i<10
    var i("i", 0, 10);

    // Declare a computation that adds 3 and 4.  This computation is done
    // within a loop that has i as iterator and is named "S0". The name is
    // useful for debugging.
    computation S0("S0", {i}, 3 + 4);
    // Since the iterator i is declared to be 0<=i<10 (i.e., the iteration space of S0 is 0<=i<10),
    // the previous declaration of S0 is equivalent to the following C code
    // for (i = 0; i < 10; i++)
    //	    S0(i) = 3 + 4;


    // ------------------------------------------------------------
    // Layer II: specify how the algorithm is optimized.
    // ------------------------------------------------------------

    // Parallelize the loop level i
    S0.parallelize(i);

    // -------------------------------------------------------
    // Layer III: allocate buffers and specify how computations
    // should be stored in these buffers.
    // -------------------------------------------------------

    // Create a buffer called "buf0".
    buffer buf0("buf0", {10}, p_uint8, a_output);
    // The second argument to the constructor is a vector that represents the
    // size of the buffer dimensions.  In this example, the vector has only one
    // element (which is expr(10)), therefore the buffer only has one dimension
    // of size 10.
    // The third argument is the type of the buffer elements (uint8).
    // The fourth argument indicates whether the buffer is in input or an output buffer.
    // Othr types are a_input (for input buffers) and a_temporary (for temporary buffers
    // allocated and freed within the function).
    // For more details about the constructor of buffers please refer to the
    // documentation https://tiramisu-compiler.github.io/doc/classtiramisu_1_1buffer.html
    // In general, all the API of Tiramisu is documented in https://tiramisu-compiler.github.io/doc/



    // Store the computation S0 to the buffer buf0.
    // That is, store the computation S0(i) in buf0[i].
    S0.store_in(&buf0);

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------

    // Compile code and store it in an object file.  Two arguments need
    // to be passed to the code generator:
    //	    - The arguments (buffers) passed to the generated function.
    //	      The buffer buf0 is supposed to be allocated by the user (caller)
    //	      and is supposed to be passed to the generated function "function0".
    //	      Any buffer of type a_output or a_input are supposed to be allocated
    //	      by the user (caller), in contrast to buffers of type a_temporary which are
    //	      allocated automatically by Tiramisu and should not be passed as arguments
    // 	      to the function).
    //	    - The name of the object file to be generated.
    tiramisu::codegen({&buf0}, "build/generated_fct_developers_tutorial_01.o");

    return 0;
}

/**

  If you want to compile the tutorial manuall, you can follow the following steps.
  
  Assuming that the variable TIRAMISU_ROOT contains the path to the root directory of Tiramisu
  and that Tiramisu and it dependences are built using the default paths.
  
  First we need to compile the generator (it requires the header files and libraries of Halide, ISL and Tiramisu which all
  should be built automatically if you follow the default installation process).
  
  cd ${TIRAMISU_ROOT}/build
  g++ -std=c++11 -fno-rtti -DHALIDE_NO_JPEG -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o developers_tutorial_01_fct_generator  -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ${TIRAMISU_ROOT}/tutorials/developers/tutorial_01/tutorial_01.cpp -ltiramisu -lisl -lHalide -ldl -lpthread -lz -lm   
  
  Run the generator to generate the object file.

  cd ${TIRAMISU_ROOT}
  ./build/developers_tutorial_01_fct_generator
  
  This generator generates the file generated_fct_developers_tutorial_01.o
  You can compile the wrapper code (code that uses the generated code) and link it to the generated object file.
  
  cd ${TIRAMISU_ROOT}/build
  g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -o wrapper_tutorial_01  -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ${TIRAMISU_ROOT}/tutorials/developers/tutorial_01/wrapper_tutorial_01.cpp  generated_fct_developers_tutorial_01.o -ltiramisu -lHalide -ldl -lpthread -lz -lm 
  
  To run the program.
  
  ./wrapper_tutorial_01
  
 
 */
