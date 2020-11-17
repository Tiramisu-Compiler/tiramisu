#include "Halide.h"
#include "tiramisu/utils.h"
#include "new.o.h"
#include <cstdlib>
#include <iostream>

int main(){

    Halide::Buffer<int64_t> input(100, 100);
    
    function_0(input.raw_buffer()) ;
    std::cout<<"hello world";

    return 0 ;
}
