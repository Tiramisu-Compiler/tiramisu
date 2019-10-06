#include "Halide.h"
#include "function10243_schedule_42_wrapper.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

#define MAX_RAND 200

int main(int, char **){
    Halide::Buffer<int32_t> image_buff(1024, 1024);
    
    for (int i = 0; i < 1024; ++i){
        for (int j = 0; j < 1024; ++j){
            image_buff(j, i) = (rand() % MAX_RAND) + 1;
        }
    }

    
    Halide::Buffer<int32_t> filter_buff(3, 3, 3);
    
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j){
            for (int k = 0; k < 3; ++k){
                filter_buff(k, j, i) = (rand() % MAX_RAND) + 1;
            }
        }
    }

    Halide::Buffer<int32_t> convolved_buff(1022, 1022, 3);
    init_buffer(convolved_buff, (int32_t)0);

    
    auto t1 = std::chrono::high_resolution_clock::now();

    function10243_schedule_42(image_buff.raw_buffer(), filter_buff.raw_buffer(), convolved_buff.raw_buffer());

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = t2 - t1;

    std::ofstream exec_times_file;
    exec_times_file.open("../data/programs/function10243/function10243_schedule_42/exec_times.txt", std::ios_base::app);
    if (exec_times_file.is_open()){
        exec_times_file << diff.count() * 1000000 << "us" <<std::endl;
        exec_times_file.close();
    }

    return 0;
}