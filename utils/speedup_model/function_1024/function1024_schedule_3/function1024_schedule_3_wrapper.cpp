#include "Halide.h"
#include "function1024_schedule_3_wrapper.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

#define MAX_RAND 200
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;
int main(int, char **){
    Halide::Buffer<int32_t> buf00(1024, 1024);
    Mat image;
    image = imread("/home/isra/tiramisu/build/ref.jpeg" , 1);
    if(! image.data ) {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    Mat gray_image;
    cvtColor( image, gray_image, COLOR_BGR2GRAY );
    
    for (int i = 0; i < 1024; ++i){
        for (int j = 0; j < 1024; ++j){
            buf00(j, i) = gray_image.data[j + i * 1024];
        }
    }

    Halide::Buffer<int32_t> buf0(1024, 1024);
    init_buffer(buf0, (int32_t)0);

    
    auto t1 = std::chrono::high_resolution_clock::now();

    function1024_schedule_3(buf00.raw_buffer(), buf0.raw_buffer());

    for (int j = 0; j < 25; ++j){
        function1024_schedule_3(buf0.raw_buffer(), buf00.raw_buffer());
        function1024_schedule_3(buf00.raw_buffer(), buf0.raw_buffer());
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = t2 - t1;

    std::ofstream exec_times_file;
    exec_times_file.open("../data/programs/function1024/function1024_schedule_3/exec_times.txt", std::ios_base::app);
    if (exec_times_file.is_open()){
        exec_times_file << diff.count() * 1000000 << "us" <<std::endl;
        exec_times_file.close();
    }

    unsigned char *modified_image;
    modified_image = (unsigned char*)malloc(32 * 1024* 1024);

    
    for (int i = 0; i < 1024; ++i){
        for (int j = 0; j < 1024; ++j){
            modified_image[j + i * 1024] = buf0(j, i);
        }
    }

    Mat img = Mat(1024, 1024, CV_8U, (unsigned*)modified_image);
    vector<int> compression_params;
    compression_params.push_back(1);
    compression_params.push_back(95);

    try {
        imwrite("../data/programs/function1024/function1024_schedule_3/modified.jpeg", img, compression_params);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to JPEG format: %s\n", ex.what());
        return 1;
    }

    return 0;
}
