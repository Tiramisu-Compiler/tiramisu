#include "Halide.h"
#include "function10243_no_schedule_wrapper.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <fstream>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define MAX_RAND 200

int main(int, char **){
    Halide::Buffer<int32_t> image_buff(1024, 1024);
    
    Mat image;
    image = imread("/home/isra/tiramisu/sample_nature.jpeg" , 1);
  
    if(! image.data ) {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    Mat gray_image;
    cvtColor( image, gray_image, COLOR_BGR2GRAY );
    for (int i = 0; i < 1024; ++i){
        for (int j = 0; j < 1024; ++j){
            image_buff(j, i) = gray_image.data[j + i * 1024];
        }
    }


    
    Halide::Buffer<int32_t> filter_buff(1, 3, 3);
    
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j){
            for (int k = 0; k < 1; ++k){
               /* if (i == 0) filter_buff(k, j, i) = 8;
		else filter_buff(k, j, i) = -1;*/
		if (j == 0) filter_buff(k, j, i) = 1;
		if (j == 1) filter_buff(k, j, i) = 0;
		if (j == 2) filter_buff(k, j, i) = -1;
            }
        }
    }

    Halide::Buffer<int32_t> convolved_buff(1, 1022, 1022);
    init_buffer(convolved_buff, (int32_t)0);

    
    auto t1 = std::chrono::high_resolution_clock::now();

    function10243_no_schedule(image_buff.raw_buffer(), filter_buff.raw_buffer(), convolved_buff.raw_buffer());

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = t2 - t1;

    std::ofstream exec_times_file;
    exec_times_file.open("../data/programs/function10243/function10243_no_schedule/exec_times.txt", std::ios_base::app);
    if (exec_times_file.is_open()){
        exec_times_file << diff.count() * 1000000 << "us" <<std::endl;
        exec_times_file.close();
    }

    
    Halide::Buffer<int32_t> expected_buff(1, 1024, 1024);
    init_buffer(expected_buff, (int32_t)0);
    for (int i = 1; i < 1023; ++i){
        for (int j = 1; j < 1023; ++j){
            for (int k = -1; k < 2; ++k){
		for (int l = -1; l < 2; ++l){
                	expected_buff(0, j, i) += image_buff(j + l, i + k) * filter_buff(0, l + 1, k + 1);
		}
	    }
        }
    }





    /*for (int i = 0; i < 1022; ++i){
        for (int j = 0; j < 1022; ++j){
            if (expected_buff(1, j + 1, i + 1) != convolved_buff(1, i, j)) cout <<  "OUPS" << std::endl ;
        }
    }*/




    vector<int> compression_params;
    compression_params.push_back(1);
    compression_params.push_back(95);


unsigned char *expected_image;
    expected_image = (unsigned char*)malloc(32 * 1022 * 1022);
    for (int i = 0; i < 1022; ++i){
        for (int j = 0; j < 1022; ++j){
            expected_image[j + i * 1022] = expected_buff(0, j, i);
        }
    }

    Mat img_exp = Mat(1022, 1022, CV_8U, (unsigned*)expected_image);

    

    try {
        imwrite("expected.jpeg", img_exp, compression_params);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        return 1;
	}




    unsigned char *modified_image;
    modified_image = (unsigned char*)malloc(32 * 1022 * 1022);
    for (int i = 0; i < 1022; ++i){
        for (int j = 0; j < 1022; ++j){
            modified_image[j + i * 1022] = convolved_buff(0, j, i);
        }
    }

    Mat img = Mat(1022, 1022, CV_8U, (unsigned*)modified_image);

    

    try {
        imwrite("convolved.jpeg", img, compression_params);
    }
    catch (runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        return 1;
	}

    return 0;
}
