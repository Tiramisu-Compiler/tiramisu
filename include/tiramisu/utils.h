#ifndef _TIRAMISU_UTILS
#define _TIRAMISU_UTILS

#include <string>
#include <chrono>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

double median(std::vector<std::chrono::duration<double,std::milli>> scores);
void print_time(std::string file_name, std::string kernel_name,
                std::vector<std::string> header_text,
                std::vector<double>      time_vector);
void init_1D_buffer(buffer_t *buf, int N, uint8_t val);
void print_1D_buffer(buffer_t *buf, int N);
void print_2D_buffer(buffer_t *buf, int N, int M);
void init_1D_buffer_val(buffer_t *buf, int N, uint8_t val);
buffer_t allocate_1D_buffer(int NN);
void print_2D_array(buffer_t buf, int N, int M);
void init_2D_buffer_val(buffer_t *buf, int N, int M, uint8_t val);
void copy_2D_buffer(uint8_t* buf, int N, int M, uint8_t* array);
buffer_t allocate_2D_buffer(int NN, int MM);
void compare_2_2D_arrays(std::string str, uint8_t *array1, uint8_t *array2, int N, int M);
void compare_2_1D_arrays(std::string str, uint8_t *array1, uint8_t *array2, int N);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
