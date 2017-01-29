#include "Halide.h"
#include "tiramisu/utils.h"
#include "tiramisu/debug.h"

#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <fstream>


double median(std::vector<std::chrono::duration<double,std::milli>> scores)
{
  double median;
  size_t size = scores.size();

  sort(scores.begin(), scores.end());

  if (size  % 2 == 0)
      median = (scores[size/2-1].count() + scores[size/2].count())/2;
  else
      median = scores[size/2].count();

  return median;
}

void print_time(std::string file_name, std::string kernel_name,
                std::vector<std::string> header_text,
                std::vector<double>      time_vector)
{
    std::ofstream file;

    file.open(file_name, std::ios::app);
    file << std::fixed << std::setprecision(6);

    file << kernel_name << " ; ";
    for (auto t: time_vector)
        file << t << " ;";
    file << std::endl;

    std::cout << "Kernel : ";
    for (auto t: header_text)
        std::cout << t << " ;";
    std::cout << std::endl;

    std::cout << kernel_name << " : ";
    for (auto t: time_vector)
        std::cout << t << " ;";
    std::cout << std::endl;

    file.close();
}

void print_1D_buffer(buffer_t *buf, int N)
{
    int i;

    for (i=0; i<N; i++)
    {
       printf("%u, ", buf->host[i]);
    }
    printf("\n");
}

void init_1D_buffer(buffer_t *buf, int N, uint8_t val)
{
    int i;

    for (i=0; i<N; i++)
    {
        buf->host[i] = val;
    }
}

void init_1D_buffer_val(buffer_t *buf, int N, uint8_t val)
{
    int i;

    for (i=0; i<N; i++)
        buf->host[i] = val;
}

void print_2D_buffer(buffer_t *buf, int N, int M)
{
    int i,j;

    for (i=0; i<N; i++)
    {
        for (j=0; j<M; j++)
        {
            printf("%u, ", buf->host[i*M+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void init_2D_buffer_val(buffer_t *buf, int N, int M, uint8_t val)
{
    int i, j;

    for (i=0; i<N; i++)
    {
        for (j=0; j<M; j++)
        {
            buf->host[i*M+j] = val;
        }
    }
}

/**
 * Create an array {val1, val2, val1, val2, val1, val2, val1,
 * val2, ...}.
 */
void init_2D_buffer_interleaving(buffer_t *buf, int N, int M,
								 uint8_t val1, uint8_t val2)
{
	int i, j;

	for (i=0; i<N; i++)
    {
		for (j=0; j<M; j++)
        {
			if (j%2 == 0)
            {
				buf->host[i*M+j] = val1;
            }
			else
            {
				buf->host[i*M+j] = val2;
            }
        }
    }
}

void copy_2D_buffer(uint8_t* buf, int N, int M, uint8_t* array)
{
    int i, j;

    for (i=0; i<N; i++)
    {
        for (j=0; j<M; j++)
        {
            buf[i*M+j] = array[i*M+j];
        }
    }
}

buffer_t allocate_1D_buffer(int NN)
{
    buffer_t input_buf = {0};
    input_buf.host = (unsigned char *) malloc(NN*sizeof(unsigned char));
    input_buf.min[0] = 0;
    input_buf.stride[0] = 1;
    input_buf.extent[0] = NN;
    input_buf.min[1] = 0;
    input_buf.stride[1] = NN;
    input_buf.extent[1] = 1;
    input_buf.elem_size = 1;
    return input_buf;
}

buffer_t allocate_2D_buffer(int NN, int MM)
{
	buffer_t input_buf = {0};
	input_buf.host = (unsigned char *) malloc(NN*MM*sizeof(unsigned char));
    input_buf.min[0] = 0;
	input_buf.stride[0] = 1;
    input_buf.extent[0] = NN;
    input_buf.min[1] = 0;
	input_buf.stride[1] = NN;
	input_buf.extent[1] = MM;
	input_buf.elem_size = 1;
	return input_buf;
}

void compare_2_2D_arrays(std::string str, uint8_t *array1, uint8_t *array2, int N, int M)
{
    bool error = false;

	for (int j=0; j<M; j++)
    {
        for (int i=0; i<N; i++)
        {
			if (array1[i+j*N] != array2[i+j*N])
            {
				error = true;
            }
        }
    }

	if (error)
	    tiramisu::error("\033[1;31mTest " + str + " failed.\033[0m\n", false);
	else
	    tiramisu::str_dump("\033[1;32mTest " + str + " succeeded.\033[0m\n");
}


void compare_2_1D_arrays(std::string str, uint8_t *array1, uint8_t *array2, int N)
{
    bool error = false;

    for (int i=0; i<N; i++)
    {
            if (array1[i] != array2[i])
            {
                error = true;
            }

    }

    if (error)
        tiramisu::error("\033[1;31mTest " + str + " failed.\033[0m\n", false);
    else
        tiramisu::str_dump("\033[1;32mTest " + str + " succeeded.\033[0m\n");
}
