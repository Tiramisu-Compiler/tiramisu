#include "Halide.h"
#include "wrapper_tutorial_02.h"
#include <cstdlib>
#include <iostream>

#define NN 10

void print_array(buffer_t buf, int N, int M)
{
	int i,j;

	for (i=0; i<N; i++)
		for (j=0; j<M; j++)
		{
			printf("%u, ", buf.host[i*M+j]);
		}
		printf("\n");
}

void init_array(buffer_t *buf, int N, int M, uint8_t val)
{
	int i, j;

	for (i=0; i<N; i++)
		for (j=0; j<M; j++)
			buf->host[i*M+j] = val;
}

int main(int, char**)
{
   buffer_t input_buf = {0};
   input_buf.host = (unsigned char *) malloc(NN*NN*sizeof(unsigned char));
   input_buf.stride[0] = 1;
   input_buf.stride[1] = NN;
   input_buf.extent[0] = NN;
   input_buf.extent[1] = NN;
   input_buf.min[0] = 0;
   input_buf.min[1] = 0;
   input_buf.elem_size = 1;

   buffer_t output_buf = {0};
   output_buf.host = (unsigned char *) malloc(NN*NN*sizeof(unsigned char));
   output_buf.stride[0] = 1;
   output_buf.stride[1] = NN;
   output_buf.extent[0] = NN;
   output_buf.extent[1] = NN;
   output_buf.min[0] = 0;
   output_buf.min[1] = 0;
   output_buf.elem_size = 1;

   init_array(&input_buf, NN, NN, 9);
   std::cout << "Array (after initialization)" << std::endl;
   print_array(input_buf, NN, NN);

   blurxy(&input_buf, &output_buf);

   std::cout << "Array after the Halide pipeline" << std::endl;
   print_array(output_buf, NN, NN);

   return 0;
}
