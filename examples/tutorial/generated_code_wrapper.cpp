#include "Halide.h"
#include "generated_code_wrapper.h"
#include <cstdlib>
#include <iostream>

#define NN 10
#define MM 10

void print_array(buffer_t buf, int N)
{
	int i;

	for (i=0; i<N; i++)
	{
	   printf("%d, ", buf.host[i]);
	}
	printf("\n");
}

void init_array(buffer_t *buf, int N, uint8_t val)
{
	int i;

	for (i=0; i<N; i++)
	    buf->host[i] = val;
}

int main(int, char**)
{
   buffer_t input_buf = {0};
   input_buf.host = (unsigned char *) malloc(NN*MM*sizeof(unsigned char));
   input_buf.stride[0] = 1;
   input_buf.stride[1] = 1;
   input_buf.extent[0] = NN;
   input_buf.extent[1] = 1;
   input_buf.min[0] = 0;
   input_buf.min[1] = 0;
   input_buf.elem_size = 1;

   init_array(&input_buf, NN, 9);
   std::cout << "Array (after initialization)" << std::endl;
   print_array(input_buf, NN);

   test1(&input_buf);

   std::cout << "Array after the Halide pipeline" << std::endl;
   print_array(input_buf, NN);

   return 0;
}
