#include "Halide.h"
#include "generated_C_pgm.h"
#include <cstdlib>
#include <iostream>

#define NN 10
#define MM 10

void print_array(buffer_t buf, int N, int M)
{
	int i, j;

	for (i=0; i<N; i++)
	{
	  for (j=0; j<M; j++)
	    printf("buf[%d][%d] = %d - ", i, j, buf.host[i*MM+j]);
	  printf("\n");
	}
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
   input_buf.host = (unsigned char *) malloc(NN*MM*sizeof(unsigned char));
   input_buf.stride[0] = 1;
   input_buf.stride[1] = NN;
   input_buf.extent[0] = NN;
   input_buf.extent[1] = MM;
   input_buf.min[0] = 0;
   input_buf.min[1] = 0;
   input_buf.elem_size = 1;

   init_array(&input_buf, NN, MM, 98);
   std::cout << "Array (after initialization)" << std::endl;
   print_array(input_buf, NN, MM);

   test1(&input_buf);

   std::cout << "Array after the Halide pipeline" << std::endl;
   print_array(input_buf, NN, MM);

   return 0;
}
