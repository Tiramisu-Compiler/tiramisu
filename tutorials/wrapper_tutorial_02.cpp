#include "Halide.h"
#include "wrapper_tutorial_02.h"
#include "halide_image_io.h"
#include <cstdlib>
#include <iostream>

#define NN 10

void print_array(buffer_t buf, int N, int M)
{
	int i,j;

	for (i=0; i<N; i++)
	{
		for (j=0; j<M; j++)
		{
			printf("%u, ", buf.host[i*M+j]);
		}
		printf("\n");
	}
	printf("\n");
}

void init_array_val(buffer_t *buf, int N, int M, uint8_t val)
{
	int i, j;

	for (i=0; i<N; i++)
		for (j=0; j<M; j++)
			buf->host[i*M+j] = val;
}

void copy_array(uint8_t* buf, int N, int M, uint8_t* array)
{
	int i, j;

	for (i=0; i<N; i++)
		for (j=0; j<M; j++)
			buf[i*M+j] = array[i*M+j];
}

int main(int, char**)
{
   Halide::Image<uint8_t> image = Halide::Tools::load_image("./tutorials/images/rgb.png");

   buffer_t input_buf = {0};
   input_buf.host = (unsigned char *) image.data();
   input_buf.stride[0] = 1;
   input_buf.stride[1] = image.extent(0);
   input_buf.extent[0] = image.extent(0);
   input_buf.extent[1] = image.extent(1);
   input_buf.min[0] = 0;
   input_buf.min[1] = 0;
   input_buf.elem_size = 1;

   buffer_t output_buf = {0};
   output_buf.host = (unsigned char *) malloc(image.extent(0)*image.extent(1)*sizeof(unsigned char));
   output_buf.stride[0] = 1;
   output_buf.stride[1] = image.extent(0);
   output_buf.extent[0] = image.extent(0);
   output_buf.extent[1] = image.extent(1);
   output_buf.min[0] = 0;
   output_buf.min[1] = 0;
   output_buf.elem_size = 1;

   blurxy(&input_buf, &output_buf);

   copy_array(image.data(), image.extent(0), image.extent(1), output_buf.host);

   Halide::Tools::save_image(image, "tutorial_02.png");

   return 0;
}
