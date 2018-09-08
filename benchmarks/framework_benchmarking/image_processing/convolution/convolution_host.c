#include <assert.h>
#include <stdio.h>
#include "ocl_utilities.h"
#include "convolution_kernel.cl"
#include <time.h>

#include "convolution.h"
#include <pencil.h>
#include <assert.h>

//#if !__PENCIL__
#include <stdlib.h>
//#endif

static void convolution( const int rows
                    , const int cols
                    , const int step
                    , const unsigned char src[rows][step][3]
                    , const int kernelX_length
                    , const float kernelX[kernelX_length]
                    , const int kernelY_length
                    , const float kernelY[kernelY_length]
                    , uint8_t conv[rows][step][3]
		    , uint8_t temp[rows][step][3]
                    )
{
    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (rows >= 1 && cols >= 1) {
      #define openclCheckReturn(ret) \
  if (ret != CL_SUCCESS) {\
    fprintf(stderr, "OpenCL error: %s\n", opencl_error_string(ret)); \
    fflush(stderr); \
    assert(ret == CL_SUCCESS);\
  }

      cl_mem dev_conv;
      cl_mem dev_kernelX;
      cl_mem dev_src;
      
      cl_device_id device;
      cl_context context;
      cl_program program;
      cl_command_queue queue;
      cl_int err;
      device = opencl_create_device(1);
      context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
      openclCheckReturn(err);
      queue = clCreateCommandQueue(context, device, 0, &err);
      openclCheckReturn(err);
      program = opencl_build_program_from_string(context, device, kernel_code, sizeof(kernel_code), "");
      
      #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
      {
        dev_conv = clCreateBuffer(context, CL_MEM_READ_WRITE, ppcg_max(sizeof(unsigned char), (rows) * (step) * (3) * sizeof(unsigned char)), NULL, &err);
        openclCheckReturn(err);
      }
      {
        dev_kernelX = clCreateBuffer(context, CL_MEM_READ_WRITE, ppcg_max(sizeof(const float), (kernelX_length) * sizeof(const float)), NULL, &err);
        openclCheckReturn(err);
      }
      {
        dev_src = clCreateBuffer(context, CL_MEM_READ_WRITE, ppcg_max(sizeof(unsigned char), (rows) * (step) * (3) * sizeof(unsigned char)), NULL, &err);
        openclCheckReturn(err);
      }
      
      if (step >= 1)
        openclCheckReturn(clEnqueueWriteBuffer(queue, dev_conv, CL_TRUE, 0, (rows) * (step) * (3) * sizeof(unsigned char), conv, 0, NULL, NULL));
      if (kernelX_length >= 1) {
        openclCheckReturn(clEnqueueWriteBuffer(queue, dev_kernelX, CL_TRUE, 0, (kernelX_length) * sizeof(const float), kernelX, 0, NULL, NULL));
        if (step >= 1)
          openclCheckReturn(clEnqueueWriteBuffer(queue, dev_src, CL_TRUE, 0, (rows) * (step) * (3) * sizeof(unsigned char), src, 0, NULL, NULL));
      }
      {
        size_t global_work_size[3] = {(ppcg_min(256, (rows + 31) / 32)) * 32, (1) * 3, 4};
        size_t block_size[3] = {32, 3, 4};
        cl_kernel kernel0 = clCreateKernel(program, "kernel0", &err);
        openclCheckReturn(err);
	clock_t begin = clock();
        openclCheckReturn(clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void *) &dev_conv));
        openclCheckReturn(clSetKernelArg(kernel0, 1, sizeof(cl_mem), (void *) &dev_kernelX));
        openclCheckReturn(clSetKernelArg(kernel0, 2, sizeof(cl_mem), (void *) &dev_src));
        openclCheckReturn(clSetKernelArg(kernel0, 3, sizeof(step), &step));
        openclCheckReturn(clSetKernelArg(kernel0, 4, sizeof(rows), &rows));
        openclCheckReturn(clSetKernelArg(kernel0, 5, sizeof(cols), &cols));
        openclCheckReturn(clSetKernelArg(kernel0, 6, sizeof(kernelX_length), &kernelX_length));
        openclCheckReturn(clEnqueueNDRangeKernel(queue, kernel0, 3, NULL, global_work_size, block_size, 0, NULL, NULL));
        openclCheckReturn(clReleaseKernel(kernel0));
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("new time : %f \n", time_spent);
        clFinish(queue);
      }
      
      if (step >= 1)
        openclCheckReturn(clEnqueueReadBuffer(queue, dev_conv, CL_TRUE, 0, (rows) * (step) * (3) * sizeof(unsigned char), conv, 0, NULL, NULL));
      openclCheckReturn(clReleaseMemObject(dev_conv));
      openclCheckReturn(clReleaseMemObject(dev_kernelX));
      openclCheckReturn(clReleaseMemObject(dev_src));
      openclCheckReturn(clReleaseCommandQueue(queue));
      openclCheckReturn(clReleaseProgram(program));
      openclCheckReturn(clReleaseContext(context));
    }
}

void pencil_convolution( const int rows
                    , const int cols
                    , const int step
                    , const uint8_t src[]
                    , const int kernelX_length
                    , const float kernelX[]
                    , const int kernelY_length
                    , const float kernelY[]
                    , uint8_t conv[]
		    , uint8_t temp[]
                    )
{
    convolution( rows, cols, step, (const uint8_t(*)[step])src
            , kernelX_length, kernelX
            , kernelY_length, kernelY
            , (uint8_t(*)[step])conv
	    , temp
            );
}
