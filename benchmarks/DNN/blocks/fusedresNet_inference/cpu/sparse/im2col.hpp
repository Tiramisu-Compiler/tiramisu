#include <omp.h>
#include "configure.h"
// This code is from BVLC/Caffe changed a little bit : link https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void im2col_cpu_addpadding(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    float* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) + 1;
  int width_col = (width + 2 * pad_w - kernel_w) + 1;
  int channels_col = channels * kernel_h * kernel_w;
  #pragma omp parallel for
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    const int hc0 = h_offset - pad_h;
    const int wc0 = w_offset - pad_w;
    for (int h = 0; h < height_col; ++h) {
      int h_pad = h + hc0;

      const int row_offset = (c * height_col + h) * width_col;
      const int srow_offset = (c_im * height + h_pad) * width;
      if (((unsigned)h_pad) < ((unsigned)height))
        for (int w = 0; w < width_col; ++w) {
          int w_pad = w + wc0;
          if (((unsigned)w_pad) < ((unsigned)width))
            data_col[row_offset + w] = data_im[srow_offset + w_pad];
          else
            data_col[row_offset + w] = 0;
        }
      else
      for (int w = 0; w < width_col; ++w)
        data_col[row_offset + w] = 0;
    }
  }
}

void im2col_cpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    float* data_col) {
  int height_col = (height - kernel_h) + 1;
  int width_col = (width - kernel_w) + 1;
  int channels_col = channels * kernel_h * kernel_w;
  #pragma omp parallel for
  for( int h_b = 0; h_b < H_NB_BL; h_b++){
    int h_start = h_b * H_BL;
    int h_end = std::min((h_b + 1) * H_BL, N);
    for( int w_b = 0; w_b < W_NB_BL; w_b++){
      int w_start = w_b * W_BL;
      int w_end =  std::min((w_b + 1) * W_BL, N);
      for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;
        const int hc0 = h_offset;
        const int wc0 = w_offset;

        for (int h = h_start; h < h_end; ++h) {
          int h_pad = h + hc0;

          const int row_offset = (c * height_col + h) * width_col;
          float* data_col_off = data_col + row_offset;
          if ((((unsigned)h_pad) < ((unsigned)height)) ){
            const int srow_offset = (c_im * height + h_pad) * width;
            float * data_im_offset = (float*)data_im + srow_offset;
            for (int w = w_start; w < w_end; ++w) {
              int w_pad = w + wc0;
              data_col_off[w] = data_im_offset[w_pad];
              if (((unsigned)w_pad) >= ((unsigned)width))
                data_col_off[w] = (float)0;
            }
          }
          else
            for (int w = w_start; w < w_end; ++w)
                data_col[row_offset + w] = (float)0;
        }
      }
    }

  }
}
