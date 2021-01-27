#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "mkl.h"
#include "mkldnn.hpp"
#undef N
#include "configure.h"

#define ACCESS_ARRAY (x + y * N + fin * N * N + n * N * N * FIn)

using namespace mkldnn;

int main()
{
    std::vector<double> duration_vector;

    srand(1);

    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    float* buf_x = (float*) malloc(BATCH_SIZE*FIn*N*N * sizeof(float));
    float* buf_y = (float*) malloc(BATCH_SIZE*FIn*N*N * sizeof(float));

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fin = 0; fin < FIn; ++fin)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                  buf_x[ACCESS_ARRAY] = 1.f;

    for (int n = 0; n < BATCH_SIZE; ++n)
        for (int fin = 0; fin < FIn; ++fin)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    buf_y[ACCESS_ARRAY] = 2.f;


    auto relu_src_md = memory::desc(
      {BATCH_SIZE, FIn, N, N},
      memory::data_type::f32,
      memory::format_tag::nchw
    );
    auto relu_usr_mem = memory(relu_src_md, cpu_engine, buf_x);

    // create a relu

    auto relu_desc = eltwise_forward::desc(prop_kind::forward_inference,
              algorithm::eltwise_relu, relu_src_md,
              0);
    auto relu_pd = eltwise_forward::primitive_desc(relu_desc, cpu_engine);
    net.push_back(eltwise_forward(relu_pd));
    net_args.push_back({
      {MKLDNN_ARG_SRC, relu_usr_mem},
      {MKLDNN_ARG_DST, relu_usr_mem}
    });
    double start, end, middle;
    for (int i = 0; i < NB_TESTS; ++i) {
        for (int n = 0; n < BATCH_SIZE; ++n)
            for (int fin = 0; fin < FIn; ++fin)
                for (int y = 0; y < N; ++y)
                    for (int x = 0; x < N; ++x)
                        buf_y[ACCESS_ARRAY] = 2.f;
        start = rtclock();

        cblas_saxpy ((MKL_INT)(N*N*FIn*BATCH_SIZE), 1, buf_x, (MKL_INT)1, buf_y, (MKL_INT)1);
        middle = rtclock();
        net[0].execute(cpu_stream, net_args[0]);
        cpu_stream.wait();

        end = rtclock();
        duration_vector.push_back((end - start) * 1000);
    }

    std::cout << "\t\tAdd ReLU : "
    << median(duration_vector) << " ms" << std::endl;
    std::cout << "\t\t\tSaxPy time : "<<(middle-start) * 1000 << " ms" <<std::endl;
    // Write results to file
    FILE* f = fopen("mkl_result.txt", "w");
    if (f == NULL) {
        printf("Error creating mkl_result.txt.\n");
        return 1;
    }

    for (int n = 0; n < BATCH_SIZE; ++n)
      for (int fin = 0; fin < FIn; ++fin)
        for (int y = 0; y < N; ++y)
          for (int x = 0; x < N; ++x)
            fprintf(f, "%.10g\n", buf_y[ACCESS_ARRAY]);

    fclose(f);

    return 0;
}
