#ifndef WRAPPER_H
#define WRAPPER_H

#include <tiramisu/utils.h>
#include <sys/time.h>

#define NB_THREAD_INIT 48

struct args {
    int *buf;
    unsigned long long int part_start;
    unsigned long long int part_end;
    int value;
};

void *init_part(void *params)
{
   int *buffer = ((struct args*) params)->buf;
   unsigned long long int start = ((struct args*) params)->part_start;
   unsigned long long int end = ((struct args*) params)->part_end;
   int val = ((struct args*) params)->value;
   for (unsigned long long int k = start; k < end; k++){
       buffer[k]=val;
   }
   pthread_exit(NULL);
}

void parallel_init_buffer(int* buf, unsigned long long int size, int value){
    pthread_t threads[NB_THREAD_INIT]; 
    struct args params[NB_THREAD_INIT];
    for (int i = 0; i < NB_THREAD_INIT; i++) {
        unsigned long long int start = i*size/NB_THREAD_INIT;
        unsigned long long int end = std::min((i+1)*size/NB_THREAD_INIT, size);
        params[i] = (struct args){buf, start, end, value};
        pthread_create(&threads[i], NULL, init_part, (void*)&(params[i])); 
    }
    for (int i = 0; i < NB_THREAD_INIT; i++) 
        pthread_join(threads[i], NULL); 
    return;
}
        


#ifdef __cplusplus
extern "C" {
#endif


int conv(halide_buffer_t *buf01,halide_buffer_t *buf02);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
double median(std::vector<double> scores)
{
    double median;
    size_t size = scores.size();

    sort(scores.begin(), scores.end());

    if (size % 2 == 0)
    {
        median = (scores[size / 2 - 1] + scores[size / 2]) / 2;
    }
    else
    {
        median = scores[size / 2];
    }

    return median;
}
#else
double median(int n, double x[])
{
    double temp;
    int i, j;

    for(i=0; i<n-1; i++) {
        for(j=i+1; j<n; j++) {
            if(x[j] < x[i]) {
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }
    }

    if(n%2==0) {
        return((x[n/2] + x[n/2 - 1]) / 2.0);
    } else {
        return x[n/2];
    }
}
#endif

double rtclock()
{
    struct timeval Tp;
    gettimeofday(&Tp, NULL);

    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

#endif
