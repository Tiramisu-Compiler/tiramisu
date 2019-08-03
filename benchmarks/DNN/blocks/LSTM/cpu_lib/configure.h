#define FEATURE_SIZE 16  // 512
#define BATCH_SIZE 64  // 64
#define NUM_LAYERS 4  // 4
#define SEQ_LENGTH 10  // 100

#define NB_TESTS 2

#if 1  // Flip to use double precision
    #define DATA_TYPE float
    #define DATA_TYPE_P p_float32
    #define DATA_TYPE_CUDNN CUDNN_DATA_FLOAT
#else
    #define DATA_TYPE double
    #define DATA_TYPE_P p_float64
    #define DATA_TYPE_CUDNN CUDNN_DATA_DOUBLE
#endif
