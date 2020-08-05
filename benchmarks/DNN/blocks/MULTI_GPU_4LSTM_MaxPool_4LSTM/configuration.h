#define FEATURE_SIZE int32_t(512)  // 512
#define BATCH_SIZE int32_t(64)  // 64
#define NUM_LAYERS int32_t(4)  // 4 DO NOT CHANGE
#define SEQ_LENGTH int32_t(100)  // 100

#define GEMM_BATCH 25
#define CHECK_ERROR_DIFFERENCE 0

#if 1  // Flip to use double precision
    #define DATA_TYPE float
    #define DATA_TYPE_P p_float32
    #define DATA_TYPE_CUDNN CUDNN_DATA_FLOAT
#else
    #define DATA_TYPE double
    #define DATA_TYPE_P p_float64
    #define DATA_TYPE_CUDNN CUDNN_DATA_DOUBLE
#endif
