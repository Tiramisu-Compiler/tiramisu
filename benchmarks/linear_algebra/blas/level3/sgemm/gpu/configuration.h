#define DATA_TYPE float
#define DATA_PTYPE p_float32

// GPU BLOCK
#define BLOCK 16
// REGISTER BLOCK
#define R_BLOCK_I 16
#define R_BLOCK_J 6

// Dimensions need to be multiples of blocksize
#define M 3072
#define N 3072
#define K 3072
