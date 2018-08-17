#define DATA_TYPE float
#define DATA_PTYPE p_float32

// GPU BLOCK
#define BLOCK 16
// REGISTER BLOCK
#define R_BLOCK_I 8
#define R_BLOCK_J 16

// Dimensions need to be multiples of blocksize
#define M 4096
#define N 4096
#define K 4096

#define alpha ((float) 3)
#define beta ((float) 2)
