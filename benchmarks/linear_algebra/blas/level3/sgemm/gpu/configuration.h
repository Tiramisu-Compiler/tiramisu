#define DATA_TYPE float
#define DATA_PTYPE p_float32

// GPU BLOCK
#define BLOCK 16
// REGISTER BLOCK
#define R_BLOCK_I 6
#define R_BLOCK_J 16
// R_BLOCK_J needs to be equal to BLOCK because of
// the way we copy B from global to shared

// Dimensions need to be multiples of blocksize
#define M 3072
#define N 3072
#define K 3072
