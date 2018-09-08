#ifdef __cplusplus
extern "C" {
#endif
void pencil_convolution( const int rows
                    , const int cols
                    , const int step
                    , const unsigned char src[]
                    , const int kernelX_length
                    , const float kernelX[]
                    , const int kernelY_length
                    , const float kernelY[]
                    , unsigned char conv[]
                    , unsigned char temp[]
                    );
#ifdef __cplusplus
} // extern "C"
#endif
