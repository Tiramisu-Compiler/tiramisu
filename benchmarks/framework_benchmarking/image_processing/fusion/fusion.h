#ifdef __cplusplus
extern "C" {
#endif
void pencil_fusion( const int rows
                    , const int cols
                    , const int step
                    , const unsigned char src[]
                    , const int kernelX_length
                    , const float kernelX[]
                    , const int kernelY_length
                    , const float kernelY[]
                    , unsigned char f[]
                    , unsigned char g[]
                    , unsigned char h[]
                    , unsigned char k[]
                    );
#ifdef __cplusplus
} // extern "C"
#endif
