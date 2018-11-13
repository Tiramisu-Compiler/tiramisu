#define SYNTHETIC_INPUT 0

#if SYNTHETIC_INPUT
    // Window size
    #define w 4
    // Image size
    #define PY_IMG_SIZE 20
#else
    // Window size
    #define w 32
    // Image size
    #define PY_IMG_SIZE 100
#endif

// Number of pyramid levels
#define npyramids 3

// Number of refinement iterations
#define niterations 1
