#set -x

source ../../../../configure_paths.sh
MKLDNNROOT=/usr/local/

export INCLUDES="-I${MKL_PREFIX}/include/ -I${MKLDNNROOT}/include"
export LIBRARIES="${MKL_FLAGS} -lisl -lz -lpthread -ldl "
export LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} -L${MKLDNNROOT}/lib"

source ${MKL_PREFIX}/bin/mklvars.sh ${MKL_LIB_PATH_SUFFIX}
g++ -DMKL_ILP64 -m64 ${INCLUDES} add_relu_mkl.cpp -o add_relu_mkl ${LIBRARIES_DIR} -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -fopenmp -lm -ldl -lmkldnn
./add_relu_mkl
