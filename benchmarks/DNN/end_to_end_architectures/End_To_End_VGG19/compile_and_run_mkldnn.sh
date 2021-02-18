#set -x

source ../../../configure_paths.sh
MKLDNNROOT=/usr/local/

export INCLUDES="-I${MKL_PREFIX}/include/ -I${MKLDNNROOT}/include"
export LIBRARIES="${MKL_FLAGS} -lisl -lz -lpthread -ldl "
export LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX} -L${MKLDNNROOT}/lib"

source ${MKL_PREFIX}/bin/mklvars.sh ${MKL_LIB_PATH_SUFFIX}
g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib end_to_end_vgg19_wrapper.cpp -o end_to_end_vgg19_mkldnn -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -fopenmp -lm -ldl -lmkldnn
./end_to_end_vgg19_mkldnn
