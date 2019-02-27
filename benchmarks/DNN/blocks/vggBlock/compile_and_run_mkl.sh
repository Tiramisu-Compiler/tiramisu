#set -x
source ../../../configure_paths.sh

export INCLUDES="-I${MKL_PREFIX}/include/"
export LIBRARIES="${MKL_FLAGS} -lisl -lz -lpthread -ldl "
export LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX}"

rm -rf _results
echo "MKL VGG"

make libintel64 function="vgg_block_generator_mkl" MKLROOT=${MKL_PREFIX} 
./_results/vgg_block_generator_mkl.exe 
