#set -x

source ../../../../configure_paths.sh

export INCLUDES="-I${MKL_PREFIX}/include/"
export LIBRARIES="${MKL_FLAGS} -lisl -lz -lpthread -ldl "
export LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX}"

rm -rf _results

echo
echo
echo "MKL ReLu"

make libintel64 function="relu_layer_generator_mkl" MKLROOT=${MKL_PREFIX} 

./_results/relu_layer_generator_mkl.exe 
