source ../../../../../configure_paths.sh

rm -rf _results

export INCLUDES="-I${MKL_PREFIX}/include/"
export LIBRARIES="${MKL_FLAGS} -lisl -lz -lpthread -ldl "
export LIBRARIES_DIR="-L${MKL_PREFIX}/lib/${MKL_LIB_PATH_SUFFIX}"

rm -rf _results

echo
echo
echo "MKL maxpool"

make libintel64 function="s_score_sample" MKLROOT=${MKL_PREFIX} 

./_results/s_score_sample.exe 
