source ../../../../../configure_paths.sh

make libintel64 compiler=clang example="s_score_sample" MKLROOT=${MKL_PREFIX} 
