make -B

echo

../compile_and_run_benchmarks.sh cg/ cg
./test_HPCCG 4096 128 128
