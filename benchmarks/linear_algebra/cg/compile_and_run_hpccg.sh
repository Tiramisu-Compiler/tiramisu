../compile_and_run_benchmarks.sh ./ cg
make -B

echo

./test_HPCCG 4096 128 128
