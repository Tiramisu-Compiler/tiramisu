../compile_and_run_benchmarks.sh ./ cg
../compile_and_run_benchmarks.sh ../blas/level1/dot/ dot
../compile_and_run_benchmarks.sh ../blas/level1/waxpby/ waxpby

make -B

echo

./test_HPCCG 4096 128 128
