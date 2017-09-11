#! /bin/bash

last_gen=(`grep -nE '^\s*build/test_[0-9]+_fct_generator\s*$' Makefile | grep -oE '[0-9]+'`)
last_bin=(`grep -nE '^\s*build/test_[0-9]+\s*$' Makefile | grep -oE '[0-9]+'`)
last_run=(`grep -nE '^\s*run_test_[0-9]+\s*$' Makefile | grep -oE '[0-9]+'`)

last_test_nums=(${last_gen[1]} ${last_bin[1]} ${last_run[1]})

test_num=${last_test_nums[0]}
for n in "${ar[@]}" ; do
    ((n > test_num)) && test_num=$n
done
test_num=`expr $test_num + 1`

if [ "$#" -ne 1 ]; then
    echo "Please specify test name" >&2
    exit 1
fi

test_name=$1
echo "Creating test \"$test_num\" with the name \"$test_name\""

if [ ! -f tests/test_${test_num}.cpp ]; then
    m4 -DTEMPLATE_TESTNAME=$test_name -DTEMPLATE_TESTNUM=$test_num templates/test_cpp.m4 > tests/test_${test_num}.cpp
fi

if [ ! -f tests/wrapper_test_${test_num}.cpp ]; then
    m4 -DTEMPLATE_TESTNAME=$test_name -DTEMPLATE_TESTNUM=$test_num templates/wrapper_test_cpp.m4 > tests/wrapper_test_${test_num}.cpp
fi

if [ ! -f tests/wrapper_test_${test_num}.h ]; then
    m4 -DTEMPLATE_TESTNAME=$test_name -DTEMPLATE_TESTNUM=$test_num templates/wrapper_test_h.m4 > tests/wrapper_test_${test_num}.h
fi

awk 'NR=='${last_run[0]}' {$0=$0" \\\n    run_test_'$test_num'"} {print}' Makefile | \
awk 'NR=='${last_bin[0]}' {$0=$0" \\\n    build/test_'$test_num'"} {print}' | \
awk 'NR=='${last_gen[0]}' {$0=$0" \\\n    build/test_'$test_num'_fct_generator"} {print}' > Makefile_new

mv Makefile Makefile.bak
mv Makefile_new Makefile
