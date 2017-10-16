#! /bin/bash

function show_help {
    cat << EndOfHelp

Adds a test to the test suite and generates the corresponding files.

Usage:
        ./new_test.sh [-o] <test name>
            Creates a new test with the name <test name>. Automatically
            detects and generates a new test number. By default modifies
            the CMake configuration file. If you wish to still use the
            old Makefile, please use the -o option.

       ./new_test.sh --help
            Shows this help text.
EndOfHelp
}

function malformed_command {
    echo -e "\e[33mMalformed command.\e[39m" >&2
    show_help
    exit 1
}

new_style=true

if [ "$#" -eq 1 ]; then
    if [[ $1 == -* ]]; then
        if [[ "$1" == "--help" ]]; then
            show_help
            exit 0
        else
            malformed_command
        fi
    fi
    test_name=$1
elif [ "$#" -eq 2 ]; then
    if [[ "$1" != "-o" ]] || [[ "$2" == -* ]]; then
        malformed_command
    fi
    new_style=false
    test_name=$2
else
    malformed_command
fi

if [[ "$new_style" == true ]]; then
    test_num=(`sort -nr test_list | head -1`)

    test_num=`expr $test_num + 1`

    echo "Creating test \"$test_num\" with the name \"$test_name\" to be added to CMake configuration file."
else
    last_gen=(`grep -nE '^\s*build/test_[0-9]+_fct_generator\s*$' Makefile | grep -oE '[0-9]+'`)
    last_bin=(`grep -nE '^\s*build/test_[0-9]+\s*$' Makefile | grep -oE '[0-9]+'`)
    last_run=(`grep -nE '^\s*run_test_[0-9]+\s*$' Makefile | grep -oE '[0-9]+'`)

    last_test_nums=(${last_gen[1]} ${last_bin[1]} ${last_run[1]})

    test_num=${last_test_nums[0]}
    for n in "${ar[@]}" ; do
        ((n > test_num)) && test_num=$n
    done
    test_num=`expr $test_num + 1`


    echo "Creating test \"$test_num\" with the name \"$test_name\" to be added to old style Makefile."
fi

if [ ! -f tests/test_${test_num}.cpp ]; then
    m4 -DTEMPLATE_TESTNAME=$test_name -DTEMPLATE_TESTNUM=$test_num templates/test_cpp.m4 > tests/test_${test_num}.cpp
fi

if [ ! -f tests/wrapper_test_${test_num}.cpp ]; then
    m4 -DTEMPLATE_TESTNAME=$test_name -DTEMPLATE_TESTNUM=$test_num templates/wrapper_test_cpp.m4 > tests/wrapper_test_${test_num}.cpp
fi

if [ ! -f tests/wrapper_test_${test_num}.h ]; then
    m4 -DTEMPLATE_TESTNAME=$test_name -DTEMPLATE_TESTNUM=$test_num templates/wrapper_test_h.m4 > tests/wrapper_test_${test_num}.h
fi

if [[ "$new_style" == true ]]; then
    echo "$test_num" >> test_list
else
    awk 'NR=='${last_run[0]}' {$0=$0" \\\n    run_test_'$test_num'"} {print}' Makefile | \
    awk 'NR=='${last_bin[0]}' {$0=$0" \\\n    build/test_'$test_num'"} {print}' | \
    awk 'NR=='${last_gen[0]}' {$0=$0" \\\n    build/test_'$test_num'_fct_generator"} {print}' > Makefile_new

    mv Makefile Makefile.bak
    mv Makefile_new Makefile
fi
