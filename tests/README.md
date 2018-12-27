## Add a New Test

To add a new test, add two files in `tests/`.  Assuming `XX` is the number
of the test that you want to add, `test_XX.cpp` should contain
the actual Tiramisu generator code while `wrapper_test_XX.cpp` should contain
wrapper code.  Wrapper code initializes the input data, calls the generated function,
and compares its output with a reference output.  You should then add the
test number `XX` in the file `tests/test_list.txt`.

## Test Descriptions
- access parsing: test_16
- clamped access: test_56
- .after(): test_43, 44, 45, 46, 47
- .allocate_at: test_27, 90, 92, 93, 130
- .allocate_and_map_buffer_automatically: test_49
- .allocate_and_map_buffers_automatically: test_50
- .between: test_58, 59
- .before(): test_27
- .store_in(): 105, 106, 107, 108, 109, 129
-  codegen(): 104
- .compute_at(): test_14, 32, 33, 34, 35, 36, 37, 38, 82, 83
- .compute_bounds(): test_86, 22, 23, 24, 25, 27, 130
- Dynamic buffer size (buffer size unknown at compile time): test_87, 88, 92
- floor operator: test_08
- .fuse_after(): test_13, 29, 30, 31
- .get_last_update(): test_68
- High level API: 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
    127, 128
- ISL-free API: 110, 111, 112, 113
- tiramisu::init(): 103, 114, 115, 116
- let statement: test_04
- lerp(): test_55
- low level separation: test_73
- RDom predicate: test_54
- .parallelize(): test_75
- saxpy: test_71
- skew(): 131, 132, 133, 134, 135, 136, 137, 138, 139,
	  140
- .store_at(): test_29, 30, 31, 38, 39, 82, 83
- .shift(): test_15
-  shift operator: test_06
- .tag_parallel_level(): test_48
- .tag_gpu_level(): test_17, 18
- .tag_unroll_level(): test_11
- .then(): test_130
- .tile(): test_01, 02, 03, 74, 80, 81
- .vectorize(): test_10, 28, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 74
- .unroll(): test_12, 74
- .update() (new way of expressing updates): test_91
- 64 bit buffers: test_97
