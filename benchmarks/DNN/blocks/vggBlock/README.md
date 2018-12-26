to compile Tiramisu code and generate executables :
add this line of code to the file tiramisu/CMakeLists.txt
  add_subdirectory(benchmarks/DNN/blocks/vggBlock)
Then rebuild the main Tiramisu library 
in the new directory tiramisu/build/benchmarks/DNN/blocks/vggBlock execute
  make 


