#set -x

if [ $# -eq 0 ]; then
      echo "Usage: TIRAMISU_SMALL=1 script.sh <KERNEL_FOLDER>"
      echo "Example: script.sh axpy"
      exit
fi

s=$1
b=${s%.*}

CXXFLAGS="-std=c++11 -O3"

echo "Compiling conv_layer_generator_tiramisu.cpp"

g++ $CXXFLAGS ${INCLUDES} $s -ltiramisu ${LIBRARIES_DIR} ${LIBRARIES} -o $b
