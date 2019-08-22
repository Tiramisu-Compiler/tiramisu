#set -x

s=$1
b=${s%.*}
g++ $1 -g -I $HalideSrc/include -L $HalideSrc/bin -lHalide -lpthread -ldl -std=c++11 -o $b
