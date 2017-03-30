#use: <script> <path_to_halide>

HALIDE=Halide/

LLVM_VERSION=3.7.0
CORES=1
USER_HOME_DIR=/Users/b/Documents/
LLVM_PREFIX=$USER_HOME_DIR/src-not-saved/llvm/llvm-${LLVM_VERSION}_prefix
export CLANG=$LLVM_PREFIX/bin/clang
export LLVM_CONFIG=$LLVM_PREFIX/bin/llvm-config

git submodule update --init --remote

if [ -z $HALIDE ]; then
	echo "USE: <script> <path_to_halide>"
else
	cd $HALIDE
	git checkout tiramisu
	make -j$CORES
fi
