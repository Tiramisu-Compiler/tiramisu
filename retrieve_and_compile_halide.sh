# Path to LLVM prefix. Please change this path to the directory where you
# installed LLVM.
LLVM_PREFIX=/Users/b/Documents/src-not-saved/llvm/llvm-3.7.0_prefix


# No need to change anything among the following.
export CLANG=$LLVM_PREFIX/bin/clang
export LLVM_CONFIG=$LLVM_PREFIX/bin/llvm-config

git submodule update --init --remote

cd Halide/
git checkout tiramisu
git pull
make -j8
