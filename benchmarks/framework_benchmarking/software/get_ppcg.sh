set -x

export LLVM_PREFIX="/Users/b/Documents/src/MIT/tiramisu/3rdParty/llvm/prefix/"

git clone git://repo.or.cz/ppcg.git
cd ppcg
./get_submodules.sh
./autogen.sh
./configure --with-clang-prefix=${LLVM_PREFIX}
make -j
