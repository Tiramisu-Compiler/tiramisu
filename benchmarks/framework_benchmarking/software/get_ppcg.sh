set -x

source ../configure.sh

git clone git://repo.or.cz/ppcg.git
cd ppcg
./get_submodules.sh
./autogen.sh
./configure --with-clang-prefix=${LLVM_PREFIX} --with-int=imath
make -j
