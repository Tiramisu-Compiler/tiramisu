#ifndef _H_TIRAMISU_BLOCK_
#define _H_TIRAMISU_BLOCK_

#include <tiramisu/core.h>

namespace tiramisu {

class computation;

/**
  * A class that represents a group of computations that share schedules up to
  * a level. This class is used for convenience when a set of scheduling
  * commands should be applied to several computations (i.e GPU kernels).
  */
class block: public computation {
private:

    /**
      * List of computations that this block contains. Actual order of the
      * computations is independent of the vector order and determined by the
      * scheduling commands.
      */
    const std::vector<computation *> children;

public:

    /**
      * \brief Constructor for block.
      *
      * \details This constructor creates a block of computations.
      *
      * In Tiramisu each computation should be scheduled separately. If the
      * user has multiple computations and wants to tile them for example,
      * the user has to apply the tiling command on each one of the computations
      * separately which is not convenient. The block class solves this problem.
      * It allows the user to create a block of computations. Any scheduling
      * command applied to the block is automatically applied to all of its
      * computations.
      *
      * \p children is the list of the computations of the block.
      *
      * The actual order of the computations is not determined by the vector
      * order. It is rather determined by the scheduling commands.
      */
    block(const std::vector<computation *> children);

    /**
      * Overriden scheduling methods from computation. These transformations
      * will be applied to all children of the block.
      */
    // @{
    void gpu_tile(var L0, var L1, int sizeX, int sizeY) override;
    void gpu_tile(var L0, var L1, int sizeX, int sizeY,
                  var L0_outer, var L1_outer,
                  var L0_inner, var L1_inner) override;
    void gpu_tile(var L0, var L1, var L2,
                  int sizeX, int sizeY, int sizeZ) override;
    void gpu_tile(var L0, var L1, var L2, int sizeX, int sizeY, int sizeZ,
                  var L0_outer, var L1_outer, var L2_outer,
                  var L0_inner, var L1_inner, var L2_inner) override;
    void interchange(var L0, var L1) override;
    void interchange(int L0, int L1) override;
    void parallelize(var L) override;
    void shift(var L0, int n) override;
    void skew(var i, var j, int f, var ni, var nj) override;
    void skew(var i, var j, var k, int factor, var ni, var nj, var nk) override;
    void skew(var i, var j, var k, var l, int factor,
              var ni, var nj, var nk, var nl) override;
    void skew(var i, var j, int factor) override;
    void skew(var i, var j, var k, int factor) override;
    void skew(var i, var j, var k, var l, int factor) override;
    void skew(int i, int j, int factor) override;
    void skew(int i, int j, int k, int factor) override;
    void skew(int i, int j, int k, int l, int factor) override;
    void split(var L0, int sizeX) override;
    void split(var L0, int sizeX, var L0_outer, var L0_inner) override;
    void split(int L0, int sizeX) override;
    void storage_fold(var dim, int f) override;
    void tile(var L0, var L1, int sizeX, int sizeY) override;
    void tile(var L0, var L1, int sizeX, int sizeY, var L0_outer, var L1_outer,
              var L0_inner, var L1_inner) override;
    void tile(var L0, var L1, var L2, int sizeX, int sizeY, int sizeZ) override;
    void tile(var L0, var L1, var L2, int sizeX, int sizeY, int sizeZ,
              var L0_outer, var L1_outer, var L2_outer, var L0_inner,
              var L1_inner, var L2_inner) override;
    void tile(int L0, int L1, int sizeX, int sizeY) override;
    void tile(int L0, int L1, int L2, int sizeX, int sizeY, int sizeZ) override;
    void unroll(var L, int fac) override;
    void unroll(var L, int fac, var L_outer, var L_inner) override;
    void unroll(int L, int fac) override;
    void vectorize(var L, int v) override;
    void vectorize(var L, int v, var L_outer, var L_inner) override;
    // @}
};  // class block

}  // namespace tiramisu

#endif  // _H_TIRAMISU_BLOCK_
