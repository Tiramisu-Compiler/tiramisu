#include <tiramisu/block.h>

namespace tiramisu {

block::block(const std::vector<computation *> children) : children(children) {
    // Block is a special child of computation. Don't call parent constructor.
}

// Overloads of scheduling commands.
void block::gpu_tile(var L0, var L1, int sizeX, int sizeY) {
    for (auto &child : this->children) {
        child->gpu_tile(L0, L1, sizeX, sizeY);
    }
}

void block::gpu_tile(var L0, var L1, int sizeX, int sizeY,
                     var L0_outer, var L1_outer, var L0_inner, var L1_inner) {
    for (auto &child : this->children) {
        child->gpu_tile(L0, L1, sizeX, sizeY, L0_outer, L1_outer, L0_inner, L1_inner);
    }
}

void block::gpu_tile(var L0, var L1, var L2, int sizeX, int sizeY, int sizeZ) {
    for (auto &child : this->children) {
        child->gpu_tile(L0, L1, L2, sizeX, sizeY, sizeZ);
    }
}

void block::gpu_tile(var L0, var L1, var L2, int sizeX, int sizeY, int sizeZ,
                     var L0_outer, var L1_outer, var L2_outer,
                     var L0_inner, var L1_inner, var L2_inner) {
    for (auto &child : this->children) {
        child->gpu_tile(L0, L1, L2, sizeX, sizeY, sizeZ, L0_outer, L1_outer,
                    L2_outer, L0_inner, L1_inner, L2_inner);
    }
}

void block::interchange(var L0, var L1) {
    for (auto &child : this->children) {
        child->interchange(L0, L1);
    }
}

void block::interchange(int L0, int L1) {
    for (auto &child : this->children) {
        child->interchange(L0, L1);
    }
}
void block::matrix_transform(std::vector<std::vector<int>> matrix) {
    for (auto &child : this->children) {
        child->matrix_transform(matrix);
    }
}

void block::parallelize(var L) {
    for (auto &child : this->children) {
        child->parallelize(L);
    }
}

void block::shift(var L0, int n) {
    for (auto &child : this->children) {
        child->shift(L0, n);
    }
}

/*
void block::skew(var i, var j, int f, var ni, var nj) {
    for (auto &child : this->children) {
        child->skew(i, j, f, ni, nj);
    }
}

void block::skew(var i, var j, var k, int factor, var ni, var nj, var nk) {
    for (auto &child : this->children) {
        child->skew(i, j, k, factor, ni, nj, nk);
    }
}

void block::skew(var i, var j, var k, var l, int factor, var ni, var nj, var nk, var nl) {
    for (auto &child : this->children) {
        child->skew(i, j, k, l, factor, ni, nj, nk, nl);
    }
}

void block::skew(var i, var j, int factor) {
    for (auto &child : this->children) {
        child->skew(i, j, factor);
    }
}

void block::skew(var i, var j, var k, int factor) {
    for (auto &child : this->children) {
        child->skew(i, j, k, factor);
    }
}

void block::skew(var i, var j, var k, var l, int factor) {
    for (auto &child : this->children) {
        child->skew(i, j, k, l, factor);
    }
}

void block::skew(int i, int j, int factor) {
    for (auto &child : this->children) {
        child->skew(i, j, factor);
    }
}

void block::skew(int i, int j, int k, int factor) {
    for (auto &child : this->children) {
        child->skew(i, j, k, factor);
    }
}

void block::skew(int i, int j, int k, int l, int factor) {
    for (auto &child : this->children) {
        child->skew(i, j, k, l, factor);
    }
}
*/
void block::skew(var i, var j, int a, int b, var ni, var nj) {
    for (auto &child : this->children) {
        child->skew(i, j, a, b, ni, nj);
    }
}


void block::skew(int i, int j, int a, int b) {
    for (auto &child : this->children) {
        child->skew(i, j, a, b);
    }
}
void block::skew(var i, var j, int a, int b, int c, int d, var ni, var nj) {
    for (auto &child : this->children) {
        child->skew(i, j, a, b, c, d, ni, nj);
    }
}


void block::skew(int i, int j, int a, int b, int c, int d) {
    for (auto &child : this->children) {
        child->skew(i, j, a, b, c, d);
    }
}
void block::split(var L0, int sizeX) {
    for (auto &child : this->children) {
        child->split(L0, sizeX);
    }
}

void block::split(var L0, int sizeX, var L0_outer, var L0_inner) {
    for (auto &child : this->children) {
        child->split(L0, sizeX, L0_outer, L0_inner);
    }
}

void block::split(int L0, int sizeX) {
    for (auto &child : this->children) {
        child->split(L0, sizeX);
    }
}

void block::storage_fold(var dim, int f) {
    for (auto &child : this->children) {
        child->storage_fold(dim, f);
    }
}

void block::tile(var L0, var L1, int sizeX, int sizeY) {
    for (auto &child : this->children) {
        child->tile(L0, L1, sizeX, sizeY);
    }
}

void block::tile(var L0, var L1, int sizeX, int sizeY,
                 var L0_outer, var L1_outer, var L0_inner, var L1_inner) {
    for (auto &child : this->children) {
        child->tile(L0, L1, sizeX, sizeY, L0_outer, L1_outer, L0_inner, L1_inner);
    }
}

void block::tile(var L0, var L1, var L2, int sizeX, int sizeY, int sizeZ) {
    for (auto &child : this->children) {
        child->tile(L0, L1, L2, sizeX, sizeY, sizeZ);
    }
}

void block::tile(var L0, var L1, var L2, int sizeX, int sizeY, int sizeZ,
                 var L0_outer, var L1_outer, var L2_outer, var L0_inner,
                 var L1_inner, var L2_inner) {
    for (auto &child : this->children) {
        child->tile(L0, L1, L2, sizeX, sizeY, sizeZ, L0_outer, L1_outer,
                    L2_outer, L0_inner, L1_inner, L2_inner);
    }
}

void block::tile(int L0, int L1, int sizeX, int sizeY) {
    for (auto &child : this->children) {
        child->tile(L0, L1, sizeX, sizeY);
    }
}

void block::tile(int L0, int L1, int L2, int sizeX, int sizeY, int sizeZ) {
    for (auto &child : this->children) {
        child->tile(L0, L1, L2, sizeX, sizeY, sizeZ);
    }
}

void block::unroll(var L, int fac) {
    for (auto &child : this->children) {
        child->unroll(L, fac);
    }
}

void block::unroll(var L, int fac, var L_outer, var L_inner) {
    for (auto &child : this->children) {
        child->unroll(L, fac, L_outer, L_inner);
    }
}

void block::unroll(int L, int fac) {
    for (auto &child : this->children) {
        child->unroll(L, fac);
    }
}

void block::vectorize(var L, int v) {
    for (auto &child : this->children) {
        child->vectorize(L, v);
    }
}

void block::vectorize(var L, int v, var L_outer, var L_inner) {
    for (auto &child : this->children) {
        child->vectorize(L, v, L_outer, L_inner);
    }
}


}  // namespace tiramisu
