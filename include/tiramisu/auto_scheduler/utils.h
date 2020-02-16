#ifndef _TIRAMISU_AUTO_SCHEDULER_UTILS_
#define _TIRAMISU_AUTO_SCHEDULER_UTILS_

#include <tiramisu/core.h>
#include <tiramisu/expr.h>

namespace tiramisu::auto_scheduler
{

class ast_node;

enum optimization_type
{
    FUSION,
    TILING,
    INTERCHANGE,
    UNROLLING,
    NB_OPTIMIZATIONS
};

struct optimization_info
{
    optimization_type type;
    std::vector<tiramisu::computation*> comps;
    
    ast_node *node;
    
    int nb_l;
    int l0, l1, l2;
    int l0_fact, l1_fact, l2_fact;
};

inline bool can_split_iterator(int it_extent, int split_fact)
{
    return it_extent > split_fact && it_extent % split_fact == 0;
}

}

#endif
