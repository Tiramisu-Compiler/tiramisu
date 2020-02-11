#ifndef _TIRAMISU_AUTO_SCHEDULER_STATES_GENERATOR_
#define _TIRAMISU_AUTO_SCHEDULER_STATES_GENERATOR_

#include "computation_graph.h"

namespace tiramisu::auto_scheduler
{

const std::vector<int> TILING_FACTORS_DEFAULT_LIST = {32, 64, 128};
const std::vector<int> UNROLLING_FACTORS_DEFAULT_LIST = {4, 8, 16};
const std::vector<optimization_type> DEFAULT_OPTIMIZATIONS_ORDER = {FUSION, TILING, INTERCHANGE, UNROLLING};

/**
 * Generate a set of computation graphs from a given
 * computation graph.
 *
 * Inherit this class to implement a state generation behavior.
 */
class states_generator
{
private:

protected:
    bool apply_fusion,
         apply_tiling,
         apply_interchange,
         apply_unrolling;

    std::vector<optimization_type> optimizations_order;

    std::vector<int> tiling_factors_list;
    std::vector<int> unrolling_factors_list;

public:
    states_generator(bool apply_fusion = true, bool apply_tiling = true,
                     bool apply_interchange = true, bool apply_unrolling = true,
                     std::vector<optimization_type> const& optimizations_order = DEFAULT_OPTIMIZATIONS_ORDER,
                     std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                     std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST)
        
        : apply_fusion(apply_fusion), apply_tiling(apply_tiling),
          apply_interchange(apply_interchange), apply_unrolling(apply_unrolling),
          optimizations_order(optimizations_order), tiling_factors_list(tiling_factors_list), 
          unrolling_factors_list(unrolling_factors_list) {}

    virtual ~states_generator() {}

    virtual std::vector<computation_graph*> generate_states(computation_graph const& cg) =0;
};

/**
 * Generate all combinations of the following optimizations :
 * Fusion, tiling, interchange, unrolling.
 */
class exhaustive_generator : public states_generator
{
private:
    void generate_fusions(std::vector<cg_node*> const& tree_level, std::vector<computation_graph*>& states, computation_graph const& cg);
    void generate_tilings(cg_node *node, std::vector<computation_graph*>& states, computation_graph const& cg);
    void generate_interchanges(cg_node *node, std::vector<computation_graph*>& states, computation_graph const& cg);
    void generate_unrollings(cg_node *node, std::vector<computation_graph*>& states, computation_graph const& cg);

protected:

public:
    exhaustive_generator(bool apply_fusion = true, bool apply_tiling = true,
                         bool apply_interchange = true, bool apply_unrolling = true,
                         std::vector<optimization_type> const& optimizations_order = DEFAULT_OPTIMIZATIONS_ORDER,
                         std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                         std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST)
        
        : states_generator(apply_fusion, apply_tiling, apply_interchange, apply_unrolling,
                           optimizations_order, tiling_factors_list, unrolling_factors_list) {}

    virtual std::vector<computation_graph*> generate_states(computation_graph const& cg);
};

}

#endif
