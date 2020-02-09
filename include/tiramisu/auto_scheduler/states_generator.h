#ifndef _TIRAMISU_AUTO_SCHEDULER_STATES_GENERATOR_
#define _TIRAMISU_AUTO_SCHEDULER_STATES_GENERATOR_

#include "computation_graph.h"

namespace tiramisu::auto_scheduler
{

const std::vector<int> TILING_FACTORS_DEFAULT_LIST = {32, 64, 128};
const std::vector<int> UNROLLING_FACTORS_DEFAULT_LIST = {4, 8, 16};

class states_generator
{
private:

protected:
    std::vector<int> tiling_factors_list;
    std::vector<int> unrolling_factors_list;

public:
    states_generator(std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                         std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST)
        : tiling_factors_list(tiling_factors_list), unrolling_factors_list(unrolling_factors_list) {}

    virtual ~states_generator() {}

    virtual std::vector<computation_graph> generate_states(computation_graph const& cg) =0;
};

class exhaustive_generator : public states_generator
{
    exhaustive_generator(std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                         std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST)
        : states_generator(tiling_factors_list, unrolling_factors_list) {}

    virtual std::vector<computation_graph> generate_states(computation_graph const& cg);
};

}

#endif