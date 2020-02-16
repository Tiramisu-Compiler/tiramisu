#ifndef _TIRAMISU_AUTO_SCHEDULER_STATES_GENERATOR_
#define _TIRAMISU_AUTO_SCHEDULER_STATES_GENERATOR_

#include "ast.h"

namespace tiramisu::auto_scheduler
{

const std::vector<int> TILING_FACTORS_DEFAULT_LIST = {32, 64, 128};
const std::vector<int> UNROLLING_FACTORS_DEFAULT_LIST = {4, 8, 16};

/**
 * Generate a set of AST from a given AST.
 *
 * Inherit this class to implement a state generation behavior.
 */
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

    virtual std::vector<syntax_tree*> generate_states(syntax_tree const& ast, optimization_type optim) =0;
};

/**
 * Generate all combinations of the following optimizations :
 * Fusion, tiling, interchange, unrolling.
 */
class exhaustive_generator : public states_generator
{
private:

protected:
    void generate_fusions(std::vector<ast_node*> const& tree_level, std::vector<syntax_tree*>& states, syntax_tree const& ast);
    void generate_tilings(ast_node *node, std::vector<syntax_tree*>& states, syntax_tree const& ast);
    void generate_interchanges(ast_node *node, std::vector<syntax_tree*>& states, syntax_tree const& ast);
    void generate_unrollings(ast_node *node, std::vector<syntax_tree*>& states, syntax_tree const& ast);

public:
    exhaustive_generator(std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                         std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST)
        
        : states_generator(tiling_factors_list, unrolling_factors_list) {}

    virtual std::vector<syntax_tree*> generate_states(syntax_tree const& ast, optimization_type optim);
};

}

#endif
