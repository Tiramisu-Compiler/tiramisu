#ifndef _TIRAMISU_AUTO_SCHEDULER_STATES_GENERATOR_
#define _TIRAMISU_AUTO_SCHEDULER_STATES_GENERATOR_

#include "ast.h"
#include "evaluator.h"

namespace tiramisu::auto_scheduler
{

const std::vector<int> TILING_FACTORS_DEFAULT_LIST = {32, 64, 128};
const std::vector<int> UNROLLING_FACTORS_DEFAULT_LIST = {4, 8, 16};
const int DEFAULT_MAX_NB_ITERATORS = 4;

/**
 * Generate a set of AST's from a given AST.
 * Inherit this class to implement a state generation behavior.
 */
class schedules_generator
{
private:

protected:
    std::vector<int> tiling_factors_list;
    std::vector<int> unrolling_factors_list;

public:
    schedules_generator(std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                     std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST)
        
        : tiling_factors_list(tiling_factors_list), unrolling_factors_list(unrolling_factors_list) {}

    virtual ~schedules_generator() {}

    virtual std::vector<syntax_tree*> generate_states(syntax_tree const& ast, optimization_type optim) =0;
};

/**
 * Generate all combinations of the following optimizations :
 * Fusion, tiling, interchange, unrolling.
 */
class exhaustive_generator : public schedules_generator
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
        
        : schedules_generator(tiling_factors_list, unrolling_factors_list) {}

    virtual std::vector<syntax_tree*> generate_states(syntax_tree const& ast, optimization_type optim);
};

/**
 * Generate tilings and interchanges applied to shared loop levels.
 * Generate unrollings applied to innermost loop levels.
 */
class tree_structured_search_space : public schedules_generator
{
private:

protected:
    int max_nb_iterators;

public:
    tree_structured_search_space(int max_nb_iterators = DEFAULT_MAX_NB_ITERATORS,
                     std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                     std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST)
        
        : schedules_generator(tiling_factors_list, unrolling_factors_list),    
          max_nb_iterators(max_nb_iterators) {}
        
    virtual std::vector<syntax_tree*> generate_states(syntax_tree const& ast, optimization_type optim);
};

}

#endif
