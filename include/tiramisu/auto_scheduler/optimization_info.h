#ifndef _TIRAMISU_AUTO_SCHEDULER_OPTIMIZATION_INFO_
#define _TIRAMISU_AUTO_SCHEDULER_OPTIMIZATION_INFO_

#include <tiramisu/core.h>

namespace tiramisu::auto_scheduler
{

class ast_node;

enum optimization_type
{
    UNFUSE,
    TILING,
    INTERCHANGE,
    UNROLLING,
    NB_OPTIMIZATIONS,
    FUSION
};

/**
 * Check apply_optimizations() to see how this structure is used.
 */
struct optimization_info
{
    /**
     * The type of this optimization.
     */
    optimization_type type;
    
    /**
     * The list of computations that this optimization will be applied to.
     */
    std::vector<tiramisu::computation*> comps;
    
    /**
     * This attribute is used when transforming the AST.
     * It indicates the node at which to start the transformation.
     */
    ast_node *node;
    
    /**
     * The number of loop levels that this optimization affects.
     * For example, a 2 level tiling affects 2 loop levels, an interchange
     * affects 2 loop levels, an unrolling affects 1 loop level.
     */
    int nb_l;
    
    /**
     * The loop levels this optimization affects.
     * nb_l indicates the number of loop levels to consider.
     * In the case of unrolling, if l0 == -1, unrolling is applied
     * on all innermost levels.
     */
    int l0, l1, l2;
    
    /**
     * Contains the factors of each loop level.
     * For example, if the optimization is a 2 level tiling,
     * l0_fact and l1_fact will contain the factors for each loop level.
     */
    int l0_fact, l1_fact, l2_fact;
};

/**
 * Tag the outermost level of each computation to be parallelized.
 */
void parallelize_outermost_levels(std::vector<tiramisu::computation*> const& comps_list);

/**
 * Tag the innermost level of each computation to be unrolled by a factor of unroll_fact.
 */
void unroll_innermost_levels(std::vector<tiramisu::computation*> const& comps_list, int unroll_fact);

/**
 * Apply the optimizations specified by the syntax tree
 * using the Tiramisu API.
 */
void apply_optimizations(syntax_tree const& ast);
    
/**
 * Apply the given optimization using the Tiramisu API.
 */
void apply_optimizations(optimization_info const& optim_info);
    
/**
 * Schedule the computations so as to be in the order specified by the AST.
 */
void apply_fusions(syntax_tree const& ast);
    
/**
 *
 */
tiramisu::computation* apply_fusions(ast_node *node, tiramisu::computation *last_comp, int dimension);

}

#endif
