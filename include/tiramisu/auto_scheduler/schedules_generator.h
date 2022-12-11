#ifndef _TIRAMISU_AUTO_SCHEDULER_SCHEDULES_GENERATOR_
#define _TIRAMISU_AUTO_SCHEDULER_SCHEDULES_GENERATOR_

#include "ast.h"
#include "evaluator.h"

namespace tiramisu::auto_scheduler
{

const std::vector<int> TILING_FACTORS_DEFAULT_LIST = {32, 64, 128};
const std::vector<int> UNROLLING_FACTORS_DEFAULT_LIST = {4, 8, 16};
const std::vector<std::tuple<int,int>> SKEWING_FACTORS_DEFAULT_LIST = {{1,1}, {1,2}, {2,1}};
const int DEFAULT_MAX_NB_ITERATORS = 7;

/**
 * Generate a set of AST's from a given AST.
 * Inherit this class to implement a new way to generate schedules.
 */

class schedules_generator
{
private:

protected:
    /**
     * A list of tiling factors to apply when tiling is applied.
     */
    std::vector<int> tiling_factors_list;
    
    /**
     * A list of unrolling factors to apply when unrolling is applied.
     */
    std::vector<int> unrolling_factors_list;

    /**
     * A list of skewing factors to apply when skewing is applied.
     */
    std::vector<std::tuple<int,int>> skewing_factors_list;

    /**
     * Max Number of dimension to explore for unrolling, starting from the innermost loop level,
    */
    int unrolling_search_deapth = 3;

    /**
     * Max Number of dimension to explore for parallelism, starting from the outermost loop level.
     * stops at first found.
    */
    int parallelism_search_deapth = 3;

    /**
     * Max Number of dimension to explore for unrolling, starting from the innermost loop level
    */
    int vectorization_search_deapth = 3;

    /**
     * The number of diffrent skewing proposed, skewing versions that enable inner parallelism in our case.
    */
    int skewing_inner_parallelism_number = 3;


public:
    schedules_generator(std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                        std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST,
                        std::vector<std::tuple<int,int>> skewing_factors_list = SKEWING_FACTORS_DEFAULT_LIST)
        
        : tiling_factors_list(tiling_factors_list), unrolling_factors_list(unrolling_factors_list), skewing_factors_list(skewing_factors_list) {}

    virtual ~schedules_generator() {}

    /**
     * Given an AST, and an optimization to apply, 
     * generate new ASTs by applying the given optimization.
     */
//    virtual std::vector<syntax_tree*> generate_schedules(syntax_tree const& ast, optimization_type optim) =0;

    /**
     * Additional default definition, it would be overriden such to generate schedules from
     * the AST and it's generator_state attribute.
    */
//    virtual std::vector<syntax_tree*> generate_schedules(syntax_tree& ast)
//    {
//        return this->generate_schedules(ast,optimization_type::TILING);
//    }
    virtual std::vector<syntax_tree*> generate_schedules(syntax_tree& ast)=0;
    virtual std::vector<syntax_tree *> generate_matrices(syntax_tree &ast)=0;
    
};

/**
 * Generate all combinations of the following optimizations :
 * Fusion, tiling, interchange, unrolling.
 */
class exhaustive_generator : public schedules_generator
{
private:

protected:
    /**
     * Given a tree level, fuse nodes that have the same name, same lower bounds,
     * and same upper bounds.
     */
    void generate_fusions(std::vector<ast_node*> const& tree_level, std::vector<syntax_tree*>& states, syntax_tree const& ast);
    
    /**
     * Try to apply tiling such as the given node is the first loop to tile, 
     * and then call this method recursively on children of the given node.
     */
    void generate_tilings(ast_node *node, std::vector<syntax_tree*>& states, syntax_tree const& ast);
    
    /**
     * Try to apply interchange by swapping the given node with one of its descendents, 
     * and then call this method recursively on children of the given node.
     */
    void generate_interchanges(ast_node *node, std::vector<syntax_tree*>& states, syntax_tree const& ast);
    
    /**
     * Apply unrolling starting from the given node, and then
     * apply unrolling recursively on children of the given node.
     */
    void generate_unrollings(ast_node *node, std::vector<syntax_tree*>& states, syntax_tree const& ast);

public:
    exhaustive_generator(std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                         std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST)
        
        : schedules_generator(tiling_factors_list, unrolling_factors_list) {}

    virtual std::vector<syntax_tree*> generate_schedules(syntax_tree const& ast, optimization_type optim);
    virtual std::vector<syntax_tree *> generate_matrices(syntax_tree &ast)=0;
};

/**
 * Generate unfuse applied to shared loop levels.
 * Generate tilings and interchanges applied to shared loop levels.
 * Generate unrollings applied to innermost loop levels.
 */
class ml_model_schedules_generator : public schedules_generator
{
private:

protected:
    /**
     * The maximum number of iterators on which to apply the optimizations.
     */
    int max_nb_iterators;

public:
    ml_model_schedules_generator(int max_nb_iterators = DEFAULT_MAX_NB_ITERATORS,
                     std::vector<int> const& tiling_factors_list = TILING_FACTORS_DEFAULT_LIST,
                     std::vector<int> const& unrolling_factors_list = UNROLLING_FACTORS_DEFAULT_LIST)
        
        : schedules_generator(tiling_factors_list, unrolling_factors_list),    
          max_nb_iterators(max_nb_iterators) {}
        
//    virtual std::vector<syntax_tree*> generate_schedules(syntax_tree const& ast, optimization_type optim);

    virtual std::vector<syntax_tree*> generate_schedules(syntax_tree& ast);
    virtual std::vector<syntax_tree *> generate_matrices(syntax_tree &ast);
};

}

#endif
