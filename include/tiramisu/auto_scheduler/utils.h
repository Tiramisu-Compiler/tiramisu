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

class dnn_iterator
{
public:
    int low_bound;
    int up_bound;
    
    dnn_iterator(int low_bound, int up_bound)
        : low_bound(low_bound), up_bound(up_bound) {}
};

class dnn_schedule
{
public:
    int nb_iterators;

    std::vector<bool> interchanged;
    std::vector<bool> tiled;
    std::vector<int> tiling_fact;
    int unrolling_fact = 0;
    
    dnn_schedule(int nb_iterators) 
        : nb_iterators(nb_iterators), interchanged(nb_iterators, false),
          tiled(nb_iterators, false), tiling_fact(nb_iterators, 0),
          unrolling_fact(0) {}
};

class dnn_access_matrix
{
protected:
    void fill_matrix_row(int i, tiramisu::expr const& e, bool minus = false);

public:
    int nb_iterators;
    int nb_dims;
    std::vector<std::vector<int>> matrix;
    
    std::string buffer_name;
    int buffer_id;
    
    tiramisu::computation *comp;
    
    dnn_access_matrix(int nb_iterators, int nb_dims);
    dnn_access_matrix(int nb_iterators, tiramisu::expr const& e, tiramisu::computation *comp);
    
    void set_buffer_id(tiramisu::function *fct);
};

class dnn_accesses
{
public:
    tiramisu::computation* comp;
    int nb_iterators;
    std::vector<dnn_access_matrix> accesses_list;
        
    dnn_accesses(tiramisu::computation *comp, int nb_iterators)
        : comp(comp), nb_iterators(nb_iterators)
    {
        create_accesses(comp->get_expr());
    }
        
    void create_accesses(tiramisu::expr const& e);
};

/**
 * Return true if an iterator having extent = it_extent can
 * be split perfectly by a factor = split_fact.
 */
inline bool can_split_iterator(int it_extent, int split_fact)
{
    return it_extent > split_fact && it_extent % split_fact == 0;
}

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
 * Schedule the computations so as to be in the order specified
 * by the AST.
 */
void apply_fusions(syntax_tree const& ast);
    
/**
 *
 */
tiramisu::computation* apply_fusions(ast_node *node, tiramisu::computation *last_comp, int dimension);

/**
 *
 */
void parallelize_outermost_level(syntax_tree const& ast);

}

#endif
