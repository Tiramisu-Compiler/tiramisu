#ifndef _H_TIRAMISU_AUTO_SCHEDULER_COMPUTATION_GRAPH_
#define _H_TIRAMISU_AUTO_SCHEDULER_COMPUTATION_GRAPH_

#include <tiramisu/core.h>
#include "utils.h"

namespace tiramisu::auto_scheduler
{

/**
  * An iterator is defined by its name, its lower
  * bound and its upper bound.
  */
class iterator
{
public:
    std::string name;
    int low_bound;
    int up_bound;
        
    iterator(std::string const& name, int low_bound, int up_bound)
        : name(name), low_bound(low_bound), up_bound(up_bound) {}
};

/**
  * Computation graph node.
  * A node in the computation graph represents a loop level.
  */
class cg_node
{
public:
    /**
     * Depth of this loop level.
     */
    int depth;
    
    /**
      * Contains the iterators from the root to this level.
      */
    std::vector<iterator> iterators;

    /**
     * List of the computations computed at this level.
     */
    std::vector<tiramisu::computation*> comp;

    std::vector<cg_node*> children;
    
    bool fused = false;
    int fused_with;
    
    bool tiled = false;
    int tiling_dim;
    
    int tiling_size1, 
        tiling_size2, 
        tiling_size3;
    
    bool interchanged = false;
    int interchanged_with;
    
    int unrolling_factor = 0;
        
    ~cg_node()
    {
        for (cg_node* child : children)
            delete child;
    }

    /**
     * Copy the tree rooted at this node into new_node and return
     * a pointer to the copied version of node_to_find.
     */
    cg_node* copy_and_return_node(cg_node *new_node, cg_node *node_to_find) const;
    
    int get_branch_depth() const;
};

class computation_graph
{
protected:
    /**
     * The function represented by the computation graph.
     */
    tiramisu::function *fct;

    /**
     * The next optimization that the states generator will apply.
     */
    optimization_type next_optimization = optimization_type::FUSION;

public:
    /**
      * Computation graph root nodes.
      */
    std::vector<cg_node*> roots;

    /**
     * An evaluation of the execution of the function represented by
     * the computation graph.
     */
    float evaluation;
        
    /**
     * Create an empty computation graph.
     */
    computation_graph() {}
    
    /**
     * Create a computation graph from the given function.
     */
    computation_graph(tiramisu::function *fct);

    /**
     * Transform a computation to a computation graph node.
     */
    cg_node* computation_to_cg_node(tiramisu::computation *comp);

    /**
     * Copy this computation graph to new_cg and return
     * a pointer to the copied version of node_to_find.
     */
    cg_node* copy_and_return_node(computation_graph& new_cg, cg_node *node_to_find) const;

    optimization_type get_next_optimization() const { return next_optimization; }
    void set_next_optimization(optimization_type next_opt) { next_optimization = next_opt; }
};

}

#endif
