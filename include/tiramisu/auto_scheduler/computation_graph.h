#ifndef _H_TIRAMISU_AUTO_SCHEDULER_COMPUTATION_GRAPH_
#define _H_TIRAMISU_AUTO_SCHEDULER_COMPUTATION_GRAPH_

#include <tiramisu/core.h>
#include "utils.h"

namespace tiramisu::auto_scheduler
{

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
     * Name of this loop level iterator.
     */
    std::string name;
    
    /**
     * Lower bound of this loop level iterator.
     */
    int low_bound;
    
    /**
     * Upper bound of this loop level iterator.
     */
    int up_bound;

    /**
     * List of the computations computed at this level.
     */
    std::vector<tiramisu::computation*> computations;

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
    
    void update_depth(int depth);

    void print_node() const;
};

class computation_graph
{
protected:
    /**
     * The function represented by the computation graph.
     */
    tiramisu::function *fct;

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
     * Index of the next optimization that the states generator
     * will apply.
     */
    int next_optim_index = 0;
        
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

    /**
     * Transform the computation graph by applying the specified
     * optimizations (see cg_node).
     */
    void transform_computation_graph();

    /**
     * Inspect the given tree level and apply any specified fusion.
     */
    static void transform_computation_graph_by_fusion(std::vector<cg_node*>& tree_level);
    
    /**
     * Inspect the tree rooted at node and apply any specified tiling.
     */
    static void transform_computation_graph_by_tiling(cg_node *node);
    
    /**
     * Inspect the tree rooted at node and apply any specified interchange.
     */
    static void transform_computation_graph_by_interchange(cg_node *node);

    void print_graph() const;
};

}

#endif
