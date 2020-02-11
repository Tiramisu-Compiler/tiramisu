#ifndef _H_TIRAMISU_AUTO_SCHEDULER_AST_
#define _H_TIRAMISU_AUTO_SCHEDULER_AST_

#include <tiramisu/core.h>
#include "utils.h"

namespace tiramisu::auto_scheduler
{

/**
  * A node in the AST represents a loop level.
  */
class ast_node
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

    std::vector<ast_node*> children;
    
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
        
    ~ast_node()
    {
        for (ast_node* child : children)
            delete child;
    }

    /**
     * Copy the tree rooted at this node into new_node and return
     * a pointer to the copied version of node_to_find.
     */
    ast_node* copy_and_return_node(ast_node *new_node, ast_node *node_to_find) const;
    
    int get_branch_depth() const;
    
    void update_depth(int depth);

    void print_node() const;
};

class syntax_tree
{
protected:
    /**
     * The function represented by the AST.
     */
    tiramisu::function *fct;

public:
    /**
      * AST root nodes.
      */
    std::vector<ast_node*> roots;

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
    syntax_tree() {}
    
    /**
     * Create a computation graph from the given function.
     */
    syntax_tree(tiramisu::function *fct);

    /**
     * Transform a computation to an AST node.
     */
    ast_node* computation_to_ast_node(tiramisu::computation *comp);

    /**
     * Copy this AST to new_ast and return
     * a pointer to the copied version of node_to_find.
     */
    ast_node* copy_and_return_node(syntax_tree& new_ast, ast_node *node_to_find) const;

    /**
     * Transform the AST by applying the specified
     * optimizations (see cg_node).
     */
    void transform_ast();

    /**
     * Inspect the given tree level and apply any specified fusion.
     */
    static void transform_ast_by_fusion(std::vector<ast_node*>& tree_level);
    
    /**
     * Inspect the tree rooted at node and apply any specified tiling.
     */
    static void transform_ast_by_tiling(ast_node *node);
    
    /**
     * Inspect the tree rooted at node and apply any specified interchange.
     */
    static void transform_ast_by_interchange(ast_node *node);

    void print_graph() const;
};

}

#endif
