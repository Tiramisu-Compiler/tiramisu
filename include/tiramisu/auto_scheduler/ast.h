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
private:

protected:

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
     * True if the following loop level has been unrolled.
     */
    bool unrolled = false;

    /**
     * List of the computations computed at this level.
     */
    std::vector<tiramisu::computation*> computations;

	/**
	 * Next loop levels.
	 */
    std::vector<ast_node*> children;
    
    /**
     * Parent of this loop level.
     */
    ast_node *parent;

	/**
	 * Create an empty AST node.
	 */
	ast_node() {}

	/**
	 * Create an AST node from the given computation.
	 */
	ast_node(tiramisu::computation *comp);
        
    ~ast_node()
    {
        for (ast_node* child : children)
            delete child;
    }
    
    /**
     * Copy this node and return the copy.
     */
    ast_node* copy_node() const;

    /**
     * Copy the tree rooted at this node into new_node and return
     * a pointer to the copied version of node_to_find.
     */
    ast_node* copy_and_return_node(ast_node *new_node, ast_node *node_to_find) const;

    /**
     * Recompute the depth of each node of the tree rooted at
     * this node, with the given depth being the depth of this node.
     */
    void update_depth(int depth);

    /**
     * Fill the given array with all the computations computed 
     * at this level and the levels below.
     */
    void get_all_computations(std::vector<tiramisu::computation*>& comps);

    /**
     *
     */
    int get_loop_levels_chain_depth() const;
    
    /**
     * Return the extent of this loop level.
     */
    int get_extent() const { return up_bound - low_bound + 1; }

    /**
     * Print the subtree rooted at this node.
     */
    void print_node() const;
};

class syntax_tree
{
private:

protected:

public:
    /**
     * The function represented by the AST.
     */
    tiramisu::function *fct;
    
    /**
      * AST root nodes.
      */
    std::vector<ast_node*> roots;

    /**
     * An evaluation of the execution of the function represented by
     * the AST.
     */
    float evaluation;
    
    /**
     * The depth of this AST in a search space procedure.
	 * Depending on the search space procedure, it can be interpreted
	 * as the number of optimizations that have been applied.
     */
    int search_depth = 0;
    
    /**
     * List of the optimizations that the AST resulted from.
     */
    std::vector<optimization_info> optims_info;
        
    /**
     * Create an empty AST.
     */
    syntax_tree() {}
    
    /**
     * Create an AST from the given function.
     */
    syntax_tree(tiramisu::function *fct);
    
    ~syntax_tree()
    {
        for (ast_node *node : roots)
            delete node;
    }
    
    /**
     * Copy this AST, and return the copy.
     */
    syntax_tree* copy_ast() const;

    /**
     * Copy this AST to new_ast and return
     * a pointer to the copied version of node_to_find.
     */
    ast_node* copy_and_return_node(syntax_tree& new_ast, ast_node *node_to_find) const;

    /**
     * Transform the AST by applying the specified optimizations.
     */
    void transform_ast();

    void transform_ast_by_fusion(optimization_info const& opt);
    void transform_ast_by_tiling(optimization_info const& opt);
    void transform_ast_by_interchange(optimization_info const& opt);
    void transform_ast_by_unrolling(optimization_info const& opt);

    /**
     * Print the AST.
     */
    void print_ast() const
	{
		for (ast_node *root : roots)
			root->print_node();
	}
};

}

#endif
