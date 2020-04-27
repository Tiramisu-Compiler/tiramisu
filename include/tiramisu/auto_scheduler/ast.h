#ifndef _H_TIRAMISU_AUTO_SCHEDULER_AST_
#define _H_TIRAMISU_AUTO_SCHEDULER_AST_

#include <tiramisu/core.h>
#include "utils.h"
#include "optimization_info.h"
#include "dnn_accesses.h"

namespace tiramisu::auto_scheduler
{

class syntax_tree;

/**
 *
 */
class computation_info
{
private:

protected:

public:
    tiramisu::computation *comp_ptr;
    
    std::vector<dnn_iterator> iters;
    dnn_accesses accesses;
    
    int buffer_nb_dims;
    bool is_reduction;
    
    int nb_additions;
    int nb_substractions;
    int nb_multiplications;
    int nb_divisions;
    
    computation_info(tiramisu::computation *comp, syntax_tree *ast);
    
    /**
     * Compute nb_additions, nb_substractions, nb_multiplications and nb_divisions
     * from the given expr.
     */
    void get_info_from_expr(tiramisu::expr const& e);
};

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
    std::vector<computation_info> computations;

	/**
	 * Next loop levels.
	 */
    std::vector<ast_node*> children;
    
    /**
     * Parent of this loop level.
     */
    ast_node *parent = nullptr;
    
	/**
	 * Create an empty AST node.
	 */
	ast_node() {}

	/**
	 * Create an AST node from the given computation.
	 */
	ast_node(tiramisu::computation *comp, syntax_tree* ast);
        
    ~ast_node()
    {
        for (ast_node* child : children)
            delete child;
    }
    
    /**
     * Return the extent of this loop level.
     */
    int get_extent() const { return up_bound - low_bound + 1; }
    
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
     * Fill the given array with the extents of the innermost loop levels
     * contained in this subtree.
     */
    void get_innermost_extents(std::vector<int>& extents) const;
    
    /**
     * Get the computations located at the innermost loop levels of this subtree.
     */
    void get_innermost_computations(std::vector<tiramisu::computation*>& comps);
    
    /**
     * Fill the given array with the nodes representing the innermost loop levels
     * contained in this subtree.
     */
    void get_innermost_nodes(std::vector<ast_node*>& nodes);
    
    /**
     * Get the root of the tree to which this node belongs to.
     */
    ast_node* get_root_node();
    
    /**
     * Get the node located at the leftmost side of the tree.
     */
    ast_node* get_leftmost_node();
    
    /**
     * Get the node located at the rightmost side of the tree.
     */
    ast_node* get_rightmost_node();

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
     * Print the subtree rooted at this node.
     */
    void print_node() const;
};

class syntax_tree
{
private:
    /**
     * Transform the AST using the order specified by the user
     * with the command "after".
     */
    void order_computations();

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
     * The list of computations contained in this AST.
     */
    std::vector<tiramisu::computation*> computations_list;
    
    /**
     * A mapping between each computation and the node where it is contained.
     */
    std::unordered_map<tiramisu::computation*, ast_node*> computations_mapping;
    
    /**
     * The list of buffers used by the program.
     */
    std::vector<std::string> buffers_list;
    
    /**
     * A mapping between each computation and the name of the buffer where
     * it is stored.
     */
    std::unordered_map<std::string, std::string> buffers_mapping;

    /**
     * An evaluation of the execution of the function represented by
     * the AST.
     */
    float evaluation;
    
    /**
     * The depth of this AST in a search space procedure.
     */
    int search_depth = 0;
    
    /**
     * The total number of explored optimizations.
     */
    int nb_explored_optims = 0;
    
    /**
     *
     */
    std::vector<optimization_info> previous_optims;
    
    /**
     *
     */
    std::vector<optimization_info> new_optims;
        
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
     * Return the node corresponding to the given loop level of the given computation.
     */
    ast_node* find_node_by_level(tiramisu::computation *comp, int level);

    /**
     * Transform the AST by applying the given optimization.
     */
    void transform_ast(optimization_info const& opt);
    
    /**
     * Transform the AST by applying the last new optimization.
     */
    void transform_ast();

    void transform_ast_by_fusion(optimization_info const& opt);
    void transform_ast_by_tiling(optimization_info const& opt);
    void transform_ast_by_interchange(optimization_info const& opt);
    void transform_ast_by_unrolling(optimization_info const& opt);
    
    /**
     * Get the extents of the loop levels shared by all computations.
     */
    std::vector<int> get_shared_levels_extents() const;
    
    /**
     * Get the extents of all the innermost loop levels.
     */
    std::vector<int> get_innermost_extents() const;
    
    /*
     * Get the computations located at the innermost loop levels.
     */
    std::vector<tiramisu::computation*> get_innermost_computations();
    
    /**
     * Return the nodes representing the innermost loop levels.
     */
    std::vector<ast_node*> get_innermost_nodes() const;
    
    /**
     * Return the schedule of this AST.
     */
    std::vector<optimization_info> get_schedule() const
    {
        std::vector<optimization_info> schedule = previous_optims;
        for (optimization_info const& optim_info : new_optims)
            schedule.push_back(optim_info);
            
        return schedule;
    }
    
    /**
     * Return the position, in the list of the buffers, of the buffer where
     * the given computation is stored.
     */
    int get_buffer_id_from_computation_name(std::string comp_name);
    
    /**
     * Return the position of the given buffer in the list of buffers.
     */
    int get_buffer_id(std::string const& buf_name) const;
    
    /**
     * Add the content of new_optims to previous_optims and
     * clear new_optims.
     */
	void clear_new_optimizations()
	{
	    for (optimization_info const& optim_info : new_optims)
	        previous_optims.push_back(optim_info);
	        
	    new_optims.clear();
	}

    /**
     * Print the AST to stdout.
     */
    void print_ast() const;
    
    std::vector<tiramisu::computation*> const& get_computations() const { return computations_list; }
};

}

#endif
