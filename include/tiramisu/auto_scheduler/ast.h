#ifndef _H_TIRAMISU_AUTO_SCHEDULER_AST_
#define _H_TIRAMISU_AUTO_SCHEDULER_AST_

#include <tiramisu/core.h>
#include "utils.h"
#include "optimization_info.h"
#include "dnn_accesses.h"
enum search_phase
{
    FUSION,
    UNIMODULAR,
    NON_UNIMODULAR
};
namespace tiramisu::auto_scheduler
{

class syntax_tree;



/**
 * a class that represent a state for schedule_generator to allow to generate a same optimisation
 * in different ast_nodes by looking at the previous states and current state.
*/
class generator_state
{

public:

    static std::vector<optimization_type> optimization_list;

    static bool initialized;

    // a list of ast_node to explore with an additional information (int).
    std::vector<std::pair<ast_node*,int>> target_ast_heads;

    // index in the vector for the explored generations.
    int current_index = 0;

    // current optimization
    int optimization_index = 0;



public:    

    /**
     *  checks if the current optimization does not have any more possible variants.
     * */
    bool is_current_optimization_fully_explored();

    /**
     *  checks if the current type of optimization is not the last element from wanted optimizations.
     * */
    bool can_move_to_next_optimization();

    /**
     * Set the new alternatives for the current optimization.
     * */
    void set_new_heads(std::vector<std::pair<ast_node*,int>>& optim_heads);

    /**
     * get the currently pointed alternative (state) for the currently pointed optimization.
    */
    std::pair<ast_node*,int> get_current_head();

    /**
     * move to the next alternative within the same optimization
    */
    void increment_index();


    /**
     *  True if the search space is empty.
    */
    bool is_search_space_empty();


};


/**
 * stores the state of the computation's schedule.
*/
class state_computation
{
    friend tiramisu::computation;

    private:
    /**
     * Isl_map that represent the schedule of the computation
    */
    isl_map * current_schedule = NULL;

   /**
    * Computation reference that points to the real computation object.
    * Useful to stage optimizations with setting current_schedule as schedule then use computations features.
   */
    tiramisu::computation * staging_computation = NULL;
    /**
     * Describes wether this state is staged inside the computation or not.
    */
    bool is_state_staged;

    protected:

    public:

    /**
     * constructors
    */
    //@{
    state_computation(tiramisu::computation * reference);
    state_computation(state_computation const& reference);
    //@}

    /**
     * returns the isl_map
    */
   isl_map * get_inner_isl_map() const;

    /**
     * moves current schedule into the computation object as staging area.
     * mermutes the schedules between the computation and this schedule.
    */
    void move_schedule_to_staging();

    /**
     * Gets the schedule back from the computation and store it in this class instance.
    */
    void recover_schedule_from_staging();

    /**
     * get real computation in which we set the current state as a schedule inside it. 
    */
    tiramisu::computation * get_computation_staged();

    /**
     * get real computation without staging a schedule inside it.
    */

    tiramisu::computation * get_computation_unstated() const;

    /**
     * 
    */
    bool is_this_state_staged() const;

    ~state_computation()
    {
        isl_map_free(current_schedule);
    }

};

/**
 * Stores information about a computation.
 */
class computation_info
{
private:

protected:

public:
    /**
     * Pointer to the corresponding computation.
     */
    tiramisu::computation *comp_ptr;
    
    /**
     * List of iterators of the computation.
     */
    std::vector<dnn_iterator> iters;
    
    /**
     * List of accesses of the computation.
     */
    dnn_accesses accesses;
    
    /**
     * Number of dimensions of the output buffer.
     */
    int buffer_nb_dims;
    
    /**
     * True if this computation is a reduction.
     */
    bool is_reduction;

    /**
     * A string representing the ISL write access relation of the computation.
     */
    std::string write_access_relation;

    /**
     * The ID of the buffer where the computation is stored
     */
    int storage_buffer_id;

    /**
     * A string representing the data type of the computation
     */
    std::string data_type_str;

    /**
     * Size of the data type in Bytes
     */
    int data_type_size;

    /**
     * Some metrics about the computation.
     */
    int nb_additions;
    int nb_substractions;
    int nb_multiplications;
    int nb_divisions;
    
    /**
     * Get info about the given computation. The AST is needed to get some info.
     */
    computation_info(tiramisu::computation *comp, syntax_tree *ast, std::vector<dnn_iterator> iterators);


    /**
     * Copy constructor
    */
    //computation_info(computation_info const& reference);

    /**
     * modifies the accesses by skewing
    */
    void set_accesses_changes_with_skewing(int first_node_depth,int alpha,int beta,int gamma,int sigma);
    
    /**
     * Compute nb_additions, nb_substractions, nb_multiplications and nb_divisions
     * from the given expr.
     */
    void get_info_from_expr(tiramisu::expr const& e);

    /**
     * Returns the size of the computation's data type in Bytes
     */
    int get_data_type_size();
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
    std::string low_bound;
    
    /**
     * Upper bound of this loop level iterator.
     */
    std::string up_bound;

    /**
     * True if the following loop level has been unrolled.
     */
    bool unrolled = false;

    /**
     * True if the loop level has been parallelized
     */
    bool parallelized = false;

    /**
    * True if the loop level has been skewed
    */
    bool skewed = false;

    /**
    * True if the loop level has been vectorized
    */
    bool vectorized = false;


    /**
     * specifies if it contains conditionals as a result of shifting
    */
    bool shifted = false;

    /**
     * List of the computations computed at this level.
     */
    std::vector<computation_info> computations;

    /**
     * Structure that holds the state of each computation inside this node.
    */
    std::vector<state_computation> isl_states;

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
    std::string get_extent() const { return up_bound + "-" + low_bound + "+1"; }
    
    /**
     * Copy this node and return the copy.
     */
    ast_node* copy_node() const;

    /**
     * Copy the tree rooted at this node into new_node and return
     * a pointer to the copied version of node_to_find.
     *
     * This function is used if you want to copy a tree, and need the new location
     * of a node.
     */
    ast_node* copy_and_return_node(ast_node *new_node, ast_node *node_to_find) const;
    
    /**
     * Fill the given array with the extents of the innermost loop levels
     * contained in this subtree.
     */
    void get_innermost_extents(std::vector<std::string>& extents) const;
    
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
     * get all the nodes starting from root that have 1 child, 
     * i.e. the shared nodes between all computations
    */
    void get_shared_nodes_from_outermost(std::vector<ast_node*>& shared) const;

    /**
     * Recompute the depth of each node of the tree rooted at
     * this node, with the given depth being the depth of this node.
     */
    void update_depth(int depth);
    /**
     * Fill the given array with all the computations computed 
     * at this level.
     */
    void get_node_computations(std::vector<tiramisu::computation*>& comps);
    /**
     * Fill the given array with all the computations computed 
     * at this level and the levels below.
     */
    void get_all_computations(std::vector<tiramisu::computation*>& comps);
     /**
     * Fill the given array with all the nodes 
     * at this level and the levels below.
     */
    void get_all_nodes(std::vector<ast_node*>   &nodes);
    /**
     * Starting from this node, get the number of nodes that have no computation,
     * and only one child.
     */
    int get_loop_levels_chain_depth() const;

    /**
     * Print the subtree rooted at this node.
     */
    void print_node() const;

    /**
     * Print the subtree of isl_states
    */
    void print_isl_states() const;

    /**
     * prints the computations's accesses of this AST
    */
    void print_computations_accesses() const;

    /**
     * Changes the access within computation_info after applying the skewing optimization
    */
    void transforme_accesses_with_skewing(int a,int b);

    void set_accesses_changes_with_skewing(int first_node_depth,int alpha,int beta,int gamma,int sigma);

    /**
     * create initial isl_states from current computations
    */
    void create_initial_states();

    /**
     * erases all isl_states from the nodes and computations recursively 
    */
    void erase_isl_states();

    /**
     * stage the isl_states to the real computations
    */
    void stage_isl_states();

    /**
     * recover the states from the computations
    */
    void recover_isl_states();

    /**
     * pushs all the computations inside this node recursively 
    */
    void collect_all_computation(std::vector<computation_info*>& vector);

    /**
     * get the extent of this node, i.e:
     * return upper_bound - lower_bound
    */
    std::string get_node_loop_extent() const;

    /**
     * checks whether or not two ast_node have the same iteration domaine +/-1 so they can possible be fuzed together.
    */
    bool have_similar_itr_domain(ast_node * other);

    /**
     * checks whether or not two ast_node have the same iteration domaine and depth starting form the commun ast_node. 
    */
    bool is_candidate_for_fusion(ast_node * other);

    /**
     *  returns the two ast_node, first one for the previous ast_node that we fuze into, and the other for this ast_node.
     *  the result should be two nodes that have the same depth.
    */
    std::pair<ast_node *,ast_node*> get_possible_fusion_candidate(ast_node * previous_node);

    /**
     * finds all loop levels from the ast until the root level.
     * returns iterators names.
    */
    std::vector<std::string> get_all_iterators();

    /**
     * Returns this ast_node or one of it's parents that posses the specified depth.
     * the specified The depth must be less or equal the local depth.
    */
   ast_node * find_node_by_depth(int depth);

   
    /**
     * returns a new node as a copy from this node but without copying the children also.
    */
    ast_node * copy_local_node(bool copy_first_computation);

    /**
     * Copies a linear branch from this node and stops with the shared parent.
     * In case the parent is null_ptr it should create a new linear branch from the root
    */
    ast_node * new_branch_leaf(ast_node * shared_node);

    /**
     * collects the nodes closer to the root (and also the root) that involves shared nodes underneath.
    */
    static std::vector<ast_node*> collect_heads_of_ast(int allowed_splits, ast_node* current);

    /**
     * get the shared nodes that start from this node
    */
    std::vector<ast_node*> collect_shared_nodes_from_head();

    /**
     * Checks if a node is already tagged and optimized.
    */
    bool is_optimized_by_tag();
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
     * the odering scheduling graph which may change during loop fusion.
     * by ptr to eleminate unnecessary duplications.
    */

    std::shared_ptr<std::unordered_map<tiramisu::computation *,
    std::unordered_map<tiramisu::computation *, int>>>  local_sched_graph = nullptr;
    
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
     * An evaluation given by a class of type evaluation_function.
     */
    float evaluation;
    
    /**
     * A set containing the names of all the iterators in the ast. Used to avoid duplicate iterator names.
     */
    std::set<std::string> iterator_names_set;
    
    /**
     * The depth of this AST in a search method.
     * Used to keep track of the depth reached by a search method.
     */
    int search_depth = 0;
    /**
     * The current exploration phase for this AST
     * 
     */
    search_phase ast_search_phase = search_phase::FUSION;
    /**
     * The depth of this AST in a search method.
     * Used to keep track of the depth reached by a search method.
     */
    int nb_explored_matrices = 0;
    /**
     * The total number of explored optimizations.
     * Used to keep track of the number of optimizations explored by a search method.
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
     * The iterators in JSON format.
     * Use by the class evaluate_by_learning_model.
     */
    std::string iterators_json;
    
    /**
     * The structure represented by this AST in JSON format.
     * Use by the class evaluate_by_learning_model.
     */
    std::string tree_structure_json;

    /**
     * a structure that saves the points of previous applied optimizations.
     * Used by the shedule_generator to explore same optimization in different deapth of the search.
    */
    generator_state search_state;
        
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

    tiramisu::function * get_function() const { return fct; } 
    
    std::vector<tiramisu::computation*> const& get_computations() const { return computations_list; }
    
    /**
     * Transform the AST by applying the last optimization found.
     */
    void transform_ast();

    /**
     * Transform the AST by applying the given optimization.
     */
    void transform_ast(optimization_info const& opt);

    /**
     * These methods are used to transform the AST given a specific type of optimization.
     */
    void transform_ast_by_fusion(optimization_info const& opt);
    void transform_ast_by_unfuse(optimization_info const& opt);
    void transform_ast_by_tiling(optimization_info const& opt);
    void transform_ast_by_interchange(optimization_info const& opt);
    void transform_ast_by_unrolling(optimization_info const& opt);
    void transform_ast_by_vectorization(const optimization_info &opt);
    void transform_ast_by_parallelism(const optimization_info &info);
    void transform_ast_by_skewing(const optimization_info &opt);
    void transform_ast_by_reversal(const optimization_info &opt);
    void transform_ast_by_shifting(const optimization_info &opt);
    void transform_ast_by_matrix(const optimization_info &opt);

    
    /**
     * Copy this AST, and return the copy.
     */
    syntax_tree* copy_ast() const;

    /**
     * Create an independent copy of sched_graph in this AST 
     * from the fct shed_graph.
     * In case a copy of another syntax_tree needs to be made, this other tree needs to be staged and swapped with fct_shed_graph before
     * invoking this methods.
    */
    void create_new_sched_graph();

    /**
     * Swaps the local sched_graph with the sched_graph within tiramisu API.
    */
    void stage_local_sched_graph() const;

    void recover_local_sched_graph() const;

    /**
     * Dump current sched graph into the tiramisu_api sched __graph
    */
    void dump_local_sched_graph_to_api() const
    {
        this->get_function()->sched_graph.clear();

        this->get_function()->sched_graph = *this->local_sched_graph;
    }

    /**
     * Copy this AST to new_ast and return a pointer to the copied version of node_to_find.
     * Can be used if you want to copy this AST, and need to find the new location of a node.
     */
    ast_node* copy_and_return_node(syntax_tree& new_ast, ast_node *node_to_find) const;
    
    /**
     * Clear computations_mapping, and recompute the nodes where computations are stored.
     */
    void recompute_computations_mapping();
    
    /**
     * Recursive subroutine used by void recompute_computations_mapping();
     */
    void recompute_computations_mapping(ast_node *node);
    
    /**
     * Return the schedule of this AST.
     */
    std::vector<optimization_info> get_schedule() const;
    
    /**
     * Add the content of new_optims to previous_optims and
     * clear new_optims.
     */
	void clear_new_optimizations();
    
    /**
     * Return the node corresponding to the given loop level of the given computation.
     */
    ast_node* find_node_by_level(tiramisu::computation *comp, int level);
    
    /**
     * Get the extents of the loop levels shared by all computations.
     */
    std::vector<std::string> get_shared_levels_extents() const;
    
    /**
     * Get the extents of all the innermost loop levels.
     */
    std::vector<std::string> get_innermost_extents() const;
    
    /*
     * Get the computations located at the innermost loop levels.
     */
    std::vector<tiramisu::computation*> get_innermost_computations();
    
    /**
     * Return the nodes representing the innermost loop levels.
     */
    std::vector<ast_node*> get_innermost_nodes() const;

    /**
     * Deletes the specified node and if the parent becomes empty it adjusts the AST recursively
    */
    void delete_duplicated_node_recursively(ast_node * node);

    /**
    * Moves the computation in this ast_node the specified node and deletes the previous ast_node.
    * The computation's original ast_node is 
    * the moved computation allways by analysis and design resides in the start of the computations list.
   */
    void move_in_computation(ast_node * new_node, tiramisu::computation * comp_ptr);


    /**
     * get all the nodes starting from root that have 1 child, 
     * i.e. the shared nodes between all computations
    */
    void get_shared_nodes_from_outermost(std::vector<ast_node*>& shared) const;
    
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
     * Print the AST to stdout.
     */
    void print_ast() const;

    /**
     * prints the computations's accesses of this AST
    */
    void print_computations_accesses() const;

    /**
     * Print information about the applied optimizations
     */
    void print_new_optims() const;

    void print_previous_optims() const;

    void print_isl_states() const;

    void create_initial_isl_state() const;

    /**
    * Returns the last (deepest in the AST) parent node shared by node1 and node2. Returns nullptr if no shared loops were found
    */
    ast_node *get_last_shared_parent(ast_node *node1, ast_node *node2) const;
    /**
     * erases all ISL states and creates new ones from the computations.
    */
    void recreate_isl_state() const;

    /**
     * push isl_states to the real computations to use tiramisu API
    */
    void stage_isl_states() const;

    /**
     * Recover the isl_states from the computation 
    */
    void recover_isl_states() const;

    /**
     * Check the correctness of the list of applied structurel optimizations
    */
    bool ast_is_legal() const;

    /**
     * Encodes the transformations applied to the ast as a string, this is used for saving the exploration trace while
     * sampling schedules
     */
    std::string get_schedule_str();

    /**
     * Predicts if the schedule applied to the ast is worth evaluating and exploring further.
     */
    bool schedule_is_prunable();
    bool ast_is_prunable();

    /**
     * Gets all the computations that are ordered before the current computation.
     * In case the computation_index is -1, it will bring all the computation within the node.
    */
    void get_previous_computations(std::vector<computation*>& result,ast_node*& node, int computation_index);

    /**
     * Checks if the AST's evaluation can be predicted using manual engineered rules
     */
    bool can_set_default_evaluation();

    /**
     * Initialize the search state with potentiel alternatives for the specified optimisation.
    */
    std::vector<std::pair<ast_node*,int>> compute_search_space_states(optimization_type optimization) const;

    /**
     * Initialize the list of explored optimizations in the search space.
    */
    void initialize_search_space_optimizations(std::vector<optimization_type> optimizations);

    /**
     * True if the search space is empty.
    */
    bool is_search_space_empty();

    /**
     * Gets the current optimization target (ast_node : that represent a branch of the AST)
     *  along with the correspending optimization_type
     * that need to be explored.
     * Must only be invoked when the search_space is not fully explored.
    */
    std::pair<ast_node*,int> get_current_optimization_target();

     /**
     * Gets the previous optimization target (ast_node : that represent a branch of the AST)
     *  along with the correspending optimization_type
     * as long as the number of alternative for this optimization is >1
     * and the current_optimization_target index is >0 
    */
    std::pair<ast_node*,int> get_previous_optimization_target();

    /**
     * Gets the current optimization type
    */
    optimization_type get_current_optimization_type() const;
    

    /**
     * moves to the next alternative or to the next optimization .
    */
    void move_to_next_optimization_target();
    /**
     * in matrix exploration, move to the next head or to the next optimization or stop exploration at this level.
    */
    void move_to_next_head();
    /**
     * moves to the next alternative.
    */
    void move_to_next_optimization_target_matrix();
    /**
     * Recomputes the states to update after a generation of a copy or application of fusion.
    */
    void refresh_states();

    /**
     * Return the computation's execution order (it's index in computation_list, index starts from 0)
     */
    int get_computation_index(computation *comp);

    /**
     * Checks whether the optimization type opt_type is  already applied to computation comp (searches in new_optims)
     */
     bool optim_already_applied_on_comp(tiramisu::computation* comp,tiramisu::auto_scheduler::optimization_type opt_type);

     /**
     * Checks whether the optimization type opt_type is  already applied to at least one computation from comp_list (searches in new_optims)
     */
     bool optim_already_applied_on_comps(const std::vector<tiramisu::computation *>comp_list, tiramisu::auto_scheduler::optimization_type opt_type);
};

/**
 * The candidate_trace class is used to recursively record the visited candidates during the search space exploration
 */
class candidate_trace
{
private:
    /**
     * A numerical id is assigned for each explored candidate
     */
    int candidate_id;

public:
    int get_candidate_id() const;

private:
    /**
     * The search depth where this candidate was visited
     */
    int exploration_depth;

    /**
     * candidate's evaluation
     */
    float evaluation;

    /**
     * The applied schedule, encoded as a string
     */
    std::string schedule_str;

    /**
     * The child candidates that are derived from this candidate
     */
    std::vector<candidate_trace*> child_candidates;

public:

    candidate_trace(syntax_tree* ast, int candidate_id);

    ~candidate_trace();

    /**
     * Initializes a new child candidate from an AST and adds it to the children list
     */
    void add_child_path(syntax_tree* ast, int candidate_id);

    /**
     * Recursively exports the exploration trace into a JSON format
     */
    std::string get_exploration_trace_json();

    /**
     * A mapping between ASTs and explored child candidates
     */
    std::unordered_map<syntax_tree*, candidate_trace*> child_mappings;

};


}

#endif
