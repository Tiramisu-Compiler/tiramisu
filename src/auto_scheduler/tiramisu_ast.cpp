#include <tiramisu/auto_scheduler/ast.h>
#include <tiramisu/auto_scheduler/evaluator.h>

namespace tiramisu::auto_scheduler
{

computation_info::computation_info(tiramisu::computation *comp, syntax_tree *ast)
    : comp_ptr(comp), iters(dnn_iterator::get_iterators_from_computation(*comp)),
      accesses(comp, iters.size(), comp->get_function()), buffer_nb_dims(iters.size()),
      nb_additions(0), nb_substractions(0), nb_multiplications(0), nb_divisions(0)
{
    get_info_from_expr(comp->get_expr());
    
    // Check if this computation is a reduction
    isl_map *storage_map = comp->access;
    buffer_nb_dims = isl_map_dim(storage_map, isl_dim_out);
    
    if (buffer_nb_dims < iters.size())
        is_reduction = true;
    else
        is_reduction = false;
        
    // Get buffer_id for the accesses of this computation
    for (dnn_access_matrix& matrix : accesses.accesses_list)
        matrix.buffer_id = ast->get_buffer_id_from_computation_name(matrix.buffer_name);
}

void computation_info::get_info_from_expr(tiramisu::expr const& e)
{
    // Not an operation, stop the search
    if (e.get_expr_type() != tiramisu::e_op)
        return ;
        
    // We have an access, stop the search
    if (e.get_op_type() == tiramisu::o_access || 
        e.get_op_type() == tiramisu::o_lin_index ||
        e.get_op_type() == tiramisu::o_address_of || 
        e.get_op_type() == tiramisu::o_dummy ||
        e.get_op_type() == tiramisu::o_buffer)
    {
        return ;
    }
    
    switch (e.get_op_type())
    {
        case o_add:
            nb_additions++;
            break;
            
        case o_sub:
            nb_substractions++;
            break;
            
        case o_mul:
            nb_multiplications++;
            break;
            
        case o_div:
            nb_divisions++;
            break;
            
        default:
            break;
    }
    
    // We have an operation, we explore its operands
    for (int i = 0; i < e.get_n_arg(); ++i)
        get_info_from_expr(e.get_operand(i));
}

// ---------------------------------------------------------------------------- //

syntax_tree::syntax_tree(tiramisu::function *fct)
    : fct(fct)
{
    const std::vector<computation*> computations = fct->get_computations();
    
    for (tiramisu::computation *comp : computations) 
    {
        // Get this computation buffer name
        isl_map *storage_map = comp->access;
        std::string buf_name = isl_map_get_tuple_name(storage_map, isl_dim_out);
        
        if (std::find(buffers_list.begin(), buffers_list.end(), buf_name) == buffers_list.end())
            buffers_list.push_back(buf_name);
            
        buffers_mapping[comp->get_name()] = buf_name;
        
        if (comp->get_expr().get_expr_type() == e_none)
            continue;
        
        // Insert this computation in the AST
        ast_node *node = new ast_node(comp, this);
        node->parent = nullptr;
        
        roots.push_back(node);
        computations_list.push_back(comp);
        computations_mapping[comp] = node->get_leftmost_node();
    }

    // Order the computations by the order specified by the user using "after" commands
    order_computations();
    
    // Get the JSON representation of this AST iterators
    for (ast_node *node : roots)
        evaluate_by_learning_model::represent_iterators_from_nodes(node, iterators_json);
        
    iterators_json.pop_back();
    
    // Get the JSON representation of this tree
    tree_structure_json = evaluate_by_learning_model::get_tree_structure_json(*this);
}

ast_node::ast_node(tiramisu::computation *comp, syntax_tree *ast)
{
    std::vector<ast_node*> nodes;

    // Get computation iterators
    isl_set *iter_domain = comp->get_iteration_domain();
    int nb_iterators = isl_set_dim(iter_domain, isl_dim_set);

    // The fist node is the one created by this constructor
    this->depth = 0;
    this->name = isl_set_get_dim_name(iter_domain, isl_dim_set, 0);
    this->low_bound = utility::get_bound(iter_domain, 0, false).get_int_val();
    this->up_bound = utility::get_bound(iter_domain, 0, true).get_int_val();

    nodes.push_back(this);
        
    // Create the other nodes, one for each iterator
    for (int i = 1; i < nb_iterators; ++i)
    {
        ast_node *node = new ast_node();
        
        node->depth = i;
        node->name = isl_set_get_dim_name(iter_domain, isl_dim_set, i);
        node->low_bound = utility::get_bound(iter_domain, i, false).get_int_val();
        node->up_bound = utility::get_bound(iter_domain, i, true).get_int_val();
        
        nodes.push_back(node);
    }

    // Chain the nodes together
    for (int i = 0; i < nodes.size() - 1; ++i)
    {
        nodes[i]->children.push_back(nodes[i + 1]);
        nodes[i + 1]->parent = nodes[i];
    }
    
    nodes.back()->computations.push_back(computation_info(comp, ast));
}

void syntax_tree::order_computations()
{
    if (roots.size() < 2)
        return ;
        
    // We use the scheduling graph (fct->sched_graph) to find the computations order
    for (auto& sched_graph_node : fct->sched_graph)
    {
        tiramisu::computation *parent_comp = sched_graph_node.first;
        
        for (auto& sched_graph_child : sched_graph_node.second)
        {
            tiramisu::computation *child_comp = sched_graph_child.first;
            int level = sched_graph_child.second;
            
            if (level < 0)
                continue;
            
            ast_node *parent_comp_ast_node = find_node_by_level(parent_comp, level);
            ast_node *child_comp_ast_node = find_node_by_level(child_comp, level);
            
            // Insert computations
            if (!child_comp_ast_node->computations.empty())
            {
                if (parent_comp_ast_node->children.empty())
                {
                    for (computation_info& comp_info : child_comp_ast_node->computations)
                    {
                        parent_comp_ast_node->computations.push_back(comp_info);
                        computations_mapping[comp_info.comp_ptr] = parent_comp_ast_node;
                    }
                }
                
                else
                {
                    // We have here a special case (see PFE manuscript page 46)
                    ast_node *new_node = new ast_node();
                    
                    new_node->depth = child_comp_ast_node->depth;
                    new_node->name = "dummy_iter";
                    new_node->low_bound = 0;
                    new_node->up_bound = 0;
                    new_node->computations = child_comp_ast_node->computations;
                    new_node->parent = parent_comp_ast_node;
                    
                    for (computation_info& comp_info : child_comp_ast_node->computations)
                        computations_mapping[comp_info.comp_ptr] = new_node;
                        
                    parent_comp_ast_node->children.push_back(new_node);
                }
            }
            
            // Insert children
            for (ast_node *child : child_comp_ast_node->children)
            {
                parent_comp_ast_node->children.push_back(child);
                child->parent = parent_comp_ast_node;
            }
                    
            ast_node *root_node = child_comp_ast_node->get_root_node();
            auto it = std::find(roots.begin(), roots.end(), root_node);
            roots.erase(it);
        }
    }
}

void syntax_tree::transform_ast()
{
    if (new_optims.size() == 0)
        return ;
        
    transform_ast(new_optims.back());
}

void syntax_tree::transform_ast(optimization_info const& opt)
{
    switch(opt.type)
    {
        case optimization_type::FUSION:
            transform_ast_by_fusion(opt);
            break;
            
        case optimization_type::UNFUSE:
            transform_ast_by_unfuse(opt);
            break;
            
        case optimization_type::TILING:
            transform_ast_by_tiling(opt);
            break;
            
        case optimization_type::INTERCHANGE:
            transform_ast_by_interchange(opt);
            break;
            
        case optimization_type::UNROLLING:
            transform_ast_by_unrolling(opt);
            break;

        case optimization_type::PARALLELIZE:
            transform_ast_by_paralellize(opt);
            break;

        default:
            break;
    }
}

void syntax_tree::transform_ast_by_fusion(optimization_info const& opt)
{
    std::vector<ast_node*> *tree_level;
    
    if (opt.node->parent != nullptr)
        tree_level = &opt.node->parent->children;
    else
        tree_level = &roots;
    
    ast_node *node1 = (*tree_level)[opt.l0];
    ast_node *node2 = (*tree_level)[opt.l1];

    for (ast_node *child : node2->children)
        node1->children.push_back(child);

    for (computation_info& comp_info : node2->computations)
    {
        node1->computations.push_back(comp_info);
        computations_mapping[comp_info.comp_ptr] = node1;
    }

    tree_level->erase(tree_level->begin() + opt.l1);
}

void syntax_tree::transform_ast_by_unfuse(optimization_info const& opt)
{
    ast_node *unfuse_node, *shared_node;
    
    int i = 0;
    shared_node = roots[0];
    
    while (shared_node->children.size() == 1)
    {
        if (i == opt.l0)
            unfuse_node = shared_node;
            
        shared_node = shared_node->children[0];
        i++;
    }
    
    std::vector<ast_node*> shared_node_children = shared_node->children;
    ast_node *removed_node = unfuse_node->children[0];
    unfuse_node->children.clear();
    
    for (ast_node *node : shared_node_children)
    {
        shared_node->children.clear();
        shared_node->children.push_back(node);
        
        unfuse_node->children.push_back(removed_node->copy_node());
    }
    
    tree_structure_json = evaluate_by_learning_model::get_tree_structure_json(*this);
}

void syntax_tree::transform_ast_by_tiling(optimization_info const& opt)
{
    ast_node *node = opt.node;
    
    // 2 level tiling
    if (opt.nb_l == 2)
    {
        // Create the new loop structure
        ast_node *i_outer = node;
        ast_node *j_outer = new ast_node();
            
        ast_node *i_inner = new ast_node();
        ast_node *j_inner = node->children[0];
            
        // Chain the nodes
        i_outer->children[0] = j_outer;
        j_outer->children.push_back(i_inner);
        i_inner->children.push_back(j_inner);
        
        j_outer->parent = i_outer;
        i_inner->parent = j_outer;
        j_inner->parent = i_inner;
            
        // Rename the nodes
        i_inner->name = i_outer->name + "_inner";
        i_outer->name = i_outer->name + "_outer";
            
        j_outer->name = j_inner->name + "_outer";
        j_inner->name = j_inner->name + "_inner";
            
        // Set lower and upper bounds
        i_outer->low_bound = 0;
        i_outer->up_bound = i_outer->get_extent() / opt.l0_fact - 1;
            
        j_outer->low_bound = 0;
        j_outer->up_bound = j_inner->get_extent() / opt.l1_fact - 1;
            
        i_inner->low_bound = 0;
        i_inner->up_bound = opt.l0_fact - 1;
            
        j_inner->low_bound = 0;
        j_inner->up_bound = opt.l1_fact - 1;
    }
        
    // 3 level tiling
    else if (opt.nb_l == 3)
    {
        // Create the new loop structure
        ast_node *i_outer = node;
        ast_node *j_outer = new ast_node();
        ast_node *k_outer = new ast_node();
            
        ast_node *i_inner = new ast_node();
        ast_node *j_inner = node->children[0];
        ast_node *k_inner = j_inner->children[0];
            
        // Chain the nodes
        i_outer->children[0] = j_outer;
        j_outer->children.push_back(k_outer);
        k_outer->children.push_back(i_inner);
        i_inner->children.push_back(j_inner);
        j_inner->children[0] = k_inner;
        
        j_outer->parent = i_outer;
        k_outer->parent = j_outer;
        i_inner->parent = k_outer;
        j_inner->parent = i_inner;
        k_inner->parent = j_inner;
            
        // Rename the nodes
        i_inner->name = i_outer->name + "_inner";
        i_outer->name = i_outer->name + "_outer";
            
        j_outer->name = j_inner->name + "_outer";
        j_inner->name = j_inner->name + "_inner";
            
        k_outer->name = k_inner->name + "_outer";
        k_inner->name = k_inner->name + "_inner";
            
        // Set lower and upper bounds
        i_outer->low_bound = 0;
        i_outer->up_bound = i_outer->get_extent() / opt.l0_fact - 1;
            
        j_outer->low_bound = 0;
        j_outer->up_bound = j_inner->get_extent() / opt.l1_fact - 1;
            
        k_outer->low_bound = 0;
        k_outer->up_bound = k_inner->get_extent() / opt.l2_fact - 1;
            
        i_inner->low_bound = 0;
        i_inner->up_bound = opt.l0_fact - 1;
            
        j_inner->low_bound = 0;
        j_inner->up_bound = opt.l1_fact - 1;
            
        k_inner->low_bound = 0;
        k_inner->up_bound = opt.l2_fact - 1;
    }

    node->update_depth(node->depth);
}

void syntax_tree::transform_ast_by_interchange(optimization_info const& opt)
{
    ast_node *node1 = opt.node;
    
    // Find the node to interchange with
    ast_node *node2 = node1;
    for (int i = opt.l0; i < opt.l1; ++i)
        node2 = node2->children[0];
            
    // Rename the two nodes
    std::string tmp_str;
    tmp_str = node1->name;
    node1->name = node2->name;
    node2->name = tmp_str;
            
    int tmp_int;
    tmp_int = node1->low_bound;
    node1->low_bound = node2->low_bound;
    node2->low_bound = tmp_int;
        
    tmp_int = node1->up_bound;
    node1->up_bound = node2->up_bound;
    node2->up_bound = tmp_int;
}

void syntax_tree::transform_ast_by_unrolling(optimization_info const& opt)
{
    std::vector<ast_node*> nodes_list;
    
    // Apply unrolling on the node provided by opt
    if (opt.l0 != -1)
        nodes_list = {opt.node};
        
    // Apply unrolling on every innermost loop level
    else
        nodes_list = get_innermost_nodes();
    
    for (ast_node *node : nodes_list)
    {
        if (node->get_extent() <= opt.l0_fact)
            node->unrolled = true;
            
        else 
        {
            // Create the new loop structure
            ast_node *i_outer = node;
            ast_node *i_inner = new ast_node();
            
            // Chain the nodes
            i_inner->computations = i_outer->computations;
            i_inner->children = i_outer->children;
            
            i_outer->computations.clear();
            i_outer->children.clear();
            i_outer->children.push_back(i_inner);
            
            i_inner->parent = i_outer;
            
            // Location of computations have changed, update computations_mapping
            for (computation_info& comp_info : i_inner->computations)
            {
                computations_mapping[comp_info.comp_ptr] = i_inner;
            }
            
            // Rename the nodes
            i_inner->name = i_outer->name + "_inner";
            i_outer->name = i_outer->name + "_outer";
            
            // Set lower and upper bounds
            i_outer->low_bound = 0;
            i_outer->up_bound = i_outer->get_extent() / opt.l0_fact - 1;
            
            i_inner->low_bound = 0;
            i_inner->up_bound = opt.l0_fact - 1;
            
            // Finalize unrolling
            i_inner->unrolled = true;
            i_inner->update_depth(i_outer->depth + 1);
        }
    }
}

void syntax_tree::transform_ast_by_paralellize(const optimization_info &info) {
    // Just sets the parallilezed tag to true
    info.node->parallelized = true;
}

syntax_tree* syntax_tree::copy_ast() const
{
    syntax_tree *ast = new syntax_tree();
    copy_and_return_node(*ast, nullptr);
    
    return ast;
}

ast_node* ast_node::copy_node() const
{
    ast_node *node = new ast_node();
    copy_and_return_node(node, nullptr);
    
    return node;
}

ast_node* syntax_tree::copy_and_return_node(syntax_tree& new_ast, ast_node *node_to_find) const
{
    ast_node *ret_node = nullptr;

    // Copy all root nodes
    for (ast_node *root : roots) 
    {
        ast_node *new_node = new ast_node();

        ast_node *tmp = root->copy_and_return_node(new_node, node_to_find);
        if (tmp != nullptr)
            ret_node = tmp;

        new_node->parent = nullptr;
        new_ast.roots.push_back(new_node);
    }

    // Copy AST data
    new_ast.fct = fct;
    new_ast.computations_list = computations_list;
    new_ast.buffers_list = buffers_list;
    new_ast.buffers_mapping = buffers_mapping;
    
    new_ast.iterators_json = iterators_json;
    new_ast.tree_structure_json = tree_structure_json;
    
    new_ast.evaluation = evaluation;
    new_ast.search_depth = search_depth;
    new_ast.nb_explored_optims = nb_explored_optims;
    new_ast.previous_optims = previous_optims;
    new_ast.new_optims = new_optims;

    // In new_ast, the location of computations have changed, so recompute computations_mapping
    new_ast.recompute_computations_mapping();    
    
    return ret_node;
}

ast_node* ast_node::copy_and_return_node(ast_node *new_node, ast_node *node_to_find) const
{
    ast_node *ret_node = nullptr;

    if (this == node_to_find)
        ret_node = new_node;

    // Recursively copy children
    for (ast_node *child : children)
    {
        ast_node *new_child = new ast_node();

        ast_node *tmp = child->copy_and_return_node(new_child, node_to_find);
        if (tmp != nullptr)
            ret_node = tmp;

        new_child->parent = new_node;
        new_node->children.push_back(new_child);
    }

    // Copy node data
    new_node->depth = depth;
    new_node->name = name;
    new_node->low_bound = low_bound;
    new_node->up_bound = up_bound;
    new_node->unrolled = unrolled;
    new_node->computations = computations;

    return ret_node;
}

void syntax_tree::recompute_computations_mapping()
{
    computations_mapping.clear();
    
    for (ast_node *root : roots)
        recompute_computations_mapping(root);
}
    
void syntax_tree::recompute_computations_mapping(ast_node *node)
{
    for (computation_info& comp_info : node->computations)
        computations_mapping[comp_info.comp_ptr] = node;
        
    for (ast_node *child : node->children)
        recompute_computations_mapping(child);
}

std::vector<optimization_info> syntax_tree::get_schedule() const
{
    std::vector<optimization_info> schedule = previous_optims;
    for (optimization_info const& optim_info : new_optims)
        schedule.push_back(optim_info);
            
    return schedule;
}

void syntax_tree::clear_new_optimizations()
{
    for (optimization_info const& optim_info : new_optims)
        previous_optims.push_back(optim_info);

    new_optims.clear();
}

ast_node* syntax_tree::find_node_by_level(tiramisu::computation *comp, int level)
{
    ast_node *node = computations_mapping[comp];
    int current_level = node->depth;
    
    while (current_level > level && node->parent != nullptr)
    {
        node = node->parent;
        current_level--;
    }
    
    return node;
}

std::vector<int> syntax_tree::get_shared_levels_extents() const
{
    std::vector<int> extents;
    if (roots.size() != 1)
        return extents;
        
    // Starting from the root, loop until we find a node with no children,
    // or with more than one child.
    ast_node *node = roots[0];
    while (true)
    {
        if (node->get_extent() <= 1)
            break;
            
        extents.push_back(node->get_extent());
        if (node->children.size() != 1 || node->computations.size() != 0)
            break;
            
        node = node->children[0];
    }
        
    return extents;
}

std::vector<int> syntax_tree::get_innermost_extents() const
{
    std::vector<int> extents;
    
    for (ast_node *node : roots)
        node->get_innermost_extents(extents);
    
    return extents;
}

void ast_node::get_innermost_extents(std::vector<int>& extents) const
{
    if (children.empty() && get_extent() > 1)
        extents.push_back(get_extent());
        
    for (ast_node *child : children)
        child->get_innermost_extents(extents);
}

std::vector<tiramisu::computation*> syntax_tree::get_innermost_computations()
{
    std::vector<tiramisu::computation*> comps;
    
    for (ast_node *node : roots)
        node->get_innermost_computations(comps);
        
    return comps;
}

void ast_node::get_innermost_computations(std::vector<tiramisu::computation*>& comps)
{
    if (children.empty() && get_extent() > 1)
    {
        for (computation_info& comp_info : computations)
            comps.push_back(comp_info.comp_ptr);
    }
    
    for (ast_node *child : children)
        child->get_innermost_computations(comps);
}

std::vector<ast_node*> syntax_tree::get_innermost_nodes() const
{
    std::vector<ast_node*> nodes;
    
    for (ast_node *node : roots)
        node->get_innermost_nodes(nodes);
        
    return nodes;
}

void ast_node::get_innermost_nodes(std::vector<ast_node*>& nodes)
{
    if (children.empty() && get_extent() > 1)
        nodes.push_back(this);
        
    for (ast_node *child : children)
        child->get_innermost_nodes(nodes);
}

ast_node* ast_node::get_root_node()
{
    ast_node *node = this;
    
    while (node->parent != nullptr)
        node = node->parent;
    
    return node;
}

ast_node* ast_node::get_leftmost_node()
{
    ast_node *node = this;
    
    while (!node->children.empty())
        node = node->children[0];
    
    return node;
}

ast_node* ast_node::get_rightmost_node()
{
    ast_node *node = this;
    
    while (!node->children.empty())
        node = node->children.back();
    
    return node;
}

int syntax_tree::get_buffer_id_from_computation_name(std::string comp_name)
{
    return get_buffer_id(buffers_mapping[comp_name]);
}

int syntax_tree::get_buffer_id(std::string const& buf_name) const
{
    auto it = std::find(buffers_list.begin(), buffers_list.end(), buf_name);
    if (it == buffers_list.end())
        return -1;
        
    return std::distance(buffers_list.begin(), it);
}

void ast_node::update_depth(int depth)
{
    if (get_extent() > 1)
        this->depth = depth;
    else
        this->depth = depth - 1;
    
    for (ast_node *child : children)
        child->update_depth(this->depth + 1);
}

void ast_node::get_all_computations(std::vector<tiramisu::computation*>& comps)
{
    for (computation_info& comp_info : computations)
        comps.push_back(comp_info.comp_ptr);
        
    for (ast_node *child : children)
        child->get_all_computations(comps);
}

int ast_node::get_loop_levels_chain_depth() const
{
    int ret = depth + 1;
    const ast_node *node = this;
    
    while (node->children.size() == 1 && node->computations.size() == 0 && !node->unrolled)
    {
        ret++;
        node = node->children[0];
    }
    
    return ret;
}

void syntax_tree::print_ast() const
{
    for (ast_node *root : roots)
	    root->print_node();
}

void syntax_tree::print_new_optims() const
{
    for (optimization_info optim: new_optims)
        print_optim(optim);
}

void syntax_tree::print_previous_optims() const
{
    for (optimization_info optim: previous_optims)
        print_optim(optim);
}

void ast_node::print_node() const
{
    if (get_extent() > 1)
    {
        for (int i = 0; i < depth; ++i)
            std::cout << "\t";

        std::cout << "for " << low_bound << " <= " << name << " < " << up_bound + 1 << " | " << unrolled;
        if (parallelized)
            std::cout << " | P";
        std::cout << std::endl;
    }
    
    for (computation_info const& comp_info : computations) 
    {
        for (int i = 0; i < depth + 1; ++i)
            std::cout << "\t";
            
        std::cout << comp_info.comp_ptr->get_name() << std::endl;
    }

    for (ast_node *child : children)
        child->print_node();
}

}
