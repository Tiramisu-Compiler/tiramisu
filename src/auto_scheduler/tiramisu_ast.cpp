#include <tiramisu/auto_scheduler/ast.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <algorithm> //for searching in comps list
#include <iostream>
#include <string.h>
#include <tiramisu/auto_scheduler/search_method.h>



namespace tiramisu::auto_scheduler
{
    std::vector<optimization_type> generator_state::optimization_list;

    bool generator_state::initialized;
    
computation_info::computation_info(tiramisu::computation *comp, syntax_tree *ast)
    : comp_ptr(comp), iters(dnn_iterator::get_iterators_from_computation(*comp)),
      accesses(comp, iters.size(), comp->get_function()), buffer_nb_dims(iters.size()),
      nb_additions(0), nb_substractions(0), nb_multiplications(0), nb_divisions(0)
{
    get_info_from_expr(comp->get_expr());
    
    // Check if this computation is a reduction
    isl_map *storage_map = comp->access;
    buffer_nb_dims = isl_map_dim(storage_map, isl_dim_out);

    write_access_relation = isl_map_to_str(storage_map);
    storage_buffer_id = ast->get_buffer_id_from_computation_name(comp_ptr->name);

    data_type_str = str_from_tiramisu_type_primitive(comp_ptr->get_data_type());
    data_type_size = get_data_type_size();
    
    if (buffer_nb_dims < iters.size())
        is_reduction = true;
    else
        is_reduction = false;
        
    // Get buffer_id for the accesses of this computation
    for (dnn_access_matrix& matrix : accesses.accesses_list)
        matrix.buffer_id = ast->get_buffer_id_from_computation_name(matrix.buffer_name);
}

/*
computation_info::computation_info(computation_info const& reference)
:accesses(reference.accesses)
{
    std::cout<<"IN123";
    this->iters = reference.iters;
    std::cout<<"ZZZZ";

    this->buffer_nb_dims = reference.buffer_nb_dims;
    this->is_reduction = reference.is_reduction;
    this->nb_additions = reference.nb_additions;
    this->nb_divisions = reference.nb_divisions;
    this->nb_multiplications = reference.nb_multiplications;
    this->nb_substractions = reference.nb_substractions;
    this->storage_buffer_id = reference.storage_buffer_id;
    this->write_access_relation = reference.write_access_relation;
    this->data_type_str = reference.data_type_str;
    this->data_type_size = reference.data_type_size;

    std::cout<<"INFO_C";
    
}
*/

int computation_info::get_data_type_size(){
    if (comp_ptr->get_data_type()==tiramisu::p_boolean)
        return 1;
    // extract the the data size from the data type string
    std::string type_str = str_from_tiramisu_type_primitive(comp_ptr->get_data_type());
    size_t i = 0;
    for ( ; i < type_str.length(); i++ ){ if ( std::isdigit(type_str[i]) ) break; }
    std::string data_size_str = type_str.substr(i, type_str.length() - i );
    int data_size = std::atoi(type_str.c_str())/8;
    return data_size;
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

void computation_info::set_accesses_changes_with_skewing(int first_node_depth,int alpha,int beta,int gamma,int sigma)
{
    this->accesses.modify_accesses_by_skewing(first_node_depth,alpha,beta,gamma,sigma);
}

// ---------------------------------------------------------------------------- //

syntax_tree::syntax_tree(tiramisu::function *fct)
    : fct(fct)
{
    local_sched_graph =  std::make_shared<std::unordered_map<tiramisu::computation *,
    std::unordered_map<tiramisu::computation *, int>>>(fct->sched_graph);
    
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
        computations_list.push_back(comp);// at this level computations are stored in their declaration order, this list will be sorted by computing order after calling order_computations()
        computations_mapping[comp] = node->get_leftmost_node();
    }

    // Order the computations by the order specified by the user using "after" commands
    order_computations();

    create_initial_isl_state();

    // INITIALIZE the generator states as uninitialized
    generator_state::initialized = false;
    
    // Get the JSON representation of this AST iterators
    for (ast_node *node : roots)
        evaluate_by_learning_model::represent_iterators_from_nodes(node, iterators_json);
        
    iterators_json.pop_back();
    
    // Get the JSON representation of this tree
    tree_structure_json = evaluate_by_learning_model::get_tree_structure_json(*this);
}

int get_number_of_iterators_from_set(isl_set *set){
    assert(set != NULL);
    
    assert(isl_set_is_empty(set) == isl_bool_false);
    isl_ast_build *ast_build;
    isl_ctx *ctx = isl_set_get_ctx(set);
    ast_build = isl_ast_build_alloc(ctx);

    // Create identity map for set.
    isl_space *sp = isl_set_get_space(set);
    isl_map *sched = isl_map_identity(isl_space_copy(isl_space_map_from_set(sp)));
    sched = isl_map_set_tuple_name(sched, isl_dim_out, "");
    isl_map *map =
        isl_map_intersect_domain(
            isl_map_copy(sched),
            isl_set_copy(set));
    int length = isl_map_dim(map, isl_dim_out);
    isl_id_list *iterators = isl_id_list_alloc(ctx, length);

    for (int i = 0; i < length; i++)
    {
        std::string name;
        if (isl_set_has_dim_name(set, isl_dim_set, i) == true)
            name = isl_set_get_dim_name(set, isl_dim_set, i);
        else
            name = generate_new_variable_name();
        isl_id *id = isl_id_alloc(ctx, name.c_str(), NULL);
        iterators = isl_id_list_add(iterators, id);
    }
    ast_build = isl_ast_build_set_iterators(ast_build, iterators);

    isl_ast_node *node = isl_ast_build_node_from_schedule_map(ast_build, isl_union_map_from_map(map));

    // handle the case when the actual number of for loops is less than the target unrolled loop
    isl_ast_node* node1 = node;
    int cpt = 0;
    bool stop = false;
    // calculate the number of for loops 
    
    while(!stop){
        if(isl_ast_node_get_type(node1) == isl_ast_node_for ){
            cpt++;
            node1 = isl_ast_node_for_get_body(node1);
        }
        else if(isl_ast_node_get_type(node1) == isl_ast_node_user ){
            stop = true;
        }
        else if(isl_ast_node_get_type(node1) == isl_ast_node_if){
            node1 = isl_ast_node_if_get_then(node1);
        }             
    }
    return cpt;
}

ast_node::ast_node(tiramisu::computation *comp, syntax_tree *ast)
{
    std::vector<ast_node*> nodes;

    // Get computation iterators
    isl_set *iter_domain = comp->get_iteration_domain();
    int nb_iterators = isl_set_dim(iter_domain, isl_dim_set);

    // The fist node is the one created by this constructor
    this->depth = 0;
    this->name = isl_set_get_dim_name(iter_domain, isl_dim_set, 0) + comp->get_name();
    
    this->low_bound = utility::get_bound(iter_domain, 0, false).to_str();


    this->up_bound = utility::get_bound(iter_domain, 0, true).to_str();

    nodes.push_back(this);
        
    // Create the other nodes, one for each iterator
    for (int i = 1; i < nb_iterators; ++i)
    {
        ast_node *node = new ast_node();
        
        node->depth = i;
        node->name = isl_set_get_dim_name(iter_domain, isl_dim_set, i) + comp->get_name();
        node->low_bound = utility::get_bound(iter_domain, i, false).to_str();
        node->up_bound = utility::get_bound(iter_domain, i, true).to_str();
        
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

    //Sort the scheduling graph (fct->sched_graph) into a list of tuples that represents the order of computations
    std::vector <tiramisu::computation*> rs_comps; //computations appearing on the right side of the ordering tuples
    std::vector <tiramisu::computation*> nrs_comps; //computations that never appear on the right side of the ordering tuples
    
    for (auto& sched_graph_node : fct->sched_graph)
        for (auto& sched_graph_child : sched_graph_node.second)
            rs_comps.push_back(sched_graph_child.first);

    for (tiramisu::computation* comp: this->computations_list)
        if(std::find(rs_comps.begin(), rs_comps.end(), comp) == rs_comps.end()) // if comp never appears on the right side of the ordering tuples
            nrs_comps.push_back(comp);

    std::vector<std::pair<tiramisu::computation*, std::unordered_map<tiramisu::computation*, int>>> sorted_sched_graph;
    std::vector<tiramisu::computation *> ordered_computations_list; // a list that will contain computations on their computing order

    for (tiramisu::computation* comp: nrs_comps){
        tiramisu::computation* current_comp= comp;
        ordered_computations_list.push_back(current_comp);
        while (fct->sched_graph.find(current_comp) != fct->sched_graph.end()) {
            auto sched_graph_l = fct->sched_graph[current_comp];
            if (sched_graph_l.empty()) // if empty
                break; //not sure if it's the right way to do it
            sorted_sched_graph.push_back(std::make_pair(current_comp, sched_graph_l));
            current_comp = sched_graph_l.begin()->first;
            ordered_computations_list.push_back(current_comp);
        }
    }
    this->computations_list = ordered_computations_list; // replace the computation list with the ordered one
    
    // We use the sorted scheduling graph to construct the computations AST
    for (auto& sched_graph_node : sorted_sched_graph)
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
                    
                    ast_node *new_node = new ast_node();

                    new_node->depth = child_comp_ast_node->depth;
                    new_node->name = "dummy_iter";
                    new_node->low_bound = "0";
                    new_node->up_bound = "0";
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
ast_node* syntax_tree::get_last_shared_parent(ast_node* node1, ast_node* node2) const
{
    std::vector<ast_node*> parent_list_1;
    std::vector<ast_node*> parent_list_2;

    // construct the list of ancestors of both nodes
    ast_node* parent_node = node1;
    while (parent_node != nullptr)
    {
        if (parent_node->name != "dummy_iter") // ignore dummy iterators
            parent_list_1.push_back(parent_node);
        parent_node = parent_node->parent;
    }
    std::reverse(parent_list_1.begin(), parent_list_1.end());

    parent_node = node2;
    while (parent_node != nullptr)
    {
        if (parent_node->name != "dummy_iter") // ignore dummy iterators
            parent_list_2.push_back(parent_node);
        parent_node = parent_node->parent;
    }
    std::reverse(parent_list_2.begin(), parent_list_2.end());

    ast_node* latest_shared_parent = nullptr;
    for (int i = 0; i< std::min(parent_list_1.size(), parent_list_2.size()); i++)
        if (parent_list_1[i]==parent_list_2[i])
            latest_shared_parent = parent_list_2[i];
        else
            break;

    return latest_shared_parent;
}

void syntax_tree::transform_ast()
{
    if (new_optims.size() == 0)
        return ;
        
    transform_ast(new_optims.back());
}

void syntax_tree::transform_ast(optimization_info const& opt)
{
    //TODOF why is transform ast by fusion desactivated
    switch(opt.type)
    {
        /*case optimization_type::FUSION:
            transform_ast_by_fusion(opt);
            break;
            
        case optimization_type::UNFUSE:
            transform_ast_by_unfuse(opt);
            break;
            */
        case optimization_type::MATRIX:
            transform_ast_by_matrix(opt);
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
            transform_ast_by_parallelism(opt);
            break;

        case optimization_type::SKEWING:
            transform_ast_by_skewing(opt);
            break;

        case optimization_type::SHIFTING:
            transform_ast_by_shifting(opt);
            break;

        default:
            break;
    }
    recompute_computations_mapping();
}

//void syntax_tree::transform_ast_by_fusion(optimization_info const& opt)
//{
//    std::vector<ast_node*> *tree_level;
//
//    if (opt.node->parent != nullptr)
//        tree_level = &opt.node->parent->children;
//    else
//        tree_level = &roots;
//
//    ast_node *node1 = (*tree_level)[opt.l0];
//    ast_node *node2 = (*tree_level)[opt.l1];
//
//    for (ast_node *child : node2->children)
//        node1->children.push_back(child);
//
//    for (computation_info& comp_info : node2->computations)
//    {
//        node1->computations.push_back(comp_info);
//        computations_mapping[comp_info.comp_ptr] = node1;
//    }
//
//    tree_level->erase(tree_level->begin() + opt.l1);
//}
//
//void syntax_tree::transform_ast_by_unfuse(optimization_info const& opt)
//{
//    ast_node *unfuse_node, *shared_node;
//
//    int i = 0;
//    shared_node = roots[0];
//
//    while (shared_node->children.size() == 1)
//    {
//        if (i == opt.l0)
//            unfuse_node = shared_node;
//
//        shared_node = shared_node->children[0];
//        i++;
//    }
//
//    std::vector<ast_node*> shared_node_children = shared_node->children;
//    ast_node *removed_node = unfuse_node->children[0];
//    unfuse_node->children.clear();
//
//    for (ast_node *node : shared_node_children)
//    {
//        shared_node->children.clear();
//        shared_node->children.push_back(node);
//
//        unfuse_node->children.push_back(removed_node->copy_node());
//    }
//
//    tree_structure_json = evaluate_by_learning_model::get_tree_structure_json(*this);
//}
/**
 * Multiply two matrices 
 */
std::vector<std::vector<int>>  multiply_mats(const std::vector<std::vector<int>> & m1, const std::vector<std::vector<int>> & m2)
{
    if (m1.size() == 0 || m2.size() == 0){
        throw std::invalid_argument( "At least one of the matrices to be multiplied is empty" );
    }
    if( m1.at(0).size() != m2.size()){
        throw std::invalid_argument( "Matrices don't have compatible sizes for multiplication" );
    }
        
    std::vector<std::vector<int>> result(m1.size(), std::vector<int>(m2.at(0).size()));

    for(std::size_t row = 0; row < result.size(); ++row) {
        for(std::size_t col = 0; col < result.at(0).size(); ++col) {
            for(std::size_t inner = 0; inner < m2.size(); ++inner) {
                result.at(row).at(col) += m1.at(row).at(inner) * m2.at(inner).at(col);
            }
        }
    }
    return result;
}
std::vector<std::vector<std::string>> get_bounds(std::vector<ast_node *> shared_nodes){
    
    std::vector<std::vector<std::string>> bounds_mat;
    for (int i=0; i <shared_nodes.size();i++){
        std::vector<std::string> vec;
        // Updating the node using isl_ast_map 
        vec.push_back(shared_nodes[i]->low_bound);
        vec.push_back(shared_nodes[i]->up_bound);
        bounds_mat.push_back(vec);
        vec.clear();
    }
    return bounds_mat;

}
/**
 * Update the loop bounds of a list of nodes
 */
void update_node(std::vector<ast_node *> shared_nodes, std::vector<std::vector<int>> bounds_mat){
    
    for (int i=0; i <shared_nodes.size();i++){
        // Updating the node using transformed bounds matrix 
        shared_nodes[i]->low_bound = bounds_mat[i][0];
        shared_nodes[i]->up_bound = bounds_mat[i][1];
    }
    
}
void syntax_tree::transform_ast_by_matrix(const optimization_info &opt)
{
    /**
     * Applying to staging
    */  
    stage_isl_states(); 
    std::vector<tiramisu::computation *> all_data;
    std::vector<ast_node *> all_nodes;
    opt.node->get_all_nodes(all_nodes);
    std::vector<ast_node*> to_change_nodes;
    std::vector<ast_node*> temp_to_change;
    std::vector<std::vector<int>> temp_matrix;
    for(ast_node* node1 : all_nodes ){
        for(computation_info info : node1->computations)
        {   
            info.comp_ptr->matrix_transform(opt.matrix);    
        }
    } 
    recover_isl_states();
    
}

void syntax_tree::transform_ast_by_tiling(optimization_info const& opt)
{
    ast_node *node = opt.node;

    stage_isl_states();
    
    // 2 level tiling
    if (opt.nb_l == 2)
    {
        // Create the new loop structure
        ast_node *i_outer = node;
        ast_node *j_outer = node->children[0];
            
        ast_node *i_inner = new ast_node();
        ast_node *j_inner = new ast_node();
            
        // Chain the nodes

        i_inner->children.push_back(j_inner);
        //i_outer->children[0] = j_outer;

        for(auto& states:j_outer->children)
        {
            j_inner->children.push_back(states);
        }

        for(auto states:j_outer->isl_states)
        {
            j_inner->isl_states.push_back(states);
        }
        

        j_inner->computations = j_outer->computations;

        j_outer->children.clear();
        j_outer->isl_states.clear();
        j_outer->computations.clear();

        j_outer->children.push_back(i_inner);
        for (auto child:j_inner->children)
            child->parent = j_inner;

        
        j_outer->parent = i_outer;
        i_inner->parent = j_outer;
        j_inner->parent = i_inner;
            
        // Rename the nodes
        i_inner->name = i_outer->name + "_inner";
        i_outer->name = i_outer->name + "_outer";
            
        j_inner->name = j_outer->name + "_inner";
        j_outer->name = j_outer->name + "_outer";
            
        // Set lower and upper bounds
        i_outer->low_bound = "0";
        i_outer->up_bound =  i_outer->get_extent() + "/" + std::to_string((double)opt.l0_fact - 1);
            
        j_outer->low_bound = "0";
        j_outer->up_bound = j_outer->get_extent() + "/" + std::to_string((double)opt.l1_fact - 1);
            
        i_inner->low_bound = "0";
        i_inner->up_bound = std::to_string(opt.l0_fact - 1);
            
        j_inner->low_bound = "0";
        j_inner->up_bound = std::to_string(opt.l1_fact - 1);

        /**
         * Applying tiling to the nodes schedule and states
        */
        std::vector<computation_info*> all_data;
        
        //collect computations to tile
        j_inner->collect_all_computation(all_data);

        for(computation_info* info:all_data)
        {
            std::vector<std::string> loop_names = info->comp_ptr->get_loop_level_names();
            
            std::string outer_name = loop_names[i_outer->depth];
            std::string inner_name = loop_names[i_outer->depth+1];

            std::string ii_outer = outer_name+"_outer";
            std::string jj_outer = inner_name+"_outer";
            std::string ii_inner = outer_name+"_inner";
            std::string jj_inner = inner_name+"_inner";

            std::string f = "";
            for(auto& str:loop_names)
            {
                f+=str+" ";
            }
            
            
            info->comp_ptr->tile(var(outer_name),var(inner_name)
                                ,opt.l0_fact,opt.l1_fact,
                                var(ii_outer),var(jj_outer),var(ii_inner),var(jj_inner));
           
           
        }


    }
        
    // 3 level tiling
    else if (opt.nb_l == 3)
    {
        // Create the new loop structure
        ast_node *i_outer = node;
        ast_node *j_outer = node->children[0];
        ast_node *k_outer = j_outer->children[0];
            
        ast_node *i_inner = new ast_node();
        ast_node *j_inner = new ast_node();
        ast_node *k_inner = new ast_node();
 
        // Chain the nodes

        i_inner->children.push_back(j_inner);
        j_inner->children.push_back(k_inner);

        for(auto& states:k_outer->children)
        {
            k_inner->children.push_back(states);
        }

        for(auto states:k_outer->isl_states)
        {
            k_inner->isl_states.push_back(states);
        }
        
        k_inner->computations = k_outer->computations;

        k_outer->children.clear();
        k_outer->isl_states.clear();
        k_outer->computations.clear();

        k_outer->children.push_back(i_inner);
        for (auto child:k_inner->children)
            child->parent = k_inner;
        
        j_outer->parent = i_outer;
        k_outer->parent = j_outer;
        i_inner->parent = k_outer;
        j_inner->parent = i_inner;
        k_inner->parent = j_inner;

            
        // Rename the nodes
        i_inner->name = i_outer->name + "_inner";
        i_outer->name = i_outer->name + "_outer";
            
        j_inner->name = j_outer->name + "_inner";
        j_outer->name = j_outer->name + "_outer";
            
        k_inner->name = k_outer->name + "_inner";
        k_outer->name = k_outer->name + "_outer";
            
        // Set lower and upper bounds
        i_outer->low_bound = "0";
        i_outer->up_bound = i_outer->get_extent() + "/" + std::to_string((double)opt.l0_fact - 1);
            
        j_outer->low_bound = "0";
        j_outer->up_bound = j_outer->get_extent() + "/" + std::to_string((double)opt.l1_fact - 1);
            
        k_outer->low_bound = "0";
        k_outer->up_bound = k_outer->get_extent() + "/" + std::to_string((double)opt.l2_fact - 1);
            
        i_inner->low_bound = "0";
        i_inner->up_bound = std::to_string(opt.l0_fact - 1);
            
        j_inner->low_bound = "0";
        j_inner->up_bound = std::to_string(opt.l1_fact - 1);
            
        k_inner->low_bound = "0";
        k_inner->up_bound = std::to_string(opt.l2_fact - 1);

        /**
         * Applying to staging
        */
        std::vector<computation_info*> all_data;
        
        //collect computations to tile
        j_inner->collect_all_computation(all_data);

        for(computation_info* info:all_data)
        {
            std::vector<std::string> loop_names = info->comp_ptr->get_loop_level_names();
            
            std::string outer_name_1 = loop_names[i_outer->depth];
            std::string outer_name_2 = loop_names[i_outer->depth+1];
            std::string inner_name_3 = loop_names[i_outer->depth+2];

            std::string ii_outer = outer_name_1+"_outer";
            std::string jj_outer = outer_name_2+"_outer";
            std::string kk_outer = inner_name_3+"_outer";
            std::string ii_inner = outer_name_1+"_inner";
            std::string jj_inner = outer_name_2+"_inner";
            std::string kk_inner = inner_name_3+"_inner";
            
            info->comp_ptr->tile(var(outer_name_1),var(outer_name_2),var(inner_name_3)
                                ,opt.l0_fact,opt.l1_fact,opt.l2_fact,
                                var(ii_outer),var(jj_outer),var(kk_outer),var(ii_inner),var(jj_inner),var(kk_inner));
           
            std::string f = "";
            for(auto& str:loop_names)
            {
                f+=str+" ";
            }
        }

    }

    node->update_depth(node->depth);

    recover_isl_states();
}


void syntax_tree::transform_ast_by_interchange(optimization_info const& opt)
{ 
    // stage_isl_states();

    // ast_node *node1 = opt.node;
    
    // // Find the node to interchange with
    // ast_node *node2 = node1;
    // //for (int i = opt.l0; i < opt.l1; ++i)
    // //    node2 = node2->children[0];
    // while(node2->depth < opt.l1)
    // {
    //     node2 = node2->children[0];
    // }
            
    // // Rename the two nodes
    // std::string tmp_str =  node1->name;
    // node1->name = node2->name;
    // node2->name = tmp_str;
            
    // int tmp_int  = node1->low_bound;
    // node1->low_bound = node2->low_bound;
    // node2->low_bound = tmp_int;
        
    // tmp_int = node1->up_bound;
    // node1->up_bound = node2->up_bound;
    // node2->up_bound = tmp_int;


    // /**
    //  * Applying to staging
    // */
    // std::vector<tiramisu::computation*> all_data;
        
    // //collect computations to tile
    // node2->get_all_computations(all_data);

    // for(computation* info:all_data)
    // {
    //     std::vector<std::string> loop_names = info->get_loop_level_names();
            
    //     std::string outer_name = loop_names[node1->depth];
    //     std::string inner_name = loop_names[node2->depth];
    
    //     info->interchange(var(outer_name),var(inner_name));
           
    //     std::string f = "";
    //     for(auto& str:loop_names)
    //     {
    //         f+=str+" ";
    //     }
    //     //std::cout<<" vars "<<f<<" interchange : "<<outer_name<<" & "<<inner_name ;
    // }

    // recover_isl_states();
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
        // TODOF check the case where the bound is N  
        if (!node->unrolled)
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
            i_outer->low_bound = "0";
            i_outer->up_bound = i_outer->get_extent() + "/" + std::to_string(opt.l0_fact - 1);
            
            i_inner->low_bound = "0";
            i_inner->up_bound = std::to_string(opt.l0_fact - 1);
            
            // Finalize unrolling
            i_inner->unrolled = true;
            i_inner->update_depth(i_outer->depth + 1);
        }
    }
}

void syntax_tree::transform_ast_by_vectorization(const optimization_info &opt)
{
    // std::vector<ast_node*> nodes_list;
    
    // // Apply unrolling on the node provided by opt
    // if (opt.l0 != -1)
    //     nodes_list = {opt.node};
        
    // // Apply unrolling on every innermost loop level
    // else
    //     nodes_list = get_innermost_nodes();
    
    // for (ast_node *node : nodes_list)
    // {
    //     if (node->get_extent() <= opt.l0_fact)
    //         node->vectorized = true;
            
    //     else 
    //     {
    //         // Create the new loop structure
    //         ast_node *i_outer = node;
    //         ast_node *i_inner = new ast_node();
            
    //         // Chain the nodes
    //         i_inner->computations = i_outer->computations;
    //         i_inner->children = i_outer->children;
            
    //         i_outer->computations.clear();
    //         i_outer->children.clear();
    //         i_outer->children.push_back(i_inner);
            
    //         i_inner->parent = i_outer;
            
    //         // Location of computations have changed, update computations_mapping
    //         for (computation_info& comp_info : i_inner->computations)
    //         {
    //             computations_mapping[comp_info.comp_ptr] = i_inner;
    //         }
            
    //         // Rename the nodes
    //         i_inner->name = i_outer->name + "_inner";
    //         i_outer->name = i_outer->name + "_outer";
            
    //         // Set lower and upper bounds
    //         i_outer->low_bound = 0;
    //         i_outer->up_bound = i_outer->get_extent() / opt.l0_fact - 1;
            
    //         i_inner->low_bound = 0;
    //         i_inner->up_bound = opt.l0_fact - 1;
            
    //         // Finalize unrolling
    //         i_inner->vectorized = true;
    //         i_inner->update_depth(i_outer->depth + 1);
    //     }
    // }
}


void syntax_tree::transform_ast_by_parallelism(const optimization_info &info) {
    // Just sets the parallilezed tag to true
    info.node->parallelized = true;
}

void syntax_tree::transform_ast_by_skewing(const optimization_info &info){
    // stage_isl_states();

    // ast_node *node_1 = info.node;
    // ast_node *node_2 = node_1->children[0];

    // int number_space_outer =   node_1->up_bound - node_1->low_bound ;
    // int inner_space =  node_2->up_bound - node_2->low_bound ;

    // std::string new_1 = "_skew_" + std::to_string(info.l0_fact) +"_"+std::to_string(info.l1_fact) ;
    // std::string new_2 = "_skew";

    // node_2->low_bound = 0;
    // //node_1->low_bound = info.l0_fact * node_1->low_bound + info.l1_fact *node_2->low_bound; 
    // node_1->low_bound = abs(info.l0_fact) * node_1->low_bound; 
    // node_1->up_bound = node_1->low_bound + abs(info.l0_fact) * number_space_outer + abs(info.l1_fact) *inner_space ;
    // node_2->up_bound =  (( number_space_outer * inner_space )/(node_1->up_bound - node_1->low_bound)) + 1;	

    // std::vector<computation_info*> all_data;
        
    // std::string outer_name = "";
    // std::string inner_name = "";

    // //collect computations to tile
    // node_2->collect_all_computation(all_data);

    // for(computation_info* info_comp:all_data)
    // {
    //     std::vector<std::string> loop_names = info_comp->comp_ptr->get_loop_level_names();
            
    //     outer_name = loop_names[node_1->depth];
    //     inner_name = loop_names[node_1->depth+1];
    
    //     info_comp->comp_ptr->skew(var(outer_name),var(inner_name),
    //         info.l0_fact,info.l1_fact,
    //         var(outer_name+new_1),var(inner_name+new_2));

    //     if(info.l2_fact == -1)
    //     {//reversal on second loop
    //         info_comp->comp_ptr->loop_reversal(var(inner_name+new_2),var(inner_name+new_2+"_R"));
    //     }
           
    //     std::string f = "";
    //     for(auto& str:loop_names)
    //     {
    //         f+=str+" ";
    //     }
    //     //std::cout<<" vars "<<f<<" Skewing : "<<outer_name<<" & "<<inner_name ;
    // }

    // node_1->name = outer_name+new_1;
    // node_2->name = inner_name+new_2;

    // node_1->skewed = true;
    // node_2->skewed = true;
    
    // node_1->transforme_accesses_with_skewing(info.l0_fact,info.l1_fact);

    // recover_isl_states();
}


void syntax_tree::transform_ast_by_shifting(const optimization_info &info){
    stage_isl_states();

    ast_node *node_1 = info.node;
    node_1->shifted = true;
    info.comps[0]->shift(info.l0,info.l0_fact);

    recover_isl_states();
}


syntax_tree* syntax_tree::copy_ast() const
{
    syntax_tree *ast = new syntax_tree();
    copy_and_return_node(*ast, nullptr);
    
    return ast;
}

void syntax_tree::create_new_sched_graph()
{
    this->local_sched_graph = std::make_shared<std::unordered_map<tiramisu::computation *,
    std::unordered_map<tiramisu::computation *, int>>>(this->fct->sched_graph);
}

void syntax_tree::stage_local_sched_graph() const
{
    this->fct->sched_graph.swap(*this->local_sched_graph);
}

void syntax_tree::recover_local_sched_graph() const
{
    this->fct->sched_graph.swap(*this->local_sched_graph);
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
    new_ast.nb_explored_matrices = nb_explored_matrices;
    new_ast.ast_search_phase = ast_search_phase;
    new_ast.fct = fct;
    new_ast.computations_list = computations_list;
    new_ast.buffers_list = buffers_list;
    new_ast.buffers_mapping = buffers_mapping;

    new_ast.local_sched_graph = local_sched_graph;
    
    new_ast.iterators_json = iterators_json;
    new_ast.tree_structure_json = tree_structure_json;
    
    new_ast.evaluation = evaluation;
    new_ast.search_depth = search_depth;
    new_ast.nb_explored_optims = nb_explored_optims;
    new_ast.nb_explored_matrices = nb_explored_matrices;
    new_ast.previous_optims = previous_optims;
    new_ast.new_optims = new_optims;

    new_ast.search_state = search_state;
    new_ast.refresh_states();

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
    new_node->skewed = skewed;
    new_node->parallelized = parallelized;
    new_node->computations = computations;

    //new_node->isl_states = isl_states;
    for(auto state:isl_states)
    {
        new_node->isl_states.push_back(state);
    }

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

    if (node->name == "dummy_iter")
        node = node->parent; // because dummy iterators are not counted as a loop level
    
    while (current_level > level && node->parent != nullptr)
    {
        node = node->parent;
        current_level--;
    }
    
    return node;
}

std::vector<std::string> syntax_tree::get_shared_levels_extents() const
{
    std::vector<std::string> extents;
    if (roots.size() != 1)
        return extents;
        
    // Starting from the root, loop until we find a node with no children,
    // or with more than one child.
    ast_node *node = roots[0];
    while (true)
    {
        if (node->get_extent()  == "0-0+1")
            break;
            
        extents.push_back(node->get_extent());
        if (node->children.size() != 1 || node->computations.size() != 0)
            break;
            
        node = node->children[0];
    }
    return extents;
}

std::vector<std::string> syntax_tree::get_innermost_extents() const
{
    std::vector<std::string> extents;
    
    for (ast_node *node : roots)
        node->get_innermost_extents(extents);
    
    return extents;
}

void ast_node::get_innermost_extents(std::vector<std::string>& extents) const
{
    if (children.empty() && get_extent() != "0-0+1")
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
    if (children.empty() && get_extent() != "0-0+1")
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

void syntax_tree::delete_duplicated_node_recursively(ast_node * node)
{
    if(node->depth == 0)
    {
        // delete a root level
        auto position = std::find(this->roots.begin(), this->roots.end(), node);
        if (position != this->roots.end())
        {
            this->roots.erase(position);
        }

        delete node;
    }
    else
    {
        // delete a node 
        auto position = std::find(node->parent->children.begin(), node->parent->children.end(), node);
        if (position != node->parent->children.end())
        {
            node->parent->children.erase(position);
        }

        if((node->parent->children.size() == 0) && (node->parent->computations.size() == 0))
        {
            delete_duplicated_node_recursively(node->parent);
        }
        else
        {
            delete node;
        }
    }       
            
}

void syntax_tree::move_in_computation(ast_node * new_node, tiramisu::computation * comp_ptr)
{
    ast_node * old_node = this->computations_mapping[comp_ptr];

    if(old_node->depth == new_node->depth)
    {
        new_node->computations.push_back(old_node->computations[0]);
        new_node->isl_states.push_back(old_node->isl_states[0]);

    }
    else
    {
        // old node have more depth than new_node(which is the node in the ast where we will fuze)
        ast_node * link_node = old_node->find_node_by_depth(new_node->depth);
        //{#}
        ast_node * new_leaf = old_node->new_branch_leaf(link_node);

        ast_node * direct_link = new_leaf->find_node_by_depth(link_node->depth + 1);

        new_node->children.push_back(direct_link);
        direct_link->parent = new_node;

    }

    old_node->computations.erase(old_node->computations.begin());
    old_node->isl_states.erase(old_node->isl_states.begin());

    if((old_node->computations.size() == 0) && (old_node->children.size() == 0))
    {
        this->delete_duplicated_node_recursively(old_node);
    }

    this->recompute_computations_mapping(); 
}

void ast_node::get_innermost_nodes(std::vector<ast_node*>& nodes)
{
    if (children.empty() && get_extent() != "0-0+1")
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

void syntax_tree::get_shared_nodes_from_outermost(std::vector<ast_node*>& shared) const
{
    if(this->roots.size() == 1)
    {
        shared.push_back(roots[0]);
        roots[0]->get_shared_nodes_from_outermost(shared);
    }
}

void ast_node::get_shared_nodes_from_outermost(std::vector<ast_node*>& shared) const
{
    if(this->children.size() == 1)
    {
        shared.push_back(children[0]);
        children[0]->get_shared_nodes_from_outermost(shared);
    }
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
 
    this->depth = depth;
   
    for (ast_node *child : children)
        child->update_depth(this->depth + 1);
}
void ast_node::get_node_computations(std::vector<tiramisu::computation*>& comps)
{
    for (computation_info& comp_info : computations)
        comps.push_back(comp_info.comp_ptr);
    
}
void ast_node::get_all_computations(std::vector<tiramisu::computation*>& comps)
{
    for (computation_info& comp_info : computations)
        comps.push_back(comp_info.comp_ptr);
        
    for (ast_node *child : children)
        child->get_all_computations(comps);
}
void ast_node::get_all_nodes(std::vector<ast_node*>& nodes)
{
    
    nodes.push_back(this);   
    for (ast_node *child : children)
        child->get_all_nodes(nodes);
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
    if (name != "dummy_iter")//get_extent() > 1
    {
        for (int i = 0; i < depth; ++i)
            std::cout << "\t";

        std::cout<<this->depth <<"- "<< "for " << low_bound << " <= " << name << " < " << up_bound + "1" << " | " << unrolled ;
        if (parallelized)
            std::cout << " | P";

        if (vectorized)
            std::cout << " | V";
        
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

state_computation::state_computation(computation * origin)
{
    this->staging_computation =  origin;
    this->current_schedule = isl_map_copy(origin->get_schedule());
    is_state_staged = false;
}


state_computation::state_computation(state_computation const& reference)
{
    this->staging_computation =  reference.get_computation_unstated();
    this->current_schedule = isl_map_copy(reference.get_inner_isl_map());
    is_state_staged = false;
}

/*
state_computation::state_computation(state_computation * reference)
{
    this->staging_computation =  reference->get_computation_unstated();
    this->current_schedule = isl_map_copy(this->staging_computation->get_schedule());
    is_state_staged = false;
}*/


void state_computation::move_schedule_to_staging()
{
    isl_map * tmp = this->staging_computation->get_schedule();

    this->staging_computation->set_schedule(this->current_schedule);

    this->current_schedule = tmp;

    is_state_staged = true;
}


void state_computation::recover_schedule_from_staging()
{
    isl_map * tmp = this->staging_computation->get_schedule();

    this->staging_computation->set_schedule(this->current_schedule);

    this->current_schedule = tmp;

    is_state_staged = false;
}

computation * state_computation::get_computation_staged()
{
    this->move_schedule_to_staging();
    return this->staging_computation;
}

computation * state_computation::get_computation_unstated() const
{
     return this->staging_computation;
}
isl_map * state_computation::get_inner_isl_map() const
{ 
    return this->current_schedule; 
}  

bool state_computation::is_this_state_staged() const
{
    return is_state_staged;
}

void ast_node::print_isl_states() const
{
    for(auto& info:this->isl_states)
    {
        std::cout<<(std::string(isl_map_to_str(info.get_inner_isl_map()))) ;
    }

    for(ast_node* child:children)
    {
        child->print_isl_states();
    }
}

void ast_node::print_computations_accesses() const
{
    std::cout<<"\n";
    for(auto const& comp:this->computations)
    {
        comp.accesses.print_all_access();
    }
    for(ast_node* child:this->children)
    {
        child->print_computations_accesses();
    }
}

void ast_node::transforme_accesses_with_skewing(int a,int b)
{
    /*
        compute isl Map of transformation here
    */
    int f_i = a;
    int f_j = b;
  
    int gamma = 0;
    int sigma = 1;

    bool found = false;

    if ((f_j == 1) || (f_i == 1)){

        gamma = f_i - 1;
        sigma = 1;
        /* Since sigma = 1  then
            f_i - gamma * f_j = 1 & using the previous condition :
             - f_i = 1 : then gamma = 0 (f_i-1) is enough
             - f_j = 1 : then gamma = f_i -1  */
    }
    else
    { 
        if((f_j == - 1) && (f_i > 1))
        {
            gamma = 1;
            sigma = 0;    
        }    
        else
        {   //General case : solving the Linear Diophantine equation & finding basic solution (sigma & gamma) for : f_i* sigma - f_j*gamma = 1 
            int i =0;
            while((i < 100) && (!found))
            {
                if (((sigma * f_i ) % abs(f_j)) ==  1){
                            found = true;
                }
                else{
                    sigma ++;
                    i++;
                }
            };

            if(!found){
                // Detect infinite loop and prevent it in case where f_i and f_j are not prime between themselfs
                ERROR(" Error in solving the Linear Diophantine equation f_i* sigma - f_j*gamma = 1  ", true);
            }

            gamma = ((sigma * f_i) - 1 ) / f_j;
        }
    }

    std::string transformation_map = "{[i,j]->["+std::to_string(f_i)+"*i"+std::to_string(f_j)+"*j ,"
                                                +std::to_string(gamma)+"*i"+std::to_string(sigma)+"*j]}";
    
    // std::cout<<"\n transformation map:"<<transformation_map;

    

    this->set_accesses_changes_with_skewing(this->depth,f_i,f_j,gamma,sigma);
}

void ast_node::set_accesses_changes_with_skewing(int first_node_depth,int alpha,int beta,int gamma,int sigma)
{
    for(auto& comp:this->computations)
    {
        comp.set_accesses_changes_with_skewing(first_node_depth,alpha,beta,gamma,sigma);
    }
    for(ast_node* child:children)
    {
        child->set_accesses_changes_with_skewing(first_node_depth,alpha,beta,gamma,sigma);
    }
}

void ast_node::create_initial_states()
{
    for(auto& info:this->computations)
    {
        this->isl_states.push_back(state_computation(info.comp_ptr));
    }

    for(ast_node* child:children)
    {
        child->create_initial_states();
    }
}

void ast_node::erase_isl_states()
{
    
    this->isl_states.clear();

    for(ast_node* child:children)
    {
        child->erase_isl_states();
    }
}


void ast_node::stage_isl_states()
{
    for(auto& obj:this->isl_states)
    {
        obj.move_schedule_to_staging();
    }
    for(ast_node* child:children)
    {
        child->stage_isl_states();
    }
}


void ast_node::recover_isl_states()
{
    for(auto& obj:this->isl_states)
    {
        obj.recover_schedule_from_staging();
    }
    for(ast_node* child:children)
    {
        child->recover_isl_states();
    }

}

void ast_node::collect_all_computation(std::vector<computation_info*>& vector)
{
    for(auto& info:this->computations)
    {
        vector.push_back(&info);
    }

    for(ast_node* child:children)
    {
        child->collect_all_computation(vector);
    }
}

bool is_number(const std::string& s)
{
    return !s.empty() && std::find_if(s.begin(), 
        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

// TODOF check if both bounds are ints: same behaviour, otherwise check if strings are equal for now?
bool ast_node::have_similar_itr_domain(ast_node * other)
{
    // check whether both bounds are ints to be able to compare. If that's not the case return true, generate fusion and leave the decision to the legality check
    if(is_number(this->low_bound) && is_number(this->up_bound) && is_number(other->low_bound) && is_number(other->up_bound)){
        int nb_itr1 = stoi(this->up_bound) - stoi(this->low_bound);
        int nb_itr2 = stoi(other->up_bound) - stoi(other->low_bound);

        if((nb_itr1 != 0) && (nb_itr2 != 0))
        {
            // nb_itr1/nb_itr is 1 or 0
            if(((nb_itr2/nb_itr1) < 2) && ((nb_itr1/nb_itr2) < 2))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }else{
        return true;
    }
    
}

bool ast_node::is_candidate_for_fusion(ast_node * other)
{
    if(other->depth != this->depth)
    {
        return false;
    }

    ast_node * current_ptr = this;
    ast_node * other_ptr = other;

    while(current_ptr != other_ptr) // nullptr == nullptr is the possible last exit manner. 
    {
        if(current_ptr->have_similar_itr_domain(other_ptr))
        {
            current_ptr = current_ptr->parent;
            other_ptr = other_ptr->parent;
        }
        else
        {
            return false;
        }
    }

    return true;
}

std::pair<ast_node *,ast_node*> ast_node::get_possible_fusion_candidate(ast_node * previous_node)
{

    ast_node * current_ptr = this;
    ast_node * previous_ptr = previous_node;

    while(current_ptr->depth != previous_ptr->depth)
    {
        if(current_ptr->depth > previous_ptr->depth)
        {
            current_ptr = current_ptr->parent;
        }
        else
        {
            previous_ptr = previous_ptr->parent;
        }
    }

    return std::make_pair(previous_ptr,current_ptr);

}

std::vector<std::string> ast_node::get_all_iterators()
{
    std::vector<std::string> iterator_names;
    ast_node * current = this;

    while(current != nullptr)
    {
        if(current->get_extent() != "0-0+1")
        { // not a dummy itr             
            iterator_names.push_back(current->name);
        }
  

        current = current->parent;
    }

    std::reverse(iterator_names.begin(),iterator_names.end());
    return iterator_names;

}

ast_node * ast_node::find_node_by_depth(int depth_val)
{
    assert(depth_val > -1);
    assert(depth_val <= this->depth);

    if(this->depth == depth_val)
    {
        return this;
    }
    else
    {
        return this->parent->find_node_by_depth(depth_val);
    }
}


void syntax_tree::get_previous_computations(std::vector<computation*>& result,ast_node*& node, int computation_index)
{
    
    if(computation_index == -1)
    {
        computation_index = node->computations.size();
    }

    for(int i=computation_index-1; i>=0; i--)
    {
        result.push_back(node->computations[i].comp_ptr);
    }

    if(node->parent != nullptr)
    {
        int superior_index = -1;

        //localize the index of current child in the parent
        for(int k=0; k<node->parent->children.size(); k++)
        {
            if(node->parent->children[k] == node)
            {
                superior_index = k;
                break;
            }
        }

        if(superior_index == -1)
        {
            ERROR("INDEX PARENT NOT FOUND ",1);
        }

        // collect computation from children before the current child

        for(int j=0; j<superior_index; j++)
        {
            node->parent->children[j]->get_all_computations(result);
        }

        this->get_previous_computations(result,node->parent,-1);
    }
    else
    {// the case of the root levels
        int root_index = -1;

        //localize the index of current root in the root vector
        for(int k=0; k<this->roots.size(); k++)
        {
            if(this->roots[k] == node)
            {
                root_index = k;
                break;
            }
        }

        if(root_index == -1)
        {
            ERROR("INDEX PARENT NOT FOUND ",1);
        }

        for(int i=0; i<root_index; i++)
        {
            this->roots[i]->get_all_computations(result);
        }
    }
}

ast_node * ast_node::copy_local_node(bool copy_first_computation)
{
    ast_node * new_node = new ast_node();
    new_node->parent = parent;
    new_node->depth = depth;
    new_node->name = name;
    new_node->low_bound = low_bound;
    new_node->up_bound = up_bound;
    new_node->unrolled = unrolled;
    new_node->skewed = skewed;
    new_node->parallelized = parallelized;
    if(copy_first_computation)
    {
        new_node->computations = {computations[0]};
        new_node->isl_states = {isl_states[0]};
    }
    

    return new_node;
}


ast_node * ast_node::new_branch_leaf(ast_node * shared_node)
{
    std::vector<ast_node*> copy_target;

    std::vector<ast_node*> new_branch;
    ast_node * current = this;

    while(current != shared_node)
    {
        copy_target.push_back(current);
    }

    for(int i=0; i<copy_target.size();i++)
    {
        if(i == (copy_target.size() - 1))
        {
            new_branch.push_back(copy_target[i]->copy_local_node(true));
        }
        else
        {
            new_branch.push_back(copy_target[i]->copy_local_node(false));
        }
    }

    for(int i=0; i < new_branch.size()-1;i++)
    {
       new_branch[i]->parent = new_branch[i+1];
    }

    return new_branch[0];


}


std::string ast_node::get_node_loop_extent() const
{
    return this->up_bound + "-" + this->low_bound;
}

void syntax_tree::print_isl_states() const
{
    for(ast_node* root:this->roots)
    {
        root->print_isl_states();
    }

}

void syntax_tree::create_initial_isl_state() const
{
    for(ast_node* root:this->roots)
    {
        root->create_initial_states();
    }

}

void syntax_tree::recreate_isl_state() const
{
    for(ast_node* root:this->roots)
    {
        // erase all isl states
        root->erase_isl_states();
    }

    this->create_initial_isl_state();
}

void syntax_tree::stage_isl_states() const
{
    for(ast_node* root:this->roots)
    {
        root->stage_isl_states();
    }
    this->stage_local_sched_graph();

}

void syntax_tree::recover_isl_states() const
{
    for(ast_node* root:this->roots)
    {
        root->recover_isl_states();
    }
    this->recover_local_sched_graph();
}

bool syntax_tree::ast_is_legal() const
{
    
    stage_isl_states();

    this->fct->prepare_schedules_for_legality_checks(true);

    bool result = this->fct->check_legality_for_function();

    recover_isl_states();
    
    return result;

}

void syntax_tree::print_computations_accesses() const
{
    for(ast_node* root:this->roots)
    {
        root->print_computations_accesses();
    }
}
int syntax_tree::get_computation_index(tiramisu::computation *comp)
{
    auto it = find(computations_list.begin(), computations_list.end(), comp);

    assert(it != computations_list.end()); // element has to be found
    int index = it - computations_list.begin();
    return index;
}
std::string syntax_tree::get_schedule_str()
{
    std::vector<optimization_info> schedule_vect = this->get_schedule();
    std::string schedule_str;
    bool transformed_by_matrix = false;
    int start_matrices = -1;
    int first_matrix = true;
    if(schedule_vect.size()<1) return schedule_str;
    std::vector<std::vector<std::vector<int>>> matrices(this->get_computations().size());
    std::vector<int> first_time(this->get_computations().size());
    
    for(int i=0;i<first_time.size();i++) first_time.at(i)=1;
    for (auto optim: schedule_vect)
    {
        std::string comps_list_str="{";
        
        for (auto comp: optim.comps)
        {
            comps_list_str+= "C"+std::to_string(get_computation_index(comp))+",";    
        }
        comps_list_str.pop_back(); //remove the last comma
        comps_list_str+="}";

        switch(optim.type) {
            case optimization_type::FUSION:
                schedule_str += "F("+comps_list_str+",L"+std::to_string(optim.l0)+")";
                break;
            
            case optimization_type::MATRIX:
                // In the case of the matrix transformation, we start by saving the transformation matrices
                // We add the final matrices after this loop
                transformed_by_matrix = true;
                if(first_matrix){
                    start_matrices = schedule_str.size();
                    first_matrix = false;
                }
                for (auto comp: optim.comps)
                {
                    int index = get_computation_index(comp);
                    if(first_time.at(index)){ 
                        first_time.at(index) = 0;
                        matrices.at(index) = optim.matrix;
                    }else{
                        if(optim.matrix.size()<matrices.at(index).size()){
                            std::vector <  std::vector<int> >  matrix(matrices.at(index).size());
                            for(int l = 0; l<matrix.size(); l++){
                                matrix.at(l)= std::vector<int>(matrices.at(index).size());
                                for(int c = 0; c<matrix.size(); c++){
                                    if (l!=c ){
                                        matrix.at(l).at(c) = 0;
                                    }else{
                                        matrix.at(l).at(c) = 1;
                                    }
                                }
                            }

                            for(int i=0 ; i<optim.matrix.size();i++){
                                for(int j=0 ; j<optim.matrix.size();j++){
                                    matrix.at(i).at(j)= optim.matrix.at(i).at(j);
                                }
                            }
                            matrices.at(index) = multiply_mats(matrix, matrices.at(index)); 
                        }else{
                            
                            matrices.at(index) = multiply_mats(optim.matrix, matrices.at(index) );
                        }
                              
                    }          
                }   
                break;

            case optimization_type::SHIFTING:
                schedule_str += "Sh("+comps_list_str+",L"+std::to_string(optim.l0)+","+std::to_string(optim.l0_fact)+")";
                break;


//            case optimization_type::UNFUSE:
//                schedule_str += "F(L"+std::to_string(optim.l0)+",L"+std::to_string(optim.l1)+"),";
//                break;

            case optimization_type::INTERCHANGE:
                schedule_str += "I("+comps_list_str+",L"+std::to_string(optim.l0)+",L"+std::to_string(optim.l1)+")";
                break;

            case optimization_type::TILING:
                if (optim.nb_l == 2)
                    schedule_str += "T2("+comps_list_str+",L"+std::to_string(optim.l0)+",L"+std::to_string(optim.l1)+","+
                            std::to_string(optim.l0_fact)+","+std::to_string(optim.l1_fact)+")";
                else if (optim.nb_l == 3)
                    schedule_str += "T3("+comps_list_str+",L"+std::to_string(optim.l0)+",L"+std::to_string(optim.l1)+",L"+std::to_string(optim.l2)+","+
                            std::to_string(optim.l0_fact)+","+std::to_string(optim.l1_fact)+","+std::to_string(optim.l2_fact)+")";
                break;

            case optimization_type::UNROLLING:
                schedule_str += "U("+comps_list_str+",L"+std::to_string(optim.l0)+","+std::to_string(optim.l0_fact)+")";
                break;

            case optimization_type::PARALLELIZE:
                schedule_str += "P("+comps_list_str+",L"+std::to_string(optim.l0)+")";
                break;

            case optimization_type::SKEWING:
                schedule_str += "S("+comps_list_str+",L"+std::to_string(optim.l0)+",L"+std::to_string(optim.l1)+","+
                                std::to_string(optim.l0_fact)+","+std::to_string(optim.l1_fact)+")";
                break;

            default:
                break;
        }
       
    
              
    }
    if(transformed_by_matrix){
        for(int index = 0; index<matrices.size();index++){
            schedule_str.insert(start_matrices,"M(");
            start_matrices+=2;
            schedule_str.insert(start_matrices,"{C"+std::to_string(index)+"}");
            start_matrices+=4;
            schedule_str.insert(start_matrices,",");
            start_matrices+=1;

            for(int i = 0; i < matrices.at(index).size(); i++){    
                for(int j = 0; j< matrices.at(index).size(); j++){ 
                    schedule_str.insert(start_matrices,std::to_string(matrices.at(index).at(i).at(j)));
                    start_matrices+=std::to_string(matrices.at(index).at(i).at(j)).size();
                    if(!(i==matrices.at(index).size()-1 && j==matrices.at(index).size()-1)){schedule_str.insert(start_matrices,",");start_matrices+=1;}
                }
            }
            schedule_str.insert(start_matrices,")");
            start_matrices+=1;
        }    
    }
    
    return schedule_str;
}

bool syntax_tree::schedule_is_prunable()
{
    // The following filtering rules are selected after a statistical analysis of inefficient schedule patterns
    std::string schedule_str = get_schedule_str();

    std::vector<int> depths(this->get_computations().size());
    int min ;
    for (optimization_info optim: new_optims){
            if(optim.type == optimization_type::MATRIX){
                for(int i=0;i<optim.comps.size();i++){
                    depths.at(this->get_computation_index(optim.comps.at(i))) = optim.matrix.size();
                }  
            }
            min= *std::min_element(depths.begin(), depths.end());
            // Only get the depths from identity matrices
            if(min>0) break;
        }
    std::string reg = "";
    for(int i=0;i<depths.size();i++){
        reg += ".*P\\(\\{(C[0-9],)*C" + std::to_string(i) + "(,C[0-9])*\\},L"+ std::to_string(depths.at(i)-1)+"\\)U.*|"; 
    }
    if(depths.size()>0){
        reg.pop_back();
    }

    
    std::regex regexp(reg);
    
    if (std::regex_search(schedule_str, regexp))
        return true;
    
    reg = "";
    for(int i=0;i<depths.size();i++){
        reg += ".*P\\(\\{(C[0-9],)*C" + std::to_string(i) + "(,C[0-9])*\\},L"+ std::to_string(depths.at(i)-1)+"\\)T2\\(\\{(C[0-9],)*C" + std::to_string(i)+ "(,C[0-9])*\\},L"+std::to_string(depths.at(i)-3)+",L"+std::to_string(depths.at(i)-2)+".*|"; 
    }
    if(depths.size()>0){
        reg.pop_back();
    }
    
    std::regex regexpt(reg);
    
    if (std::regex_search(schedule_str, regexpt))
        return true;

    return false;
}
bool syntax_tree::ast_is_prunable()
{
    std::vector<int> optims(this->get_computations().size());
    for (optimization_info optim: new_optims){
            if(optim.type == optimization_type::MATRIX){
                for(int i=0;i<optim.comps.size();i++){
                    optims.at(this->get_computation_index(optim.comps.at(i))) += 1;
                }  
        }
    }
    
    int max = *std::max_element(optims.begin(), optims.end());
    // compare the maximum number of applied matrices on any computation with MAX_MAT_DEPTH+1 (the plus 1 is to take into considiration the identity matrix)
    if(max>MAX_MAT_DEPTH+1) return true;

    return false;
}
bool syntax_tree::can_set_default_evaluation()
{
    std::vector<int> depths(this->get_computations().size());
    int min;
    for (optimization_info optim: new_optims){
            
            if(optim.type == optimization_type::MATRIX){
                for(int i=0;i<optim.comps.size();i++){
                    depths.at(this->get_computation_index(optim.comps.at(i))) = optim.matrix.size();
                }  
            }

            min= *std::min_element(depths.begin(), depths.end());
            // Only get the depths from identity matrices
            if(min>0) break;
        }
    
    std::string schedule_str = get_schedule_str();
    
    //check if innermost loop is parallelized, if yes set the speedup to 0.001 
    std::string reg = "";
    for(int i=0;i<depths.size();i++){
        reg += ".*P\\(\\{(C[0-9],)*C" + std::to_string(i) + "(,C[0-9])*\\},L"+ std::to_string(depths.at(i)-1)+"\\)$|"; 
    }
    if(depths.size()>0){
        reg.pop_back();
    }
    std::regex regexp(reg);
    

    if (std::regex_search(schedule_str,regexp))
    {
        evaluation =  std::atof(read_env_var("INIT_EXEC_TIME"))*1000;
        return true;
    }
    
    

    return false;
}

std::vector<ast_node*> ast_node::collect_heads_of_ast(int allowed_splits, ast_node* current)
{
    std::vector<ast_node*> result1;

    if(allowed_splits < 0)
    {
        return result1;
    }

    if(current->get_extent() != "0-0+1")
    {
        result1.push_back(current);
    }

    ast_node * current_itr = current;

    while(current_itr->children.size() == 1)
    {
        current_itr = current_itr->children[0];
    }

    if(current_itr->children.size() == 0)
    {
        return result1;
    }
    else
    {
        for(ast_node * child : current_itr->children)
        {
            auto res_i = collect_heads_of_ast(allowed_splits-1,child);

            if(res_i.size() > 0)
            {
                result1.insert(result1.end(), res_i.begin(), res_i.end());
            }
        }

        return result1;
    }
}

std::vector<ast_node*> ast_node::collect_shared_nodes_from_head()
{
    std::vector<ast_node*> result;

    ast_node * current = this;

    result.push_back(this);
    // if a node has one or more computation or more than one child we stop
    // in the case where we find a computation the branch must stop here since we cannot apply transformations with other branches 
    while(current->children.size() == 1 && current->computations.size()==0)
    {
        current = current->children[0];
        result.push_back(current);
    }

    return result;
}

bool ast_node::is_optimized_by_tag()
{
    return this->parallelized||this->vectorized||this->unrolled;
}

void collect_computation_states_for_fusion(ast_node * node, std::vector<std::pair<ast_node*,int>>& fusion_candidates)
{

    for(int i=0;i<node->computations.size(); i++)
    {
        fusion_candidates.push_back(std::make_pair(node,i));
    }
    for(ast_node * child:node->children)
    {
        collect_computation_states_for_fusion(child,fusion_candidates);
    }
}

std::vector<std::pair<ast_node*,int>> syntax_tree::compute_search_space_states(optimization_type optimization) const
{

    std::vector<ast_node*> result_vect;

    std::vector<std::pair<ast_node*,int>> heads;
    // heads : 
    // represent the node along with the computation's index in the computations's list in that
    // specific node.

    int allowed_splits = 1;
    // number of allowed splits when defining the target nodes to explore.
    // when allowed_split=0 we will explore optimizations on shared nodes only.
                 

    switch (optimization)
    {
        case optimization_type::FUSION:

            {
                for(ast_node * node: this->roots)
                {
                    collect_computation_states_for_fusion(node,heads);
                }
            }

        break;

        // case of other optimisations
        default:

        {

            if(roots.size() > 1)
            {
                for(ast_node * root : roots)
                {
                    auto res_i = ast_node::collect_heads_of_ast(allowed_splits-1,root);

                    result_vect.insert(result_vect.end(), res_i.begin(), res_i.end());
                }
            }
            else
            {
                result_vect = ast_node::collect_heads_of_ast(allowed_splits,roots[0]);
            }

            for(auto& node:result_vect)
            {
                heads.push_back(std::make_pair(node,0)); // why always first comp?
            }

        }        

        break;

    }

    return heads;

}

void syntax_tree::initialize_search_space_optimizations(std::vector<optimization_type> optimizations)
{
    generator_state::initialized = true;
    generator_state::optimization_list = optimizations;
    
    auto first_optim_alternatives = this->compute_search_space_states(generator_state::optimization_list[0]);
    this->search_state.set_new_heads(first_optim_alternatives);
}

bool syntax_tree::is_search_space_empty()
{
    return this->search_state.is_search_space_empty();
}

void syntax_tree::refresh_states()
{
    auto optim_alternatives 
                = this->compute_search_space_states(
                    generator_state::optimization_list[this->search_state.optimization_index]
                    );
    this->search_state.set_new_heads(optim_alternatives);
}


std::pair<ast_node*,int> syntax_tree::get_current_optimization_target()
{
    return this->search_state.get_current_head();
}

std::pair<ast_node*,int> syntax_tree::get_previous_optimization_target()
{
    assert(this->search_state.current_index > 0);

    return this->search_state.target_ast_heads[this->search_state.current_index-1];
}

optimization_type syntax_tree::get_current_optimization_type() const
{
    return generator_state::optimization_list[this->search_state.optimization_index]; 
}
void syntax_tree::move_to_next_optimization_target()
{

    this->search_state.increment_index();


    if(this->search_state.current_index >= this->search_state.target_ast_heads.size())
    {
        this->search_state.optimization_index++;

        if(this->search_state.optimization_index < generator_state::optimization_list.size())
        {
            auto optim_alternatives 
                = this->compute_search_space_states(
                    generator_state::optimization_list[this->search_state.optimization_index]
                    );
            this->search_state.set_new_heads(optim_alternatives);
            this->search_state.current_index = 0;
        }
    }
}
void syntax_tree::move_to_next_head()
{

    this->search_state.increment_index();


    if(this->search_state.current_index >= this->search_state.target_ast_heads.size())
    {
        //this->search_state.optimization_index++;

        if(this->search_state.optimization_index < generator_state::optimization_list.size())
        {
            auto optim_alternatives 
                = this->compute_search_space_states(
                    generator_state::optimization_list[this->search_state.optimization_index]
                    );
            this->search_state.set_new_heads(optim_alternatives);
            this->search_state.current_index = 0;
        }else if(generator_state::optimization_list.size()==1 && generator_state::optimization_list.at(0) == optimization_type::MATRIX){
            
            auto optim_alternatives   = this->compute_search_space_states( optimization_type::MATRIX);
            
            this->search_state.set_new_heads(optim_alternatives);
            this->search_state.current_index = 0;
        }
    }
}

bool syntax_tree::optim_already_applied_on_comp(tiramisu::computation *comp, tiramisu::auto_scheduler::optimization_type opt_type) {
    for (auto opt_info:new_optims) {
        if (opt_info.type != opt_type) //if different optimization, skip to next
            continue;
        if (std::find(opt_info.comps.begin(), opt_info.comps.end(), comp) != opt_info.comps.end()) // if comp in computations list
            return true;
    }
    return false;
}

bool syntax_tree::optim_already_applied_on_comps(const std::vector<tiramisu::computation *>comp_list, tiramisu::auto_scheduler::optimization_type opt_type) {
    for (auto comp:comp_list)
        if (optim_already_applied_on_comp(comp,opt_type))
            return true;
    return false;
}

bool generator_state::is_current_optimization_fully_explored()
{
    if(this->current_index < this->target_ast_heads.size()-1)
    {
        return false;
    }
    else
    {
        return true;
    }
}


bool generator_state::can_move_to_next_optimization()
{
    if(this->optimization_index < (generator_state::optimization_list.size() - 1))
    {
        return true;
    }
    else
    {
        return false;
    }
}


void generator_state::set_new_heads(std::vector<std::pair<ast_node*,int>>& optim_heads)
{
    this->target_ast_heads = optim_heads;
}


std::pair<ast_node*,int> generator_state::get_current_head()
{
    return this->target_ast_heads[this->current_index];
}


void generator_state::increment_index()
{
    this->current_index++;
}



bool generator_state::is_search_space_empty()
{
    if(this->current_index < this->target_ast_heads.size())
    {
        
        return false;
    }
    else
    {// we are in the last optimization
        if(this->optimization_index < generator_state::optimization_list.size())
        {
            
            return false;
        }
        else
        {
            
            return true;
        }
    }
}


candidate_trace::candidate_trace(syntax_tree *ast, int candidate_id)
{
    this->evaluation = ast->evaluation;
    this->exploration_depth = ast->search_depth+1;
    this->candidate_id = candidate_id;
    this->schedule_str = ast->get_schedule_str();
}

candidate_trace::~candidate_trace() {
    for (candidate_trace *child_candidate:this->child_candidates)
        delete child_candidate;
}

void candidate_trace::add_child_path(syntax_tree *ast, int candidate_id)
{
    candidate_trace *child_candidate = new candidate_trace(ast, candidate_id);
    this->child_candidates.push_back(child_candidate);
    this->child_mappings.insert({ast, child_candidate});
}

std::string candidate_trace::get_exploration_trace_json()
{
    std::string trace_json = "{ \"id\": "+std::to_string(this->candidate_id)+
            ", \"schedule\": \"" + this->schedule_str + "\"" +
            ", \"depth\": " + std::to_string(this->exploration_depth);
            if (std::isfinite(this->evaluation)){ // the evaluation is not finite mean that the schedule didn't run
                trace_json+=", \"evaluation\": " + std::to_string(this->evaluation) +
            ", \"children\": [";
            }else{
                trace_json+=", \"evaluation\": null , \"children\": [";
            }

    if (!this->child_candidates.empty())
    {
        trace_json += "\n";
        for (auto child_candidate: this->child_candidates)
            trace_json += child_candidate->get_exploration_trace_json() + ",";
        trace_json.pop_back(); // remove last comma
    }

    trace_json += "]}\n";

    return trace_json;
}

int candidate_trace::get_candidate_id() const {
    return candidate_id;
}

}
