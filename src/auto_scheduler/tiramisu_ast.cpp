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
        computations_list.push_back(comp);
        computations_mapping[comp] = node->get_leftmost_node();
    }

    // Order the computations by the order specified by the user using "after" commands
    order_computations();

    create_initial_isl_state();
    
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

    std::unordered_map<std::string,std::unordered_map<std::string,int>> sched_string;

    //Sort the scheduling graph (fct->sched_graph) into a list of tuples that represents the order of computations
    std::vector <tiramisu::computation*> rs_comps; //computations appearing on the right side of the ordering tuples
    std::vector <tiramisu::computation*> nrs_comps; //computations that never appear on the right side of the ordering tuples
    for (auto& sched_graph_node : fct->sched_graph)
        for (auto& sched_graph_child : sched_graph_node.second)
        {
            rs_comps.push_back(sched_graph_child.first);

            sched_string[sched_graph_node.first->get_name()][sched_graph_child.first->get_name()] = sched_graph_child.second;
        }
            

    for(auto* comp:this->computations_list)
    {
        bool found = false;
        for(auto& comp_rs:rs_comps)
        {
            if(comp_rs->get_name() == comp->get_name())
            {
                found = true;
                break;
            }
        }

        if(!found)
        {
            nrs_comps.push_back(comp);
        }
    }

    /*for (tiramisu::computation* comp: this->computations_list)
        if(std::find(rs_comps.begin(), rs_comps.end(), comp) == rs_comps.end()) // if comp never appears on the right side of the ordering tuples
            nrs_comps.push_back(comp);
            */

    std::vector<std::pair<tiramisu::computation*, std::unordered_map<tiramisu::computation*, int>>> sorted_sched_graph;

    //first computation
    tiramisu::computation* current_comp= nrs_comps[0];

    while (sched_string.find(current_comp->get_name()) != sched_string.end()) 
    {
        auto sched_graph_l = fct->sched_graph[fct->get_computation_by_name(current_comp->get_name())[0]];

        sorted_sched_graph.push_back(std::make_pair(current_comp, sched_graph_l));

        current_comp = sched_graph_l.begin()->first;
    }


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
            transform_ast_by_parallelism(opt);
            break;

        case optimization_type::SKEWING:
            transform_ast_by_skewing(opt);
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


        
        j_outer->parent = i_outer;
        i_inner->parent = j_outer;
        j_inner->parent = i_inner;
            
        // Rename the nodes
        i_inner->name = i_outer->name + "_inner";
        i_outer->name = i_outer->name + "_outer";
            
        j_inner->name = j_outer->name + "_inner";
        j_outer->name = j_outer->name + "_outer";
            
        // Set lower and upper bounds
        i_outer->low_bound = 0;
        i_outer->up_bound =  (int)ceil((double)i_outer->get_extent() / (double)opt.l0_fact) - 1;
            
        j_outer->low_bound = 0;
        j_outer->up_bound = (int)ceil((double)j_outer->get_extent() / (double)opt.l1_fact) - 1;
            
        i_inner->low_bound = 0;
        i_inner->up_bound = opt.l0_fact - 1;
            
        j_inner->low_bound = 0;
        j_inner->up_bound = opt.l1_fact - 1;

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
        i_outer->low_bound = 0;
        i_outer->up_bound = (int)ceil((double)i_outer->get_extent() / (double)opt.l0_fact) - 1;
            
        j_outer->low_bound = 0;
        j_outer->up_bound = (int)ceil((double)j_outer->get_extent() / (double)opt.l1_fact) - 1;
            
        k_outer->low_bound = 0;
        k_outer->up_bound = (int)ceil((double)k_outer->get_extent() / (double)opt.l2_fact) - 1;
            
        i_inner->low_bound = 0;
        i_inner->up_bound = opt.l0_fact - 1;
            
        j_inner->low_bound = 0;
        j_inner->up_bound = opt.l1_fact - 1;
            
        k_inner->low_bound = 0;
        k_inner->up_bound = opt.l2_fact - 1;

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
            //std::cout<<"Tiling3 loop names: "<<f<<" deapth of outer is:"<<std::to_string(i_outer->depth)<<" test : "<<outer_name_1<<" & "<<outer_name_2 ;
        }

    }

    node->update_depth(node->depth);

    recover_isl_states();
}


void syntax_tree::transform_ast_by_interchange(optimization_info const& opt)
{ 
    stage_isl_states();

    ast_node *node1 = opt.node;
    
    // Find the node to interchange with
    ast_node *node2 = node1;
    //for (int i = opt.l0; i < opt.l1; ++i)
    //    node2 = node2->children[0];
    while(node2->depth < opt.l1)
    {
        node2 = node2->children[0];
    }
            
    // Rename the two nodes
    std::string tmp_str =  node1->name;
    node1->name = node2->name;
    node2->name = tmp_str;
            
    int tmp_int  = node1->low_bound;
    node1->low_bound = node2->low_bound;
    node2->low_bound = tmp_int;
        
    tmp_int = node1->up_bound;
    node1->up_bound = node2->up_bound;
    node2->up_bound = tmp_int;


    /**
     * Applying to staging
    */
    std::vector<tiramisu::computation*> all_data;
        
    //collect computations to tile
    node2->get_all_computations(all_data);

    for(computation* info:all_data)
    {
        std::vector<std::string> loop_names = info->get_loop_level_names();
            
        std::string outer_name = loop_names[node1->depth];
        std::string inner_name = loop_names[node2->depth];
    
        info->interchange(var(outer_name),var(inner_name));
           
        std::string f = "";
        for(auto& str:loop_names)
        {
            f+=str+" ";
        }
        std::cout<<" vars "<<f<<" interchange : "<<outer_name<<" & "<<inner_name ;
    }

    recover_isl_states();
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

void syntax_tree::transform_ast_by_vectorization(const optimization_info &opt)
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
            node->vectorized = true;
            
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
            i_inner->vectorized = true;
            i_inner->update_depth(i_outer->depth + 1);
        }
    }
}


void syntax_tree::transform_ast_by_parallelism(const optimization_info &info) {
    // Just sets the parallilezed tag to true
    info.node->parallelized = true;
}

void syntax_tree::transform_ast_by_skewing(const optimization_info &info){
    stage_isl_states();

    ast_node *node_1 = info.node;
    ast_node *node_2 = node_1->children[0];

    int number_space_outer =   node_1->up_bound - node_1->low_bound ;
    int inner_space =  node_2->up_bound - node_2->low_bound ;

    std::string new_1 = "_skew_" + std::to_string(info.l0_fact) +"_"+std::to_string(info.l1_fact) ;
    std::string new_2 = "_skew";

    node_2->low_bound = 0;
    //node_1->low_bound = info.l0_fact * node_1->low_bound + info.l1_fact *node_2->low_bound; 
    node_1->low_bound = abs(info.l0_fact) * node_1->low_bound; 
    node_1->up_bound = node_1->low_bound + abs(info.l0_fact) * number_space_outer + abs(info.l1_fact) *inner_space ;
    node_2->up_bound =  (( number_space_outer * inner_space )/(node_1->up_bound - node_1->low_bound)) + 1;	

    std::vector<computation_info*> all_data;
        
    std::string outer_name = "";
    std::string inner_name = "";

    //collect computations to tile
    node_2->collect_all_computation(all_data);

    for(computation_info* info_comp:all_data)
    {
        std::vector<std::string> loop_names = info_comp->comp_ptr->get_loop_level_names();
            
        outer_name = loop_names[node_1->depth];
        inner_name = loop_names[node_1->depth+1];
    
        info_comp->comp_ptr->skew(var(outer_name),var(inner_name),
            info.l0_fact,info.l1_fact,
            var(outer_name+new_1),var(inner_name+new_2));

        if(info.l2_fact == -1)
        {//reversal on second loop
            info_comp->comp_ptr->loop_reversal(var(inner_name+new_2),var(inner_name+new_2+"_R"));
        }
           
        std::string f = "";
        for(auto& str:loop_names)
        {
            f+=str+" ";
        }
        std::cout<<" vars "<<f<<" Skewing : "<<outer_name<<" & "<<inner_name ;
    }

    node_1->name = outer_name+new_1;
    node_2->name = inner_name+new_2;

    node_1->skewed = true;
    node_2->skewed = true;
    
    node_1->transforme_accesses_with_skewing(info.l0_fact,info.l1_fact);

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
    std::unordered_map<tiramisu::computation *, int>>>(this->local_sched_graph.get());
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

    new_ast.local_sched_graph = local_sched_graph;
    
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
    if (true)//get_extent() > 1
    {
        for (int i = 0; i < depth; ++i)
            std::cout << "\t";

        std::cout<<this->depth <<"- "<< "for " << low_bound << " <= " << name << " < " << up_bound + 1 << " | " << unrolled <<" v "<<vectorized;
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
    
    std::cout<<"\n transformation map:"<<transformation_map;

    

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


void ast_node::get_previous_computations_foreign_nodes(std::vector<computation_info*>& vector) 
{
    ast_node * parent_ptr = nullptr;

    ast_node * current_node = this;

    parent_ptr = parent;

    while(parent_ptr != nullptr)
    {
        // add the parent computations
        for(computation_info& computation:parent->computations)
        {
            vector.push_back(&computation);
        }

        for(auto& child:parent->children)
        {
            if(child == current_node)
            {
                //current computation, there is no previous child available anymore.
                break;
            }
            else
            {
                child->collect_all_computation(vector);
            }
        }

        current_node = parent_ptr;
        parent_ptr = parent_ptr->parent;

    }

}

bool ast_node::have_similar_itr_domain(ast_node * other)
{
    int nb_itr1 = this->up_bound - this->low_bound;
    int nb_itr2 = other->up_bound - other->low_bound;

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
    ast_node * previous_ptr = previous_ptr;

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
        if(current->get_extent() > 0)
        { // not a dummy itr
            iterator_names.push_back(current->name);
        }
  

        current = current->parent;
    }

    std::reverse(iterator_names.begin(),iterator_names.end());

    return iterator_names;

}

void ast_node::move_computation_for_fusion(ast_node * adjusted_previous_node, computation_info * computation)
{

    /*
    adjusted_previous_node->computations.push_back(computation_info(*computation));
    
    auto iterator = this->computations.begin();
    while (iterator != this->computations.end())
    {
        if((*iterator).comp_ptr->get_name() == computation->comp_ptr->get_name())
        {
            iterator = this->computations.erase(iterator);
            break;
        }
    }

    // check if the node needs to be deleted
    if(this->computations.empty() && this->children.empty())
    {
        ast_node * current_node = this;
    }*/
    
}


int ast_node::get_node_loop_extent() const
{
    return this->up_bound - this->low_bound;
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

void syntax_tree::stage_isl_states() const
{
    for(ast_node* root:this->roots)
    {
        root->stage_isl_states();
    }
    this->fct->sched_graph.swap(*this->local_sched_graph);

}

void syntax_tree::recover_isl_states() const
{
    for(ast_node* root:this->roots)
    {
        root->recover_isl_states();
    }
    this->fct->sched_graph.swap(*this->local_sched_graph);
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

std::string syntax_tree::get_schedule_str()
{
    std::vector<optimization_info> schedule_vect = this->get_schedule();
    std::string schedule_str;

    for (auto optim: schedule_vect)
    {
        switch(optim.type) {
            case optimization_type::FUSION:
                schedule_str += "F(L"+std::to_string(optim.l0)+",L"+std::to_string(optim.l1)+"),";
                break;

            case optimization_type::UNFUSE:
                schedule_str += "F(L"+std::to_string(optim.l0)+",L"+std::to_string(optim.l1)+"),";
                break;

            case optimization_type::INTERCHANGE:
                schedule_str += "I(L"+std::to_string(optim.l0)+",L"+std::to_string(optim.l1)+"),";
                break;

            case optimization_type::TILING:
                if (optim.nb_l == 2)
                    schedule_str += "T2(L"+std::to_string(optim.l0)+",L"+std::to_string(optim.l1)+","+
                            std::to_string(optim.l0_fact)+","+std::to_string(optim.l1_fact)+"),";
                else if (optim.nb_l == 3)
                    schedule_str += "T3(L"+std::to_string(optim.l0)+",L"+std::to_string(optim.l1)+",L"+std::to_string(optim.l2)+","+
                            std::to_string(optim.l0_fact)+","+std::to_string(optim.l1_fact)+","+std::to_string(optim.l2_fact)+"),";
                break;

            case optimization_type::UNROLLING:
                schedule_str += "U(L"+std::to_string(optim.l0)+","+std::to_string(optim.l0_fact)+"),";
                break;

            case optimization_type::PARALLELIZE:
                schedule_str += "P(L"+std::to_string(optim.l0)+"),";
                break;

            case optimization_type::SKEWING:
                schedule_str += "S(L"+std::to_string(optim.l0)+",L"+std::to_string(optim.l1)+","+
                                std::to_string(optim.l0_fact)+","+std::to_string(optim.l1_fact)+"),";
                break;

            default:
                break;
        }
        if (!schedule_vect.empty())
            schedule_str.pop_back(); // remove last comma
    }

    return schedule_str;
}

bool syntax_tree::schedule_is_prunable()
{
    // Please note that this function currently only works for single computation programs
    // The following filtering rules are selected after a statistical analysis of inefficient schedule patterns on single computation programs

    assert(computations_list.size()==1 && "current implementation of syntax_tree::schedule_is_prunable() supports only single computation programs");  // assuming the ast has only one computation

    int original_ast_depth = computations_list[0]->get_loop_levels_number();
    std::string schedule_str = get_schedule_str();

    if (std::regex_search(schedule_str, std::regex(R"(P\(L2\)U\(L3,\d+\))")))
        return true;

    if (original_ast_depth==2)
        if (std::regex_search(schedule_str, std::regex(R"(P\(L1\)U)")))
            return true;

    if (original_ast_depth==3)
        if (std::regex_search(schedule_str, std::regex(R"(P\(L2\)(?:U|T2\(L0,L1))")))
            return true;

    return false;
}

bool syntax_tree::can_set_default_evaluation()
{
    // Please note that this function currently only works for single computation programs
    // The following filtering rules are selected after a statistical analysis of inefficient schedule patterns on single computation programs
    assert(computations_list.size()==1 && "current implementation of syntax_tree::schedule_is_prunable() supports only single computation programs");  // assuming the ast has only one computation

    int original_ast_depth = computations_list[0]->get_loop_levels_number();
    std::string schedule_str = get_schedule_str();

    //check if innermost loop is parallelized, if yes set the speedup to 0.001
    if (original_ast_depth==2)
        if (std::regex_search(schedule_str, std::regex(R"(P\(L1\)$)")))
        {
            evaluation =  std::atof(read_env_var("INIT_EXEC_TIME"))*1000;
            return true;
        }

    if (original_ast_depth==3)
        if (std::regex_search(schedule_str, std::regex(R"(P\(L2\)$)")))
        {
            evaluation =  std::atof(read_env_var("INIT_EXEC_TIME"))*1000;
            return true;
        }

    return false;
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
            ", \"depth\": " + std::to_string(this->exploration_depth) +
            ", \"evaluation\": " + std::to_string(this->evaluation) +
            ", \"children\": [";

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
