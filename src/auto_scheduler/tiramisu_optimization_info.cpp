#include <tiramisu/auto_scheduler/optimization_info.h>
#include <tiramisu/auto_scheduler/ast.h>
#include <tiramisu/block.h>

namespace tiramisu::auto_scheduler
{

void parallelize_outermost_levels(std::vector<tiramisu::computation*> const& comps_list)
{
    for (tiramisu::computation *comp : comps_list)
        comp->tag_parallel_level(0);
}

void unroll_innermost_levels(std::vector<tiramisu::computation*> const& comps_list, int unroll_fact)
{
    std::vector<int> innermost_indices; 
    
    // For each computation, get the indice of its innermost loop level.
    for (tiramisu::computation *comp : comps_list)
        innermost_indices.push_back(comp->get_loop_levels_number() - 1);
                
    // Apply unrolling to innermost loop levels.
    for (int i = 0; i < innermost_indices.size(); ++i)
        comps_list[i]->unroll(innermost_indices[i], unroll_fact);
}

void apply_optimizations(syntax_tree const& ast)
{
    // Check ast.h for the difference between ast.previous_optims and ast.new_optims
    for (optimization_info const& optim_info : ast.previous_optims){
        apply_optimizations(optim_info);
    }
        
        
    for (optimization_info const& optim_info : ast.new_optims){
            apply_optimizations(optim_info);
    }
        

    // Fusion is a particular case, and we use apply_fusions() to apply it.
    // apply_fusions() uses the structure of the AST to correctly order the computations.
    // The structure of the program is only represented abstractly even when fusion is not applied.
    // We need to call apply_fusion on all ASTs to go from the abstract representation of the structure to changing the actual order of computations using the .after command
    apply_fusions(ast);

    // Parallelization needs to be applied after the other transformations in order to have the accurate loop depth of
    // the tagged ast_nodes
    apply_parallelization(ast);
}

void apply_optimizations(optimization_info const& optim_info)
{
    // tiramisu::block can be used to apply the same optimization to a set of computations
    tiramisu::block block(optim_info.comps);
                                
    switch (optim_info.type)
    {
        case optimization_type::TILING:
            if (optim_info.nb_l == 2)
                block.tile(optim_info.l0, optim_info.l1, 
                           optim_info.l0_fact, optim_info.l1_fact);
                
            else if (optim_info.nb_l == 3)
                block.tile(optim_info.l0, optim_info.l1, optim_info.l2,
                           optim_info.l0_fact, optim_info.l1_fact, optim_info.l2_fact);
            break;
                
        case optimization_type::INTERCHANGE:
            block.interchange(optim_info.l0, optim_info.l1);
            break;
                
        case optimization_type::UNROLLING:
            // Apply unrolling on the level indicated by l0
            if (optim_info.l0 != -1)
                block.unroll(optim_info.l0, optim_info.l0_fact);
                
            // Apply unrolling on all innermost levels
            else
                unroll_innermost_levels(optim_info.comps, optim_info.l0_fact);
            break;

        case optimization_type::VECTORIZATION:
            // Apply unrolling on the level indicated by l0
            
            for(auto const& computation: optim_info.comps)
            {
                computation->vectorize(optim_info.l0, optim_info.l0_fact);
                
            }
            break;

        case optimization_type::SKEWING:
            block.skew(optim_info.l0, optim_info.l1, optim_info.l0_fact, optim_info.l1_fact);
            if(optim_info.l2_fact == -1)
                for(auto const& computation: optim_info.comps)
                    computation->loop_reversal(optim_info.l1);
            break;
        case optimization_type::MATRIX:
            block.matrix_transform(optim_info.matrix);
            break;
        case optimization_type::SHIFTING:
            for(auto const& computation: optim_info.comps)
            {
                computation->shift(optim_info.l0,optim_info.l0_fact);   
            }
            
            break;

        default:
            break;
    }
}

    void apply_fusions(syntax_tree const& ast)
    {
        // Use the "after" scheduling command to replicate the structure of the AST
        // on the computations order.

        tiramisu::computation *previous_comp = nullptr;

        for (tiramisu::computation* current_comp : ast.computations_list) // iterate over the ordered computations list
        {
            if (previous_comp == nullptr) // if current comp is the first computation
            {
                previous_comp = current_comp;
                continue;
            }

            ast_node *current_comp_node = ast.computations_mapping.at(current_comp);
            ast_node *previous_comp_node = ast.computations_mapping.at(previous_comp);
            ast_node *last_shared_parent = ast.get_last_shared_parent(current_comp_node, previous_comp_node);


            int fusion_level;
            if (last_shared_parent != nullptr){
                fusion_level = last_shared_parent->depth;
            }else
                fusion_level = tiramisu::computation::root_dimension;

            current_comp->after(*previous_comp, fusion_level);
            previous_comp = current_comp;

        }

    }

void apply_parallelization(syntax_tree const& ast)
{
    for (ast_node *root : ast.roots)
        apply_parallelization(root);
}

void apply_parallelization(ast_node* node)
{
    // if the ast_node is tagged for parallelization, get the child computations and tag them using tag_parallel_level()
    if (node->parallelized)
    {
        std::vector<tiramisu::computation*> involved_computations;
        node->get_all_computations(involved_computations);
        for (tiramisu::computation* comp: involved_computations)
            comp->tag_parallel_level(node->depth);
    }
    for (ast_node *child : node->children)
        apply_parallelization(child);

}

void print_optim(optimization_info optim)
{
    switch(optim.type) {
        case optimization_type::FUSION:
            std::cout << "Fusion" << " L" << optim.l0 << " " << optim.comps[0]->get_name() << " then "<< optim.comps[1]->get_name() <<std::endl;
            break;

        case optimization_type::SHIFTING:
            std::cout << "SHIFTING"
                      << " L" << optim.l0 << " +"<<optim.l0_fact<<" " << optim.comps[0]->get_name() <<std::endl;
            break;

        case optimization_type::MATRIX:
            std::cout << "Matrix Transform " << std::to_string(optim.matrix.size()) << "x" << std::to_string(optim.matrix.size())<< " {";
            for (auto comp:optim.comps)
                std::cout<< comp->get_name() <<", ";
            std::cout<< "} [";
            for(int i = 0; i < optim.matrix.size(); i++){
                for(int j = 0; j< optim.matrix.size(); j++){
                    std::cout << std::to_string(optim.matrix.at(i).at(j));
                    if(!(i==optim.matrix.size()-1 && j==optim.matrix.size()-1)) std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
            break;

        case optimization_type::INTERCHANGE:
            std::cout << "Interchange" << " L" << optim.l0 << " " << " L" << optim.l1 <<" { " ;
            for (auto comp:optim.comps)
                std::cout<< comp->get_name() <<", ";
            std::cout<<"}"<< std::endl;
            break;

        case optimization_type::TILING:
            std::cout << "Tiling" << " L" << optim.l0 << " " << optim.l0_fact << " L" << optim.l1 << " " << optim.l1_fact;
            if (optim.nb_l == 3)
                std::cout << " L" << optim.l2 << " " << optim.l2_fact;
            std::cout<<" { ";
            for (auto comp:optim.comps)
                std::cout<< comp->get_name() <<", ";
            std::cout<<"}"<< std::endl;
            break;

        case optimization_type::UNROLLING:
            std::cout << "Unrolling" << " L" << optim.l0 << " " << optim.l0_fact ;
            std::cout<<" { ";
            for (auto comp:optim.comps)
                std::cout<< comp->get_name() <<", ";
            std::cout<<"}"<< std::endl;
            break;

        case optimization_type::PARALLELIZE:
            std::cout << "Parallelize" << " L" << optim.l0;
            std::cout<<" { ";
            for (auto comp:optim.comps)
                std::cout<< comp->get_name() <<", ";
            std::cout<<"}"<< std::endl;
            break;

        case optimization_type::SKEWING:
            std::cout << "Skewing" << " L" << optim.l0 << " " << optim.l0_fact << " L" << optim.l1 << " " << optim.l1_fact;
            std::cout<<" { ";
            for (auto comp:optim.comps)
                std::cout<< comp->get_name() <<", ";
            std::cout<<"}"<< std::endl;
            break;
        case optimization_type::VECTORIZATION:
            std::cout << "VECTORIZATION" << " L" << optim.l0 << " " << optim.l0_fact;
            std::cout<<" { ";
            for (auto comp:optim.comps)
                std::cout<< comp->get_name() <<", ";
            std::cout<<"}"<< std::endl;
            break;

        default:
            break;
    }
}
}
