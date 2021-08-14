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
    for (optimization_info const& optim_info : ast.previous_optims)
        apply_optimizations(optim_info);
        
    for (optimization_info const& optim_info : ast.new_optims)
        apply_optimizations(optim_info);

    // Fusion is a particular case, and we use apply_fusions() to apply it.
    // apply_fusions() uses the structure of the AST to correctly order the computations.
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
            break;

        default:
            break;
    }
}

void apply_fusions(syntax_tree const& ast)
{
    tiramisu::computation *next_comp = nullptr;
    
    // Use the "after" scheduling command to replicate the structure of the AST
    // on the computations order.
    for (ast_node *root : ast.roots)
        next_comp = apply_fusions(root, next_comp, tiramisu::computation::root_dimension);
}

tiramisu::computation* apply_fusions(ast_node *node, tiramisu::computation *last_comp, int dimension)
{
    tiramisu::computation *next_comp;
    
    if (node->computations.size() > 0)
    {
        next_comp = node->computations[0].comp_ptr;
        
        if (last_comp != nullptr)
            next_comp->after(*last_comp, dimension);
            
        last_comp = next_comp;
        for (int i = 1; i < node->computations.size(); ++i)
        {
            next_comp = node->computations[i].comp_ptr;
            next_comp->after(*last_comp, node->depth);
        }
    }
    
    else
        next_comp = last_comp;
    
    int new_dimension = dimension;
    if (node->children.size() >= 2 || node->computations.size() >= 1)
        new_dimension = node->depth;
    
    for (ast_node *child : node->children)
        next_comp = apply_fusions(child, next_comp, new_dimension);
    
    return next_comp;
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
            std::cout << "Fusion" << " L" << optim.l0 << " " << " L" << optim.l1 << std::endl;
            break;

        case optimization_type::UNFUSE:
            std::cout << "Fusion" << " L" << optim.l0 << " " << " L" << optim.l1 << std::endl;
            break;

        case optimization_type::INTERCHANGE:
            std::cout << "Interchange" << " L" << optim.l0 << " " << " L" << optim.l1  << std::endl;
            break;

        case optimization_type::TILING:
            std::cout << "Tiling" << " L" << optim.l0 << " " << optim.l0_fact << " L" << optim.l1 << " " << optim.l1_fact;
            if (optim.nb_l == 3)
                std::cout << " L" << optim.l2 << " " << optim.l2_fact;
            std::cout << std::endl;
            break;

        case optimization_type::UNROLLING:
            std::cout << "Unrolling" << " L" << optim.l0 << " " << optim.l0_fact << std::endl;
            break;

        case optimization_type::PARALLELIZE:
            std::cout << "Parallelize" << " L" << optim.l0 << std::endl;
            break;

        case optimization_type::SKEWING:
            std::cout << "Skewing" << " L" << optim.l0 << " " << optim.l0_fact << " L" << optim.l1 << " " << optim.l1_fact << std::endl;
            break;
        case optimization_type::VECTORIZATION:
            std::cout << "VECTORIZATION" << " L" << optim.l0 << " " << optim.l0_fact << std::endl;
            break;

        default:
            break;
    }
}
}
