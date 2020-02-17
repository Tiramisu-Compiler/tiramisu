#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/block.h>

#include <cstdio>
#include <cstdlib>

namespace tiramisu::auto_scheduler
{

evaluate_by_execution::evaluate_by_execution(tiramisu::function *fct, 
                                             std::vector<tiramisu::buffer*> const& arguments, 
                                             std::string const& obj_filename, 
                                             std::string const& wrapper_cmd)
    : fct(fct), obj_filename(obj_filename), wrapper_cmd(wrapper_cmd)
{
    // Set Halide compilation features
    halide_target = Halide::get_host_target();
    halide_target.set_features(halide_features);
    
    // Set input and output buffers
    fct->set_arguments(arguments);
    for (auto const& buf : arguments)
    {
        Halide::Argument buffer_arg(
                buf->get_name(),
                halide_argtype_from_tiramisu_argtype(buf->get_argument_type()),
                halide_type_from_tiramisu_type(buf->get_elements_type()),
                buf->get_n_dims());
                
       halide_arguments.push_back(buffer_arg);
    }
}

void evaluate_by_execution::apply_optimizations(syntax_tree const& ast)
{
    for (optimization_info optim_info : ast.optims_info)
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
                block.unroll(optim_info.l0, optim_info.l0_fact);
                break;
                
            default:
                break;
        }
    }
    
    // Apply fusions
    if (ast.optims_info.size() > 0 && ast.optims_info.back().type == optimization_type::FUSION)
    {
        syntax_tree *ast_copy = ast.copy_ast();
        ast_copy->transform_ast_by_fusion(ast.optims_info.back());
        
        apply_fusions(*ast_copy);
        delete ast_copy;
    }
    
    else
        apply_fusions(ast);
}

void evaluate_by_execution::apply_fusions(syntax_tree const& ast)
{
    tiramisu::computation *next_comp = nullptr;
    
    for (ast_node *root : ast.roots)
        next_comp = apply_fusions(root, next_comp, tiramisu::computation::root_dimension);
}

tiramisu::computation* evaluate_by_execution::apply_fusions(ast_node *node, tiramisu::computation *last_comp, int dimension)
{
    tiramisu::computation *next_comp;
    
    if (node->computations.size() > 0)
    {
        next_comp = node->computations[0];
        
        if (last_comp != nullptr)
            next_comp->after(*last_comp, dimension);
            
        last_comp = next_comp;
        for (int i = 1; i < node->computations.size(); ++i)
        {
            next_comp = node->computations[i];
            next_comp->after(*last_comp, node->depth);
        }
    }
    
    else
        next_comp = last_comp;
    
    int new_dimension = dimension;
    if (node->children.size() >= 2)
        new_dimension = node->depth;
    
    for (ast_node *child : node->children)
        next_comp = apply_fusions(child, next_comp, new_dimension);
    
    return next_comp;
}

float evaluate_by_execution::evaluate(syntax_tree const& ast)
{
    // Apply all the optimizations
    apply_optimizations(ast);
    
    // Compile the program to an object file
    fct->lift_dist_comps();
    fct->gen_time_space_domain();
    fct->gen_isl_ast();
    fct->gen_halide_stmt();
    
    Halide::Module m = lower_halide_pipeline(fct->get_name(), halide_target, halide_arguments,
                                             Halide::Internal::LoweredFunc::External,
                                             fct->get_halide_stmt());
                                             
    m.compile(Halide::Outputs().object(obj_filename));
    
    // Turn the object file to a shared library
    std::string gcc_cmd = "g++ -shared -o " + obj_filename + ".so " + obj_filename;
    int status = system(gcc_cmd.c_str());
    
    // Execute the wrapper and get execution time
    double exec_time = 0.f;
    FILE *pipe = popen(wrapper_cmd.c_str(), "r");
    
    fscanf(pipe, "%lf", &exec_time);
    pclose(pipe);
    
    std::cout << exec_time << std::endl;
    
    // Remove all the optimizations
    for (tiramisu::computation *comp : fct->get_computations())
        comp->set_identity_schedule_based_on_iteration_domain();
        
    fct->remove_dimension_tags();
    fct->clear_sched_graph();
    
    return exec_time;
}

}
