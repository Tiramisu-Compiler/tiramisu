#include <tiramisu/auto_scheduler/evaluator.h>

#include <cstdio>
#include <cstdlib>

namespace tiramisu::auto_scheduler
{

evaluate_by_execution::evaluate_by_execution(tiramisu::function *fct, 
                                             std::vector<tiramisu::buffer*> const& arguments, 
                                             std::string const& obj_filename, 
                                             std::string const& wrapper_cmd)
    : evaluator(), fct(fct), obj_filename(obj_filename), wrapper_cmd(wrapper_cmd)
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
    
    // Remove all the optimizations
    fct->reset_schedules();
    
    return exec_time;
}

simple_rnn_evaluator::simple_rnn_evaluator(std::string const& model_path) 
    : evaluator()
{
    model = torch::jit::load(model_path);
}

float simple_rnn_evaluator::evaluate(syntax_tree const& ast)
{
    std::vector<tiramisu::computation*> const& computations = ast.get_computations();
    int nb_computations = computations.size();
    
    at::Tensor input = torch::zeros({1, nb_computations, COMPUTATION_REPR_SIZE});
    at::Tensor length = nb_computations * torch::ones({1});
    
    // Get information about the schedule
    std::vector<bool> interchanged(MAX_NB_ITERATORS, false);
    std::vector<bool> tiled(MAX_NB_ITERATORS, false);
    std::vector<int> tiling_fact(MAX_NB_ITERATORS, 0);
    int unrolling_fact = 0;
    
    for (optimization_info const& optim_info : ast.new_optims)
    {
        switch (optim_info.type)
        {
            case optimization_type::TILING:
                if (optim_info.nb_l == 2)
                {
                    tiled[optim_info.l0] = true;
                    tiled[optim_info.l1] = true;
                    
                    tiling_fact[optim_info.l0] = optim_info.l0_fact;
                    tiling_fact[optim_info.l1] = optim_info.l1_fact;
                }
                
                else if (optim_info.nb_l == 3)
                {
                    tiled[optim_info.l0] = true;
                    tiled[optim_info.l1] = true;
                    tiled[optim_info.l2] = true;
                    
                    tiling_fact[optim_info.l0] = optim_info.l0_fact;
                    tiling_fact[optim_info.l1] = optim_info.l1_fact;
                    tiling_fact[optim_info.l2] = optim_info.l2_fact;
                }
                break;
                
            case optimization_type::INTERCHANGE:
                interchanged[optim_info.l0] = true;
                interchanged[optim_info.l1] = true;
                break;
                
            case optimization_type::UNROLLING:
                unrolling_fact = optim_info.l0_fact;
                break;
                
            default:
                break;
        }
    }
    
    // Apply previous optimizations
    for (optimization_info const& optim_info : ast.previous_optims)
        if (optim_info.type != optimization_type::UNROLLING)
            apply_optimizations(optim_info);
    
    // Create the vector representation of each computation
    for (int i = 0; i < nb_computations; ++i)
    {
        // Get information about iterators
        isl_set *iter_domain = computations[i]->get_iteration_domain();
        int nb_iterators = isl_set_dim(iter_domain, isl_dim_set);
        
        dnn_schedule dnn_sched(nb_iterators);
        for (int j = 0; j < nb_iterators; ++j)
        {
            dnn_sched.iterators[j].name = isl_set_get_dim_name(iter_domain, isl_dim_set, j);
            dnn_sched.iterators[j].low_bound = utility::get_bound(iter_domain, j, false).get_int_val();
            dnn_sched.iterators[j].up_bound = utility::get_bound(iter_domain, j, true).get_int_val();
            
            dnn_sched.iterators[j].interchanged = interchanged[j];
            dnn_sched.iterators[j].tiled = tiled[j];
            dnn_sched.iterators[j].tiling_fact = tiling_fact[j];
        }
        
        dnn_sched.unrolling_fact = unrolling_fact;
        
        // Get information about accesses
        std::vector<dnn_access_matrix> accesses;
        dnn_access_matrix::create_accesses(computations[i]->get_expr(), nb_iterators, accesses, computations[i]);
        
        for (dnn_access_matrix& acc_matrix : accesses)
            acc_matrix.set_buffer_id(ast.fct);
        
        input[0][i] = get_computation_repr(dnn_sched, accesses);
    }
    
    // Remove all the optimizations
    ast.fct->reset_schedules();
    
    // Call the DNN model
    std::vector<torch::jit::IValue> params = {input, length};
    at::Tensor output = model.forward(params).toTensor();
    
    return output.item().to<float>();
}

at::Tensor simple_rnn_evaluator::get_computation_repr(dnn_schedule const& sched, std::vector<dnn_access_matrix> const& accesses)
{
    at::Tensor ret = torch::zeros({COMPUTATION_REPR_SIZE});
    int offset = 0;
    
    // Add iterators, interchange and tiling
    for (int i = 0; i < sched.nb_iterators; ++i)
    {
        ret[offset + 0] = sched.iterators[i].low_bound;
        ret[offset + 1] = sched.iterators[i].up_bound;
        
        ret[offset + 2] = sched.iterators[i].interchanged;
        ret[offset + 3] = sched.iterators[i].tiled;
        ret[offset + 4] = sched.iterators[i].tiling_fact;
        
        offset += ITERATORS_REPR_SIZE;
    }
    
    offset = MAX_NB_ITERATORS * ITERATORS_REPR_SIZE;
    
    // Add accesses
    for (int i = 0; i < accesses.size(); ++i)
    {
        dnn_access_matrix const& access = accesses[i];
        
        ret[offset] = access.buffer_id;
        offset++;
        
        for (int j = 0; j < access.matrix.size(); ++j)
        {
            for (int k = 0; k < access.matrix[j].size(); ++k)
                ret[offset + k] = access.matrix[j][k];
            
            offset += MAX_NB_ITERATORS + 1;
        }
        
        offset += (MAX_NB_ITERATORS - access.matrix.size()) * MAX_NB_ITERATORS * (MAX_NB_ITERATORS + 1);
    }
    
    offset = MAX_NB_ITERATORS * ITERATORS_REPR_SIZE + MAX_NB_ACCESSES * ACCESS_REPR_SIZE;
    
    // Add unrolling factor
    if (sched.unrolling_fact > 0)
        ret[offset] = 1.0;
        
    ret[offset + 1] = sched.unrolling_fact;
    
    return ret;
}

}
