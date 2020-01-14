#include <tiramisu/auto_scheduler/evaluator.h>

#define ITERATORS_REPR_SIZE 5
#define ACCESS_REPR_SIZE MAX_NB_ITERATORS * (MAX_NB_ITERATORS + 1) + 1
#define COMPUTATION_REPR_SIZE 379

namespace tiramisu::auto_scheduler
{

simple_rnn_evaluator::simple_rnn_evaluator(std::string const& model_path) 
    : evaluator()
{
    model = torch::jit::load(model_path);
}

float simple_rnn_evaluator::evaluate(computation_graph const& cg, schedule_info const& sched)
{
    std::vector<torch::jit::IValue> params;
    int nb_computations = cg.roots.size();
    
    at::Tensor input = torch::zeros({1, nb_computations, COMPUTATION_REPR_SIZE});
    at::Tensor length = nb_computations * torch::ones({1});
    
    for (int i = 0; i < nb_computations; ++i)
        input[0][i] = build_computation_repr(cg.roots[i], sched);
    
    params.push_back(input);
    params.push_back(length);
    
    at::Tensor output = model.forward(params).toTensor();
    return output.item().to<float>();
}

at::Tensor simple_rnn_evaluator::build_computation_repr(cg_node *node, schedule_info const& sched)
{
    at::Tensor ret = torch::zeros({COMPUTATION_REPR_SIZE});
    int offset = 0;
    
    // Add iterators, interchange and tiling
    for (int i = 0; i < node->iterators.size(); ++i)
    {
        ret[offset + 0] = node->iterators[i].low_bound;
        ret[offset + 1] = node->iterators[i].up_bound;
        
        ret[offset + 2] = sched.interchanged[i];
        ret[offset + 3] = sched.tiled[i];
        ret[offset + 4] = sched.tiling_factors[i];
        
        offset += ITERATORS_REPR_SIZE;
    }
    
    offset = MAX_NB_ITERATORS * ITERATORS_REPR_SIZE;
    
    // Add accesses
    for (int i = 0; i < node->accesses.size(); ++i)
    {
        access_matrix const& access = node->accesses[i];
        
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
    if (sched.unrolling_factor > 0)
        ret[offset] = 1.0;
        
    ret[offset + 1] = sched.unrolling_factor;
    
    return ret;
}

}
