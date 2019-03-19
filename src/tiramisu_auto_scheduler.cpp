#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler.h>

namespace tiramisu
{
    computation_graph auto_scheduler::apply_local_optimizations_phase_I(computation_graph &g)
    {
	return g;
    }

    computation_graph auto_scheduler::apply_global_optimzations(computation_graph &g)
    {
	return g;
    }

    computation_graph auto_scheduler::apply_local_optimizations_phase_II(computation_graph &g)
    {
	return g;
    }

    void auto_scheduler::apply_computation_ordering(function *fct, computation_graph &g)
    {

    }

    computation_graph auto_scheduler::create_initial_computation_graph(function *fct)
    {

    }

    void auto_scheduler::parallelism_apply(block &b)
    {

    }

    bool auto_scheduler::parallelism_is_legal(block &b)
    {
	return true;
    }

    bool auto_scheduler::parallelism_is_profitable(block &b)
    {
	return true;
    }

    void auto_scheduler::vectorization_apply(block &b)
    {
	
    }

    bool auto_scheduler::vectorization_is_legal(block &b)
    {
	return true;
    }

    bool auto_scheduler::vectorization_is_profitable(block &b)
    {
	return true;	
    }

    void auto_scheduler::run_cpu_scheduler()
    {
	function *fct = global::get_implicit_function();
	computation_graph g = create_initial_computation_graph(fct);
	g = apply_local_optimizations_phase_I(g);
	g = apply_global_optimzations(g);
	g = apply_local_optimizations_phase_II(g);
	apply_computation_ordering(fct, g);
	fct->computation_graph = g;
    }
}
