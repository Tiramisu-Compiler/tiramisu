#ifndef _H_TIRAMISU_AUTO_SCHEDULER_
#define _H_TIRAMISU_AUTO_SCHEDULER_

#include <tiramisu/tiramisu.h>
#include <tiramisu/computation_graph.h>

#include <map>
#include <string.h>
#include <stdint.h>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

namespace tiramisu
{

/**
 * A class to keep track of the schedule of a block. This includes
 * information such as whether the block is parallelized, vectoirzed,
 * unrolled, ...
 * This information is mainly useful for the auto scheduler so that
 * if certain optimization is already applied, no need to explore it.
 */
class schedule_tracker
{

public:

    /**
     * Set to true if the block is parallel.
     */
    bool parallel;

    /**
     * Set to true if the block is vectorized.
     */
    bool vectorized;

    /**
      * Set to true if the block is unrolled.
      */
    bool unrolled;

    /**
      * Class constructor.
      */
    schedule_tracker()
    {
	parallel = false;
	vectorized = false;
	unrolled = false;
    }
};

/**
  * Main class for the auto scheduler.
  * To run the auto scheduler, use
  *
    \code
	scheduler::run();
    \endcode
  *
  **/
class auto_scheduler
{
    /**
      * Apply basic local optimizations on each node
      * in the computation graph.
      *
      * Examples of basic local optimizations include
      * parallelization, vectorization, and tiling.
      */
    static computation_graph apply_local_optimizations_phase_I(computation_graph &g);

    /**
      * Apply global optimizations to the computation graph.
      * Global optimizations include fusion, inlining and compute_at
      * (i.e., fusion with redundancy).
      */
    static computation_graph apply_global_optimzations(computation_graph &g);

    /**
      * Apply remaining local optimizations. This includes
      * interchange, unrolling, data layout optimizations, ...
      */
    static computation_graph apply_local_optimizations_phase_II(computation_graph &g);

    /**
      * Apply the order of computations defined by the graph of
      * computations \p g.
      *
      * This function will call ordering functions such as
      * .then(), .before() and .between(), ...
      */
    static void apply_computation_ordering(function *fct, computation_graph &g);

    /**
      * Create an initial graph of blocks (non-fused blocks)
      */
    static computation_graph create_initial_computation_graph(function *fct);

    /**
      * Parallelize block.
      */
    static void parallelism_apply(block &b);

    /**
      * Return true if it is legal to parallelize \p b.
      */
    static bool parallelism_is_legal(block &b);

    /**
      * Return true if \p b is profitable to parallelize.
      */
    static bool parallelism_is_profitable(block &b);

    /**
      * Vectorize block.
      */
    static void vectorization_apply(block &b);

    /**
      * Return true if it is legal to vectorize \p b.
      */
    static bool vectorization_is_legal(block &b);

    /**
      * Return true if it is profitable to vectorize \p b.
      */
    static bool vectorization_is_profitable(block &b);

public:

    /**
      * Run CPU autoscheduler.
      */
    static void run_cpu_scheduler();
};

}

#endif
