#include <algorithm>
#include <iostream>

#include <tiramisu/debug.h>
#include <Halide.h>

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::map;
using std::vector;

namespace tiramisu
{

namespace
{

string stmt_to_string(const string &str, const Stmt &s)
{
    std::ostringstream stream;
    stream << str << s << "\n";
    return stream.str();
}

} // anonymous namespace

Module lower_halide_pipeline(const string &pipeline_name,
                             const Target &t,
                             const vector<Argument> &args,
                             const Internal::LoweredFunc::LinkageType linkage_type,
                             Stmt s)
{
    Module result_module(pipeline_name, t);

    // TODO(tiramisu): Compute the env (function DAG). This is needed for
    // the sliding window and storage folding passes.
    map<string, Function> env;

    if (ENABLE_DEBUG)
    {
        std::cout << "Lower halide pipeline...\n" << s << "\n";
        std::flush(std::cout);
    }

    DEBUG(3, tiramisu::str_dump("Performing sliding window optimization...\n"));
    s = sliding_window(s, env);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after sliding window:\n", s)));

    DEBUG(3, tiramisu::str_dump("Removing code that depends on undef values...\n"));
    s = remove_undef(s);
    DEBUG(4, tiramisu::str_dump(
              stmt_to_string("Lowering after removing code that depends on undef values:\n", s)));

    // This uniquifies the variable names, so we're good to simplify
    // after this point. This lets later passes assume syntactic
    // equivalence means semantic equivalence.
    DEBUG(3, tiramisu::str_dump("Uniquifying variable names...\n"));
    s = uniquify_variable_names(s);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after uniquifying variable names:\n", s)));

    DEBUG(3, tiramisu::str_dump("Simplifying...\n")); // without removing dead lets, because storage flattening needs the strides
    s = simplify(s, false);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after simplification:\n", s)));
    
    DEBUG(3, tiramisu::str_dump("Performing storage folding optimization...\n"));
    s = storage_folding(s, env);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after storage folding:\n", s)));

    DEBUG(3, tiramisu::str_dump("Simplifying...\n")); // without removing dead lets, because storage flattening needs the strides
    s = simplify(s, false);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after simplification:\n", s)));

/*    DEBUG(3, tiramisu::str_dump("Injecting prefetches...\n"));
    s = inject_prefetch(s, env);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after injecting prefetches:\n", s)));
*/
    DEBUG(3, tiramisu::str_dump("Destructuring tuple-valued realizations...\n"));
    s = split_tuples(s, env);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after destructuring tuple-valued realizations:\n", s)));
    DEBUG(3, tiramisu::str_dump("\n\n"));

    // TODO(tiramisu): This pass is important to figure out all the buffer symbols.
    // Maybe we should put it somewhere else instead of here.
    DEBUG(3, tiramisu::str_dump("Unpacking buffer arguments...\n"));
    s = unpack_buffers(s);
    DEBUG(0, tiramisu::str_dump(stmt_to_string("Lowering after unpacking buffer arguments:\n", s)));

    if (t.has_gpu_feature() ||
        t.has_feature(Target::OpenGLCompute) ||
        t.has_feature(Target::OpenGL) ||
        (t.arch != Target::Hexagon && (t.features_any_of({Target::HVX_64, Target::HVX_128})))) {
        DEBUG(3, tiramisu::str_dump("Selecting a GPU API for GPU loops...\n"));
        s = select_gpu_api(s, t);
        DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after selecting a GPU API:\n", s)));

        DEBUG(3, tiramisu::str_dump("Injecting host <-> dev buffer copies...\n"));
        s = inject_host_dev_buffer_copies(s, t);
        DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after injecting host <-> dev buffer copies:\n",
                                    s)));
    }

    if (t.has_feature(Target::OpenGL))
    {
        DEBUG(3, tiramisu::str_dump("Injecting OpenGL texture intrinsics...\n"));
        s = inject_opengl_intrinsics(s);
        DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after OpenGL intrinsics:\n", s)));
    }

    if (t.has_gpu_feature() ||
            t.has_feature(Target::OpenGLCompute))
    {
        DEBUG(3, tiramisu::str_dump("Injecting per-block gpu synchronization...\n"));
        s = fuse_gpu_thread_loops(s);
        DEBUG(4, tiramisu::str_dump(
                  stmt_to_string("Lowering after injecting per-block gpu synchronization:\n", s)));
    }

    DEBUG(3, tiramisu::str_dump("Simplifying...\n"));
    s = simplify(s);
    s = unify_duplicate_lets(s);
    s = remove_trivial_for_loops(s);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after second simplifcation:\n", s)));

    DEBUG(3, tiramisu::str_dump("Reduce prefetch dimension...\n"));
    s = reduce_prefetch_dimension(s, t);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after reduce prefetch dimension:\n", s)));

    DEBUG(3, tiramisu::str_dump("Unrolling...\n"));
    s = unroll_loops(s);
    s = simplify(s);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after unrolling:\n", s)));

    DEBUG(3, tiramisu::str_dump("Vectorizing...\n"));
    s = vectorize_loops(s, t);
    s = simplify(s);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after vectorizing:\n", s)));

    DEBUG(3, tiramisu::str_dump("Detecting vector interleavings...\n"));
    s = rewrite_interleavings(s);
    s = simplify(s);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after rewriting vector interleavings:\n", s)));

    DEBUG(3, tiramisu::str_dump("Partitioning loops to simplify boundary conditions...\n"));
    s = partition_loops(s);
    s = simplify(s);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after partitioning loops:\n", s)));

    DEBUG(3, tiramisu::str_dump("Trimming loops to the region over which they do something...\n"));
    s = trim_no_ops(s);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after loop trimming:\n", s)));

    DEBUG(3, tiramisu::str_dump("Injecting early frees...\n"));
    s = inject_early_frees(s);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after injecting early frees:\n", s)));

    if (t.has_feature(Target::FuzzFloatStores))
    {
        DEBUG(3, tiramisu::str_dump("Fuzzing floating point stores...\n"));
        s = fuzz_float_stores(s);
        DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after fuzzing floating point stores:\n", s)));
    }

    DEBUG(3, tiramisu::str_dump("Simplifying...\n"));
    s = common_subexpression_elimination(s);

    if (t.has_feature(Target::OpenGL))
    {
        DEBUG(3, tiramisu::str_dump("Detecting varying attributes...\n"));
        s = find_linear_expressions(s);
        DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after detecting varying attributes:\n", s)));

        DEBUG(3, tiramisu::str_dump("Moving varying attribute expressions out of the shader...\n"));
        s = setup_gpu_vertex_buffer(s);
        DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after removing varying attributes:\n", s)));
    }

    s = remove_dead_allocations(s);
    s = remove_trivial_for_loops(s);
    s = simplify(s);
    // s = loop_invariant_code_motion(s);
    if (ENABLE_DEBUG)
    {
        std::cout << "Lowering after final simplification:\n" << s << "\n";
        std::flush(std::cout);
    }

    DEBUG(3, tiramisu::str_dump("Splitting off Hexagon offload...\n"));
    s = inject_hexagon_rpc(s, t, result_module);
    DEBUG(4, tiramisu::str_dump(stmt_to_string("Lowering after splitting off Hexagon offload:\n", s)));


    vector<Argument> public_args = args;

    // We're about to drop the environment and outputs vector, which
    // contain the only strong refs to Functions that may still be
    // pointed to by the IR. So make those refs strong.
    class StrengthenRefs : public IRMutator {
        using IRMutator::visit;
        void visit(const Call *c) {
            IRMutator::visit(c);
            c = expr.as<Call>();
            //internal_assert(c);
            if (c->func.defined()) {
                FunctionPtr ptr = c->func;
                ptr.strengthen();
                expr = Call::make(c->type, c->name, c->args, c->call_type,
                                  ptr, c->value_index,
                                  c->image, c->param);
            }
        }
    };
    s = StrengthenRefs().mutate(s);

    LoweredFunc main_func(pipeline_name, public_args, s, linkage_type);

    result_module.append(main_func);

    // Append a wrapper for this pipeline that accepts old buffer_ts
    // and upgrades them. It will use the same name, so it will
    // require C++ linkage. We don't need it when jitting.
    if (!t.has_feature(Target::JIT)) {
        add_legacy_wrapper(result_module, main_func);
    }

    // Also append any wrappers for extern stages that expect the old buffer_t
    wrap_legacy_extern_stages(result_module);

    return result_module;
}

}
