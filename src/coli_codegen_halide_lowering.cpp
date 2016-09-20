#include <algorithm>
#include <iostream>

#include <coli/debug.h>
#include <Halide.h>

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::map;

namespace coli
{

namespace
{

string stmt_to_string(const string &str, const Stmt &s) {
    std::ostringstream stream;
    stream << str << s << "\n";
    return stream.str();
}

} // anonymous namespace

Stmt lower_halide_pipeline(const Target &t, Stmt s) {
    map<string, Function> env; //TODO(psuriana): compute the env (function DAG)

    DEBUG(3, coli::str_dump("Performing sliding window optimization...\n"));
    s = sliding_window(s, env);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after sliding window:\n", s)));

    DEBUG(3, coli::str_dump("Removing code that depends on undef values...\n"));
    s = remove_undef(s);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after removing code that depends on undef values:\n", s)));

    // This uniquifies the variable names, so we're good to simplify
    // after this point. This lets later passes assume syntactic
    // equivalence means semantic equivalence.
    DEBUG(3, coli::str_dump("Uniquifying variable names...\n"));
    s = uniquify_variable_names(s);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after uniquifying variable names:\n", s)));

    DEBUG(3, coli::str_dump("Performing storage folding optimization...\n"));
    s = storage_folding(s, env);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after storage folding:\n", s)));

    DEBUG(3, coli::str_dump("Simplifying...\n")); // without removing dead lets, because storage flattening needs the strides
    s = simplify(s, false);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after first simplification:\n", s)));

    //TODO(psuriana): might be applicable to COLi?
    /*DEBUG(3, coli::str_dump("Dynamically skipping stages...\n"));
    s = skip_stages(s, order);
    DEBUG(3, coli::str_dump("Lowering after dynamically skipping stages:\n", s)));*/

    if (t.has_feature(Target::OpenGL) || t.has_feature(Target::Renderscript)) {
        DEBUG(3, coli::str_dump("Injecting image intrinsics...\n"));
        s = inject_image_intrinsics(s, env);
        DEBUG(4, coli::str_dump(stmt_to_string("Lowering after image intrinsics:\n", s)));
    }

    if (t.has_gpu_feature() ||
        t.has_feature(Target::OpenGLCompute) ||
        t.has_feature(Target::OpenGL) ||
        t.has_feature(Target::Renderscript) ||
        (t.arch != Target::Hexagon && (t.features_any_of({Target::HVX_64, Target::HVX_128})))) {
        DEBUG(3, coli::str_dump("Selecting a GPU API for GPU loops...\n"));
        s = select_gpu_api(s, t);
        DEBUG(4, coli::str_dump(stmt_to_string("Lowering after selecting a GPU API:\n", s)));

        DEBUG(3, coli::str_dump("Injecting host <-> dev buffer copies...\n"));
        s = inject_host_dev_buffer_copies(s, t);
        DEBUG(4, coli::str_dump(stmt_to_string("Lowering after injecting host <-> dev buffer copies:\n", s)));
    }

    if (t.has_feature(Target::OpenGL)) {
        DEBUG(3, coli::str_dump("Injecting OpenGL texture intrinsics...\n"));
        s = inject_opengl_intrinsics(s);
        DEBUG(4, coli::str_dump(stmt_to_string("Lowering after OpenGL intrinsics:\n", s)));
    }

    if (t.has_gpu_feature() ||
        t.has_feature(Target::OpenGLCompute) ||
        t.has_feature(Target::Renderscript)) {
        DEBUG(3, coli::str_dump("Injecting per-block gpu synchronization...\n"));
        s = fuse_gpu_thread_loops(s);
        DEBUG(4, coli::str_dump(stmt_to_string("Lowering after injecting per-block gpu synchronization:\n", s)));
    }

    DEBUG(3, coli::str_dump("Simplifying...\n"));
    s = simplify(s);
    s = unify_duplicate_lets(s);
    s = remove_trivial_for_loops(s);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after second simplifcation:\n", s)));

    DEBUG(3, coli::str_dump("Unrolling...\n"));
    s = unroll_loops(s);
    s = simplify(s);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after unrolling:\n", s)));

    DEBUG(3, coli::str_dump("Vectorizing...\n"));
    s = vectorize_loops(s);
    s = simplify(s);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after vectorizing:\n", s)));

    DEBUG(3, coli::str_dump("Detecting vector interleavings...\n"));
    s = rewrite_interleavings(s);
    s = simplify(s);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after rewriting vector interleavings:\n", s)));

    DEBUG(3, coli::str_dump("Partitioning loops to simplify boundary conditions...\n"));
    s = partition_loops(s);
    s = simplify(s);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after partitioning loops:\n", s)));

    DEBUG(3, coli::str_dump("Trimming loops to the region over which they do something...\n"));
    s = trim_no_ops(s);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after loop trimming:\n", s)));

    DEBUG(3, coli::str_dump("Injecting early frees...\n"));
    s = inject_early_frees(s);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after injecting early frees:\n", s)));

    if (t.has_feature(Target::FuzzFloatStores)) {
        DEBUG(3, coli::str_dump("Fuzzing floating point stores...\n"));
        s = fuzz_float_stores(s);
        DEBUG(4, coli::str_dump(stmt_to_string("Lowering after fuzzing floating point stores:\n", s)));
    }

    DEBUG(3, coli::str_dump("Simplifying...\n"));
    s = common_subexpression_elimination(s);

    if (t.has_feature(Target::OpenGL)) {
        DEBUG(3, coli::str_dump("Detecting varying attributes...\n"));
        s = find_linear_expressions(s);
        DEBUG(4, coli::str_dump(stmt_to_string("Lowering after detecting varying attributes:\n", s)));

        DEBUG(3, coli::str_dump("Moving varying attribute expressions out of the shader...\n"));
        s = setup_gpu_vertex_buffer(s);
        DEBUG(4, coli::str_dump(stmt_to_string("Lowering after removing varying attributes:\n", s)));
    }

    s = remove_dead_allocations(s);
    s = remove_trivial_for_loops(s);
    s = simplify(s);
    //DEBUG(3, coli::str_dump(stmt_to_string("Lowering after final simplification:\n", s)));
    std::cout << "Lowering after final simplification:\n" << s << "\n";

    DEBUG(3, coli::str_dump("Splitting off Hexagon offload...\n"));
    s = inject_hexagon_rpc(s, t);
    DEBUG(4, coli::str_dump(stmt_to_string("Lowering after splitting off Hexagon offload:\n", s)));

    return s;
}

}
