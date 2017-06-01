#include <isl/ctx.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/constraint.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

#include <string>
#include <algorithm>

namespace tiramisu
{

std::map<std::string, computation *> computations_list;
bool global::auto_data_mapping;

// Used for the generation of new variable names.
int id_counter = 0;

/**
 * Retrieve the access function of the ISL AST leaf node (which represents a
 * computation). Store the access in computation->access.
 */
isl_ast_node *stmt_code_generator(
    isl_ast_node *node, isl_ast_build *build, void *user);

isl_ast_node *for_code_generator_after_for(
    isl_ast_node *node, isl_ast_build *build, void *user);


std::string generate_new_variable_name();
void get_rhs_accesses(const tiramisu::function *func, const tiramisu::computation *comp,
                      std::vector<isl_map *> &accesses, bool);
tiramisu::expr traverse_expr_and_replace_non_affine_accesses(tiramisu::computation *comp,
        const tiramisu::expr &exp);

/**
 * Create an equality constraint and add it to the schedule \p sched.
 * Edit the schedule as follows: assuming that y and y' are the input
 * and output dimensions of sched in dimensions \p dim0.
 * This function function add the constraint:
 *   in_dim_coefficient*y = out_dim_coefficient*y' + const_conefficient;
 */
isl_map *add_eq_to_schedule_map(int dim0, int in_dim_coefficient, int out_dim_coefficient,
                                int const_conefficient, isl_map *sched);

/**
 * Create an inequality constraint and add it to the schedule \p sched
 * of the duplicate computation that has \p duplicate_ID as an ID.
 * Edit the schedule as follows: assuming that y and y' are the input
 * and output dimensions of sched in dimensions \p dim0.
 * This function function add the constraint:
 *   in_dim_coefficient*y <= out_dim_coefficient*y' + const_conefficient;
 */
isl_map *add_ineq_to_schedule_map(int duplicate_ID, int dim0, int in_dim_coefficient,
                                  int out_dim_coefficient, int const_conefficient, isl_map *sched);

/**
  * Add a buffer to the function.
  */
void function::add_buffer(std::pair<std::string, tiramisu::buffer *> buf)
{
    assert(!buf.first.empty() && ("Empty buffer name."));
    assert((buf.second != NULL) && ("Empty buffer."));

    this->buffers_list.insert(buf);
}

/**
 * Construct a function with the name \p name.
 */
function::function(std::string name)
{
    assert(!name.empty() && ("Empty function name"));

    this->name = name;
    halide_stmt = Halide::Internal::Stmt();
    ast = NULL;
    context_set = NULL;

    // Allocate an ISL context.  This ISL context will be used by
    // the ISL library calls within Tiramisu.
    ctx = isl_ctx_alloc();
};

/**
  * Get the arguments of the function.
  */
// @{
const std::vector<tiramisu::buffer *> &function::get_arguments() const
{
    return function_arguments;
}
// @}

isl_union_map *tiramisu::function::compute_dep_graph()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_union_map *result = NULL;

    for (const auto &consumer : this->get_computations())
    {
        DEBUG(3, tiramisu::str_dump("Computing the dependences involving the computation " +
                                    consumer->get_name() + "."));
        DEBUG(3, tiramisu::str_dump("Computing the accesses of the computation."));

        isl_union_map *accesses_union_map = NULL;
        std::vector<isl_map *> accesses_vector;
        get_rhs_accesses(this, consumer, accesses_vector, false);

        DEBUG(3, tiramisu::str_dump("Vector of accesses computed."));

        if (!accesses_vector.empty())
        {
            // Create a union map of the accesses to the producer.
            if (accesses_union_map == NULL)
            {
                isl_space *space = isl_map_get_space(accesses_vector[0]);
                assert(space != NULL);
                accesses_union_map = isl_union_map_empty(space);
            }

            for (size_t i = 0; i < accesses_vector.size(); ++i)
            {
                isl_map *reverse_access = isl_map_reverse(accesses_vector[i]);
                accesses_union_map = isl_union_map_union(isl_union_map_from_map(reverse_access),
                                                         accesses_union_map);
            }

            //accesses_union_map = isl_union_map_intersect_range(accesses_union_map, isl_union_set_from_set(isl_set_copy(consumer->get_iteration_domain())));
            //accesses_union_map = isl_union_map_intersect_domain(accesses_union_map, isl_union_set_from_set(isl_set_copy(consumer->get_iteration_domain())));

            DEBUG(3, tiramisu::str_dump("Accesses after filtering."));
            DEBUG(3, tiramisu::str_dump(isl_union_map_to_str(accesses_union_map)));

            if (result == NULL)
            {
                result = isl_union_map_copy(accesses_union_map);
                isl_union_map_free(accesses_union_map);
            }
            else
            {
                result = isl_union_map_union(result, accesses_union_map);
            }
        }
    }

    DEBUG(3, tiramisu::str_dump("Dep graph:"));
    DEBUG(3, tiramisu::str_dump(isl_union_map_to_str(result)));

    DEBUG_INDENT(-4);
    DEBUG(3, tiramisu::str_dump("End of function"));

    return result;
}

std::vector<tiramisu::computation *> tiramisu::function::get_last_consumers()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert((this->get_computations().size()>0) && "The function should have at least one computation.");

    std::vector<tiramisu::computation *> last;
    isl_union_map *deps = this->compute_dep_graph();

    if (deps != NULL)
    {
        if (isl_union_map_is_empty(deps) == isl_bool_false)
        {
            // The domains and the ranges of the dependences
            isl_union_set *domains = isl_union_map_domain(isl_union_map_copy(deps));
            isl_union_set *ranges = isl_union_map_range(isl_union_map_copy(deps));

            DEBUG(3, tiramisu::str_dump("Ranges of the dependence graph.", isl_union_set_to_str(ranges)));
            DEBUG(3, tiramisu::str_dump("Domains of the dependence graph.", isl_union_set_to_str(domains)));

            /** In a dependence graph, since dependences create a chain (i.e., the end of
             *  a dependence is the beginning of the following), then each range of
             *  a dependence has a set domains that correspond to it (i.e., that their
             *  union is equal to it).  If range exists but does not have domains that
             *  are equal to it, then that range is the last range.
             *
             *  To compute those ranges that do not have corresponding domains, we
             *  compute (ranges - domains).
             */
            isl_union_set *last_ranges = isl_union_set_subtract(ranges, domains);
            DEBUG(3, tiramisu::str_dump("Ranges - Domains :", isl_union_set_to_str(last_ranges)));

            if (isl_union_set_is_empty(last_ranges) == isl_bool_false)
            {
                for (const auto &c : this->get_computations())
                {
                    isl_space *sp = isl_set_get_space(c->get_iteration_domain());
                    isl_set *s = isl_set_universe(sp);
                    isl_union_set *intersect =
                        isl_union_set_intersect(isl_union_set_from_set(s),
                                                isl_union_set_copy(last_ranges));

                    if (isl_union_set_is_empty(intersect) == isl_bool_false)
                    {
                        last.push_back(c);
                    }
                    isl_union_set_free(intersect);
                }

                DEBUG(3, tiramisu::str_dump("Last computations:"));
                for (const auto &c : last)
                {
                    DEBUG(3, tiramisu::str_dump(c->get_name() + " "));
                }
            }
            else
            {
                // If the difference between ranges and domains is empty, then
                // all the computations of the program are recursive (assuming
                // that the set of dependences is empty).
                last = this->get_computations();
            }

            isl_union_set_free(last_ranges);
        }
        else
        {
            // If the program does not have any dependence, then
            // all the computations should be considered as the last
            // computations.
            last = this->get_computations();
        }

        isl_union_map_free(deps);
    }

    assert((last.size()>0) && "The function should have at least one last computation.");

    DEBUG_INDENT(-4);

    return last;
}

bool tiramisu::computation::is_update() const
{
    bool is_update = false;

    std::string name = this->get_name();
    assert(name.size() > 0);

    std::vector<tiramisu::computation *> computations =
            this->get_function()->get_computation_by_name(name);

    // If many computations of the same name exist, then this
    // computation is an update, otherwise it is a pure definition.
    if (computations.size() > 1)
        is_update = true;
    else
        is_update = false;

    return is_update;
}

void tiramisu::function::compute_bounds()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_union_map *Dep = this->compute_dep_graph();
    DEBUG(3, tiramisu::str_dump("Dependences computed."));
    isl_union_map *Reverse = isl_union_map_reverse(isl_union_map_copy(Dep));
    DEBUG(3, tiramisu::str_dump("Reverse of dependences:", isl_union_map_to_str(Reverse)));
    // Compute the vector of the last computations in the dependence graph
    // (i.e., the computations that do not have any consumer).
    std::vector<tiramisu::computation *> last = this->get_last_consumers();

    assert(last.size() > 0);

    isl_union_set *Domains = NULL;
    Domains = isl_union_set_empty(isl_set_get_space(last[0]->get_iteration_domain()));
    for (auto c : last)
    {
        Domains = isl_union_set_union(Domains,
                                      isl_union_set_from_set(isl_set_copy(c->get_iteration_domain())));
    }
    DEBUG(3, tiramisu::str_dump("The domains of the last computations are:",
                                isl_union_set_to_str(Domains)));

    // Compute "Producers", the union of the iteration domains of the computations
    // that computed "last".
    isl_union_set *Producers = isl_union_set_apply(isl_union_set_copy(Domains),
                               isl_union_map_copy(Reverse));
    DEBUG(3, tiramisu::str_dump("The producers of the last computations are:",
                                isl_union_set_to_str(Producers)));

    // If the graph of dependences has recursive dependences, then the intersection of
    // the old producers and the new producers will not be empty (i.e., the old producer and the new producer
    // are the same).
    // In this case, we should subtract the common domain so that in the next iterations of the
    // the algorithm we do not get the same computation again and again (since we have a recursive dependence).
    // This is equivalent to removing the recursive dependence (we remove its image instead of removing it).
    isl_union_set *old_Producers = isl_union_set_copy(Domains);
    isl_union_set *intersection = isl_union_set_intersect(old_Producers, isl_union_set_copy(Producers));
    if (isl_union_set_is_empty(intersection) == isl_bool_false)
    {
        isl_union_set *common_computations = isl_union_set_universe(intersection);
        Producers = isl_union_set_subtract(Producers, common_computations);
        DEBUG(3, tiramisu::str_dump("After eliminating the effect of recursive dependences.",
                                isl_union_set_to_str(Producers)));
    }


    // Propagation of bounds
    DEBUG(3, tiramisu::str_dump("Propagating the bounds over all computations."));
    DEBUG_INDENT(4);
    while (isl_union_set_is_empty(Producers) == isl_bool_false)
    {
        for (auto c : this->get_computations())
        {
            DEBUG(3, tiramisu::str_dump("Computing the domain (bounds) of the computation: " + c->get_name()));
            isl_union_set *c_dom = isl_union_set_from_set(isl_set_copy(c->get_iteration_domain()));
            DEBUG(3, tiramisu::str_dump("Domain of the computation: ", isl_union_set_to_str(c_dom)));
            isl_union_set *prods = isl_union_set_copy(Producers);
            DEBUG(3, tiramisu::str_dump("Producers : ", isl_union_set_to_str(prods)));
            // Filter the producers to remove the domains of all the computations except the domain of s1
            // Keep only the computations that have the same space as s1.
            isl_union_set *filter = isl_union_set_universe(isl_union_set_copy(c_dom));
            isl_union_set *c_prods = isl_union_set_intersect(isl_union_set_copy(filter), prods);
            DEBUG(3, tiramisu::str_dump("After keeping only the producers that have the same space as the domain.",
                                        isl_union_set_to_str(c_prods)));

            // If this is not an update operation, we can update its domain, otherwise
            // we do not update the domain and keep the one provided by the user.
            if (c->is_update() == false)
            {
                // REC TODO: in the documentation of compute_bounds indicate that compute_bounds does not update the bounds of update operations
                if ((isl_union_set_is_empty(c_prods) == isl_bool_false))
                {
                    if ((isl_set_plain_is_universe(c->get_iteration_domain()) == isl_bool_true))
                    {
                        DEBUG(3, tiramisu::str_dump("The iteration domain of the computation is a universe."));
                        DEBUG(3, tiramisu::str_dump("The new domain of the computation = ",
                                                    isl_union_set_to_str(c_prods)));
                        c->set_iteration_domain(isl_set_from_union_set(isl_union_set_copy(c_prods)));
                    }
                    else
                    {
                        DEBUG(3, tiramisu::str_dump("The iteration domain of the computation is NOT a universe."));
                        isl_union_set *u = isl_union_set_union(isl_union_set_copy(c_prods),
                                                               isl_union_set_copy(c_dom));
                        c->set_iteration_domain(isl_set_from_union_set(isl_union_set_copy(u)));
                        DEBUG(3, tiramisu::str_dump("The new domain of the computation = ",
                                                    isl_union_set_to_str(u)));
                    }
                }
            }
            else
            {
                assert((isl_set_plain_is_universe(c->get_iteration_domain()) == isl_bool_false) && "The iteration domain of an update should not be universe.");
                assert((isl_set_is_empty(c->get_iteration_domain()) == isl_bool_false) && "The iteration domain of an update should not be empty.");
            }

            DEBUG(3, tiramisu::str_dump(""));
        }

        old_Producers = isl_union_set_copy(Producers);
        Producers = isl_union_set_apply(isl_union_set_copy(Producers), isl_union_map_copy(Reverse));
        DEBUG(3, tiramisu::str_dump("The new Producers : ", isl_union_set_to_str(Producers)));

        // If the graph of dependences has recursive dependences, then the intersection of
        // the old producers and the new producers will not be empty (i.e., the old producer and the new producer
        // are the same).
        // In this case, we should subtract the common domain so that in the next iterations of the
        // the algorithm we do not get the same computation again and again (since we have a recursive dependence).
        // This is equivalent to removing the recursive dependence (we remove its image instead of removing it).
        intersection = isl_union_set_intersect(old_Producers, isl_union_set_copy(Producers));
        if (isl_union_set_is_empty(intersection) == isl_bool_false)
        {
            isl_union_set *common_computations = isl_union_set_universe(intersection);
            Producers = isl_union_set_subtract(Producers, common_computations);
            DEBUG(3, tiramisu::str_dump("After eliminating the effect of recursive dependences.",
                                    isl_union_set_to_str(Producers)));
        }

    }
    DEBUG_INDENT(-4);

    DEBUG(3, tiramisu::str_dump("After propagating bounds:"));
    for (auto c : this->get_computations())
    {
        DEBUG(3, tiramisu::str_dump(isl_set_to_str(c->get_iteration_domain())));
    }

    DEBUG_INDENT(-4);
    DEBUG(3, tiramisu::str_dump("End of function"));
}

tiramisu::computation *tiramisu::computation::add_computations(std::string iteration_domain_str, tiramisu::expr e,
                        bool schedule_this_computation, tiramisu::primitive_t t,
                        tiramisu::function *fct)
{
    tiramisu::computation *C =
            new tiramisu::computation(iteration_domain_str, e,
                                      schedule_this_computation, t, fct);

    return C;
}

void tiramisu::function::dump_dep_graph()
{

    tiramisu::str_dump("Dependence graph:\n");
    isl_union_map *deps = isl_union_map_copy(this->compute_dep_graph());
    isl_union_map_dump(deps);
}

/**
  * Return a map that represents the buffers of the function.
  * The buffers of the function are buffers that are either passed
  * to the function as arguments or are buffers that are declared
  * and allocated within the function itself.
  * The names of the buffers are used as a key for the map.
  */
// @{
const std::map<std::string, tiramisu::buffer *> &function::get_buffers() const
{
    return buffers_list;
}
// @}

/**
   * Return a vector of the computations of the function.
   * The order of the computations in the vector does not have any
   * effect on the actual order of execution of the computations.
   * The order of execution of computations is specified through the
   * schedule.
   */
// @{
const std::vector<computation *> &function::get_computations() const
{
    return body;
}
// @}

/**
  * Return the context of the function. i.e. an ISL set that
  * represents constraints over the parameters of the functions
  * (a parameter is an invariant of the function).
  * An example of a context set is the following:
  *          "[N,M]->{: M>0 and N>0}"
  * This context set indicates that the two parameters N and M
  * are strictly positive.
  */
isl_set *function::get_program_context() const
{
    if (context_set != NULL)
    {
        return isl_set_copy(context_set);
    }
    else
    {
        return NULL;
    }
}

/**
  * Get the name of the function.
  */
const std::string &function::get_name() const
{
    return name;
}

/**
  * Return a vector representing the invariants of the function
  * (symbolic constants or variables that are invariant to the
  * function i.e. do not change their value during the execution
  * of the function).
  */
// @{
const std::vector<tiramisu::constant> &function::get_invariants() const
{
    return invariants;
}
// @}

/**
  * Return the Halide statement that represents the whole
  * function.
  * The Halide statement is generated by the code generator.
  * This function should not be called before calling the code
  * generator.
  */
Halide::Internal::Stmt function::get_halide_stmt() const
{
    assert(halide_stmt.defined() && ("Empty Halide statement"));

    return halide_stmt;
}

/**
  * Return the isl context associated with this function.
  */
isl_ctx *function::get_isl_ctx() const
{
    return ctx;
}

/**
  * Return the isl ast associated with this function.
  */
isl_ast_node *function::get_isl_ast() const
{
    assert((ast != NULL) && ("You should generate an ISL ast first (gen_isl_ast())."));

    return ast;
}

/**
  * Get the iterator names of the function.
  */
const std::vector<std::string> &function::get_iterator_names() const
{
    return iterator_names;
}

/**
  * Return true if the computation \p comp should be parallelized
  * at the loop level \p lev.
  */
bool function::should_parallelize(const std::string &comp, int lev) const
{
    assert(!comp.empty());
    assert(lev >= 0);

    for (const auto &pd : this->parallel_dimensions)
    {
        if ((pd.first == comp) && (pd.second == lev))
        {
            return true;
        }
    }
    return false;
}

/**
* Return true if the computation \p comp should be vectorized
* at the loop level \p lev.
*/
bool function::should_vectorize(const std::string &comp, int lev) const
{
    assert(!comp.empty());
    assert(lev >= 0);

    for (const auto &pd : this->vector_dimensions)
    {
        if ((pd.first == comp) && (pd.second == lev))
        {
            return true;
        }
    }
    return false;
}

void function::set_context_set(isl_set *context)
{
    assert((context != NULL) && "Context is NULL");

    this->context_set = context;
}

void function::set_context_set(const std::string &context_str)
{
    assert((!context_str.empty()) && "Context string is empty");

    this->context_set = isl_set_read_from_str(this->get_isl_ctx(), context_str.c_str());
    assert((context_set != NULL) && "Context set is NULL");
}

void function::add_context_constraints(const std::string &context_str)
{
    assert((!context_str.empty()) && "Context string is empty");

    if (this->context_set != NULL)
    {
        this->context_set =
            isl_set_intersect(this->context_set,
                              isl_set_read_from_str(this->get_isl_ctx(), context_str.c_str()));
    }
    else
    {
        this->context_set = isl_set_read_from_str(this->get_isl_ctx(), context_str.c_str());
    }
    assert((context_set != NULL) && "Context set is NULL");
}

/**
  * Set the iterator names of the function.
  */
void function::set_iterator_names(const std::vector<std::string> &iteratorNames)
{
    iterator_names = iteratorNames;
}

/**
  * Add an iterator to the function.
  */
void function::add_iterator_name(const std::string &iteratorName)
{
    iterator_names.push_back(iteratorName);
}

/**
  * Generate an object file.  This object file will contain the compiled
  * function.
  * \p obj_file_name indicates the name of the generated file.
  * \p os indicates the target operating system.
  * \p arch indicates the architecture of the target (the instruction set).
  * \p bits indicate the bit-width of the target machine.
  *    must be 0 for unknown, or 32 or 64.
  * For a full list of supported value for \p os and \p arch please
  * check the documentation of Halide::Target
  * (http://halide-lang.org/docs/struct_halide_1_1_target.html).
  * If the machine parameters are not supplied, it will detect one automatically.
  */
// @{
void function::gen_halide_obj(const std::string &obj_file_name) const
{
    Halide::Target target = Halide::get_host_target();
    gen_halide_obj(obj_file_name, target.os, target.arch, target.bits);
}
// @}


void tiramisu::computation::rename_computation(std::string new_name)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(this->get_function()->get_computation_by_name(new_name).empty());

    std::string old_name = this->get_name();

    this->set_name(new_name);

    // Rename the iteration domain.
    isl_set *dom = this->get_iteration_domain();
    dom = isl_set_set_tuple_name(dom, new_name.c_str());
    DEBUG(10, tiramisu::str_dump("Setting the iteration domain to ", isl_set_to_str(dom)));
    this->set_iteration_domain(dom);

    // Rename the access relation of the computation.
    isl_map *access = this->get_access_relation();
    access = isl_map_set_tuple_name(access, isl_dim_in, new_name.c_str());
    DEBUG(10, tiramisu::str_dump("Setting the access relation to ", isl_map_to_str(access)));
    this->set_access(access);

    // Rename the schedule
    isl_map *sched = this->get_schedule();
    sched = isl_map_set_tuple_name(sched, isl_dim_in, new_name.c_str());
    sched = isl_map_set_tuple_name(sched, isl_dim_out, new_name.c_str());
    DEBUG(10, tiramisu::str_dump("Setting the schedule relation to ", isl_map_to_str(sched)));
    this->set_schedule(sched);

    DEBUG_INDENT(-4);
}


/**
 * A pass to rename computations.
 * Computation that are defined multiple times need to be renamed, because
 * those computations in general have different expressions and the code
 * generator expects that computations that have the same name always have
 * the same expression and access relation. So we should rename them to avoid
 * any ambiguity for the code generator.
 *
 */
void tiramisu::function::rename_computations()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Computations that have been defined multiple times should
    // be renamed. ISL code generator expects computations with the
    // same name to have the same expressions and the same access
    // relation. So, "update" computations that have the same name
    // but have different expressions should be renamed first so
    // that we can use the original code generator without any
    // modification.
    for (auto const comp: this->get_computations())
    {
        std::vector<tiramisu::computation *> same_name_computations =
                this->get_computation_by_name(comp->get_name());

        int i = 0;

        if (same_name_computations.size() > 1)
            for (auto c: same_name_computations)
            {
                std::string new_name = "_" + c->get_name() + "_update_" + std::to_string(i);
                c->rename_computation(new_name);
                i++;
            }
    }

    DEBUG(3, tiramisu::str_dump("After renaming the computations."));
    DEBUG(3, this->dump(false));

    DEBUG_INDENT(-4);
}



/**
  * Generate an isl AST for the function.
  */
void function::gen_isl_ast()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Check that time_processor representation has already been computed,
    // TODO: check that the access was provided.
    assert(this->get_trimmed_time_processor_domain() != NULL);
    assert(this->get_aligned_identity_schedules() != NULL);

    isl_ctx *ctx = this->get_isl_ctx();
    isl_ast_build *ast_build;

    if (this->get_program_context() == NULL)
    {
        ast_build = isl_ast_build_alloc(ctx);
    }
    else
    {
        ast_build = isl_ast_build_from_context(isl_set_copy(this->get_program_context()));
    }

    isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
    isl_options_get_ast_build_exploit_nested_bounds(ctx);
    ast_build = isl_ast_build_set_after_each_for(ast_build, &tiramisu::for_code_generator_after_for,
                NULL);
    ast_build = isl_ast_build_set_at_each_domain(ast_build, &tiramisu::stmt_code_generator, this);

    // Set iterator names
    isl_id_list *iterators = isl_id_list_alloc(ctx, this->get_iterator_names().size());
    if (this->get_iterator_names().size() > 0)
    {

        std::string name = generate_new_variable_name();
        isl_id *id = isl_id_alloc(ctx, name.c_str(), NULL);
        iterators = isl_id_list_add(iterators, id);

        for (int i = 0; i < this->get_iterator_names().size(); i++)
        {
            name = this->get_iterator_names()[i];
            id = isl_id_alloc(ctx, name.c_str(), NULL);
            iterators = isl_id_list_add(iterators, id);

            name = generate_new_variable_name();
            id = isl_id_alloc(ctx, name.c_str(), NULL);
            iterators = isl_id_list_add(iterators, id);
        }

        ast_build = isl_ast_build_set_iterators(ast_build, iterators);
    }

    // Intersect the iteration domain with the domain of the schedule.
    isl_union_map *umap =
        isl_union_map_intersect_domain(
            isl_union_map_copy(this->get_aligned_identity_schedules()),
            isl_union_set_copy(this->get_trimmed_time_processor_domain()));

    DEBUG(3, tiramisu::str_dump("Schedule:", isl_union_map_to_str(this->get_schedule())));
    DEBUG(3, tiramisu::str_dump("Iteration domain:",
                                isl_union_set_to_str(this->get_iteration_domain())));
    DEBUG(3, tiramisu::str_dump("Trimmed Time-Processor domain:",
                                isl_union_set_to_str(this->get_trimmed_time_processor_domain())));
    DEBUG(3, tiramisu::str_dump("Trimmed Time-Processor aligned identity schedule:",
                                isl_union_map_to_str(this->get_aligned_identity_schedules())));
    DEBUG(3, tiramisu::str_dump("Identity schedule intersect trimmed Time-Processor domain:",
                                isl_union_map_to_str(umap)));
    DEBUG(3, tiramisu::str_dump("\n"));

    this->ast = isl_ast_build_node_from_schedule_map(ast_build, umap);

    isl_ast_build_free(ast_build);

    DEBUG_INDENT(-4);
}

/**
  * A helper function to split a string.
  */
// TODO: Test this function
void split_string(std::string str, std::string delimiter,
                  std::vector<std::string> &vector)
{
    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos)
    {
        token = str.substr(0, pos);
        vector.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    token = str.substr(0, pos);
    vector.push_back(token);
}

std::string generate_new_variable_name()
{
    return "t" + std::to_string(id_counter++);
}

/**
  * Methods for the computation class.
  */
void tiramisu::computation::tag_parallel_level(int par_dim)
{
    assert(par_dim >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    this->get_function()->add_parallel_dimension(this->get_name(), par_dim);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_gpu_level(int dim0, int dim1)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, -1, -1);
    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim1, -1, -1);
}

void tiramisu::computation::tag_gpu_level(int dim0, int dim1, int dim2, int dim3)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, dim1, -1);
    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim2, dim3, -1);
}

void tiramisu::computation::tag_gpu_level(int dim0, int dim1, int dim2, int dim3, int dim4,
        int dim5)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, dim1, dim2);
    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim3, dim4, dim5);
}

void tiramisu::computation::tag_gpu_block_level(int dim0)
{
    assert(dim0 >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, -1, -1);
}

void tiramisu::computation::tag_gpu_block_level(int dim0, int dim1)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, dim1, -1);
}

void tiramisu::computation::tag_gpu_block_level(int dim0, int dim1, int dim2)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_block_dimensions(this->get_name(), dim0, dim1, dim2);
}

void tiramisu::computation::tag_gpu_thread_level(int dim0)
{
    assert(dim0 >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim0, -1);
}

void tiramisu::computation::tag_gpu_thread_level(int dim0, int dim1)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim0, dim1);
}

void tiramisu::computation::tag_gpu_thread_level(int dim0, int dim1, int dim2)
{
    assert(dim0 >= 0);
    assert(dim1 >= 0);
    assert(dim1 == dim0 + 1);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_gpu_thread_dimensions(this->get_name(), dim0, dim1, dim2);
}

void tiramisu::computation::tag_vector_level(int dim)
{
    assert(dim >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_vector_dimension(this->get_name(), dim);
}

void tiramisu::computation::tag_unroll_level(int level)
{
    assert(level >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    this->get_function()->add_unroll_dimension(this->get_name(), level);

    DEBUG_INDENT(-4);
}

tiramisu::computation *tiramisu::computation::copy()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::computation *new_c =
        new tiramisu::computation(isl_set_to_str(isl_set_copy(this->get_iteration_domain())),
                                  this->get_expr(),
                                  this->should_schedule_this_computation(),
                                  this->get_data_type(),
                                  this->get_function());

    new_c->set_schedule(isl_map_copy(this->get_schedule()));

    new_c->access = isl_map_copy(this->access);
    new_c->relative_order = this->relative_order;
    new_c->is_let = this->is_let;

    DEBUG_INDENT(-4);

    return new_c;
}

std::vector<tiramisu::computation *> tiramisu::computation::separate(int dim, tiramisu::constant &C)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(this->get_function()->get_computation_by_name("_" + this->get_name()).empty());

    // Create the separated computation.
    // First, create the domain of the separated computation (which is identical to
    // the domain of the original computation).
    std::string domain_str = std::string(isl_set_to_str(this->get_iteration_domain()));
    int pos0 = domain_str.find(this->get_name());
    int len0 = this->get_name().length();
    domain_str.replace(pos0, len0, "_" + this->get_name());

    // TODO: create copy functions for all the classes so that we can copy the objects
    // we need to have this->get_expr().copy()
    tiramisu::computation *new_c = new tiramisu::computation(domain_str,
            this->get_expr(),
            this->should_schedule_this_computation(),
            this->get_data_type(),
            this->get_function());

    DEBUG(3, tiramisu::str_dump("Separated computation:\n"); new_c->dump());

    // Create the access relation of the separated computation (by replacing its name).
    std::string access_c_str = std::string(isl_map_to_str(this->get_access_relation()));
    int pos1 = access_c_str.find(this->get_name());
    int len1 = this->get_name().length();
    access_c_str.replace(pos1, len1, "_" + this->get_name());
    new_c->set_access(access_c_str);

    // TODO: for now we are not adding the new parameter to all the access functions,
    // iteration domains, schedules, ... We should either add it every where or transform
    // it into a variable (which is a way better method since it will allow us to
    // vectorize code that has a variable as loop bound (i<j).
    // We can use isl_space_align_params to align all the parameters.

    DEBUG(3, tiramisu::str_dump("Access of the separated computation:",
                                isl_map_to_str(new_c->get_access_relation())));

    // Create the constraints i<M and i>=M. To do so, first we need to create
    // the space of the constraints, which is identical to the space of the
    // iteration domain plus a new dimension that represents the separator
    // parameter.

    // First we create the space.
    isl_space *sp = isl_space_copy(isl_set_get_space(this->get_iteration_domain()));
    sp = isl_space_add_dims(sp, isl_dim_param, 1);
    int pos = isl_space_dim(sp, isl_dim_param) - 1;
    sp = isl_space_set_dim_name(sp, isl_dim_param, pos, C.get_name().c_str());
    isl_local_space *ls = isl_local_space_from_space(isl_space_copy(sp));

    // Second, we create the constraint i<M and add it to the original computation.
    // Since constraints in ISL are of the form X>=y, we transform the previous
    // constraint as follows
    // i <  M
    // i <= M-1
    // M-1-i >= 0
    isl_constraint *cst_upper = isl_constraint_alloc_inequality(isl_local_space_copy(ls));
    cst_upper = isl_constraint_set_coefficient_si(cst_upper, isl_dim_set, dim, -1);
    cst_upper = isl_constraint_set_coefficient_si(cst_upper, isl_dim_param, pos, 1);
    cst_upper = isl_constraint_set_constant_si(cst_upper, -1);

    this->set_iteration_domain(isl_set_add_constraint(this->get_iteration_domain(), cst_upper));

    // Third, we create the constraint i>=M and add it to the newly created computation.
    // i >= M
    // i - M >= 0
    isl_space *sp2 = isl_space_copy(isl_set_get_space(new_c->get_iteration_domain()));
    sp2 = isl_space_add_dims(sp2, isl_dim_param, 1);
    int pos2 = isl_space_dim(sp2, isl_dim_param) - 1;
    sp2 = isl_space_set_dim_name(sp2, isl_dim_param, pos2, C.get_name().c_str());
    isl_local_space *ls2 = isl_local_space_from_space(isl_space_copy(sp2));
    isl_constraint *cst_lower = isl_constraint_alloc_inequality(isl_local_space_copy(ls2));
    cst_lower = isl_constraint_set_coefficient_si(cst_lower, isl_dim_set, dim, 1);
    cst_lower = isl_constraint_set_coefficient_si(cst_lower, isl_dim_param, pos, -1);

    new_c->set_iteration_domain(isl_set_add_constraint(new_c->get_iteration_domain(), cst_lower));

    // Mark the separated computation to be executed after the original (full)
    // computation.
    new_c->after(*this, dim);

    DEBUG_INDENT(-4);

    return {this, new_c};
}

void tiramisu::computation::set_iteration_domain(isl_set *domain)
{
    this->iteration_domain = domain;
}

/*
 * Create a new Tiramisu constant M = v*floor(N/v) and use it as
 * a separator.
 *
 * Add the following constraints about the separator to the context:
 *  -  separator%v = 0
 *  -  separator <= loop_upper_bound
 *
 * The separator is used to separate a computation. That
 * is, it is used to create two identical computations where we have
 * a constraint like i<M in the first and i>=M in the second.
 * The first is called the full computation while the second is called
 * the separated computation.
 *
 * This function is used in vectorize and unroll mainly.
 */
tiramisu::constant *
tiramisu::computation::create_separator_and_add_constraints_to_context (
    const tiramisu::expr &loop_upper_bound, int v)
{
    /*
     * Create a new Tiramisu constant M = v*floor(N/v). This is the biggest
     * multiple of w that is still smaller than N.  Add this constant to
     * the list of invariants.
     */
    std::string separator_name = tiramisu::generate_new_variable_name();
    tiramisu::expr div_expr = tiramisu::expr(o_div, loop_upper_bound, tiramisu::expr(v));
    tiramisu::expr cast_expr = tiramisu::expr(o_cast, tiramisu::p_float32, div_expr);
    tiramisu::expr floor_expr = tiramisu::expr(o_floor, cast_expr);
    tiramisu::expr cast2_expr = tiramisu::expr(o_cast, tiramisu::p_int32, floor_expr);
    tiramisu::expr separator_expr = tiramisu::expr(o_mul, tiramisu::expr(v), cast2_expr);
    tiramisu::constant *separation_param = new tiramisu::constant(
        separator_name, separator_expr, p_uint32, true, NULL, 0, this->get_function());

    /**
     * Add the following constraints about the separator to the context:
     *  -  separator%v = 0
     *  -  separator <= loop_upper_bound
     */
    // Create a new context set.
    std::string constraint_parameters = "[" + separation_param->get_name()
                                        + "," + loop_upper_bound.get_name() + "]->";
    std::string constraint = constraint_parameters + "{ : ("
                             + separation_param->get_name() + ") % " + std::to_string(v)
                             + " = 0 and " + "(" + separation_param->get_name() + ") <= "
                             + loop_upper_bound.get_name() + " and ("
                             + separation_param->get_name() + ") > 0 and "
                             + loop_upper_bound.get_name() + " > 0 " + " }";
    isl_set *new_context_set = isl_set_read_from_str(this->get_ctx(), constraint.c_str());

    /*
     * Align the parameters of this set with the parameters of the iteration domain
     * (that is, we have to add the new parameter to the context and then take
     * its space as a model for alignment).
     */
    isl_set *original_context = this->get_function()->get_program_context();
    if (original_context != NULL)
    {
        // Create a space from the context and add a parameter.
        isl_space *sp = isl_set_get_space(original_context);
        sp = isl_space_add_dims(sp, isl_dim_param, 1);
        int pos = isl_space_dim(sp, isl_dim_param) - 1;
        sp = isl_space_set_dim_name(sp, isl_dim_param, pos, separator_name.c_str());
        this->set_iteration_domain(
            isl_set_align_params(isl_set_copy(this->get_iteration_domain()),
                                 isl_space_copy(sp)));
        this->get_function()->set_context_set(
            isl_set_align_params(
                isl_set_copy(this->get_function()->get_program_context()), sp));
        this->get_function()->set_context_set(
            isl_set_intersect(isl_set_copy(original_context), new_context_set));
    }
    else
    {
        this->get_function()->set_context_set(new_context_set);
    }
    return separation_param;
}

// TODO: support the vectorization of loops that has a constant (tiramisu::expr(10))
// as bound. Currently only loops that have a symbolic constant bound can be vectorized
// this is mainly because the vectorize function expects a "tiramisu::expr loop_upper_bound"
// as input.
// Idem for unroll.
// TODO: make vectorize and unroll retrieve the loop bound automatically.
void tiramisu::computation::vectorize(int L0, int v,
                                      tiramisu::expr loop_upper_bound)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    /*
     * Create a new Tiramisu constant M = v*floor(N/v) and use it as
     * a separator.
     */
    tiramisu::constant *separation_param =
        create_separator_and_add_constraints_to_context(loop_upper_bound, v);

    /*
     * Separate this computation using the parameter separation_param. That
     * is create two identical computations where we have a constraint like
     * i<M in the first and i>=M in the second.
     * The first is called the full computation while the second is called
     * the separated computation.
     * The names of the two computations is different. The name of the separated
     * computation is equal to the name of the full computation prefixed with "_".
     */
    this->separate(L0, *separation_param);

    /**
     * Split the full computation since the full computation will be vectorized.
     */
    this->split(L0, v);

    // Tag the inner loop after splitting to be vectorized. That loop
    // is supposed to have a constant extent.
    this->tag_vector_level(L0 + 1);
    this->get_function()->align_schedules();

    DEBUG_INDENT(-4);
}

void tiramisu::computation::unroll(int L0, int v,
                                   tiramisu::expr loop_upper_bound)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    /*
     * Create a new Tiramisu constant M = v*floor(N/v) and use it as
     * a separator.
     */
    tiramisu::constant *separation_param =
        create_separator_and_add_constraints_to_context(loop_upper_bound, v);

    /*
     * Separate this computation using the parameter separation_param. That
     * is create two identical computations where we have a constraint like
     * i<M in the first and i>=M in the second.
     * The first is called the full computation while the second is called
     * the separated computation.
     * The names of the two computations is different. The name of the separated
     * computation is equal to the name of the full computation prefixed with "_".
     */
    this->separate(L0, *separation_param);

    /**
     * Split the full computation since the full computation will be unrolled.
     */
    this->split(L0, v);

    // Tag the inner loop after splitting to be unrolled. That loop
    // is supposed to have a constant extent.
    this->tag_unroll_level(L0 + 1);
    this->get_function()->align_schedules();

    DEBUG_INDENT(-4);
}

void computation::dump_iteration_domain() const
{
    if (ENABLE_DEBUG)
    {
        isl_set_dump(this->get_iteration_domain());
    }
}

void function::dump_halide_stmt() const
{
    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("\n\n");
        tiramisu::str_dump("\nGenerated Halide Low Level IR:\n");
        std::cout << this->get_halide_stmt();
        tiramisu::str_dump("\n\n\n\n");
    }
}

void function::dump_trimmed_time_processor_domain() const
{
    // Create time space domain

    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("\n\nTrimmed Time-processor domain:\n");

        tiramisu::str_dump("Function " + this->get_name() + ":\n");
        for (const auto &comp : this->get_computations())
        {
            isl_set_dump(comp->get_trimmed_time_processor_domain());
        }

        tiramisu::str_dump("\n\n");
    }
}

void function::dump_time_processor_domain() const
{
    // Create time space domain

    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("\n\nTime-processor domain:\n");

        tiramisu::str_dump("Function " + this->get_name() + ":\n");
        for (const auto &comp : this->get_computations())
        {
            isl_set_dump(comp->get_time_processor_domain());
        }

        tiramisu::str_dump("\n\n");
    }
}

const bool tiramisu::computation::has_multiple_definitions() const
{
    return this->multiple_definitions;
}

void tiramisu::computation::set_has_multiple_definitions(bool val)
{
    this->multiple_definitions = val;
}

void function::gen_time_processor_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Rename updates so that they have different names because
    // the code generator expects each uniqueme name to have
    // an expression, different computations that have the same
    // name cannot have different expressions.
    this->rename_computations();

    this->align_schedules();

    for (auto &comp : this->get_computations())
    {
        comp->gen_time_processor_domain();
    }

    DEBUG_INDENT(-4);
}

void computation::dump_schedule() const
{
    DEBUG_INDENT(4);

    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("Dumping the schedule of the computation " + this->get_name() + " : ");

        std::flush(std::cout);
        isl_map_dump(this->get_schedule());
    }

    DEBUG_INDENT(-4);
}

void computation::dump() const
{
    if (ENABLE_DEBUG)
    {
        std::cout << std::endl << "Dumping the computation \"" + this->get_name() + "\" :" << std::endl;
        std::cout << "Iteration domain of the computation \"" << this->name << "\" : ";
        std::flush(std::cout);
        isl_set_dump(this->get_iteration_domain());
        std::flush(std::cout);
        this->dump_schedule();

        std::flush(std::cout);
        std::cout << "Expression of the computation : "; std::flush(std::cout);
        this->get_expr().dump(false);
        std::cout << std::endl; std::flush(std::cout);

        std::cout << "Access relation of the computation : "; std::flush(std::cout);
        isl_map_dump(this->get_access_relation());
        if (this->get_access_relation() == NULL)
            std::cout << "\n";
        std::flush(std::cout);

        if (this->get_time_processor_domain() != NULL)
        {
            std::cout << "Time-space domain " << std::endl; std::flush(std::cout);
            isl_set_dump(this->get_time_processor_domain());
        }
        else
        {
            std::cout << "Time-space domain : NULL." << std::endl;
        }

        std::cout << "Computation to be scheduled ? " << (this->schedule_this_computation) << std::endl;

        for (const auto &e : this->index_expr)
        {
            tiramisu::str_dump("Access expression:", (const char *)isl_ast_expr_to_C_str(e));
            tiramisu::str_dump("\n");
        }

        tiramisu::str_dump("Halide statement: ");
        if (this->stmt.defined())
        {
            std::cout << this->stmt;
        }
        else
        {
            tiramisu::str_dump("NULL");
        }
        tiramisu::str_dump("\n");
        tiramisu::str_dump("\n");
    }
}


int max_elem(std::vector<int> vec)
{
    int res = -1;

    for (auto v: vec)
        res = std::max(v, res);

    return res;
}

tiramisu::computation *buffer::allocate_at(tiramisu::computation *C, int level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(C != NULL);
    assert(level >= tiramisu::computation::root_dimension);
    assert(level < isl_set_dim(C->get_iteration_domain(), isl_dim_set));

    isl_set *iter = C->get_iteration_domain();
    int projection_dimension = level + 1;
    iter = isl_set_project_out(isl_set_copy(iter),
                               isl_dim_set,
                               projection_dimension,
                               isl_set_dim(iter, isl_dim_set) - projection_dimension);
    std::string new_name = "_allocation_" + generate_new_variable_name();
    iter = isl_set_set_tuple_name(iter, new_name.c_str());
    std::string iteration_domain_str = isl_set_to_str(iter);

    DEBUG(3, tiramisu::str_dump(
              "Computed iteration domain for the allocate() operation",
              isl_set_to_str(iter)));

    tiramisu::expr *new_expression = new tiramisu::expr(tiramisu::o_allocate, this->get_name());

    tiramisu::computation *alloc = new tiramisu::computation(iteration_domain_str,
                                                             *new_expression,
                                                             true, p_none, C->get_function());

    this->set_auto_allocate(false);

    DEBUG(3, tiramisu::str_dump("The computation representing the allocate() operator:");
          alloc->dump());

    DEBUG_INDENT(-4);

    return alloc;
}

void buffer::set_auto_allocate(bool auto_allocation)
{
    this->auto_allocate = auto_allocation;
}

bool buffer::get_auto_allocate()
{
    return this->auto_allocate;
}

void computation::set_schedule(std::string map_str)
{
    assert(!map_str.empty());
    assert(this->ctx != NULL);

    isl_map *map = isl_map_read_from_str(this->ctx, map_str.c_str());
    assert(map != NULL);

    this->set_schedule(map);
}

void computation::apply_transformation_on_schedule(std::string map_str)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(!map_str.empty());
    assert(this->ctx != NULL);

    isl_map *map = isl_map_read_from_str(this->ctx, map_str.c_str());
    assert(map != NULL);

    DEBUG(3, tiramisu::str_dump("Applying the following transformation on the schedule : "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(map)));

    isl_map *sched = this->get_schedule();
    sched = isl_map_apply_range(isl_map_copy(sched), isl_map_copy(map));
    this->set_schedule(sched);

    DEBUG(3, tiramisu::str_dump("Schedule after transformation : "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::apply_transformation_on_schedule_domain(std::string map_str)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(!map_str.empty());
    assert(this->ctx != NULL);

    isl_map *map = isl_map_read_from_str(this->ctx, map_str.c_str());
    assert(map != NULL);

    DEBUG(3, tiramisu::str_dump("Applying the following transformation on the domain of the schedule : "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(map)));

    isl_map *sched = this->get_schedule();
    sched = isl_map_apply_domain(isl_map_copy(sched), isl_map_copy(map));

    this->set_schedule(sched);

    DEBUG(3, tiramisu::str_dump("Schedule after transformation : "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::add_schedule_constraint(std::string domain_constraints,
                                          std::string range_constraints)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);


    assert(this->ctx != NULL);
    isl_map *sched = this->get_schedule();

    if (!domain_constraints.empty())
    {
        isl_set *domain_cst = isl_set_read_from_str(this->ctx, domain_constraints.c_str());
        assert(domain_cst != NULL);

        DEBUG(3, tiramisu::str_dump("Adding the following constraints to the domain of the schedule : "));
        DEBUG(3, tiramisu::str_dump(isl_set_to_str(domain_cst)));

        sched = isl_map_intersect_domain(isl_map_copy(sched), isl_set_copy(domain_cst));

    }

    if (!range_constraints.empty())
    {
        isl_set *range_cst = isl_set_read_from_str(this->ctx, range_constraints.c_str());

        DEBUG(3, tiramisu::str_dump("Adding the following constraints to the range of the schedule : "));
        DEBUG(3, tiramisu::str_dump(isl_set_to_str(range_cst)));

        sched = isl_map_intersect_range(isl_map_copy(sched), isl_set_copy(range_cst));
    }

    this->set_schedule(sched);

    DEBUG(3, tiramisu::str_dump("Schedule after transformation : "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

/**
  * Set the schedule of the computation.
  *
  * \p map is a string that represents a mapping from the iteration domain
  *  to the time-processor domain (the mapping is in the ISL format:
  *  http://isl.gforge.inria.fr/user.html#Sets-and-Relations).
  *
  * The name of the domain and range space must be identical.
  */
void tiramisu::computation::set_schedule(isl_map *map)
{
    this->schedule = map;
}

struct param_pack_1
{
    int in_dim;
    int out_constant;
};

/**
 * Take a basic map as input, go through all of its constraints,
 * identifies the constraint of the static dimension param_pack_1.in_dim
 * (passed in user) and replace the value of param_pack_1.out_constant if
 * the static dimension is bigger than that value.
 */
isl_stat extract_static_dim_value_from_bmap(__isl_take isl_basic_map *bmap, void *user)
{
    struct param_pack_1 *data = (struct param_pack_1 *) user;

    isl_constraint_list *list = isl_basic_map_get_constraint_list(bmap);
    int n_constraints = isl_constraint_list_n_constraint(list);

    for (int i = 0; i < n_constraints; i++)
    {
        isl_constraint *cst = isl_constraint_list_get_constraint(list, i);
        isl_val *val = isl_constraint_get_coefficient_val(cst, isl_dim_out, data->in_dim);
        if (isl_val_is_one(val)) // i.e., the coefficient of the dimension data->in_dim is 1
        {
            isl_val *val2 = isl_constraint_get_constant_val(cst);
            int const_val = (-1) * isl_val_get_num_si(val2);
            data->out_constant = const_val;
            DEBUG(3, tiramisu::str_dump("Dimensions found.  Constant = " +
                                        std::to_string(const_val)));
        }
    }

    return isl_stat_ok;
}

isl_stat extract_constant_value_from_bset(__isl_take isl_basic_set *bset, void *user)
{
    struct param_pack_1 *data = (struct param_pack_1 *) user;

    isl_constraint_list *list = isl_basic_set_get_constraint_list(bset);
    int n_constraints = isl_constraint_list_n_constraint(list);

    for (int i = 0; i < n_constraints; i++)
    {
        isl_constraint *cst = isl_constraint_list_get_constraint(list, i);
        if (isl_constraint_is_equality(cst) &&
                isl_constraint_involves_dims(cst, isl_dim_set, data->in_dim, 1))
        {
            isl_val *val = isl_constraint_get_coefficient_val(cst, isl_dim_out, data->in_dim);
            assert(isl_val_is_one(val));
            // assert that the coefficients of all the other dimension spaces are zero.

            isl_val *val2 = isl_constraint_get_constant_val(cst);
            int const_val = (-1) * isl_val_get_num_si(val2);
            data->out_constant = const_val;
            DEBUG(3, tiramisu::str_dump("Dimensions found.  Constant = " +
                                        std::to_string(const_val)));
        }
    }

    return isl_stat_ok;
}

/**
 * Return the value of the static dimension.
 *
 * For example, if we have a map M = {S0[i,j]->[0,0,i,1,j,2]; S0[i,j]->[1,0,i,1,j,3]}
 * and call isl_map_get_static_dim(M, 5, 1), it will return 3.
 */
int isl_map_get_static_dim(isl_map *map, int dim_pos)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(map != NULL);
    assert(dim_pos >= 0);
    assert(dim_pos <= (signed int) isl_map_dim(map, isl_dim_out));

    DEBUG(3, tiramisu::str_dump("Getting the constant coefficient of ",
                                isl_map_to_str(map));
          tiramisu::str_dump(" at dimension ");
          tiramisu::str_dump(std::to_string(dim_pos)));

    struct param_pack_1 *data = (struct param_pack_1 *) malloc(sizeof(struct param_pack_1));
    data->out_constant = 0;
    data->in_dim = dim_pos;

    isl_map_foreach_basic_map(isl_map_copy(map),
                              &extract_static_dim_value_from_bmap,
                              data);

    DEBUG(3, tiramisu::str_dump("The constant is: ");
          tiramisu::str_dump(std::to_string(data->out_constant)));

    DEBUG_INDENT(-4);

    return data->out_constant;
}

int isl_set_get_const_dim(isl_set *set, int dim_pos)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(set != NULL);
    assert(dim_pos >= 0);
    assert(dim_pos <= (signed int) isl_set_dim(set, isl_dim_out));

    DEBUG(3, tiramisu::str_dump("Getting the constant coefficient of ",
                                isl_set_to_str(set));
          tiramisu::str_dump(" at dimension ");
          tiramisu::str_dump(std::to_string(dim_pos)));

    struct param_pack_1 *data = (struct param_pack_1 *) malloc(sizeof(struct param_pack_1));
    data->out_constant = 0;
    data->in_dim = dim_pos;

    isl_set_foreach_basic_set(isl_set_copy(set),
                              &extract_constant_value_from_bset,
                              data);

    DEBUG(3, tiramisu::str_dump("The constant is: ");
          tiramisu::str_dump(std::to_string(data->out_constant)));

    DEBUG_INDENT(-4);

    return data->out_constant;
}

isl_map *isl_map_set_const_dim(isl_map *map, int dim_pos, int val)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(map != NULL);
    assert(dim_pos >= 0);
    assert(dim_pos <= (signed int) isl_map_dim(map, isl_dim_out));

    DEBUG(3, tiramisu::str_dump("Setting the constant coefficient of ",
                                isl_map_to_str(map));
          tiramisu::str_dump(" at dimension ");
          tiramisu::str_dump(std::to_string(dim_pos));
          tiramisu::str_dump(" into ");
          tiramisu::str_dump(std::to_string(val)));

    isl_space *sp = isl_map_get_space(map);
    isl_local_space *lsp =
        isl_local_space_from_space(isl_space_copy(sp));

    isl_map *identity = isl_set_identity(isl_map_range(isl_map_copy(map)));
    identity = isl_map_universe(isl_map_get_space(identity));

    sp = isl_map_get_space(identity);
    lsp = isl_local_space_from_space(isl_space_copy(sp));

    for (int i = 0; i < isl_map_dim(identity, isl_dim_out); i++)
        if (i == dim_pos)
        {
            isl_constraint *cst = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, dim_pos, 1);
            cst = isl_constraint_set_constant_si(cst, (-1) * (val));
            identity = isl_map_add_constraint(identity, cst);
        }
        else
        {
            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
            identity = isl_map_add_constraint(identity, cst2);
        }

    DEBUG(3, tiramisu::str_dump("Transformation map ", isl_map_to_str(identity)));

    map = isl_map_apply_range(map, identity);

    DEBUG(3, tiramisu::str_dump("After applying the transformation map: ",
                                isl_map_to_str(map)));

    DEBUG_INDENT(-4);

    return map;
}

isl_map *isl_map_add_dim_and_eq_constraint(isl_map *map, int dim_pos, int constant)
{
    assert(map != NULL);
    assert(dim_pos >= 0);
    assert(dim_pos <= (signed int) isl_map_dim(map, isl_dim_out));

    map = isl_map_insert_dims(map, isl_dim_out, dim_pos, 1);
    map = isl_map_set_tuple_name(map, isl_dim_out, isl_map_get_tuple_name(map, isl_dim_in));

    isl_space *sp = isl_map_get_space(map);
    isl_local_space *lsp =
        isl_local_space_from_space(isl_space_copy(sp));
    isl_constraint *cst = isl_constraint_alloc_equality(lsp);
    cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, dim_pos, 1);
    cst = isl_constraint_set_constant_si(cst, (-1) * constant);
    map = isl_map_add_constraint(map, cst);

    return map;
}

/**
 * Transform the loop level into its corresponding dynamic schedule
 * dimension.
 *
 * In the example below, the dynamic dimension that corresponds
 * to the loop level 0 is 1, and to 1 it is 3, ...
 *
 * The first dimension is the duplication dimension, the following
 * dimensions are static, dynamic, static, dynamic, ...
 *
 * Loop level               :    -1         0      1      2
 * Schedule dimension number:        0, 1   2  3   4  5   6  7
 * Schedule:                        [0, 0, i1, 0, i2, 0, i3, 0]
 */
int loop_level_into_dynamic_dimension(int level)
{
    return 1 + (level * 2 + 1);
}

/**
 * Transform the loop level into the first static schedule
 * dimension after its corresponding dynamic dimension.
 *
 * In the example below, the first static dimension that comes
 * after the corresponding dynamic dimension for
 * the loop level 0 is 3, and to 1 it is 5, ...
 *
 * Loop level               :    -1         0      1      2
 * Schedule dimension number:        0, 1   2  3   4  5   6  7
 * Schedule:                        [0, 0, i1, 0, i2, 0, i3, 0]
 */
int loop_level_into_static_dimension(int level)
{
    return loop_level_into_dynamic_dimension(level) + 1;
}

void computation::after(computation &comp, std::vector<int> levels)
{
    for (auto level: levels)
        this->after(comp, level);
}


/**
  * Implementation internals.
  *
  * This function gets as input a loop level and translates it
  * automatically to the appropriate schedule dimension by:
  * (1) getting the dynamic schedule dimension that corresponds to
  * that loop level, then adding +1 which corresponds to the first
  * static dimension that comes after the dynamic dimension.
  *
  * Explanation of what static and dynamic dimensions are:
  * In the time-processor domain, dimensions can be either static
  * or dynamic.  Static dimensions are used to order statements
  * within a given loop level while dynamic dimensions represent
  * the actual loop levels.  For example, the computations c0 and
  * c1 in the following loop nest
  *
  * for (i=0; i<N: i++)
  *   for (j=0; j<N; j++)
  *   {
  *     c0;
  *     c1;
  *   }
  *
  * have the following representations in the iteration domain
  *
  * {c0(i,j): 0<=i<N and 0<=j<N}
  * {c1(i,j): 0<=i<N and 0<=j<N}
  *
  * and the following representation in the time-processor domain
  *
  * {c0[0,i,0,j,0]: 0<=i<N and 0<=j<N}
  * {c1[0,i,0,j,1]: 0<=i<N and 0<=j<N}
  *
  * The first dimension (dimension 0) in the time-processor
  * representation (the leftmost dimension) is a static dimension,
  * the second dimension (dimension 1) is a dynamic dimension that
  * represents the loop level i, ..., the forth dimension is a dynamic
  * dimension that represents the loop level j and the last dimension
  * (dimension 4) is a static dimension and allows the ordering of
  * c1 after c0 in the loop nest.
  *
  * \p dim has to be a static dimension, i.e. 0, 2, 4, 6, ...
  */
void computation::after(computation &comp, int level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // for loop level i return 2*i+1 which represents the
    // the static dimension just after the dynamic dimension that
    // represents the loop level i.
    int dim = loop_level_into_static_dimension(level);

    DEBUG(3, tiramisu::str_dump("Setting the schedule of ");
          tiramisu::str_dump(this->get_name());
          tiramisu::str_dump(" after ");
          tiramisu::str_dump(comp.get_name());
          tiramisu::str_dump(" at dimension ");
          tiramisu::str_dump(std::to_string(dim)));

    comp.get_function()->align_schedules();

    DEBUG(3, tiramisu::str_dump("Preparing to adjust the schedule of the computation ");
          tiramisu::str_dump(this->get_name()));
    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(this->get_schedule())));

    assert(this->get_schedule() != NULL);
    DEBUG(3, tiramisu::str_dump("Dimension level in which ordering dimensions will be inserted : ");
          tiramisu::str_dump(std::to_string(dim)));
    assert(dim < (signed int) isl_map_dim(this->get_schedule(), isl_dim_out));
    assert(dim >= computation::root_dimension);

    // Get the constant in comp, add +1 to it and set it to sched1
    int order = isl_map_get_static_dim(comp.get_schedule(), dim);
    isl_map *new_sched = isl_map_copy(this->get_schedule());
    new_sched = add_eq_to_schedule_map(dim, 0, -1, order + 1, new_sched);
    this->set_schedule(new_sched);
    DEBUG(3, tiramisu::str_dump("Schedule adjusted: ",
                                isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}


void computation::before(computation &comp, std::vector<int> dims)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    for (auto dim: dims)
        comp.after(*this, dim);

    DEBUG_INDENT(-4);
}


void computation::before(computation &comp, int dim)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    comp.after(*this, dim);

    DEBUG_INDENT(-4);
}

void computation::gpu_tile(int L0, int L1, int sizeX, int sizeY)
{
    assert(L0 >= 0);
    assert(L1 >= 0);
    assert((L1 == L0 + 1));
    assert(sizeX > 0);
    assert(sizeY > 0);

    this->tile(L0, L1, sizeX, sizeY);
    this->tag_gpu_block_level(L0, L1);
    this->tag_gpu_thread_level(L0 + 2, L1 + 2);
}

void computation::gpu_tile(int L0, int L1, int L2, int sizeX, int sizeY, int sizeZ)
{
    assert(L0 >= 0);
    assert(L1 >= 0);
    assert(L2 >= 0);
    assert((L1 == L0 + 1));
    assert((L2 == L1 + 1));
    assert(sizeX > 0);
    assert(sizeY > 0);
    assert(sizeZ > 0);

    this->tile(L0, L1, L2, sizeX, sizeY, sizeZ);
    this->tag_gpu_block_level(L0, L1, L2);
    this->tag_gpu_thread_level(L0 + 3, L1 + 3, L2 + 3);
}

void computation::tile(int L0, int L1, int sizeX, int sizeY)
{
    // Check that the two dimensions are consecutive.
    // Tiling only applies on a consecutive band of loop dimensions.
    assert(L0 >= 0);
    assert(L1 >= 0);
    assert((L1 == L0 + 1));
    assert(sizeX > 0);
    assert(sizeY > 0);
    assert(this->get_iteration_domain() != NULL);

    assert(loop_level_into_dynamic_dimension(L1) < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    this->split(L0, sizeX);
    this->split(L1 + 1, sizeY);
    this->interchange(L0 + 1, L1 + 1);

    DEBUG_INDENT(-4);
}

void computation::tile(int L0, int L1, int L2, int sizeX, int sizeY, int sizeZ)
{
    // Check that the two dimensions are consecutive.
    // Tiling only applies on a consecutive band of loop dimensions.
    assert(L0 >= 0);
    assert(L1 >= 0);
    assert(L2 >= 0);
    assert((L1 == L0 + 1));
    assert((L2 == L1 + 1));
    assert(sizeX > 0);
    assert(sizeY > 0);
    assert(sizeZ > 0);
    assert(this->get_iteration_domain() != NULL);

    assert(loop_level_into_dynamic_dimension(L0) < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));
    assert(loop_level_into_dynamic_dimension(L1) < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));
    assert(loop_level_into_dynamic_dimension(L2) < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    //  Original loops
    //  L0
    //    L1
    //      L2

    this->split(L0, sizeX); // Split L0 into L0 and L0_prime
    // Compute the new L1 and the new L2 and the newly created L0 (called L0 prime)
    int L0_prime = L0 + 1;
    L1 = L1 + 1;
    L2 = L2 + 1;

    //  Loop after transformation
    //  L0
    //    L0_prime
    //      L1
    //        L2

    this->split(L1, sizeY);
    int L1_prime = L1 + 1;
    L2 = L2 + 1;

    //  Loop after transformation
    //  L0
    //    L0_prime
    //      L1
    //        L1_prime
    //          L2

    this->split(L2, sizeZ);

    //  Loop after transformation
    //  L0
    //    L0_prime
    //      L1
    //        L1_prime
    //          L2
    //            L2_prime

    this->interchange(L0_prime, L1);
    // Change the position of L0_prime to the new position
    int temp = L0_prime;
    L0_prime = L1;
    L1 = temp;

    //  Loop after transformation
    //  L0
    //    L1
    //      L0_prime
    //        L1_prime
    //          L2
    //            L2_prime

    this->interchange(L0_prime, L2);
    // Change the position of L0_prime to the new position
    temp = L0_prime;
    L0_prime = L2;
    L2 = temp;

    //  Loop after transformation
    //  L0
    //    L1
    //      L2
    //        L1_prime
    //          L0_prime
    //            L2_prime

    this->interchange(L1_prime, L0_prime);

    //  Loop after transformation
    //  L0
    //    L1
    //      L2
    //        L0_prime
    //          L1_prime
    //            L2_prime

    DEBUG_INDENT(-4);
}

/**
 * This function modifies the schedule of the computation so that the two loop
 * levels L0 and L1 are interchanged (swapped).
 */
void computation::interchange(int L0, int L1)
{
    int inDim0 = loop_level_into_dynamic_dimension(L0);
    int inDim1 = loop_level_into_dynamic_dimension(L1);

    assert(inDim0 >= 0);
    assert(inDim0 < isl_space_dim(isl_map_get_space(this->get_schedule()),
                                  isl_dim_out));
    assert(inDim1 >= 0);
    assert(inDim1 < isl_space_dim(isl_map_get_space(this->get_schedule()),
                                  isl_dim_out));

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_map *schedule = this->get_schedule();

    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(schedule)));
    DEBUG(3, tiramisu::str_dump("Interchanging the dimensions " + std::to_string(
                                    L0) + " and " + std::to_string(L1)));

    int n_dims = isl_map_dim(schedule, isl_dim_out);

    std::string inDim0_str = isl_map_get_dim_name(schedule, isl_dim_out, inDim0);
    std::string inDim1_str = isl_map_get_dim_name(schedule, isl_dim_out, inDim1);

    std::vector<isl_id *> dimensions;

    // ------------------------------------------------------------
    // Create a map for the duplicate schedule.
    // ------------------------------------------------------------

    std::string map = "{ " + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            int duplicate_ID = isl_map_get_static_dim(schedule, 0);
            map = map + std::to_string(duplicate_ID);
        }
        else
        {
            if (isl_map_get_dim_name(schedule, isl_dim_out, i) == NULL)
            {
                isl_id *new_id = isl_id_alloc(this->get_ctx(), generate_new_variable_name().c_str(), NULL);
                schedule = isl_map_set_dim_id(schedule, isl_dim_out, i, new_id);
            }

            map = map + isl_map_get_dim_name(schedule, isl_dim_out, i);
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] ->" + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            int duplicate_ID = isl_map_get_static_dim(schedule, 0);
            map = map + std::to_string(duplicate_ID);
        }
        else
        {
            if ((i != inDim0) && (i != inDim1))
            {
                map = map + isl_map_get_dim_name(schedule, isl_dim_out, i);
                dimensions.push_back(isl_map_get_dim_id(schedule, isl_dim_out, i));
            }
            else if (i == inDim0)
            {
                map = map + inDim1_str;
                isl_id *id1 = isl_id_alloc(this->get_ctx(), inDim1_str.c_str(), NULL);
                dimensions.push_back(id1);
            }
            else if (i == inDim1)
            {
                map = map + inDim0_str;
                isl_id *id1 = isl_id_alloc(this->get_ctx(), inDim0_str.c_str(), NULL);
                dimensions.push_back(id1);
            }
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "]}";

    DEBUG(3, tiramisu::str_dump("A map that transforms the duplicate"));
    DEBUG(3, tiramisu::str_dump(map.c_str()));

    isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());


    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_in, isl_map_get_tuple_id(isl_map_copy(schedule), isl_dim_out));
    isl_id *id_range = isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL);
    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_out, id_range);


    // Check that the names of each dimension is well set
    for (int i = 1; i < isl_map_dim(transformation_map, isl_dim_in); i++)
    {
        isl_id *dim_id = isl_id_copy(dimensions[i - 1]);
        transformation_map = isl_map_set_dim_id(transformation_map, isl_dim_out, i, dim_id);
        assert(isl_map_has_dim_name(transformation_map, isl_dim_in, i));
        assert(isl_map_has_dim_name(transformation_map, isl_dim_out, i));
    }

    DEBUG(3, tiramisu::str_dump("Final transformation map : ", isl_map_to_str(transformation_map)));

    schedule = isl_map_apply_range(isl_map_copy(schedule), isl_map_copy(transformation_map));

    DEBUG(3, tiramisu::str_dump("Schedule after interchange: ", isl_map_to_str(schedule)));

    this->set_schedule(schedule);

    DEBUG_INDENT(-4);
}

/**
 * Get a map as input.  Go through all the basic maps, keep only
 * the basic map of the duplicate ID.
 */
isl_map *isl_map_filter_bmap_by_dupliate_ID(int ID, isl_map *map)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_map *identity = isl_map_universe(isl_map_get_space(map));
    identity = isl_set_identity(isl_map_range(isl_map_copy(map)));
    DEBUG(3, tiramisu::str_dump("Identity created from the range of the map: ",
                                isl_map_to_str(identity)));

    identity = isl_map_set_const_dim(identity, 0, ID);

    return isl_map_apply_range(isl_map_copy(map), identity);

    DEBUG_INDENT(-4);
}

/**
 * domain_constraints_set: a set defined on the space of the domain of the
 * schedule.
 *
 * range_constraints_set: a set defined on the space of the range of the
 * schedule.
 */
tiramisu::computation *computation::duplicate(std::string domain_constraints_set,
        std::string range_constraints_set)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);


    DEBUG(3, tiramisu::str_dump("Creating a schedule that duplicates ");
          tiramisu::str_dump(this->get_name()););
    DEBUG(3, tiramisu::str_dump("The duplicate is defined with the following constraints on the domain of the schedule: ");
          tiramisu::str_dump(domain_constraints_set));
    DEBUG(3, tiramisu::str_dump("and the following constraints on the range of the schedule: ");
          tiramisu::str_dump(range_constraints_set));

    this->get_function()->align_schedules();


    DEBUG(3, tiramisu::str_dump("Preparing to adjust the schedule of the computation ");
          tiramisu::str_dump(this->get_name()));
    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(this->get_schedule())));

    assert(this->get_schedule() != NULL);

    DEBUG(3, tiramisu::str_dump("The ID of the last duplicate of this computation (i.e., number of duplicates) is : "
                                + std::to_string(this->get_duplicates_number())));

    DEBUG(3, tiramisu::str_dump("Now creating a map for the new duplicate."));
    int new_ID = this->get_duplicates_number() + 1;
    this->duplicate_number++; // Increment the duplicate number.
    isl_map *new_sched = isl_map_copy(this->get_schedule());
    DEBUG(3, tiramisu::str_dump("The map of the original: ", isl_map_to_str(new_sched)));

    // Intersecting the range of the schedule with the domain and range provided by the user.
    isl_set *domain_set = NULL;
    if (domain_constraints_set.length() > 0)
    {
        domain_set = isl_set_read_from_str(this->get_ctx(), domain_constraints_set.c_str());
    }
    isl_set *range_set = NULL;
    if (range_constraints_set.length() > 0)
    {
        range_set = isl_set_read_from_str(this->get_ctx(), range_constraints_set.c_str());
    }

    if (domain_set != NULL)
    {
        DEBUG(3, tiramisu::str_dump("Intersecting the following schedule and set on the domain."));
        DEBUG(3, tiramisu::str_dump("Schedule: ", isl_map_to_str(new_sched)));
        DEBUG(3, tiramisu::str_dump("Set: ", isl_set_to_str(domain_set)));

        new_sched = isl_map_intersect_domain(new_sched, domain_set);
    }

    if (range_set != NULL)
    {
        DEBUG(3, tiramisu::str_dump("Intersecting the following schedule and set on the range."));
        DEBUG(3, tiramisu::str_dump("Schedule: ", isl_map_to_str(new_sched)));
        DEBUG(3, tiramisu::str_dump("Set: ", isl_set_to_str(range_set)));

        new_sched = isl_map_intersect_range(new_sched, range_set);
    }

    new_sched = this->simplify(new_sched);
    DEBUG(3, tiramisu::str_dump("Resulting schedule: ", isl_map_to_str(new_sched)));


    // Setting the duplicate dimension
    new_sched = isl_map_set_const_dim(new_sched, 0, new_ID);
    DEBUG(3, tiramisu::str_dump("After setting the dimension 0 to the new_ID: ",
                                isl_map_to_str(new_sched)));
    DEBUG(3, tiramisu::str_dump("The map of the new duplicate is now: ", isl_map_to_str(new_sched)));

    // Create the duplicate computation.
    tiramisu::computation *new_c = this->copy();
    new_c->set_schedule(isl_map_copy(new_sched));

    DEBUG(3, tiramisu::str_dump("The schedule of the original computation: "));
    isl_map_dump(this->get_schedule());
    DEBUG(3, tiramisu::str_dump("The schedule of the duplicate: "));
    isl_map_dump(new_c->get_schedule());

    DEBUG_INDENT(-4);

    return new_c;
}

// TODO: fix this function
isl_map *add_eq_to_schedule_map(int dim0, int in_dim_coefficient, int out_dim_coefficient,
                                int const_conefficient, isl_map *sched)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("The schedule :", isl_map_to_str(sched)));
    DEBUG(3, tiramisu::str_dump("Editing the dimension " + std::to_string(dim0)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the input dimension " + std::to_string(
                                    in_dim_coefficient)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the output dimension " + std::to_string(
                                    out_dim_coefficient)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the constant " + std::to_string(const_conefficient)));

    isl_map *identity = isl_set_identity(isl_map_range(isl_map_copy(sched)));
    identity = isl_map_universe(isl_map_get_space(identity));
    isl_space *sp = isl_map_get_space(identity);
    isl_local_space *lsp = isl_local_space_from_space(isl_space_copy(sp));

    // Create a transformation map that transforms the schedule.
    for (int i = 0; i < isl_map_dim (identity, isl_dim_out); i++)
        if (i == dim0)
        {
            isl_constraint *cst = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, dim0, in_dim_coefficient);
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, dim0, -out_dim_coefficient);
            // TODO: this should be inverted into const_conefficient.
            cst = isl_constraint_set_constant_si(cst, -const_conefficient);
            identity = isl_map_add_constraint(identity, cst);
            DEBUG(3, tiramisu::str_dump("Setting the constraint for dimension " + std::to_string(dim0)));
            DEBUG(3, tiramisu::str_dump("The identity schedule is now: ", isl_map_to_str(identity)));
        }
        else
        {
            // Set equality constraints for dimensions
            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
            identity = isl_map_add_constraint(identity, cst2);
        }

    isl_map *final_identity = identity;
    DEBUG(3, tiramisu::str_dump("The transformation map is: ", isl_map_to_str(final_identity)));
    sched = isl_map_apply_range (sched, final_identity);
    DEBUG(3, tiramisu::str_dump("The schedule after being transformed: ", isl_map_to_str(sched)));

    DEBUG_INDENT(-4);

    return sched;
}

isl_map *add_ineq_to_schedule_map(int duplicate_ID, int dim0, int in_dim_coefficient,
                                  int out_dim_coefficient, int const_conefficient, isl_map *sched)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Editing the duplicate " + std::to_string(
                                    duplicate_ID) + " of the schedule :", isl_map_to_str(sched)));
    DEBUG(3, tiramisu::str_dump("Editing the dimension " + std::to_string(dim0)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the input dimension " + std::to_string(
                                    in_dim_coefficient)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the output dimension " + std::to_string(
                                    out_dim_coefficient)));
    DEBUG(3, tiramisu::str_dump("Coefficient of the constant " + std::to_string(const_conefficient)));

    isl_map *identity = isl_set_identity(isl_map_range(isl_map_copy(sched)));
    identity = isl_map_universe(isl_map_get_space(identity));
    isl_space *sp = isl_map_get_space(identity);
    isl_local_space *lsp = isl_local_space_from_space(isl_space_copy(sp));

    // Create a transformation map that applies only on the map that have
    // duplicate_ID as an ID.
    for (int i = 0; i < isl_map_dim (identity, isl_dim_out); i++)
        if (i == 0)
        {
            isl_constraint *cst = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, 0, 1);
            cst = isl_constraint_set_constant_si(cst, -duplicate_ID);
            identity = isl_map_add_constraint(identity, cst);

            // Set equality constraints for the first dimension (to keep the value of the duplicate ID)
            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
            identity = isl_map_add_constraint(identity, cst2);

            DEBUG(3, tiramisu::str_dump("Setting the constant " + std::to_string(
                                            duplicate_ID) + " for dimension 0."));
        }
        else if (i == dim0)
        {
            isl_constraint *cst = isl_constraint_alloc_inequality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, dim0, in_dim_coefficient);
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, dim0, -out_dim_coefficient);
            cst = isl_constraint_set_constant_si(cst, -const_conefficient);
            identity = isl_map_add_constraint(identity, cst);
            DEBUG(3, tiramisu::str_dump("Setting the constraint for dimension " + std::to_string(dim0)));
            DEBUG(3, tiramisu::str_dump("The identity schedule is now: ", isl_map_to_str(identity)));
        }
        else
        {
            // Set equality constraints for dimensions
            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
            identity = isl_map_add_constraint(identity, cst2);
        }

    isl_map *final_identity = identity;
    DEBUG(3, tiramisu::str_dump("The identity schedule is now: ", isl_map_to_str(final_identity)));

    isl_map *identity2;

    // Now set map that keep schedules of the other duplicates without any modification.
    DEBUG(3, tiramisu::str_dump("Setting a map to keep the schedules of the other duplicates that have an ID > this duplicate"));
    identity2 = isl_set_identity(isl_map_range(isl_map_copy(sched)));
    identity2 = isl_map_universe(isl_map_get_space(identity2));
    sp = isl_map_get_space(identity2);
    lsp = isl_local_space_from_space(isl_space_copy(sp));
    for (int i = 0; i < isl_map_dim (identity2, isl_dim_out); i++)
    {
        if (i == 0)
        {
            isl_constraint *cst = isl_constraint_alloc_inequality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, 0, 1);
            cst = isl_constraint_set_constant_si(cst, -duplicate_ID - 1);
            identity2 = isl_map_add_constraint(identity2, cst);
        }
        isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
        cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
        cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
        identity2 = isl_map_add_constraint(identity2, cst2);
    }

    DEBUG(3, tiramisu::str_dump("The identity schedule is now: ", isl_map_to_str(identity2)));
    final_identity = isl_map_union (final_identity, identity2);

    if (duplicate_ID > 0)
    {
        DEBUG(3, tiramisu::str_dump("Setting a map to keep the schedules of the other duplicates that have an ID < this duplicate"));
        identity2 = isl_set_identity(isl_map_range(isl_map_copy(sched)));
        identity2 = isl_map_universe(isl_map_get_space(identity2));
        sp = isl_map_get_space(identity2);
        lsp = isl_local_space_from_space(isl_space_copy(sp));
        for (int i = 0; i < isl_map_dim (identity2, isl_dim_out); i++)
        {
            if (i == 0)
            {
                isl_constraint *cst = isl_constraint_alloc_inequality(isl_local_space_copy(lsp));
                cst = isl_constraint_alloc_inequality(isl_local_space_copy(lsp));
                cst = isl_constraint_set_coefficient_si(cst, isl_dim_in, 0, -1);
                cst = isl_constraint_set_constant_si(cst, duplicate_ID - 1);
                identity2 = isl_map_add_constraint(identity2, cst);
            }
            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_in, i, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_out, i, -1);
            identity2 = isl_map_add_constraint(identity2, cst2);
        }
        DEBUG(3, tiramisu::str_dump("The identity schedule is now: ", isl_map_to_str(identity2)));
        final_identity = isl_map_union (final_identity, identity2);
    }

    DEBUG(3, tiramisu::str_dump("The final transformation map is: ", isl_map_to_str(final_identity)));
    sched = isl_map_apply_range (sched, final_identity);
    DEBUG(3, tiramisu::str_dump("The schedule after being modified: ", isl_map_to_str(sched)));

    DEBUG_INDENT(-4);

    return sched;
}

void computation::shift(int L0, int n)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int dim0 = loop_level_into_dynamic_dimension(L0);

    assert(this->get_schedule() != NULL);
    assert(dim0 >= 0);
    assert(dim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));


    DEBUG(3, tiramisu::str_dump("Creating a schedule that shifts the loop level ");
          tiramisu::str_dump(std::to_string(L0));
          tiramisu::str_dump(" of the computation ");
          tiramisu::str_dump(this->get_name());
          tiramisu::str_dump(" by ");
          tiramisu::str_dump(std::to_string(n)));

    this->get_function()->align_schedules();
    assert(this->get_schedule() != NULL);

    DEBUG(3, tiramisu::str_dump("Original schedule: ",
                                isl_map_to_str(this->get_schedule())));

    isl_map *new_sched = isl_map_copy(this->get_schedule());
    new_sched = add_eq_to_schedule_map(dim0, -1, -1, n, new_sched);
    this->set_schedule(new_sched);
    DEBUG(3, tiramisu::str_dump("Schedule after shifting: ",
                                isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

isl_set *computation::simplify(isl_set *set)
{
    set = this->intersect_set_with_context(set);
    set = isl_set_coalesce(set);
    set = isl_set_remove_redundancies(set);

    return set;
}

isl_map *computation::simplify(isl_map *map)
{
    map = this->intersect_map_domain_with_context(map);
    map = isl_map_coalesce(map);

    return map;
}

isl_set *computation::intersect_set_with_context(isl_set *set)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Unify the space of the context and the "missing" set so that we can intersect them.
    isl_set *context = isl_set_copy(this->get_function()->get_program_context());
    if (context != NULL)
    {
        isl_space *model = isl_set_get_space(isl_set_copy(context));
        set = isl_set_align_params(set, isl_space_copy(model));
        DEBUG(10, tiramisu::str_dump("Context: ", isl_set_to_str(context)));
        DEBUG(10, tiramisu::str_dump("Set after aligning its parameters with the context parameters: ",
                                     isl_set_to_str (set)));

        isl_id *missing_id1 = NULL;
        if (isl_set_has_tuple_id(set) == isl_bool_true)
        {
            missing_id1 = isl_set_get_tuple_id(set);
        }
        else
        {
            std::string name = isl_set_get_tuple_name(set);
            assert(name.size() > 0);
            missing_id1 = isl_id_alloc(this->get_ctx(), name.c_str(), NULL);
        }

        int nb_dims = isl_set_dim(set, isl_dim_set);
        context = isl_set_add_dims(context, isl_dim_set, nb_dims);
        DEBUG(10, tiramisu::str_dump("Context after adding dimensions to make it have the same number of dimensions as missing: ",
                                     isl_set_to_str (context)));
        context = isl_set_set_tuple_id(context, isl_id_copy(missing_id1));
        DEBUG(10, tiramisu::str_dump("Context after setting its tuple ID to be equal to the tuple ID of missing: ",
                                     isl_set_to_str (context)));
        set = isl_set_intersect(set, isl_set_copy(context));
        DEBUG(10, tiramisu::str_dump("Set after intersecting with the program context: ",
                                     isl_set_to_str (set)));
    }

    DEBUG_INDENT(-4);

    return set;
}

isl_map *computation::intersect_map_domain_with_context(isl_map *map)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Unify the space of the context and the "missing" set so that we can intersect them.
    isl_set *context = isl_set_copy(this->get_function()->get_program_context());
    if (context != NULL)
    {
        isl_space *model = isl_set_get_space(isl_set_copy(context));
        map = isl_map_align_params(map, isl_space_copy(model));
        DEBUG(10, tiramisu::str_dump("Context: ", isl_set_to_str(context)));
        DEBUG(10, tiramisu::str_dump("Map after aligning its parameters with the context parameters: ",
                                     isl_map_to_str(map)));

        isl_id *missing_id1 = NULL;
        if (isl_map_has_tuple_id(map, isl_dim_in) == isl_bool_true)
        {
            missing_id1 = isl_map_get_tuple_id(map, isl_dim_in);
        }
        else
        {
            std::string name = isl_map_get_tuple_name(map, isl_dim_in);
            assert(name.size() > 0);
            missing_id1 = isl_id_alloc(this->get_ctx(), name.c_str(), NULL);
        }

        int nb_dims = isl_map_dim(map, isl_dim_in);
        context = isl_set_add_dims(context, isl_dim_set, nb_dims);
        DEBUG(10, tiramisu::str_dump("Context after adding dimensions to make it have the same number of dimensions as missing: ",
                                     isl_set_to_str (context)));
        context = isl_set_set_tuple_id(context, isl_id_copy(missing_id1));
        DEBUG(10, tiramisu::str_dump("Context after setting its tuple ID to be equal to the tuple ID of missing: ",
                                     isl_set_to_str (context)));
        map = isl_map_intersect_domain(map, isl_set_copy(context));
        DEBUG(10, tiramisu::str_dump("Map after intersecting with the program context: ",
                                     isl_map_to_str(map)));
    }

    DEBUG_INDENT(-4);

    return map;
}

/**
 * Assuming the set missing is the set of missing computations that will be
 * duplicated. The duplicated computations may needed to be shifted so that
 * they are executed with the original computation rather than being executed
 * after the original computation.
 * This function figures out the shift degree for each dimension of the missing
 * set.
 *
 * - For each dimension d in [0 to L]:
 *      * Project all the dimensions of the missing set except the dimension d.
 *      * The shift factor is obtained as follows:
 *              For the remaining the negative of the constant value of that dimension.
 */
std::vector<int> get_shift_degrees(isl_set *missing, int L)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<int> shifts;

    DEBUG(3, tiramisu::str_dump("Getting the shift degrees for the missing set."));
    DEBUG(3, tiramisu::str_dump("The missing set is: ", isl_set_to_str(missing)));
    DEBUG(3, tiramisu::str_dump("Get the shift degrees up to the loop level : " + std::to_string(L)));

    for (int i = 0; i <= L; i++)
    {
        isl_set *m = isl_set_copy(missing);
        int dim = loop_level_into_dynamic_dimension(i);
        int max_dim = loop_level_into_dynamic_dimension(L);
        DEBUG(3, tiramisu::str_dump("The current dynamic dimension is: " + std::to_string(dim)));

        DEBUG(3, tiramisu::str_dump("Projecting out all the dimensions of the set except the dimension " +
                                    std::to_string(dim)));

        if (dim != 0)
        {
            m = isl_set_project_out(m, isl_dim_set, 0, dim);
            DEBUG(10, tiramisu::str_dump("Projecting " + std::to_string(dim) +
                                         " dimensions starting from dimension 0."));
        }

        DEBUG(10, tiramisu::str_dump("After projection: ", isl_set_to_str(m)));

        if (dim != max_dim)
        {
            int last_dim = isl_set_dim(m, isl_dim_set);
            DEBUG(10, tiramisu::str_dump("Projecting " + std::to_string(last_dim - 1) +
                                         " dimensions starting from dimension 1."));
            m = isl_set_project_out(m, isl_dim_set, 1, last_dim - 1);
        }

        DEBUG(3, tiramisu::str_dump("After projection: ", isl_set_to_str(m)));

        /**
         * TODO: We assume that the set after projection is of the form
         * [T0]->{[i0]: i0 = T0 + 1}
         * which is in general the case, but we need to check that this
         * is the case. If it is not the case, the computed shifts are wrong.
         * i.e., check that we do not have any other dimension or parameter is
         * involved in the constraint. The constraint should have the form
         * dynamic_dimension = fixed_dimension + constant
         * where tile_dimension is a fixed dimension and where constant is
         * a literal constant not a symbolic constant. This constant will
         * become the shift degree.
         */
        int c = (-1) * isl_set_get_const_dim(isl_set_copy(m), 0);

        shifts.push_back(c);

        DEBUG(3, tiramisu::str_dump("The constant value of the remaining dimension is: " + std::to_string(
                                        c)));
    }

    if (ENABLE_DEBUG && DEBUG_LEVEL >= 3)
    {
        DEBUG_NO_NEWLINE(3, tiramisu::str_dump("Shift degrees are: "));
        for (auto c : shifts)
        {
            tiramisu::str_dump(std::to_string(c) + " ");
        }
        tiramisu::str_dump("\n");
    }

    DEBUG_INDENT(-4);

    return shifts;
}

/**
 * - Get the access function of the consumer (access to computations).
 * - Apply the schedule on the iteration domain and access functions.
 * - Keep only the access function to the producer.
 * - Compute the iteration space of the consumer with all dimensions after L projected out.
 * - Project out the dimensions after L in the access function.
 * - Compute the image of the iteration space with the access function.
 *   //This is called the "needed".
 *
 * - Project out the dimensions that are after L in the iteration domain of the producer.
 *   // This is called the "produced".
 *
 * -  missing = needed - produced.
 *
 * - Add universal dimensions to the missing set.
 *
 * - Use the missing set as an argument to create the redundant computation.
 *
 * - How to shift:
 *      max(needed) - max(produced) at the level L. The result should be an integer.
 *
 * - Order the redundant computation after the original at level L.
 * - Order the consumer after the redundant at level L.
 */
void computation::compute_at(computation &consumer, int L)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int dim = loop_level_into_static_dimension(L);

    assert(this->get_schedule() != NULL);
    assert(dim < (signed int) isl_map_dim(isl_map_copy(this->get_schedule()), isl_dim_out));
    assert(dim >= computation::root_dimension);

    this->get_function()->align_schedules();

    DEBUG(3, tiramisu::str_dump("Setting the schedule of the producer ");
          tiramisu::str_dump(this->get_name());
          tiramisu::str_dump(" to be computed at the loop nest of the consumer ");
          tiramisu::str_dump(consumer.get_name());
          tiramisu::str_dump(" at dimension ");
          tiramisu::str_dump(std::to_string(dim)));
    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(this->get_schedule())));

    // Compute the access relation of the consumer computation.
    std::vector<isl_map *> accesses_vector;
    get_rhs_accesses(consumer.get_function(), &consumer, accesses_vector, false);
    assert(accesses_vector.size() > 0);

    DEBUG(3, tiramisu::str_dump("Vector of accesses computed."));

    // Create a union map of the accesses to the producer.
    isl_map *consumer_accesses = NULL;
    isl_space *space = NULL;
    space = isl_map_get_space(isl_map_copy(accesses_vector[0]));
    assert(space != NULL);
    consumer_accesses = isl_map_empty(isl_space_copy(space));
    for (const auto a : accesses_vector)
    {
        std::string range_name = isl_map_get_tuple_name(isl_map_copy(consumer_accesses), isl_dim_out);

        if (range_name == this->get_name())
        {
            consumer_accesses = isl_map_union(isl_map_copy(a), consumer_accesses);
        }
    }
    consumer_accesses = isl_map_intersect_range(consumer_accesses,
                        isl_set_copy(this->get_iteration_domain()));
    consumer_accesses = isl_map_intersect_domain(consumer_accesses,
                        isl_set_copy(consumer.get_iteration_domain()));
    consumer_accesses = this->simplify(consumer_accesses);

    DEBUG(3, tiramisu::str_dump("Accesses after keeping only those that have the producer in the range: "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(consumer_accesses)));

    // Get the consumer domain and schedule and the producer domain and schedule
    isl_set *consumer_domain = isl_set_copy(consumer.get_iteration_domain());
    isl_map *consumer_sched = isl_map_copy(consumer.get_schedule());
    isl_set *producer_domain = isl_set_copy(this->get_iteration_domain());
    isl_map *producer_sched = isl_map_copy(this->get_schedule());

    DEBUG(3, tiramisu::str_dump("Consumer domain (in iteration space): ",
                                isl_set_to_str(consumer_domain)));
    DEBUG(3, tiramisu::str_dump("Consumer schedule (in iteration space): ",
                                isl_map_to_str(consumer_sched)));
    DEBUG(3, tiramisu::str_dump("Producer domain (in iteration space): ",
                                isl_set_to_str(producer_domain)));
    DEBUG(3, tiramisu::str_dump("Producer schedule (in iteration space): ",
                                isl_map_to_str(producer_sched)));

    // Simplify
    consumer_domain = this->simplify(consumer_domain);
    consumer_sched = this->simplify(consumer_sched);
    producer_sched = this->simplify(producer_sched);
    producer_domain = this->simplify(producer_domain);

    // Transform, into time-processor, the consumer domain and schedule and the producer domain and schedule and the access relation
    consumer_domain = isl_set_apply(consumer_domain, isl_map_copy(consumer_sched));
    producer_domain = isl_set_apply(producer_domain, isl_map_copy(producer_sched));
    consumer_accesses = isl_map_apply_domain(isl_map_copy(consumer_accesses),
                        isl_map_copy(consumer_sched));
    consumer_accesses = isl_map_apply_range(isl_map_copy(consumer_accesses),
                                            isl_map_copy(producer_sched));

    DEBUG(3, tiramisu::str_dump("")); DEBUG(3, tiramisu::str_dump(""));
    DEBUG(3, tiramisu::str_dump("Consumer domain (in time-processor): ",
                                isl_set_to_str(consumer_domain)));
    DEBUG(3, tiramisu::str_dump("Consumer accesses (in time-processor): ",
                                isl_map_to_str(consumer_accesses)));
    DEBUG(3, tiramisu::str_dump("Producer domain (in time-processor): ",
                                isl_set_to_str(producer_domain)));

    std::vector<std::string> param_names;

    // Add parameter dimensions and equate the dimensions on the left of dim to these parameters
    if (L + 1 > 0)
    {
        int pos_last_param0 = isl_set_dim(consumer_domain, isl_dim_param);
        int pos_last_param1 = isl_set_dim(producer_domain, isl_dim_param);
        consumer_domain = isl_set_add_dims(consumer_domain, isl_dim_param, L + 1);
        producer_domain = isl_set_add_dims(producer_domain, isl_dim_param, L + 1);

        // Set the names of the new parameters
        for (int i = 0; i <= L; i++)
        {
            std::string new_param = generate_new_variable_name();
            consumer_domain = isl_set_set_dim_name(consumer_domain, isl_dim_param, pos_last_param0 + i,
                                                   new_param.c_str());
            producer_domain = isl_set_set_dim_name(producer_domain, isl_dim_param, pos_last_param1 + i,
                                                   new_param.c_str());

            // Save the parameter names for later use (to eliminate them again and replace them with existential variables).
            param_names.push_back(new_param);
        }

        isl_space *sp = isl_set_get_space(isl_set_copy(consumer_domain));
        isl_local_space *lsp = isl_local_space_from_space(isl_space_copy(sp));

        isl_space *sp2 = isl_set_get_space(isl_set_copy(producer_domain));
        isl_local_space *lsp2 = isl_local_space_from_space(isl_space_copy(sp2));

        for (int i = 0; i <= L; i++)
        {
            // Assuming that i is the dynamic dimension and T is the parameter.
            // We want to create the following constraint: i - T = 0
            int pos = loop_level_into_dynamic_dimension(i);
            isl_constraint *cst = isl_constraint_alloc_equality(isl_local_space_copy(lsp));
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_set, pos, 1);
            cst = isl_constraint_set_coefficient_si(cst, isl_dim_param, pos_last_param0 + i, -1);
            consumer_domain = isl_set_add_constraint(consumer_domain, cst);

            isl_constraint *cst2 = isl_constraint_alloc_equality(isl_local_space_copy(lsp2));
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_set, pos, 1);
            cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_param, pos_last_param1 + i, -1);
            producer_domain =  isl_set_add_constraint(producer_domain, cst2);
        }
    }
    DEBUG(3, tiramisu::str_dump("Consumer domain after fixing left dimensions to parameters: ",
                                isl_set_to_str(consumer_domain)));
    DEBUG(3, tiramisu::str_dump("Producer domain after fixing left dimensions to parameters: ",
                                isl_set_to_str(producer_domain)));


    // Compute needed = consuler_access(consumer_domain)
    isl_set *needed = isl_set_apply(isl_set_copy(consumer_domain), isl_map_copy(consumer_accesses));
    needed = this->simplify(needed);
    DEBUG(3, tiramisu::str_dump("Needed in time-processor = consumer_access(consumer_domain) in time-processor: ",
                                isl_set_to_str(needed)));


    // Compute missing = needed - producer
    // First, rename the needed to have the same space name as produced
    needed = isl_set_set_tuple_name(needed, isl_set_get_tuple_name(isl_set_copy(producer_domain)));

    /*
     * The isl_set_subtract function is not well documented. Here is a test that indicates what is does exactly.
     * S1: { S[i, j] : i >= 0 and i <= 100 and j >= 0 and j <= 100 }
     * S2: { S[i, j] : i >= 0 and i <= 50 and j >= 0 and j <= 50 }
     * isl_set_subtract(S2, S1): { S[i, j] : 1 = 0 }
     * isl_set_subtract(S1, S2): { S[i, j] : (i >= 51 and i <= 100 and j >= 0 and j <= 100) or (i >= 0 and i <= 50 and j >= 51 and j <= 100) }
     *
     * So isl_set_subtract(S1, S2) = S1 - S2.
     */
    isl_set *missing = isl_set_subtract(isl_set_copy(needed), isl_set_copy(producer_domain));
    missing = this->simplify(missing);
    DEBUG(3, tiramisu::str_dump("Missing = needed - producer = ", isl_set_to_str(missing)));
    DEBUG(3, tiramisu::str_dump("")); DEBUG(3, tiramisu::str_dump(""));
    isl_set *original_missing = isl_set_copy(missing);

    std::vector<int> shift_degrees = get_shift_degrees(isl_set_copy(missing), L);

    // Now replace the parameters by existential variables and remove them
    if (L + 1 > 0)
    {
        int pos_last_dim = isl_set_dim(missing, isl_dim_set);
        std::string space_name = isl_set_get_tuple_name(missing);
        missing = isl_set_add_dims(missing, isl_dim_set, L + 1);
        missing = isl_set_set_tuple_name(missing, space_name.c_str());

        // Set the names of the new dimensions.
        for (int i = 0; i <= L; i++)
        {
            missing = isl_set_set_dim_name(missing, isl_dim_set, pos_last_dim + i,
                                           ("p" + param_names[i]).c_str());
        }

        /* Go through all the constraints of the set "missing" and replace them with new constraints.
         * In the new constraints, each coefficient of a param is replaced by a coefficient to the new
         * dynamic variables. Later, these dynamic variables are projected out to create existential
         * variables.
         *
         * For each basic set in a set
         *      For each constraint in a basic set
         *          For each parameter variable created previously
         *              If the constraint involves that parameter
         *                  Read the coefficient of the parameter.
         *                  Set the coefficient of the corresponding variable into that coefficient
         *                  and set the coefficient of the parameter to 0.
         * Project out the dynamic variables.  The parameters are kept but are not used at all in the
         * constraints of "missing".
         */
        isl_set *new_missing = isl_set_universe(isl_space_copy(isl_set_get_space(isl_set_copy(missing))));
        isl_basic_set_list *bset_list = isl_set_get_basic_set_list(isl_set_copy(missing));
        for (int i = 0; i < isl_set_n_basic_set(missing); i++)
        {
            isl_basic_set *bset = isl_basic_set_list_get_basic_set(isl_basic_set_list_copy(bset_list), i);
            isl_basic_set *new_bset = isl_basic_set_universe(isl_space_copy(isl_basic_set_get_space(
                                          isl_basic_set_copy(bset))));
            isl_constraint_list *cst_list = isl_basic_set_get_constraint_list(bset);
            isl_space *sp = isl_basic_set_get_space(bset);
            DEBUG(10, tiramisu::str_dump("Retrieving the constraints of the bset:",
                                         isl_set_to_str(isl_set_from_basic_set(isl_basic_set_copy(bset)))));
            DEBUG(10, tiramisu::str_dump("Number of constraints: " + std::to_string(
                                             isl_constraint_list_n_constraint(cst_list))));
            DEBUG(10, tiramisu::str_dump("List of constraints: "); isl_constraint_list_dump(cst_list));

            for (int j = 0; j < isl_constraint_list_n_constraint(cst_list); j++)
            {
                DEBUG(10, tiramisu::str_dump("Checking the constraint number " + std::to_string(j)));
                isl_constraint *cst = isl_constraint_list_get_constraint(cst_list, j);
                DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Constraint: "); isl_constraint_dump(cst));
                for (auto const p : param_names)
                {
                    int pos = isl_space_find_dim_by_name(sp, isl_dim_param, p.c_str());
                    if (isl_constraint_involves_dims(cst, isl_dim_param, pos, 1))
                    {
                        DEBUG(10, tiramisu::str_dump("Does the constraint involve the parameter " + p + "? Yes."));
                        DEBUG_NO_NEWLINE(10, tiramisu::str_dump("Modifying the constraint. The original constraint:");
                                         isl_constraint_dump(cst));
                        isl_val *coeff = isl_constraint_get_coefficient_val(cst, isl_dim_param, pos);
                        cst = isl_constraint_set_coefficient_si(cst, isl_dim_param, pos, 0);
                        int pos2 = isl_space_find_dim_by_name(sp, isl_dim_set, ("p" + p).c_str());
                        cst = isl_constraint_set_coefficient_val(cst, isl_dim_set, pos2, isl_val_copy(coeff));
                        DEBUG_NO_NEWLINE(10, tiramisu::str_dump("The new constraint:"); isl_constraint_dump(cst));
                    }
                    else
                    {
                        DEBUG(10, tiramisu::str_dump("Does the constraint involve the parameter " + p + "? No."));
                    }
                }
                DEBUG(10, tiramisu::str_dump(""));

                new_bset = isl_basic_set_add_constraint(new_bset, isl_constraint_copy(cst));
            }

            DEBUG(10, tiramisu::str_dump("The basic set after modifying the constraints:");
                  isl_basic_set_dump(new_bset));

            // In the first time, restrict the universal new_missing with the new bset,
            // in the next times compute the union of the bset with new_missing.
            if (i == 0)
            {
                new_missing = isl_set_intersect(new_missing, isl_set_from_basic_set(new_bset));
            }
            else
            {
                new_missing = isl_set_union(new_missing, isl_set_from_basic_set(new_bset));
            }

            DEBUG(10, tiramisu::str_dump("The new value of missing (after intersecting with the new bset):");
                  isl_set_dump(new_missing));

        }
        missing = new_missing;

        // Project out the set dimensions to make them existential variables
        missing = isl_set_project_out(missing, isl_dim_set, pos_last_dim, L + 1);
        int pos_first_param = isl_space_find_dim_by_name(isl_set_get_space(missing), isl_dim_param,
                              param_names[0].c_str());
        missing = isl_set_project_out(missing, isl_dim_param, pos_first_param, L + 1);
        missing = isl_set_set_tuple_name(missing, space_name.c_str());

        DEBUG(3, tiramisu::str_dump("Missing before replacing the parameters with existential variables: ",
                                    isl_set_to_str(original_missing)));
        DEBUG(3, tiramisu::str_dump("Missing after replacing the parameters with existential variables: ",
                                    isl_set_to_str(missing)));
        DEBUG(3, tiramisu::str_dump(""));
    }

    // Duplicate the producer using the missing set which is in the time-processor domain.
    tiramisu::computation *original_computation = this;
    tiramisu::computation *duplicated_computation = this->duplicate("", isl_set_to_str(missing));
    DEBUG(3, tiramisu::str_dump("Producer duplicated. Dumping the schedule of the original computation."));
    original_computation->dump_schedule();
    DEBUG(3, tiramisu::str_dump("Dumping the schedule of the duplicate computation."));
    duplicated_computation->dump_schedule();

    DEBUG(3, tiramisu::str_dump("Now setting the duplicate with regard to the other computations."));
    original_computation->after((*duplicated_computation), L);
    consumer.after((*duplicated_computation), L);
    consumer.after((*original_computation), L);
    DEBUG(3, tiramisu::str_dump("Dumping the schedule of the producer and consumer."));
    this->dump_schedule();
    consumer.dump_schedule();

    // Computing the shift degrees.
    for (int i = 0; i < L; i++)
        if (shift_degrees[i] != 0)
        {
            DEBUG(3, tiramisu::str_dump("Now shifting the duplicate by " + std::to_string(
                                            shift_degrees[i]) + " at loop level " + std::to_string(i)));
            duplicated_computation->shift(i, shift_degrees[i]);
        }

    DEBUG_INDENT(-4);
}

/**
 * Modify the schedule of this computation so that it splits the
 * loop level L0 into two new loop levels.
 * The size of the inner dimension created is sizeX.
 */
void computation::split(int L0, int sizeX)
{
    int inDim0 = loop_level_into_dynamic_dimension(L0);

    assert(this->get_schedule() != NULL);
    assert(inDim0 >= 0);
    assert(inDim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));
    assert(sizeX >= 1);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_map *schedule = this->get_schedule();
    int duplicate_ID = isl_map_get_static_dim(schedule, 0);

    schedule = isl_map_copy(schedule);
    schedule = isl_map_set_tuple_id(schedule, isl_dim_out,
                                    isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL));


    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(schedule)));
    DEBUG(3, tiramisu::str_dump("Splitting dimension " + std::to_string(inDim0)
                                + " with split size " + std::to_string(sizeX)));

    std::string inDim0_str;

    std::string outDim0_str = generate_new_variable_name();
    std::string outDim1_str = generate_new_variable_name();

    int n_dims = isl_map_dim(this->get_schedule(), isl_dim_out);
    std::vector<isl_id *> dimensions;
    std::vector<std::string> dimensions_str;
    std::string map = "{";

    // -----------------------------------------------------------------
    // Preparing a map to split the duplicate computation.
    // -----------------------------------------------------------------

    map = map + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            std::string dim_str = generate_new_variable_name();
            dimensions_str.push_back(dim_str);
            map = map + dim_str;
        }
        else
        {
            std::string dim_str = generate_new_variable_name();
            dimensions_str.push_back(dim_str);
            map = map + dim_str;

            if (i == inDim0)
            {
                inDim0_str = dim_str;
            }
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] -> " + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i == 0)
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else if (i != inDim0)
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else
        {
            map = map + outDim0_str + ", 0, " + outDim1_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim0_str.c_str(), NULL);
            isl_id *id1 = isl_id_alloc(this->get_ctx(),
                                       outDim1_str.c_str(), NULL);
            dimensions.push_back(id0);
            dimensions.push_back(id1);
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID) + " and " +
          outDim0_str + " = floor(" + inDim0_str + "/" +
          std::to_string(sizeX) + ") and " + outDim1_str + " = (" +
          inDim0_str + "%" + std::to_string(sizeX) + ")}";

    isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());

    for (int i = 0; i < dimensions.size(); i++)
        transformation_map = isl_map_set_dim_id(
                                 transformation_map, isl_dim_out, i, isl_id_copy(dimensions[i]));

    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_in,
                             isl_map_get_tuple_id(isl_map_copy(schedule), isl_dim_out));
    isl_id *id_range = isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL);
    transformation_map = isl_map_set_tuple_id(transformation_map, isl_dim_out, id_range);

    DEBUG(3, tiramisu::str_dump("Transformation map : ",
                                isl_map_to_str(transformation_map)));

    schedule = isl_map_apply_range(isl_map_copy(schedule), isl_map_copy(transformation_map));

    DEBUG(3, tiramisu::str_dump("Schedule after splitting: ", isl_map_to_str(schedule)));

    this->set_schedule(schedule);

    DEBUG_INDENT(-4);
}

// Methods related to the tiramisu::function class.

std::string tiramisu::function::get_gpu_thread_iterator(const std::string &comp, int lev0) const
{
    assert(!comp.empty());
    assert(lev0 >= 0);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::string res = "";

    for (const auto &pd : this->gpu_thread_dimensions)
    {
        if ((pd.first == comp) && ((std::get<0>(pd.second) == lev0) || (std::get<1>(pd.second) == lev0) ||
                                   (std::get<2>(pd.second) == lev0)))
        {
            if (lev0 == std::get<0>(pd.second))
            {
                res = "__thread_id_x";
            }
            else if (lev0 == std::get<1>(pd.second))
            {
                res = "__thread_id_y";
            }
            else if (lev0 == std::get<2>(pd.second))
            {
                res = "__thread_id_z";
            }
            else
            {
                tiramisu::error("Level not mapped to GPU.", true);
            }

            std::string str = "Dimension " + std::to_string(lev0) +
                              " should be mapped to iterator " + res;
            str = str + ". It was compared against: " + std::to_string(std::get<0>(pd.second)) +
                  ", " + std::to_string(std::get<1>(pd.second)) + " and " +
                  std::to_string(std::get<2>(pd.second));
            DEBUG(3, tiramisu::str_dump(str));
        }
    }

    DEBUG_INDENT(-4);
    return res;
}

std::string tiramisu::function::get_gpu_block_iterator(const std::string &comp, int lev0) const
{
    assert(!comp.empty());
    assert(lev0 >= 0);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::string res = "";;

    for (const auto &pd : this->gpu_block_dimensions)
    {
        if ((pd.first == comp) && ((std::get<0>(pd.second) == lev0) || (std::get<1>(pd.second) == lev0) ||
                                   (std::get<2>(pd.second) == lev0)))
        {
            if (lev0 == std::get<0>(pd.second))
            {
                res = "__block_id_x";
            }
            else if (lev0 == std::get<1>(pd.second))
            {
                res = "__block_id_y";
            }
            else if (lev0 == std::get<2>(pd.second))
            {
                res = "__block_id_z";
            }
            else
            {
                tiramisu::error("Level not mapped to GPU.", true);
            }

            std::string str = "Dimension " + std::to_string(lev0) +
                              " should be mapped to iterator " + res;
            str = str + ". It was compared against: " + std::to_string(std::get<0>(pd.second)) +
                  ", " + std::to_string(std::get<1>(pd.second)) + " and " +
                  std::to_string(std::get<2>(pd.second));
            DEBUG(3, tiramisu::str_dump(str));
        }
    }

    DEBUG_INDENT(-4);
    return res;
}

bool tiramisu::function::should_unroll(const std::string &comp, int lev0) const
{
    assert(!comp.empty());
    assert(lev0 >= 0);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    bool found = false;
    for (const auto &pd : this->unroll_dimensions)
    {
        if ((pd.first == comp) && (pd.second == lev0))
        {
            found = true;
        }
    }

    std::string str = "Dimension " + std::to_string(lev0) +
                      (found ? " should" : " should not") +
                      " be unrolled.";
    DEBUG(3, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);
    return found;
}

bool tiramisu::function::should_map_to_gpu_block(const std::string &comp, int lev0) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev0 >= 0);

    bool found = false;
    for (const auto &pd : this->gpu_block_dimensions)
    {
        if ((pd.first == comp) && ((std::get<0>(pd.second) == lev0) || (std::get<1>(pd.second) == lev0) ||
                                   (std::get<2>(pd.second) == lev0)))
        {
            found = true;
        }
    }

    std::string str = "Dimension " + std::to_string(lev0) +
                      (found ? " should" : " should not")
                      + " be mapped to GPU block.";
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);
    return found;
}

bool tiramisu::function::should_map_to_gpu_thread(const std::string &comp, int lev0) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev0 >= 0);

    bool found = false;
    for (const auto &pd : this->gpu_thread_dimensions)
    {
        if ((pd.first == comp) && ((std::get<0>(pd.second) == lev0) || (std::get<1>(pd.second) == lev0) ||
                                   (std::get<2>(pd.second) == lev0)))
        {
            found = true;
        }
    }

    std::string str = "Dimension " + std::to_string(lev0) +
                      (found ? " should" : " should not")
                      + " be mapped to GPU thread.";
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);
    return found;
}

int tiramisu::function::get_max_identity_schedules_range_dim() const
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int max_dim = 0;
    for (const auto &comp : this->get_computations())
    {
        isl_map *sched = comp->gen_identity_schedule_for_time_space_domain();
        int m = isl_map_dim(sched, isl_dim_out);
        max_dim = std::max(max_dim, m);
    }

    DEBUG_INDENT(-4);

    return max_dim;
}

int tiramisu::function::get_max_schedules_range_dim() const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    int max_dim = 0;
    for (const auto &comp : this->get_computations())
    {
        isl_map *sched = comp->get_schedule();
        int m = isl_map_dim(sched, isl_dim_out);
        max_dim = std::max(max_dim, m);
    }

    DEBUG_INDENT(-4);

    return max_dim;
}

isl_map *isl_map_align_range_dims(isl_map *map, int max_dim)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(map != NULL);
    int mdim = isl_map_dim(map, isl_dim_out);
    assert(max_dim >= mdim);

    DEBUG(10, tiramisu::str_dump("Input map:", isl_map_to_str(map)));

    const char *original_range_name = isl_map_get_tuple_name(map, isl_dim_out);

    map = isl_map_add_dims(map, isl_dim_out, max_dim - mdim);

    for (int i = mdim; i < max_dim; i++)
    {
        isl_space *sp = isl_map_get_space(map);
        isl_local_space *lsp = isl_local_space_from_space(sp);
        isl_constraint *cst = isl_constraint_alloc_equality(lsp);
        cst = isl_constraint_set_coefficient_si(cst, isl_dim_out, i, 1);
        map = isl_map_add_constraint(map, cst);
    }

    map = isl_map_set_tuple_name(map, isl_dim_out, original_range_name);

    DEBUG(10, tiramisu::str_dump("After alignment, map = ",
                                 isl_map_to_str(map)));

    DEBUG_INDENT(-4);
    return map;
}

isl_union_map *tiramisu::function::get_aligned_identity_schedules() const
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_union_map *result;
    isl_space *space;

    if (this->body.empty() == false)
    {
        space = isl_map_get_space(this->body[0]->gen_identity_schedule_for_time_space_domain());
    }
    else
    {
        return NULL;
    }
    assert(space != NULL);
    result = isl_union_map_empty(space);

    int max_dim = this->get_max_identity_schedules_range_dim();

    for (const auto &comp : this->get_computations())
    {
        if (comp->should_schedule_this_computation())
        {
            isl_map *sched = comp->gen_identity_schedule_for_time_space_domain();
            DEBUG(3, tiramisu::str_dump("Identity schedule for time space domain: ", isl_map_to_str(sched)));
            assert((sched != NULL) && "Identity schedule could not be computed");
            sched = isl_map_align_range_dims(sched, max_dim);
            result = isl_union_map_union(result, isl_union_map_from_map(sched));
        }
    }

    DEBUG_INDENT(-4);
    DEBUG(3, tiramisu::str_dump("End of function"));

    return result;
}

void tiramisu::function::align_schedules()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int max_dim = this->get_max_schedules_range_dim();

    for (auto &comp : this->get_computations())
    {
            isl_map *dup_sched = comp->get_schedule();
            assert((dup_sched != NULL) && "Schedules should be set before calling align_schedules");
            dup_sched = isl_map_align_range_dims(dup_sched, max_dim);
            comp->set_schedule(dup_sched);
    }

    DEBUG_INDENT(-4);
    DEBUG(3, tiramisu::str_dump("End of function"));
}

void tiramisu::function::add_invariant(tiramisu::constant invar)
{
    invariants.push_back(invar);
}

void tiramisu::function::add_computation(computation *cpt)
{
    assert(cpt != NULL);

    this->body.push_back(cpt);
}

void tiramisu::function::dump(bool exhaustive) const
{
    if (ENABLE_DEBUG)
    {
        std::cout << "\n\nFunction \"" << this->name << "\"" << std::endl << std::endl;

        if (this->function_arguments.size() > 0)
        {
            std::cout << "Function arguments (tiramisu buffers):" << std::endl;
            for (const auto &buf : this->function_arguments)
            {
                buf->dump(exhaustive);
            }
            std::cout << std::endl;
        }

        if (this->invariants.size() > 0)
        {
            std::cout << "Function invariants:" << std::endl;
            for (const auto &inv : this->invariants)
            {
                inv.dump(exhaustive);
            }
            std::cout << std::endl;
        }

        if (this->get_program_context() != NULL)
        {
            std::cout << "Function context set: "
                      << isl_set_to_str(this->get_program_context())
                      << std::endl;
        }

        std::cout << "Body " << std::endl;
        for (const auto &cpt : this->body)
        {
            cpt->dump();
        }
        std::cout << std::endl;

        if (this->halide_stmt.defined())
        {
            std::cout << "Halide stmt " << this->halide_stmt << std::endl;
        }

        std::cout << "Buffers" << std::endl;
        for (const auto &buf : this->buffers_list)
        {
            std::cout << "Buffer name: " << buf.second->get_name() << std::endl;
        }

        std::cout << std::endl << std::endl;
    }
}

void tiramisu::function::dump_iteration_domain() const
{
    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("\nIteration domain:\n");
        for (const auto &cpt : this->body)
        {
            cpt->dump_iteration_domain();
        }
        tiramisu::str_dump("\n");
    }
}

void tiramisu::function::dump_schedule() const
{
    if (ENABLE_DEBUG)
    {
        tiramisu::str_dump("\nDumping schedules of the function " + this->get_name() + " :");

        for (const auto &cpt : this->body)
        {
            cpt->dump_schedule();
        }

        std::cout << "Parallel dimensions: ";
        for (const auto &par_dim : parallel_dimensions)
        {
            std::cout << par_dim.first << "(" << par_dim.second << ") ";
        }

        std::cout << std::endl;

        std::cout << "Vector dimensions: ";
        for (const auto &vec_dim : vector_dimensions)
        {
            std::cout << vec_dim.first << "(" << vec_dim.second << ") ";
        }

        std::cout << std::endl << std::endl << std::endl;
    }
}

Halide::Argument::Kind halide_argtype_from_tiramisu_argtype(tiramisu::argument_t type)
{
    Halide::Argument::Kind res;

    if (type == tiramisu::a_temporary)
    {
        tiramisu::error("Buffer type \"temporary\" can't be translated to Halide.\n", true);
    }

    if (type == tiramisu::a_input)
    {
        res = Halide::Argument::InputBuffer;
    }
    else
    {
        assert(type == tiramisu::a_output);
        res = Halide::Argument::OutputBuffer;
    }

    return res;
}

void tiramisu::function::set_arguments(const std::vector<tiramisu::buffer *> &buffer_vec)
{
    this->function_arguments = buffer_vec;
}

void tiramisu::function::add_vector_dimension(std::string stmt_name, int vec_dim)
{
    assert(vec_dim >= 0);
    assert(!stmt_name.empty());

    this->vector_dimensions.push_back({stmt_name, vec_dim});
}

void tiramisu::function::add_parallel_dimension(std::string stmt_name, int vec_dim)
{
    assert(vec_dim >= 0);
    assert(!stmt_name.empty());

    this->parallel_dimensions.push_back({stmt_name, vec_dim});
}

void tiramisu::function::add_unroll_dimension(std::string stmt_name, int level)
{
    assert(level >= 0);
    assert(!stmt_name.empty());

    this->unroll_dimensions.push_back({stmt_name, level});
}

void tiramisu::function::add_gpu_block_dimensions(std::string stmt_name, int dim0,
        int dim1, int dim2)
{
    assert(!stmt_name.empty());
    assert(dim0 >= 0);
    // dim1 and dim2 can be -1 if not set.

    this->gpu_block_dimensions.push_back(
        std::pair<std::string, std::tuple<int, int, int>>(
            stmt_name,
            std::tuple<int, int, int>(dim0, dim1, dim2)));
}

void tiramisu::function::add_gpu_thread_dimensions(std::string stmt_name, int dim0,
        int dim1, int dim2)
{
    assert(!stmt_name.empty());
    assert(dim0 >= 0);
    // dim1 and dim2 can be -1 if not set.

    this->gpu_thread_dimensions.push_back(
        std::pair<std::string, std::tuple<int, int, int>>(
            stmt_name,
            std::tuple<int, int, int>(dim0, dim1, dim2)));
}

isl_union_set *tiramisu::function::get_trimmed_time_processor_domain() const
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_union_set *result = NULL;
    isl_space *space = NULL;
    if (!this->body.empty())
    {
        space = isl_set_get_space(this->body[0]->get_trimmed_time_processor_domain());
    }
    else
    {
        DEBUG_INDENT(-4);
        return NULL;
    }
    assert(space != NULL);

    result = isl_union_set_empty(space);

    for (const auto &cpt : this->body)
    {
        if (cpt->should_schedule_this_computation())
        {
            isl_set *cpt_iter_space = isl_set_copy(cpt->get_trimmed_time_processor_domain());
            result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
        }
    }

    DEBUG_INDENT(-4);

    return result;
}

isl_union_set *tiramisu::function::get_time_processor_domain() const
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_union_set *result = NULL;
    isl_space *space = NULL;

    if (!this->body.empty())
    {
        space = isl_set_get_space(this->body[0]->get_time_processor_domain());
    }
    else
    {
        return NULL;
    }

    assert(space != NULL);
    result = isl_union_set_empty(space);

    for (const auto &cpt : this->body)
    {
        if (cpt->should_schedule_this_computation())
        {
            isl_set *cpt_iter_space = isl_set_copy(cpt->get_time_processor_domain());
            result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
        }
    }

    DEBUG_INDENT(-4);

    return result;
}

isl_union_set *tiramisu::function::get_iteration_domain() const
{
    isl_union_set *result = NULL;
    isl_space *space = NULL;

    if (!this->body.empty())
    {
        space = isl_set_get_space(this->body[0]->get_iteration_domain());
    }
    else
    {
        return NULL;
    }

    assert(space != NULL);
    result = isl_union_set_empty(space);

    for (const auto &cpt : this->body)
    {
        if (cpt->should_schedule_this_computation())
        {
            isl_set *cpt_iter_space = isl_set_copy(cpt->get_iteration_domain());
            result = isl_union_set_union(isl_union_set_from_set(cpt_iter_space), result);
        }
    }

    return result;
}

isl_union_map *tiramisu::function::get_schedule() const
{
    isl_union_map *result = NULL;
    isl_space *space = NULL;

    if (!this->body.empty())
    {
        space = isl_map_get_space(this->body[0]->get_schedule());
    }
    else
    {
        return NULL;
    }

    assert(space != NULL);
    result = isl_union_map_empty(isl_space_copy(space));

    for (const auto &cpt : this->body)
    {
        isl_map *m = isl_map_copy(cpt->get_schedule());
        result = isl_union_map_union(isl_union_map_from_map(m), result);
    }

    result = isl_union_map_intersect_domain(result, this->get_iteration_domain());

    return result;
}

isl_union_map *tiramisu::function::get_trimmed_schedule() const
{
    isl_union_map *result = NULL;
    isl_space *space = NULL;

    if (!this->body.empty())
    {
        space = isl_map_get_space(this->body[0]->get_trimmed_union_of_schedules());
    }
    else
    {
        return NULL;
    }

    assert(space != NULL);
    result = isl_union_map_empty(isl_space_copy(space));

    for (const auto &cpt : this->body)
    {
        isl_map *m = isl_map_copy(cpt->get_trimmed_union_of_schedules());
        result = isl_union_map_union(isl_union_map_from_map(m), result);
    }

    result = isl_union_map_intersect_domain(result, this->get_iteration_domain());

    return result;
}

// Function for the buffer class

std::string str_tiramisu_type_op(tiramisu::op_t type)
{
    switch (type)
    {
    case tiramisu::o_logical_and:
        return "and";
    case tiramisu::o_logical_or:
        return "or";
    case tiramisu::o_max:
        return "max";
    case tiramisu::o_min:
        return "min";
    case tiramisu::o_minus:
        return "minus";
    case tiramisu::o_add:
        return "add";
    case tiramisu::o_sub:
        return "sub";
    case tiramisu::o_mul:
        return "mul";
    case tiramisu::o_div:
        return "div";
    case tiramisu::o_mod:
        return "mod";
    case tiramisu::o_select:
        return "select";
    case tiramisu::o_cond:
        return "ternary_cond";
    case tiramisu::o_logical_not:
        return "not";
    case tiramisu::o_eq:
        return "eq";
    case tiramisu::o_ne:
        return "ne";
    case tiramisu::o_le:
        return "le";
    case tiramisu::o_lt:
        return "lt";
    case tiramisu::o_ge:
        return "ge";
    case tiramisu::o_call:
        return "call";
    case tiramisu::o_access:
        return "access";
    case tiramisu::o_address:
        return "address";
    case tiramisu::o_right_shift:
        return "right-shift";
    case tiramisu::o_left_shift:
        return "left-shift";
    case tiramisu::o_floor:
        return "floor";
    case tiramisu::o_allocate:
        return "allocate";
    case tiramisu::o_free:
        return "free";
    case tiramisu::o_cast:
        return "cast";
    case tiramisu::o_sin:
        return "sin";
    case tiramisu::o_cos:
        return "cos";
    case tiramisu::o_tan:
        return "tan";
    case tiramisu::o_asin:
        return "asin";
    case tiramisu::o_acos:
        return "acos";
    case tiramisu::o_atan:
        return "atan";
    case tiramisu::o_abs:
        return "abs";
    case tiramisu::o_sqrt:
        return "sqrt";
    case tiramisu::o_expo:
        return "exp";
    case tiramisu::o_log:
        return "log";
    case tiramisu::o_ceil:
        return "ceil";
    case tiramisu::o_round:
        return "round";
    case tiramisu::o_trunc:
        return "trunc";
    default:
        tiramisu::error("Tiramisu op not supported.", true);
        return "";
    }
}

const bool tiramisu::buffer::is_allocated() const
{
    return this->allocated;
}

void tiramisu::buffer::mark_as_allocated()
{
    this->allocated = true;
}

std::string str_from_tiramisu_type_expr(tiramisu::expr_t type)
{
    switch (type)
    {
    case tiramisu::e_val:
        return "val";
    case tiramisu::e_op:
        return "op";
    case tiramisu::e_var:
        return "var";
    default:
        tiramisu::error("Tiramisu type not supported.", true);
        return "";
    }
}

std::string str_from_tiramisu_type_argument(tiramisu::argument_t type)
{
    switch (type)
    {
    case tiramisu::a_input:
        return "input";
    case tiramisu::a_output:
        return "output";
    case tiramisu::a_temporary:
        return "temporary";
    default:
        tiramisu::error("Tiramisu type not supported.", true);
        return "";
    }
}

std::string str_from_tiramisu_type_primitive(tiramisu::primitive_t type)
{
    switch (type)
    {
    case tiramisu::p_uint8:
        return "uint8";
    case tiramisu::p_int8:
        return "int8";
    case tiramisu::p_uint16:
        return "uint16";
    case tiramisu::p_int16:
        return "int16";
    case tiramisu::p_uint32:
        return "uin32";
    case tiramisu::p_int32:
        return "int32";
    case tiramisu::p_uint64:
        return "uint64";
    case tiramisu::p_int64:
        return "int64";
    case tiramisu::p_float32:
        return "float32";
    case tiramisu::p_float64:
        return "float64";
    case tiramisu::p_boolean:
        return "bool";
    default:
        tiramisu::error("Tiramisu type not supported.", true);
        return "";
    }
}

std::string str_from_is_null(void *ptr)
{
    return (ptr != NULL) ? "Not NULL" : "NULL";
}

/**
  * Create a tiramisu buffer.
  * Buffers have two use cases:
  * - used to store the results of computations, and
  * - used to represent input arguments to functions.
  *
  * \p name is the name of the buffer.
  * \p nb_dims is the number of dimensions of the buffer.
  * A scalar is a one dimensional buffer that has a size of one
  * element.
  * \p dim_sizes is a vector of integers that represent the size
  * of each dimension in the buffer.  The first vector element
  * represents the rightmost array dimension, while the last vector
  * element represents the leftmost array dimension.
  * For example, in the buffer buf[N0][N1][N2], the first element
  * in the vector \p dim_sizes represents the size of rightmost
  * dimension of the buffer (i.e. N2), the second vector element
  * is N1, and the last vector element is N0.
  * Buffer dimensions in Tiramisu have the same semantics as in
  * C/C++.
  * \p type is the type of the elements of the buffer.
  * It must be a primitive type (i.e. p_uint8, p_uint16, ...).
  * Possible types are declared in tiramisu::primitive_t (type.h).
  * \p data is the data stored in the buffer.  This is useful
  * if an already allocated buffer is passed to Tiramisu.
  * \p fct is a pointer to a Tiramisu function where the buffer is
  * declared or used.
  * \p is_argument indicates whether the buffer is passed to the
  * function as an argument.  All the buffers passed as arguments
  * to the function should be allocated by the user outside the
  * function.  Buffers that are not passed to the function as
  * arguments are allocated automatically at the beginning of
  * the function and deallocated at the end of the function.
  * They are called temporary buffers (of type a_temporary).
  * Temporary buffers cannot be used outside the function
  * in which they were allocated.
  */
tiramisu::buffer::buffer(std::string name, int nb_dims, std::vector<tiramisu::expr> dim_sizes,
                         tiramisu::primitive_t type, uint8_t *data,
                         tiramisu::argument_t argt, tiramisu::function *fct):
        allocated(false), argtype(argt), auto_allocate(true), data(data), dim_sizes(dim_sizes), fct(fct),
        name(name), nb_dims(nb_dims), type(type)
{
    assert(!name.empty() && "Empty buffer name");
    assert(nb_dims > 0 && "Buffer dimensions <= 0");
    assert(nb_dims == dim_sizes.size() && "Mismatch in the number of dimensions");
    assert(fct != NULL && "Input function is NULL");

    // Check that the buffer does not already exist.
    assert((fct->get_buffers().count(name) == 0) && ("Buffer already exists"));

    fct->add_buffer(std::pair<std::string, tiramisu::buffer *>(name, this));
};

/**
  * Return the type of the argument (if the buffer is an argument).
  * Three possible types:
  *  - a_input: for inputs of the function,
  *  - a_output: for outputs of the function,
  *  - a_temporary: for buffers used as temporary buffers within
  *  the function (any temporary buffer is allocated automatically by
  *  the Tiramisu runtime at the entry of the function and is
  *  deallocated at the exit of the function).
  */
tiramisu::argument_t buffer::get_argument_type() const
{
    return argtype;
}

/**
  * Return a pointer to the data stored within the buffer.
  */
uint8_t *buffer::get_data()
{
    return data;
}

/**
  * Return the name of the buffer.
  */
const std::string &buffer::get_name() const
{
    return name;
}

/**
  * Get the number of dimensions of the buffer.
  */
int buffer::get_n_dims() const
{
    return nb_dims;
}

/**
  * Return the type of the elements of the buffer.
  */
tiramisu::primitive_t buffer::get_elements_type() const
{
    return type;
}

/**
  * Return the sizes of the dimensions of the buffer.
  * Assuming the following buffer: buf[N0][N1][N2].  The first
  * vector element represents the size of rightmost dimension
  * of the buffer (i.e. N2), the second vector element is N1,
  * and the last vector element is N0.
  */
const std::vector<tiramisu::expr> &buffer::get_dim_sizes() const
{
    return dim_sizes;
}

void tiramisu::buffer::dump(bool exhaustive) const
{
    if (ENABLE_DEBUG)
    {
        std::cout << "Buffer \"" << this->name
                  << "\", Number of dimensions: " << this->nb_dims
                  << std::endl;

        std::cout << "Dimension sizes: ";
        for (const auto &size : dim_sizes)
        {
            // TODO: create_halide_expr_from_tiramisu_expr does not support
            // the case where the buffer size is a computation access.
            std::vector<isl_ast_expr *> ie = {};
            std::cout << halide_expr_from_tiramisu_expr(NULL, ie, size) << ", ";
        }
        std::cout << std::endl;

        std::cout << "Elements type: "
                  << str_from_tiramisu_type_primitive(this->type) << std::endl;

        std::cout << "Function field: "
                  << str_from_is_null(this->fct) << std::endl;

        std::cout << "Argument type: "
                  << str_from_tiramisu_type_argument(this->argtype) << std::endl;

        std::cout << std::endl << std::endl;
    }
}

Halide::Type halide_type_from_tiramisu_type(tiramisu::primitive_t type)
{
    Halide::Type t;

    switch (type)
    {
    case tiramisu::p_uint8:
        t = Halide::UInt(8);
        break;
    case tiramisu::p_int8:
        t = Halide::Int(8);
        break;
    case tiramisu::p_uint16:
        t = Halide::UInt(16);
        break;
    case tiramisu::p_int16:
        t = Halide::Int(16);
        break;
    case tiramisu::p_uint32:
        t = Halide::UInt(32);
        break;
    case tiramisu::p_int32:
        t = Halide::Int(32);
        break;
    case tiramisu::p_uint64:
        t = Halide::UInt(64);
        break;
    case tiramisu::p_int64:
        t = Halide::Int(64);
        break;
    case tiramisu::p_float32:
        t = Halide::Float(32);
        break;
    case tiramisu::p_float64:
        t = Halide::Float(64);
        break;
    case tiramisu::p_boolean:
        t = Halide::Bool();
        break;
    default:
        tiramisu::error("Tiramisu type cannot be translated to Halide type.", true);
    }
    return t;
}

//----------------

/**
  * Initialize a computation
  *  This is a private function that should not be called explicitly
  * by users.
  */
void tiramisu::computation::init_computation(std::string iteration_space_str,
        tiramisu::function *fct,
        const tiramisu::expr &e,
        bool schedule_this_computation,
        tiramisu::primitive_t t)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Constructing the computation: " + iteration_space_str));

    assert(fct != NULL);
    assert(iteration_space_str.length() > 0 && ("Empty iteration space"));

    // Initialize all the fields to NULL (useful for later asserts)
    access = NULL;
    stmt = Halide::Internal::Stmt();
    time_processor_domain = NULL;
    relative_order = 0;
    duplicate_number = 0;

    this->schedule_this_computation = schedule_this_computation;
    this->data_type = t;

    this->ctx = fct->get_isl_ctx();

    iteration_domain = isl_set_read_from_str(ctx, iteration_space_str.c_str());
    name = std::string(isl_space_get_tuple_name(isl_set_get_space(iteration_domain),
                       isl_dim_type::isl_dim_set));
    function = fct;
    function->add_computation(this);
    this->set_identity_schedule_based_on_iteration_domain();
    this->set_expression(e);

    // If there are computations that have already been defined and that
    // have the same name, check that they constraints over their iteration
    // domains.
    std::vector<tiramisu::computation *> same_name_computations =
            this->get_function()->get_computation_by_name(name);
    if (same_name_computations.size() > 1)
    {
        if (isl_set_plain_is_universe(this->get_iteration_domain()))
            tiramisu::error("Computations defined multiple times should"
                    " have bounds on their iteration domain", true);

        for (auto c: same_name_computations)
        {
            c->set_has_multiple_definitions(true);

            if (isl_set_plain_is_universe(c->get_iteration_domain()))
                tiramisu::error("Computations defined multiple times should"
                        " have bounds on their iteration domain", true);
        }
    }
    else
        this->set_has_multiple_definitions(false);

    DEBUG_INDENT(-4);
}

/**
 * Dummy constructor for derived classes.
 */
tiramisu::computation::computation()
{
    this->access = NULL;
    this->schedule = NULL;
    this->stmt = Halide::Internal::Stmt();
    this->time_processor_domain = NULL;
    this->relative_order = 0;
    this->duplicate_number = 0;

    this->schedule_this_computation = false;
    this->data_type = p_none;
    this->expression = tiramisu::expr();

    this->ctx = NULL;

    this->iteration_domain = NULL;
    this->name = "";
    this->function = NULL;
    this->is_let = false;
    this->multiple_definitions = false;
}

/**
  * Constructor for computations.
  *
  * \p iteration_domain_str is a string that represents the iteration
  * domain of the computation.  The iteration domain should be written
  * in the ISL format (http://isl.gforge.inria.fr/user.html#Sets-and-Relations).
  *
  * The iteration domain of a statement is a set that contains
  * all of the execution instances of the statement (a statement in a
  * loop has an execution instance for each loop iteration in which
  * it executes). Each execution instance of a statement in a loop
  * nest is uniquely represented by an identifier and a tuple of
  * integers  (typically,  the  values  of  the  outer  loop  iterators).
  *
  * For example, the iteration space of the statement S0 in the following
  * loop nest
  * for (i=0; i<2; i++)
  *   for (j=0; j<3; j++)
  *      S0;
  *
  * is {S0(0,0), S0(0,1), S0(0,2), S0(1,0), S0(1,1), S0(1,2)}
  *
  * S0(0,0) is the execution instance of S0 in the iteration (0,0).
  *
  * The previous set of integer tuples can be compactly described
  * by affine constraints as follows
  *
  * {S0(i,j): 0<=i<2 and 0<=j<3}
  *
  * In general, the loop nest
  *
  * for (i=0; i<N; i++)
  *   for (j=0; j<M; j++)
  *      S0;
  *
  * has the following iteration domain
  *
  * {S0(i,j): 0<=i<N and 0<=j<M}
  *
  * This should be read as: the set of points (i,j) such that
  * 0<=i<N and 0<=j<M.
  *
  * \p e is the expression computed by the computation.
  *
  * \p schedule_this_computation should be set to true if the computation
  * is supposed to be schedule and code is supposed to be generated from
  * the computation.  Set it to false if you just want to use the
  * computation to represent a buffer (that is passed as an argument
  * to the function) and you do not intend to generate code for the
  * computation.
  *
  * \p t is the type of the computation, i.e. the type of the expression
  * computed by the computation. Example of types include (p_uint8,
  * p_uint16, p_uint32, ...).
  *
  * \p fct is a pointer to the Tiramisu function where this computation
  * should be added.
  *
  * TODO: copy ISL format for sets.
  */
tiramisu::computation::computation(std::string iteration_domain_str, tiramisu::expr e,
                                   bool schedule_this_computation, tiramisu::primitive_t t,
                                   tiramisu::function *fct)
{

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    init_computation(iteration_domain_str, fct, e, schedule_this_computation, t);
    is_let = false;

    DEBUG_INDENT(-4);
}

/**
  * Return true if the this computation is supposed to be scheduled
  * by Tiramisu.
  */
bool tiramisu::computation::should_schedule_this_computation() const
{
    return schedule_this_computation;
}

/**
  * Return the access function of the computation.
  */
isl_map *tiramisu::computation::get_access_relation() const
{
    return access;
}

/**
  * Return the access function of the computation after transforming
  * it to the time-processor domain.
  * The domain of the access function is transformed to the
  * time-processor domain using the schedule, and then the transformed
  * access function is returned.
  */
isl_map *tiramisu::computation::get_access_relation_adapted_to_time_processor_domain() const
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert((this->has_accesses() == true) && ("This computation must have accesses."));

    isl_map *access = isl_map_copy(this->get_access_relation());

    if (!this->is_let_stmt())
    {
        DEBUG(10, tiramisu::str_dump("Original access:", isl_map_to_str(access)));

        if (global::is_auto_data_mapping_set())
        {
            if (access != NULL)
            {
                assert(this->get_trimmed_union_of_schedules() != NULL);

                DEBUG(10, tiramisu::str_dump("Original schedule:", isl_map_to_str(this->get_schedule())));
                DEBUG(10, tiramisu::str_dump("Trimmed schedule to apply:",
                                             isl_map_to_str(this->get_trimmed_union_of_schedules())));
                access = isl_map_apply_domain(
                             isl_map_copy(access),
                             isl_map_copy(this->get_trimmed_union_of_schedules()));
                DEBUG(10, tiramisu::str_dump("Transformed access:", isl_map_to_str(access)));
            }
            else
            {
                DEBUG(10, tiramisu::str_dump("Not access relation to transform."));
            }
        }
        else
        {
            DEBUG(10, tiramisu::str_dump("Access not transformed"));
        }
    }
    else
    {
        DEBUG(10, tiramisu::str_dump("This is a let statement."));
    }

    DEBUG_INDENT(-4);

    return access;
}

/**
 * Return the Tiramisu expression associated with the computation.
 */
const tiramisu::expr &tiramisu::computation::get_expr() const
{
    return expression;
}

/**
  * Return the function where the computation is declared.
  */
tiramisu::function *tiramisu::computation::get_function() const
{
    return function;
}

/**
  * Return vector of isl_ast_expr representing the indices of the array where
  * the computation will be stored.
  */
std::vector<isl_ast_expr *> &tiramisu::computation::get_index_expr()
{
    return index_expr;
}

/**
  * Return the iteration domain of the computation.
  * In this representation, the order of execution of computations
  * is not specified, the computations are also not mapped to memory.
  */
isl_set *tiramisu::computation::get_iteration_domain() const
{
    // Every computation should have an iteration space.
    assert(iteration_domain != NULL);

    return iteration_domain;
}

/**
  * Return the time-processor domain of the computation.
  * In this representation, the logical time of execution and the
  * processor where the computation will be executed are both
  * specified.
  */
isl_set *tiramisu::computation::get_time_processor_domain() const
{
    return time_processor_domain;
}

/**
  * Return the trimmed schedule of the computation.
  * The trimmed schedule is the schedule without the
  * duplication dimension.
  */
isl_map *tiramisu::computation::get_trimmed_union_of_schedules() const
{
    isl_map *trimmed_sched = isl_map_copy(this->get_schedule());
    const char *name = isl_map_get_tuple_name(this->get_schedule(), isl_dim_out);
    trimmed_sched = isl_map_project_out(trimmed_sched, isl_dim_out, 0, 1);
    trimmed_sched = isl_map_set_tuple_name(trimmed_sched, isl_dim_out, name);

    return trimmed_sched;
}

/**
 * Return if this computation represents a let statement.
 */
bool tiramisu::computation::is_let_stmt() const
{
    return is_let;
}

/**
  * Return the name of the computation.
  */
const std::string &tiramisu::computation::get_name() const
{
    return name;
}

/**
  * Return the context of the computations.
  */
isl_ctx *tiramisu::computation::get_ctx() const
{
    return ctx;
}

/**
 * Get the number of dimensions of the iteration
 * domain of the computation.
 */
int tiramisu::computation::get_n_dimensions()
{
    assert(iteration_domain != NULL);

    return isl_set_n_dim(this->iteration_domain);
}

/**
 * Get the data type of the computation.
 */
tiramisu::primitive_t tiramisu::computation::get_data_type() const
{
    return data_type;
}

/**
  * Return the Halide statement that assigns the computation to a buffer location.
  */
Halide::Internal::Stmt tiramisu::computation::get_generated_halide_stmt() const
{
    return stmt;
}

/**
 * Compare two computations.
 *
 * Two computations are considered to be equal if they have the
 * same name.
 */
bool tiramisu::computation::operator==(tiramisu::computation comp1)
{
    return (this->get_name() == comp1.get_name());
}

/**
  * Generate the time-processor domain of the computation.
  *
  * In this representation, the logical time of execution and the
  * processor where the computation will be executed are both
  * specified.  The memory location where computations will be
  * stored in memory is not specified at the level.
  */
void tiramisu::computation::gen_time_processor_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(this->get_iteration_domain() != NULL);
    assert(this->get_schedule() != NULL);

    time_processor_domain = isl_set_apply(
                                isl_set_copy(this->get_iteration_domain()),
                                isl_map_copy(this->get_schedule()));

    DEBUG(3, tiramisu::str_dump("Iteration domain:", isl_set_to_str(this->get_iteration_domain())));
    DEBUG(3, tiramisu::str_dump("Schedule:", isl_map_to_str(this->get_schedule())));
    DEBUG(3, tiramisu::str_dump("Generated time-space domain:", isl_set_to_str(time_processor_domain)));

    DEBUG_INDENT(-4);
}

void tiramisu::computation::set_access(isl_map *access)
{
    assert(access != NULL);

    this->access = access;
}

/**
 * Set the access function of the computation.
 *
 * The access function is a relation from computations to buffer locations.
 * \p access_str is a string that represents the relation (in ISL format,
 * http://isl.gforge.inria.fr/user.html#Sets-and-Relations).
 */
void tiramisu::computation::set_access(std::string access_str)
{
    assert(!access_str.empty());

    this->access = isl_map_read_from_str(this->ctx, access_str.c_str());

    /**
     * Search for any other computation that starts with
     * "_" and that has the same name.  That computation
     * was split from this computation.
     *
     * For now we assume that only one such computation exists
     * (we check in the separate function that each computation
     * is separated only once, separated computations cannot be
     * separated themselves).
     */
    std::vector<tiramisu::computation *> separated_computation_vec =
        this->get_function()->get_computation_by_name("_" + this->get_name());

    for (auto separated_computation : separated_computation_vec)
    {
        if (separated_computation != NULL)
        {
            int pos = access_str.find(this->get_name());
            int len = this->get_name().length();

            access_str.replace(pos, len, "_" + this->get_name());
            separated_computation->access =
                isl_map_read_from_str(separated_computation->ctx,
                                      access_str.c_str());

            assert(separated_computation->get_access_relation() != NULL);
        }
    }

    /**
     * Check that if there are other computations that have the same name
     * as this computation, then the access of all of these computations
     * should be the same.
     */
    std::vector<tiramisu::computation *> computations =
            this->get_function()->get_computation_by_name(this->get_name());
    for (auto c: computations)
        if (isl_map_is_equal(this->get_access_relation(), c->get_access_relation()) == isl_bool_false)
            tiramisu::error("Computations that have the same name should also have the same access relation.", true);
}

/**
 * Generate an identity schedule for the computation.
 *
 * This identity schedule is an identity relation created from the iteration
 * domain.
 */
isl_map *tiramisu::computation::gen_identity_schedule_for_iteration_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_space *sp = isl_set_get_space(this->get_iteration_domain());
    isl_map *sched = isl_map_identity(isl_space_map_from_set(sp));
    sched = isl_map_intersect_domain(
                sched, isl_set_copy(this->get_iteration_domain()));
    sched = isl_map_coalesce(sched);

    // Add Beta dimensions.
    for (int i = 0; i < isl_space_dim(sp, isl_dim_out) + 1; i++)
    {
        sched = isl_map_add_dim_and_eq_constraint(sched, 2 * i, 0);
    }

    // Add the duplication dimension.
    sched = isl_map_add_dim_and_eq_constraint(sched, 0, 0);

    DEBUG_INDENT(-4);

    return sched;
}

isl_set *tiramisu::computation::get_trimmed_time_processor_domain()
{
    isl_set *tp_domain = isl_set_copy(this->get_time_processor_domain());
    const char *name = isl_set_get_tuple_name(isl_set_copy(tp_domain));
    isl_set *tp_domain_without_duplicate_dim =
        isl_set_project_out(isl_set_copy(tp_domain), isl_dim_set, 0, 1);
    tp_domain_without_duplicate_dim = isl_set_set_tuple_name(tp_domain_without_duplicate_dim, name);
    return tp_domain_without_duplicate_dim ;
}

/**
 * Generate an identity schedule for the computation.
 *
 * This identity schedule is an identity relation created from the
 * time-processor domain.  It removes the "duplicate" dimension (i.e.,
 * the dimension used to identify duplicate computations).
 */
isl_map *tiramisu::computation::gen_identity_schedule_for_time_space_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_set *tp_domain = this->get_trimmed_time_processor_domain();
    isl_space *sp = isl_set_get_space(tp_domain);
    isl_map *sched = isl_map_identity(isl_space_map_from_set(sp));
    sched = isl_map_intersect_domain(
                sched, isl_set_copy(this->get_trimmed_time_processor_domain()));
    sched = isl_map_set_tuple_name(sched, isl_dim_out, "");
    sched = isl_map_coalesce(sched);

    DEBUG_INDENT(-4);

    return sched;
}

/**
 * Set an identity schedule for the computation.
 *
 * This identity schedule is an identity relation created from the iteration
 * domain.
 */
void tiramisu::computation::set_identity_schedule_based_on_iteration_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_map *sched = this->gen_identity_schedule_for_iteration_domain();
    DEBUG(3, tiramisu::str_dump("The following identity schedule is generated (setting schedule 0): "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(sched)));
    this->set_schedule(sched);
    DEBUG(3, tiramisu::str_dump("The identity schedule for the original computation is set."));

    DEBUG_INDENT(-4);
}

int computation::get_duplicates_number() const
{
    return this->duplicate_number;
}

isl_map *computation::get_schedule() const
{
    return this->schedule;
}

void tiramisu::computation::add_associated_let_stmt(std::string variable_name, tiramisu::expr e)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!variable_name.empty());
    assert(e.is_defined());

    DEBUG(3, tiramisu::str_dump("Adding a let statement associated to the computation " +
                                this->get_name() + "."));
    DEBUG(3, tiramisu::str_dump("The name of the variable of the let statement: " + variable_name +
                                "."));
    DEBUG(3, tiramisu::str_dump("Expression: ")); e.dump(false);

    this->associated_let_stmts.push_back({variable_name, e});

    DEBUG_INDENT(-4);
}

const std::vector<std::pair<std::string, tiramisu::expr>> &tiramisu::computation::get_associated_let_stmts() const
{
    return this->associated_let_stmts;
}

bool tiramisu::computation::has_accesses() const
{
    if ((this->get_expr().get_op_type() == tiramisu::o_allocate) ||
        (this->get_expr().get_op_type() == tiramisu::o_free))
        return false;
    else
        return true;
}

/**
 * Set the expression of the computation.
 */
void tiramisu::computation::set_expression(const tiramisu::expr &e)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("The original expression is: "));
    e.dump(false);
    DEBUG(3, tiramisu::str_dump(""));

    DEBUG(3, tiramisu::str_dump("Traversing the expression to replace non-affine accesses by a constant definition."));
    tiramisu::expr modified_e = traverse_expr_and_replace_non_affine_accesses(this, e);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("The new expression is: "); modified_e.dump(false););
    DEBUG(3, tiramisu::str_dump(""));

    this->expression = modified_e.copy();

    DEBUG_INDENT(-4);
}

/**
 * Set the name of the computation.
 */
void tiramisu::computation::set_name(const std::string &n)
{
    this->name = n;
}

/**
  * Bind the computation to a buffer.
  * i.e. create a one-to-one data mapping between the computation
  * the buffer.
  */
void tiramisu::computation::bind_to(buffer *buff)
{
    assert(buff != NULL);

    isl_space *sp = isl_set_get_space(this->get_iteration_domain());
    isl_map *map = isl_map_identity(isl_space_map_from_set(sp));
    map = isl_map_intersect_domain(map, isl_set_copy(this->get_iteration_domain()));
    map = isl_map_set_tuple_name(map, isl_dim_out, buff->get_name().c_str());
    map = isl_map_coalesce(map);
    DEBUG(2, tiramisu::str_dump("\nBinding. The following access function is set: ",
                                isl_map_to_str(map)));
    this->set_access(isl_map_to_str(map));
    isl_map_free(map);
}

void tiramisu::computation::mark_as_let_statement()
{
    this->is_let = true;
}

/****************************************************************************
 ****************************************************************************
 ***************************** Constant class *******************************
 ****************************************************************************
 ****************************************************************************/

tiramisu::constant::constant(
    std::string param_name, const tiramisu::expr &param_expr,
    tiramisu::primitive_t t,
    bool function_wide,
    tiramisu::computation *with_computation,
    int at_loop_level,
    tiramisu::function *func): tiramisu::computation()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(!param_name.empty() && "Parameter name empty");
    assert((func != NULL) && "Function undefined");
    assert(((function_wide && !with_computation) || (!function_wide && with_computation)) &&
           "with_computation, should be set only if function_wide is false");

    DEBUG(3, tiramisu::str_dump("Constructing a constant."));

    if (function_wide)
    {
        this->set_name(param_name);
        this->set_expression(param_expr);
        func->add_invariant(*this);
        this->mark_as_let_statement();

        DEBUG(3, tiramisu::str_dump("The constant is function wide, its name is : "));
        DEBUG(3, tiramisu::str_dump(this->get_name()));
    }
    else
    {
        assert((with_computation != NULL) &&
               "A valid computation should be provided.");
        assert((at_loop_level >= computation::root_dimension) &&
               "Invalid root dimension.");

        isl_set *iter = with_computation->get_iteration_domain();
        int projection_dimension = at_loop_level + 1;
        iter = isl_set_project_out(isl_set_copy(iter),
                                   isl_dim_set,
                                   projection_dimension,
                                   isl_set_dim(iter, isl_dim_set) - projection_dimension);
        iter = isl_set_set_tuple_name(iter, param_name.c_str());
        std::string iteration_domain_str = isl_set_to_str(iter);

        DEBUG(3, tiramisu::str_dump(
                  "Computed iteration space for the constant assignment",
                  isl_set_to_str(iter)));

        init_computation(iteration_domain_str, func, param_expr, true, t);

        this->mark_as_let_statement();

        DEBUG_NO_NEWLINE(10,
                         tiramisu::str_dump("The computation representing the assignment:");
                         this->dump(true));

        // Set the schedule of this computation to be executed
        // before the computation.
        this->before(*with_computation, at_loop_level);

        DEBUG(3, tiramisu::str_dump("The constant is not function wide, the iteration domain of the constant is: "));
        DEBUG(3, tiramisu::str_dump(isl_set_to_str(this->get_iteration_domain())));
    }

    DEBUG_INDENT(-4);
}

void tiramisu::constant::dump(bool exhaustive) const
{
    if (ENABLE_DEBUG)
    {
        std::cout << "Invariant \"" << this->get_name() << "\"" << std::endl;
        std::cout << "Expression: ";
        this->get_expr().dump(false);
        std::cout << std::endl;
    }
}

}
