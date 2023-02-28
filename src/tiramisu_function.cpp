#include <isl/ctx.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/flow.h>
#include <isl/id.h>
#include <isl/constraint.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/ilp.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>

namespace tiramisu
{

/**
 * Retrieve the access function of the ISL AST leaf node (which represents a
 * computation). Store the access in computation->access.
 */
isl_ast_node *for_code_generator_after_for(
        isl_ast_node *node, isl_ast_build *build, void *user);

isl_map *isl_map_align_range_dims(isl_map *map, int max_dim)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(map != NULL);
    int mdim = isl_map_dim(map, isl_dim_out);
    //assert(max_dim >= mdim);

    // in case where the max_dim is bigger than this map dimension, we add zeros to the schedule.
    if(max_dim >= mdim)
    {
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
    }
    else
    {
      // in case where the max_dim is smaller than this map dimension, we project_out (delete) additional dimensions
       
        DEBUG(10, tiramisu::str_dump("Input map:", isl_map_to_str(map)));
        map = isl_map_project_out(map,isl_dim_out,max_dim,mdim-max_dim);
        DEBUG(10, tiramisu::str_dump("After alignment, map = ",
                                    isl_map_to_str(map)));
    }


    DEBUG_INDENT(-4);
    return map;
}

/**
  * Add a buffer to the function.
  */
void function::add_buffer(std::pair <std::string, tiramisu::buffer *> buf)
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
    this->halide_stmt = Halide::Internal::Stmt();
    this->ast = NULL;
    this->context_set = NULL;
    this->use_low_level_scheduling_commands = false;
    this->_needs_rank_call = false;

    // Allocate an ISL context.  This ISL context will be used by
    // the ISL library calls within Tiramisu.
    this->ctx = isl_ctx_alloc();
};

isl_union_map *tiramisu::function::compute_dep_graph() {
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_union_map *result = NULL;

    for (const auto &consumer : this->get_computations()) {
        DEBUG(3, tiramisu::str_dump("Computing the dependences involving the computation " +
                                    consumer->get_name() + "."));
        DEBUG(3, tiramisu::str_dump("Computing the accesses of the computation."));

        isl_union_map *accesses_union_map = NULL;
        std::vector < isl_map * > accesses_vector;
        generator::get_rhs_accesses(this, consumer, accesses_vector, false);

        DEBUG(3, tiramisu::str_dump("Vector of accesses computed."));

        if (!accesses_vector.empty()) {
            // Create a union map of the accesses to the producer.
            if (accesses_union_map == NULL) {
                isl_space *space = isl_map_get_space(accesses_vector[0]);
                assert(space != NULL);
                accesses_union_map = isl_union_map_empty(space);
            }

            for (size_t i = 0; i < accesses_vector.size(); ++i) {
                isl_map *reverse_access = isl_map_reverse(accesses_vector[i]);
                accesses_union_map = isl_union_map_union(isl_union_map_from_map(reverse_access),
                                                         accesses_union_map);
            }

            //accesses_union_map = isl_union_map_intersect_range(accesses_union_map, isl_union_set_from_set(isl_set_copy(consumer->get_iteration_domain())));
            //accesses_union_map = isl_union_map_intersect_domain(accesses_union_map, isl_union_set_from_set(isl_set_copy(consumer->get_iteration_domain())));

            DEBUG(3, tiramisu::str_dump("Accesses after filtering."));
            DEBUG(3, tiramisu::str_dump(isl_union_map_to_str(accesses_union_map)));

            if (result == NULL) {
                result = isl_union_map_copy(accesses_union_map);
                isl_union_map_free(accesses_union_map);
            } else {
                result = isl_union_map_union(result, accesses_union_map);
            }
        }
    }

    DEBUG(3, tiramisu::str_dump("Dep graph: "));
    if (result != NULL)
    {
        DEBUG(3, tiramisu::str_dump(isl_union_map_to_str(result)));
    }
    else
    {
        DEBUG(3, tiramisu::str_dump("Null."));
    }

    DEBUG_INDENT(-4);
    DEBUG(3, tiramisu::str_dump("End of function"));

    return result;
}

void tiramisu::function::calculate_dep_flow()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(this->get_computations().size() > 0);
    assert(this->get_computations()[0]->get_schedule() != NULL);

    DEBUG(3, tiramisu::str_dump(" generating depandencies graph"));

    isl_union_map * ref_res = this->compute_dep_graph();

    if(ref_res == NULL)
    {
        // no deps fill with empty union maps

        std::string str_map = "{}";

        this->dep_read_after_write = isl_union_map_read_from_str(this->get_isl_ctx(),str_map.c_str());

        this->dep_write_after_read = isl_union_map_read_from_str(this->get_isl_ctx(),str_map.c_str());;

        this->dep_write_after_write = isl_union_map_read_from_str(this->get_isl_ctx(),str_map.c_str());;

        this->live_in_access = isl_union_map_read_from_str(this->get_isl_ctx(),str_map.c_str());;

        this->live_out_access = isl_union_map_read_from_str(this->get_isl_ctx(),str_map.c_str());;

        DEBUG(3, tiramisu::str_dump(" No deps detected just empty maps "));

        DEBUG_INDENT(-4);

        return ;
    }

    isl_union_map * ref_graph = isl_union_map_reverse(ref_res);

    

    DEBUG(3, tiramisu::str_dump(" the referencing union map is for dependecy analysis: "+std::string(isl_union_map_to_str(ref_graph))));

    int time_space_dim = isl_map_dim(this->get_computations()[0]->get_schedule(), isl_dim_out);

    std::string to_time_space_map_str = "[";

    std::string to_time_space_map_str_2 = "[";

    for(int i=0; i < time_space_dim; i++)
    {
        to_time_space_map_str+="t"+std::to_string(i);
        to_time_space_map_str_2+="t"+std::to_string(i);

        if(i != (time_space_dim - 1))
        {
            to_time_space_map_str+=",";
            to_time_space_map_str_2+=",";
            
        }

    }
    std::string ready_time_str = to_time_space_map_str+"]->" + to_time_space_map_str_2+"]";// without {} yet

    DEBUG(3, tiramisu::str_dump(" using to generate time stamp tmp map "+ready_time_str));


    std::string access_start = "{}";

    // S0[i,j] -> buff[i] the writing stmt
    isl_union_map * write_access = isl_union_map_read_from_str(this->get_isl_ctx(),access_start.c_str());

    isl_union_map * isl_schedule = isl_union_map_read_from_str(this->get_isl_ctx(),access_start.c_str());


    std::string identity = "";

    isl_map * isl_identity = NULL;
    
    for(auto& comput : this->get_computations())
    {
        identity = "{"+comput->get_name() +ready_time_str + "}";

        isl_identity = isl_map_read_from_str(this->get_isl_ctx(),identity.c_str());

        // TODO : use default schedule instead when save/restore states is implemented 
        isl_map * corrected = isl_map_apply_range(isl_map_copy(comput->get_schedule()),isl_identity);

        DEBUG(10, tiramisu::str_dump(" - > compuatation's schedule to time stamp op result is : "+std::string(isl_map_to_str(corrected))));

        isl_schedule = isl_union_map_union(isl_schedule , isl_union_map_from_map(corrected));

        write_access = isl_union_map_union(write_access,isl_union_map_from_map(isl_map_copy(comput->get_access_relation())));
        
    } 

    isl_union_set * iteration_domains = this->get_iteration_domain();

    isl_union_map * write_acccess_without_domain = isl_union_map_copy(write_access);

    write_access = isl_union_map_intersect_domain(write_access, isl_union_set_copy(iteration_domains));

    isl_schedule = isl_union_map_intersect_domain(isl_schedule, isl_union_set_copy(iteration_domains));
    
    isl_union_map * read_access = isl_union_map_apply_range(
        isl_union_map_copy(ref_graph),
        write_acccess_without_domain
    );

    read_access = isl_union_map_intersect_domain(read_access, isl_union_set_copy(iteration_domains));

    //combine reads previous with their access to establish the read access S0[i,j] -> buf2[j] in read 

    DEBUG(3, tiramisu::str_dump("the overall function schedule is : "+std::string(isl_union_map_to_str(isl_schedule))));

    DEBUG(3, tiramisu::str_dump("the write access for computations is : "+std::string(isl_union_map_to_str(write_access))));

    DEBUG(3, tiramisu::str_dump(" The read access for computations : "+std::string(isl_union_map_to_str(read_access))));

    isl_union_access_info *info = isl_union_access_info_from_sink( isl_union_map_copy(read_access));

    info = isl_union_access_info_set_schedule_map(info,isl_union_map_copy(isl_schedule));

    info = isl_union_access_info_set_must_source(info,isl_union_map_copy(write_access));

    isl_union_flow * flow = isl_union_access_info_compute_flow(info);

    //DEBUG(3, tiramisu::str_dump(" dependency analysis with must for read after write ( no predicats ) result  : "+std::string(isl_union_flow_to_str(flow))));

    isl_union_map * read_after_write_dep = isl_union_flow_get_full_must_dependence(flow);

    isl_union_map * read_from_outside = isl_union_flow_get_must_no_source(flow);

    DEBUG(3, tiramisu::str_dump(" read after write True dependencies are in the form { last_write_access -> the read statement } : "+std::string(isl_union_map_to_str(read_after_write_dep))));
       
    DEBUG(3, tiramisu::str_dump(" live-in : the computations / statement with these read access have not been written in this function (outside value)  : "+std::string(isl_union_map_to_str(read_from_outside))));
    

    info = isl_union_access_info_from_sink(isl_union_map_copy(write_access));

    info = isl_union_access_info_set_schedule_map(info,isl_union_map_copy(isl_schedule));

    info = isl_union_access_info_set_must_source(info,isl_union_map_copy(write_access));

    flow = isl_union_access_info_compute_flow(info);

    isl_union_map * write_after_write_dep = isl_union_flow_get_full_must_dependence(flow);

    DEBUG(3, tiramisu::str_dump(" write after write dependencies are { last_previous_write -> new write stmt } : "+std::string(isl_union_map_to_str(write_after_write_dep))));


    isl_union_map * not_last_writes = isl_union_map_range_factor_range( isl_union_map_copy(write_after_write_dep));
    
    isl_union_map * live_out = isl_union_map_subtract(
        isl_union_map_copy(write_access),
        isl_union_map_copy(not_last_writes)
    );

    live_out = isl_union_map_intersect_domain(live_out, this->get_iteration_domain());

    DEBUG(3, tiramisu::str_dump(" live out last access are : "+std::string(isl_union_map_to_str(live_out))));

    isl_union_map * read_without_write_stmt = isl_union_map_subtract(isl_union_map_copy(read_access), isl_union_map_copy(write_access));

    info = isl_union_access_info_from_sink(isl_union_map_copy(write_access));

    info = isl_union_access_info_set_schedule_map(info,isl_union_map_copy(isl_schedule));

    info = isl_union_access_info_set_may_source(info,isl_union_map_copy(read_without_write_stmt));

    info = isl_union_access_info_set_kill(info,isl_union_map_copy(write_access));

    flow = isl_union_access_info_compute_flow(info);

    //DEBUG(3, tiramisu::str_dump(" dependency analysis for WAR dep : "+std::string(isl_union_flow_to_str(flow))));

    isl_union_map * anti_dependencies = isl_union_flow_get_full_may_dependence(flow);

    DEBUG(3, tiramisu::str_dump(" write after read anti_dependencies are in the form { last_previous_read -> new write stmt } : "+std::string(isl_union_map_to_str(anti_dependencies))));

    //DEBUG(3, tiramisu::str_dump(" the initialisation stmt writes with no previous read before are : "+std::string(isl_union_map_to_str(initialisation_access))));
      
    this->dep_read_after_write = read_after_write_dep;

    this->dep_write_after_read = anti_dependencies;

    this->dep_write_after_write = write_after_write_dep;

    this->live_in_access = read_from_outside;

    this->live_out_access = live_out;
    
    DEBUG_INDENT(-4);

}

const std::map<std::string, tiramisu::buffer *> tiramisu::function::get_mapping() const
{
  return this->mapping;
}

void  tiramisu::function::add_mapping(std::pair<std::string,tiramisu::buffer *> p)
{
  this->mapping.insert(p);
}

/**
 * This function takes computation pts to C1 and C2 which are the first and the last computation
 * of the fucntion.
 * Automatic_communication fonction verifies if the user wants to manage the copies manually or not.
 * By default, the data management will be done automatically, and the copies will be inserted
 * in the beginning and at the end of the gpu funcion.
 * The inputs of the function will be copied to the device before the computations, and the
 * outputs of the functions will be copied back to the host at the end.
 * We have two cases copies to constant memory and the default case which is copies to the global memory.
 */
void function::Automatic_communication(tiramisu::computation* c1, tiramisu::computation* c2)
{
    assert(c1 != nullptr && "C1 = NULL ");
    assert(c2 != nullptr && "C2 = NULL ");
    assert(c1->get_predecessor() == nullptr && "C1 must be the computation that hasn't a predessessor ");
    assert(c2->get_successor() == nullptr && "C2 must be the computation that hasn't a successor ");
    std::map<std::string, tiramisu::buffer*> buff = this->get_buffers();
    std::map<std::string, tiramisu::buffer*> mp = this->mapping;
    std::map<std::string, tiramisu::buffer*>::iterator it;
    std::string name, cpt_name;
    int i = 1;
    tiramisu::computation* first_cpt = c1;
    tiramisu::computation* last_cpt = c2;

    for (it = mp.begin(); it != mp.end(); ++it)
    {
        name = it->first;
        assert(it->second->get_argument_type() == tiramisu::a_temporary  && "Mapping field should contain a string corresponding to the name of a cpu buffer and a ptr to the corresponding gpu buffer ");
        if (it->second->automatic_gpu_copy == true)
        {
            cpt_name= "cpt" + std::to_string(i);
            i++;
            switch (it->second->location)
            {
                case cuda_ast::memory_location::constant:
                    if (buff.find(name)->second->get_argument_type() == tiramisu::a_input)
                    {
                        tiramisu::computation* c =  new tiramisu::computation(cpt_name, {},
                            memcpy(*(buff.find(name)->second),(*(it->second))));
                        (*c).then((*first_cpt), computation::root);
                        first_cpt = c;
                    }
                    break;
                default:
                    if (buff.find(name)->second->get_argument_type() == tiramisu::a_input)
                    {
                        tiramisu::computation* c =  new tiramisu::computation(cpt_name, {},
                            memcpy(*(buff.find(name)->second),(*(it->second))));
                        (*c).then((*first_cpt), computation::root);
                        first_cpt = c;
                    }
                    if (buff.find(name)->second->get_argument_type() == tiramisu::a_output)
                    {
                        tiramisu::computation* c =  new tiramisu::computation(cpt_name, {},
                            memcpy((*(it->second)),*(buff.find(name)->second)));
                        (*last_cpt).then((*c), computation::root);
                        last_cpt = c;
                    }
                    break;
            }
        }
        else
        {
            DEBUG(3, tiramisu::str_dump("Communication should be done manually !"));
        }
    }
}

// TODO: get_live_in_computations() does not consider the case of "maybe"
// live-out (non-affine control flow, ...).
std::vector<tiramisu::computation *> tiramisu::function::get_live_in_computations()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert((this->get_computations().size() > 0) &&
           "The function should have at least one computation.");

    std::vector < tiramisu::computation * > first;
    isl_union_map *deps = this->compute_dep_graph();

    if (deps != NULL) {
        if (isl_union_map_is_empty(deps) == isl_bool_false) {
            // The domains and the ranges of the dependences
            isl_union_set *domains = isl_union_map_domain(isl_union_map_copy(deps));
            isl_union_set *ranges = isl_union_map_range(isl_union_map_copy(deps));

            DEBUG(3, tiramisu::str_dump("Ranges of the dependence graph.", isl_union_set_to_str(ranges)));
            DEBUG(3, tiramisu::str_dump("Domains of the dependence graph.", isl_union_set_to_str(domains)));

            /** In a dependence graph, since dependences create a chain (i.e., the end of
             *  a dependence is the beginning of the following), then each range of
             *  a dependence has a set domains that correspond to it (i.e., that their
             *  union is equal to it).  If a domain exists but does not have ranges that
             *  are equal to it, then that domain is the first domain.
             *
             *  To compute those domains that do not have corresponding ranges, we
             *  compute (domains - ranges).
             *
             *  These domains that do not have a corresponding range (i.e., are not
             *  produced by previous computations) and that are not defined (i.e., do
             *  not have any expression) are live-in.
             */
            isl_union_set *first_domains = isl_union_set_subtract(domains, ranges);
            DEBUG(3, tiramisu::str_dump("Domains - Ranges :", isl_union_set_to_str(first_domains)));

            if (isl_union_set_is_empty(first_domains) == isl_bool_false) {
                for (const auto &c : this->body) {
                    isl_space *sp = isl_set_get_space(c->get_iteration_domain());
                    isl_set *s = isl_set_universe(sp);
                    isl_union_set *intersect =
                            isl_union_set_intersect(isl_union_set_from_set(s),
                                                    isl_union_set_copy(first_domains));

                    if ((isl_union_set_is_empty(intersect) == isl_bool_false) &&
                        (c->get_expr().is_defined() == false))
                    {
                        first.push_back(c);
                    }
                    isl_union_set_free(intersect);
                }

                DEBUG(3, tiramisu::str_dump("First computations:"));
                for (const auto &c : first) {
                    DEBUG(3, tiramisu::str_dump(c->get_name() + " "));
                }
            } else {
                // If the difference between domains and ranges is empty, then
                // all the computations of the program are recursive (assuming
                // that the set of dependences is not empty).
                first = this->body;
            }

            isl_union_set_free(first_domains);
        } else {
            // If the program does not have any dependence, then
            // all the computations should be considered as the first
            // computations.
            first = this->body;
        }

        isl_union_map_free(deps);
    }

    DEBUG(3, tiramisu::str_dump("Removing inline computations."));
    std::vector<computation* > result;
    for (computation * c: first)
        if (! c->is_inline_computation())
            result.push_back(c);

    DEBUG_INDENT(-4);

    return result;
}

// TODO: get_live_out_computations() does not consider the case of "maybe"
// live-out (non-affine control flow, ...).
std::vector<tiramisu::computation *> tiramisu::function::get_live_out_computations()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert((this->get_computations().size() > 0) &&
           "The function should have at least one computation.");

    std::vector<tiramisu::computation *> first;
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
             *  union is equal to it).  If a range exists but does not have domains that
             *  are equal to it, then that range is the last range.
             *
             *  To compute those ranges that do not have corresponding domains, we
             *  compute (ranges - domains).
             */
            isl_union_set *last_ranges = isl_union_set_subtract(ranges, domains);
            DEBUG(3, tiramisu::str_dump("Ranges - Domains :", isl_union_set_to_str(last_ranges)));

            if (isl_union_set_is_empty(last_ranges) == isl_bool_false)
            {
                for (const auto &c : this->body)
                {
                    isl_space *sp = isl_set_get_space(c->get_iteration_domain());
                    isl_set *s = isl_set_universe(sp);
                    isl_union_set *intersect =
                        isl_union_set_intersect(isl_union_set_from_set(s),
                                                isl_union_set_copy(last_ranges));

                    if (isl_union_set_is_empty(intersect) == isl_bool_false)
                    {
                        first.push_back(c);
                    }
                    isl_union_set_free(intersect);
                }

                DEBUG(3, tiramisu::str_dump("Last computations:"));
                for (const auto &c : first)
                {
                    DEBUG(3, tiramisu::str_dump(c->get_name() + " "));
                }
            }
            else
            {
                // If the difference between ranges and domains is empty, then
                // all the computations of the program are recursive (assuming
                // that the set of dependences is not empty).
                first = this->body;
            }

            isl_union_set_free(last_ranges);
        }
        else
        {
            // If the program does not have any dependence, then
            // all the computations should be considered as the last
            // computations.
            first = this->body;
        }
        isl_union_map_free(deps);
    }
    else
    {
        // If the program does not have any dependence, then
        // all the computations should be considered as the last
        // computations.
        first = this->body;
    }

    DEBUG(3, tiramisu::str_dump("Removing inline computations."));
    std::vector<computation* > result;
    for (computation * c: first)
        if (! c->is_inline_computation())
            result.push_back(c);

    assert((result.size() > 0) && "The function should have at least one last computation.");

    DEBUG_INDENT(-4);


    return result;
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
    std::vector<tiramisu::computation *> first = this->get_live_out_computations();

    assert(first.size() > 0);

    isl_union_set *Domains = NULL;
    Domains = isl_union_set_empty(isl_set_get_space(first[0]->get_iteration_domain()));
    for (auto c : first)
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
            if (c->has_multiple_definitions() == false)
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
                assert((isl_set_plain_is_universe(c->get_iteration_domain()) == isl_bool_false) &&
                       "The iteration domain of an update should not be universe.");
                assert((isl_set_is_empty(c->get_iteration_domain()) == isl_bool_false) &&
                       "The iteration domain of an update should not be empty.");
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

void tiramisu::function::dump_dep_graph()
{

    tiramisu::str_dump("Dependence graph:\n");
    isl_union_map *deps = isl_union_map_copy(this->compute_dep_graph());
    isl_union_map_dump(deps);
}

void function::dump_halide_stmt() const
{
    tiramisu::str_dump("\n\n");
    tiramisu::str_dump("\nGenerated Halide Low Level IR:\n");
    std::cout << this->get_halide_stmt();
    tiramisu::str_dump("\n\n\n\n");
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

void function::dump_sched_graph_dfs(computation * comp,
                                    std::unordered_set<computation *> &visited)
{
    // Do not visit anything that was already returned
    if (visited.find(comp) != visited.end())
        return;

    visited.insert(comp);

    for (auto &edge: this->sched_graph[comp])
    {
        const std::string level = ((edge.second == computation::root_dimension) ?
                                   "root" :
                                   std::to_string(edge.second));

        DEBUG(3, tiramisu::str_dump(comp->get_unique_name() +
                                    "=[" + level + "]=>" +
                                    edge.first->get_unique_name()));

        dump_sched_graph_dfs(edge.first, visited);
    }
}

void function::dump_sched_graph()
{
    DEBUG(3, tiramisu::str_dump("Number of schedule graph roots is " +
                                std::to_string(this->starting_computations.size())));
    DEBUG(3, tiramisu::str_dump("The roots are:"));

    for (auto root: this->starting_computations)
        DEBUG(3, tiramisu::str_dump(" * " + root->get_unique_name()));

    // Contains all nodes that have been visited
    std::unordered_set<computation *> visited;

    DEBUG(3, tiramisu::str_dump("Displaying schedule graph"));

    for (auto &comp: this->starting_computations)
    {
        dump_sched_graph_dfs(comp, visited);
    }

    DEBUG(3, tiramisu::str_dump("Finished displaying schedule graph"));
}

bool function::is_sched_graph_tree_dfs(computation * comp,
                                       std::unordered_set<computation *> &visited)
{
    // Do not visit anything that was already returned
    if (visited.find(comp) != visited.end())
        return false;

    visited.insert(comp);

    for (auto &edge: this->sched_graph[comp])
    {
        if (!is_sched_graph_tree_dfs(edge.first, visited))
            return false;
    }

    return true;
}

void function::clear_sched_graph()
{
    sched_graph.clear();
    sched_graph_reversed.clear();
}

bool function::is_sched_graph_tree()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if (this->starting_computations.size() != 1)
    {
        DEBUG_INDENT(-4);
        return false;
    }

    // Contains all nodes that have been visited
    std::unordered_set<computation *> visited;

    for (auto &comp: this->starting_computations)
    {
        if (!is_sched_graph_tree_dfs(comp, visited))
        {
            DEBUG_INDENT(-4);
            return false;
        }
    }

    DEBUG_INDENT(-4);
    return true;
}


/**
  * Get the arguments of the function.
  */
// @{
const std::vector<tiramisu::buffer *> &function::get_arguments() const {
    return function_arguments;
}
// @}

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
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev >= 0);

    bool found = false;

    DEBUG(10, tiramisu::str_dump("Checking if the computation " + comp +
                                 " should be parallelized" +
                                 " at the loop level " + std::to_string(lev)));

    for (const auto &pd : this->parallel_dimensions)
    {
        DEBUG(10, tiramisu::str_dump("Checking if the computation " + comp +
                                     " at the loop level " + std::to_string(lev) +
                                     " is equal to the tagged computation " +
                                     pd.first + " at the level " + std::to_string(pd.second)));

        if ((pd.first == comp) && (pd.second == lev))
        {
            found = true;

            DEBUG(10, tiramisu::str_dump("Yes equal."));
        }
    }

    std::string str = "Dimension " + std::to_string(lev) +
                      (found ? " should" : " should not")
                       + " be mapped to CPU thread.";
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);

    return found;
}

/**
* Return the unrolling factor used to unroll the computation \p comp
* at the loop level \p lev.
*/
int function::get_unrolling_factor(const std::string &comp, int lev) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev >= 0);

    int unrolling_factor = -1;
    bool found = false;

    for (const auto &pd : this->unroll_dimensions)
    {
        if ((std::get<0>(pd) == comp) && (std::get<1>(pd) == lev))
        {
            unrolling_factor = std::get<2>(pd);
            found = true;
        }
    }

    std::string str = "Dimension " + std::to_string(lev) +
                      (found ? " should" : " should not")
                       + " be unrolled with a factor of " +
                       std::to_string(unrolling_factor);
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);

    return unrolling_factor;
}

/**
* Return the vector length of the computation \p comp at
* at the loop level \p lev.
*/
int function::get_vector_length(const std::string &comp, int lev) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev >= 0);

    int vector_length = -1;
    bool found = false;

    for (const auto &pd : this->vector_dimensions)
    {
        if ((std::get<0>(pd) == comp) && (std::get<1>(pd) == lev))
        {
            vector_length = std::get<2>(pd);
            found = true;
        }
    }

    std::string str = "Dimension " + std::to_string(lev) +
                      (found ? " should" : " should not")
                       + " be vectorized with a vector length of " +
                       std::to_string(vector_length);
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);

    return vector_length;
}

computation * function::get_first_cpt() {
    if (this->is_sched_graph_tree()) {
        tiramisu::computation* cpt = this->sched_graph.begin()->first;
        while (cpt->get_predecessor() != NULL) {
            cpt = cpt->get_predecessor();
        }
        return cpt;
    } else {
        DEBUG(3, tiramisu::str_dump(" this->is_sched_graph_tree(): false."));
        return NULL;
    }
}

computation * function::get_last_cpt() {
    if (this->is_sched_graph_tree()) {
        tiramisu::computation* cpt = this->sched_graph.begin()->first;
        while (cpt->get_successor() != NULL) {
            cpt = cpt->get_successor();
        }
        return cpt;
    } else {
        DEBUG(3, tiramisu::str_dump("this->is_sched_graph_tree(): false."));
        return NULL;
    }
}

/**
* Return true if the computation \p comp should be vectorized
* at the loop level \p lev.
*/
bool function::should_vectorize(const std::string &comp, int lev) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev >= 0);

    bool found = false;

    DEBUG(10, tiramisu::str_dump("Checking if the computation " + comp +
                                 " should be vectorized" +
                                 " at the loop level " + std::to_string(lev)));

    DEBUG_INDENT(4);

    for (const auto &pd : this->vector_dimensions)
    {
        DEBUG(10, tiramisu::str_dump("Comparing " + comp + " to " + std::get<0>(pd)));
        DEBUG(10, tiramisu::str_dump(std::get<0>(pd) + " is marked for vectorization at level " + std::to_string(std::get<1>(pd))));

        if ((std::get<0>(pd) == comp) && (std::get<1>(pd) == lev))
            found = true;
    }

    std::string str = "Dimension " + std::to_string(lev) +
                      (found ? " should" : " should not")
                       + " be vectorized.";
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);
    DEBUG_INDENT(-4);

    return found;
}

bool function::should_distribute(const std::string &comp, int lev) const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(!comp.empty());
    assert(lev >= 0);

    bool found = false;

    DEBUG(10, tiramisu::str_dump("Checking if the computation " + comp +
                                 " should be distributed" +
                                 " at the loop level " + std::to_string(lev)));

    for (const auto &pd : this->distributed_dimensions)
    {
        DEBUG(10, tiramisu::str_dump("Comparing " + comp + " to " + std::get<0>(pd)));
        DEBUG(10, tiramisu::str_dump(std::get<0>(pd) + " is marked for distribution at level " + std::to_string(std::get<1>(pd))));

        if ((std::get<0>(pd) == comp) && (std::get<1>(pd) == lev))
            found = true;
    }

    std::string str = "Dimension " + std::to_string(lev) +
                      (found ? " should" : " should not")
                      + " be distributed.";
    DEBUG(10, tiramisu::str_dump(str));

    DEBUG_INDENT(-4);

    return found;
}

bool tiramisu::function::needs_rank_call() const
{
    return _needs_rank_call;
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
  * Set the iterator names of the function.
  */
void function::set_original_number_of_computations()
{
    original_number_of_computations = this->body.size();
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

void function::gen_halide_obj(const std::string &obj_file_name, Halide::Target::OS os, Halide::Target::Arch arch, int bits) const
{
  Halide::Target target = Halide::get_host_target();
  gen_halide_obj(obj_file_name, target.os, target.arch, target.bits, tiramisu::hardware_architecture_t::arch_cpu);
}

void function::gen_halide_obj(const std::string &obj_file_name) const
{
    Halide::Target target = Halide::get_host_target();
    gen_halide_obj(obj_file_name, target.os, target.arch, target.bits);
}

void function::gen_halide_obj(const std::string &obj_file_name, const tiramisu::hardware_architecture_t hw_architecture) const
{
  Halide::Target target = Halide::get_host_target();
  gen_halide_obj(obj_file_name, target.os, target.arch, target.bits, hw_architecture);
}


// @}

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
    for (auto const comp : this->get_computations())
    {
        std::vector<tiramisu::computation *> same_name_computations =
            this->get_computation_by_name(comp->get_name());

        int i = 0;

        if (same_name_computations.size() > 1)
            for (auto c : same_name_computations)
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
    assert(this->get_trimmed_time_processor_domain() != NULL);
    assert(this->get_aligned_identity_schedules() != NULL);

    isl_ctx *ctx = this->get_isl_ctx();
    assert(ctx != NULL);
    isl_ast_build *ast_build;

    // Rename updates so that they have different names because
    // the code generator expects each unique name to have
    // an expression, different computations that have the same
    // name cannot have different expressions.

    this->rename_computations();

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
    isl_options_set_ast_build_group_coscheduled(ctx, 1);

    ast_build = isl_ast_build_set_after_each_for(ast_build, &tiramisu::for_code_generator_after_for,
                NULL);
    ast_build = isl_ast_build_set_at_each_domain(ast_build, &tiramisu::generator::stmt_code_generator,
                this);

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

void tiramisu::function::allocate_and_map_buffers_automatically()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(10, tiramisu::str_dump("Computing live-out computations."));
    // Compute live-in and live-out buffers
    std::vector<tiramisu::computation *> liveout = this->get_live_out_computations();
    DEBUG(10, tiramisu::str_dump("Allocating/Mapping buffers for live-out computations."));
    for (auto &comp: liveout)
        if (comp->get_automatically_allocated_buffer() == NULL)
            comp->allocate_and_map_buffer_automatically(a_output);

    DEBUG(10, tiramisu::str_dump("Computing live-in computations."));
    // Compute live-in and live-out buffers
    std::vector<tiramisu::computation *> livein =
            this->get_live_in_computations();
    DEBUG(10, tiramisu::str_dump("Allocating/Mapping buffers for live-in computations."));
    // Allocate each live-in computation that is not also live-out (we check that
    // by checking that it was not allocated yet)
    for (auto &comp: livein)
        if (comp->get_automatically_allocated_buffer() == NULL)
            comp->allocate_and_map_buffer_automatically(a_input);

    DEBUG(10, tiramisu::str_dump("Allocating/Mapping buffers for non live-in and non live-out computations."));
    // Allocate the buffers automatically for non inline computations
    // Allocate each computation that is not live-in or live-out (we check that
    // by checking that it was not allocated)
    for (int b = 0; b < this->body.size(); b++)
    {
        DEBUG(3, tiramisu::str_dump("Inline " + this->body[b]->get_name() + " " + std::to_string(this->body[b]->is_inline_computation())));
        if (this->body[b]->is_inline_computation()) {
            DEBUG(3, tiramisu::str_dump("Skipping inline computation " + this->body[b]->get_name()));
            continue;
        }
        DEBUG(10, tiramisu::str_dump("Allocating/Mapping buffers for " + this->body[b]->get_name()));
        if ((this->body[b]->get_expr().get_expr_type() == tiramisu::e_op))
        {
            if (this->body[b]->get_expr().get_op_type() != tiramisu::o_allocate)
            {
                if (this->body[b]->get_automatically_allocated_buffer() == NULL)
                    this->body[b]->allocate_and_map_buffer_automatically(a_temporary);
            }
        }
        else
        {
            if (this->body[b]->get_automatically_allocated_buffer() == NULL) {
                this->body[b]->allocate_and_map_buffer_automatically(a_temporary);
            }
        }
    }

    DEBUG_INDENT(-4);
}

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
                res = ".__thread_id_z";
            }
            else if (lev0 == std::get<1>(pd.second))
            {
                res = ".__thread_id_y";
            }
            else if (lev0 == std::get<2>(pd.second))
            {
                res = ".__thread_id_x";
            }
            else
            {
                ERROR("Level not mapped to GPU.", true);
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
                res = ".__block_id_z";
            }
            else if (lev0 == std::get<1>(pd.second))
            {
                res = ".__block_id_y";
            }
            else if (lev0 == std::get<2>(pd.second))
            {
                res = ".__block_id_x";
            }
            else
            {
                ERROR("Level not mapped to GPU.", true);
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

    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    bool found = false;
    for (const auto &pd : this->unroll_dimensions)
    {
        if ((std::get<0>(pd) == comp) && (std::get<1>(pd) == lev0))
        {
            found = true;
        }
    }

    std::string str = "Dimension " + std::to_string(lev0) +
                      (found ? " should" : " should not") +
                      " be unrolled.";
    DEBUG(10, tiramisu::str_dump(str));

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

int tiramisu::function::get_max_iteration_domains_dim() const
{
    int max_dim = 0;
    for (const auto &comp : this->get_computations())
    {
        isl_set *domain = comp->get_iteration_domain();
        int m = isl_set_dim(domain, isl_dim_set);
        max_dim = std::max(max_dim, m);
    }

    return max_dim;
}

int tiramisu::function::get_max_schedules_range_dim() const
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    int max_dim = 0;
    for (const auto &comp : this->get_computations())
    {
        if(comp->schedule_this_computation){

            isl_map *sched = comp->get_schedule();
            int m = isl_map_dim(sched, isl_dim_out);
            max_dim = std::max(max_dim, m);
        }
    }

    DEBUG_INDENT(-4);

    return max_dim;
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
        comp->name_unnamed_time_space_dimensions();
        
    }

    DEBUG_INDENT(-4);
    DEBUG(3, tiramisu::str_dump("End of function"));
}
void tiramisu::function::reset_computations()
{
    // remove the added computations
    if(get_computations().size()>this->original_number_of_computations){
        this->body = std::vector<tiramisu::computation *>(this->body.begin(),this->body.end()-(get_computations().size()-this->original_number_of_computations));
    }

    // we also need to reset the names for the computations that were duplicated since
    // they were renamed using the function rename_computations

    for (computation *comp : get_computations()){
        // for each computation we also remove all the added updates added using the add_definitions function
        comp->definitions_number -= comp->updates.size()-1;
        comp->updates = std::vector<tiramisu::computation *>(comp->updates.begin(),comp->updates.begin()+1);
        // we extract the correct name for the computation
        int pos = comp->name.find("_update_");
        std::string correct_name; 
        if( pos != std::string::npos){
            correct_name = comp->name.substr(1,pos-1);
            // call the rename function to correctly make all the changes to the name
            comp->rename_computation(correct_name);
        }

    }
}
void tiramisu::function::reset_schedules()
{
    for (computation *comp : get_computations())
        comp->set_identity_schedule_based_on_iteration_domain();
        
    remove_dimension_tags();
    clear_sched_graph();
    reset_computations();
}

void tiramisu::function::add_invariant(tiramisu::constant invar)
{
    invariants.push_back(invar);
}

void tiramisu::function::add_computation(computation *cpt)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(cpt != NULL);

    this->body.push_back(cpt);
    if (cpt->should_schedule_this_computation())
        this->starting_computations.insert(cpt);

    DEBUG_INDENT(-4);
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
            buf.second->dump(false);
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
        tiramisu::str_dump("\nDumping schedules of the function " + this->get_name() + " :\n");

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
            std::cout << std::get<0>(vec_dim) << "(" << std::get<1>(vec_dim) << ") ";
        }

        std::cout << std::endl << std::endl << std::endl;
    }
}

void tiramisu::function::set_arguments(const std::vector<tiramisu::buffer *> &buffer_vec)
{
    this->function_arguments = buffer_vec;
    // Implicit buffers are set to type a_temporary by default. Change them to
    // a_input or a_output, so that they don't get autoallocated.
    for (auto &buffer : buffer_vec)
    {
        assert((buffer != nullptr) && "Buffer argument is null!");
        if (buffer->get_argument_type() == a_temporary)
        {
            // Determine if it's an input function.
            // If there are any scheduled computation that uses this buffer,
            // buffer is marked as output.
            bool is_input = true;
            for (auto const &comp : this->body)
            {
                if (comp->get_buffer() == buffer
                    && comp->should_schedule_this_computation())
                {
                    is_input = false;
                    break;
                }
            }
            DEBUG(3, tiramisu::str_dump("Setting type of buffer "
                     + buffer->get_name() + " to "
                     + (is_input ? "input" : "output")));
            buffer->set_argument_type(is_input ? a_input : a_output);
        }
    }
}

void tiramisu::function::add_vector_dimension(std::string stmt_name, int vec_dim, int vector_length)
{
    assert(vec_dim >= 0);
    assert(!stmt_name.empty());

    this->vector_dimensions.push_back(std::make_tuple(stmt_name, vec_dim, vector_length));
}

void tiramisu::function::add_distributed_dimension(std::string stmt_name, int dim)
{
    assert(dim >= 0);
    assert(!stmt_name.empty());

    this->distributed_dimensions.push_back({stmt_name, dim});
}

void tiramisu::function::add_parallel_dimension(std::string stmt_name, int vec_dim)
{
    assert(vec_dim >= 0);
    assert(!stmt_name.empty());

    this->parallel_dimensions.push_back({stmt_name, vec_dim});
}

void tiramisu::function::add_unroll_dimension(std::string stmt_name, int level, int factor)
{
    assert(level >= 0);
    assert(!stmt_name.empty());
    assert(factor >= 0);

    this->unroll_dimensions.push_back(std::make_tuple(stmt_name, level, factor));
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

void tiramisu::function::remove_dimension_tags()
{
    parallel_dimensions.clear();
    vector_dimensions.clear();
    distributed_dimensions.clear();
    gpu_block_dimensions.clear();
    gpu_thread_dimensions.clear();
    unroll_dimensions.clear();
}

isl_union_set *tiramisu::function::get_trimmed_time_processor_domain() const
{
    DEBUG_FCT_NAME(10);
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

void tiramisu::function::lift_dist_comps() {
    for (std::vector<tiramisu::computation *>::iterator comp = body.begin(); comp != body.end(); comp++) {
        if ((*comp)->is_send() || (*comp)->is_recv() || (*comp)->is_wait() || (*comp)->is_send_recv()) {
            xfer_prop chan = static_cast<tiramisu::communicator *>(*comp)->get_xfer_props();
            if (chan.contains_attr(MPI)) {
                lift_mpi_comp(*comp);
            } else {
                ERROR("Can only lift MPI library calls", 0);
            }
        }
    }
}

void tiramisu::function::lift_mpi_comp(tiramisu::computation *comp) {
    if (comp->is_send()) {
        send *s = static_cast<send *>(comp);
        tiramisu::expr num_elements(s->get_num_elements());
        tiramisu::expr send_type(s->get_xfer_props().get_dtype());
        bool isnonblock = s->get_xfer_props().contains_attr(NONBLOCK);
        // Determine the appropriate number of function args and set ones that we can already know
        s->rhs_argument_idx = 3;
        s->library_call_args.resize(isnonblock ? 5 : 4);
        s->library_call_args[0] = tiramisu::expr(tiramisu::o_cast, p_int32, num_elements);
        s->library_call_args[1] = tiramisu::expr(tiramisu::o_cast, p_int32, s->get_dest());
        s->library_call_args[2] = tiramisu::expr(tiramisu::o_cast, p_int32, s->get_msg_tag());
        if (isnonblock) {
            // This additional RHS argument is to the request buffer. It is really more of a side effect.
            s->wait_argument_idx = 4;
        }
    } else if (comp->is_recv()) {
        recv *r = static_cast<recv *>(comp);
        send *s = r->get_matching_send();
        tiramisu::expr num_elements(r->get_num_elements());
        tiramisu::expr recv_type(s->get_xfer_props().get_dtype());
        bool isnonblock = r->get_xfer_props().contains_attr(NONBLOCK);
        // Determine the appropriate number of function args and set ones that we can already know
        r->lhs_argument_idx = 3;
        r->library_call_args.resize(isnonblock ? 5 : 4);
        r->library_call_args[0] = tiramisu::expr(tiramisu::o_cast, p_int32, num_elements);
        r->library_call_args[1] = tiramisu::expr(tiramisu::o_cast, p_int32, r->get_src());
        r->library_call_args[2] = tiramisu::expr(tiramisu::o_cast, p_int32, r->get_msg_tag().is_defined() ?
                                                                            r->get_msg_tag() : s->get_msg_tag());
        r->lhs_access_type = tiramisu::o_address_of;
        if (isnonblock) {
            // This RHS argument is to the request buffer. It is really more of a side effect.
          r->wait_argument_idx = 4;
        }
    } else if (comp->is_wait()) {
        wait *w = static_cast<wait *>(comp);
        // Determine the appropriate number of function args and set ones that we can already know
        w->rhs_argument_idx = 0;
        w->library_call_args.resize(1);
        w->library_call_name = "tiramisu_MPI_Wait";
    }
}

void function::gen_ordering_schedules()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if (this->use_low_level_scheduling_commands)
    {
        DEBUG(3, tiramisu::str_dump("Low level scheduling commands were used."));
        DEBUG(3, tiramisu::str_dump("Discarding high level scheduling commands."));
        return;
    }

    this->dump_sched_graph();

    if(this->is_sched_graph_tree())
    {
        DEBUG(3, tiramisu::str_dump("this->is_sched_graph_tree(): true."));

        std::priority_queue<int> level_to_check;
        std::unordered_map<int, std::deque<computation *>> level_queue;

        auto current_comp = *(this->starting_computations.begin());

        auto init_sched = automatically_allocated;
        init_sched.push_back(current_comp);

        for (auto it = init_sched.begin(); it != init_sched.end() && it + 1 != init_sched.end(); it++)
            (*(it+1))->after_low_level(**it, computation::root_dimension);

        bool comps_remain = true;
        while(comps_remain)
        {
            for (auto &edge: this->sched_graph[current_comp])
            {
                if (level_queue[edge.second].size() == 0)
                    level_to_check.push(edge.second);

                level_queue[edge.second].push_back(edge.first);
            }

            comps_remain = level_to_check.size() > 0;
            // If we haven't exhausted all computations
            if (comps_remain)
            {
                int fuse_level = level_to_check.top();
                auto next_comp = level_queue[fuse_level].front();
                level_queue[fuse_level].pop_front();

                // assert(this->get_max_iteration_domains_dim() > fuse_level);

                next_comp->after_low_level((*current_comp), fuse_level);

                current_comp = next_comp;
                if (level_queue[fuse_level].size() == 0)
                    level_to_check.pop();
            }
        }
    }
    else
    {
        DEBUG(3, tiramisu::str_dump("this->is_sched_graph_tree(): false."));
    }

    DEBUG_INDENT(-4);
}

void function::gen_time_space_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Generate the ordering based on calls to .after() and .before().
    this->gen_ordering_schedules();

    this->align_schedules();

    for (auto &comp : this->get_computations())
    {
        comp->gen_time_space_domain();
    }

    DEBUG_INDENT(-4);
}

// ADD:FLEXNLP
// TODO:FLEXNLP (Fix docs)
void tiramisu::function::gen_flexnlp_autocopy(){
  // If automatic copy is on (if we pass parameters to flexnlp_lstm_cell),
  // this code copies the weights, input and cell state at the right moment
  // which is inside the layers loop for the weights and cell state
  // and inside the sequence loop for the input and the output
  std::cout << "Generating flexnlp autocopy" << std::endl;

  // All the memory transfer computations will be mapped to this same buffer
  auto b_dummy_flexnlp = new buffer("dummy_memcpy_flexnlp", {1}, p_int32, a_temporary);

  for (auto comp : this->get_computations()){
    auto const &e = comp->get_expr(); // Get the computation's expression
    if (e.get_op_type() == tiramisu::o_call){ // If the computation is a function call
      auto call_arguments = e.get_arguments(); // Get the function call's arguments (o_call arguments)

      // Get the computation's predecessor and successor
      computation *pred = comp->get_predecessor();
      computation *succ = comp->get_successor();

      if (e.get_name() == "tiramisu_flexnlp_lstm_cell"){ // If this is an lstm call
          std::vector<tiramisu::var> v = comp->get_iteration_variables();

          int number_of_loops = v.size();
          if (number_of_loops == 0){ // Only one cell execution, just copy the weights and the input beforehand

          }
          else if (number_of_loops == 1){ // Consider the loop as a sequence loop

          }
          else{ // case >=2 (take two innermost loops and consider them as (LAYERS LOOP, SEQUENCE LOOP))
            // Prepare vectors of variables for weights copy and final cell state copy
            auto v_weights_and_cell_state = std::vector<tiramisu::var>(v.begin(), v.end() - 1); // All iteration variables but the last (sequence number)

            // declare the data movement computations
            tiramisu::computation* copy_weights = new tiramisu::computation(comp->get_name() + "_copy_weights", v_weights_and_cell_state,
                                                            expr(o_call, "tiramisu_flexnlp_copy_weights_to_device", {
                                                              call_arguments[0], // W_x
                                                              call_arguments[1], // W_h
                                                              call_arguments[2], // b_x
                                                              call_arguments[3], // b_h
                                                              expr(v[number_of_loops - 2]), // layer number (l)
                                                              call_arguments[8] // device number
                                                            }, p_int32));
            // copy the initial cell state
            tiramisu::computation* copy_cell_state = new tiramisu::computation(comp->get_name() + "_copy_cell_state", v_weights_and_cell_state,
                                                            expr(o_call, "tiramisu_flexnlp_copy_cell_state_to_device", {
                                                              call_arguments[7], // c_in
                                                              expr(v[number_of_loops - 2]), // layer number (l)
                                                              call_arguments[8] // device number
                                                            }, p_int32));

            // First copy input computation, for the first layer (l==0)
            tiramisu::computation* copy_input = new tiramisu::computation(comp->get_name() + "_copy_input", v,
                                                            expr(o_call, "tiramisu_flexnlp_copy_input_to_device", {
                                                              call_arguments[4], // x
                                                              call_arguments[5], // h_in
                                                              expr(v[number_of_loops - 1]), // sequence number (s)
                                                              expr(v[number_of_loops - 2]), // layer number (l)
                                                              call_arguments[8] // device number
                                                            }, p_int32));
            copy_input->add_predicate(expr(o_eq, expr(v[number_of_loops - 2]), 0)); // Predicate to add an if statement (the computation is executed only if l==0)

            // Second copy input computation, for the subsequent layers (l>0)
            tiramisu::computation* copy_input2 = new tiramisu::computation(comp->get_name() + "copy_input2", v,
                                                            expr(o_call, "tiramisu_flexnlp_copy_input_to_device", {
                                                              call_arguments[6], // previous layer's output
                                                              call_arguments[5], // h_in
                                                              expr(o_max, 0, v[number_of_loops - 1]), // sequence number (s)
                                                              expr(v[number_of_loops - 2]), // layer number (l)

                                                              call_arguments[8] // device number
                                                            }, p_int32));
            copy_input2->add_predicate(expr(o_ne, expr(v[number_of_loops - 2]), 0)); // Predicate to add an if statement (the computation is executed only if l!=0)

            // Create the copy output computation
            tiramisu::computation* copy_output = new tiramisu::computation(comp->get_name() + "_copy_output", v,
                                                            expr(o_call, "tiramisu_flexnlp_copy_output_to_host", {
                                                              call_arguments[6], // h_out
                                                              expr(v[number_of_loops - 1]), // sequence number (s)
                                                              call_arguments[8] // device number
                                                            }, p_int32));

            tiramisu::computation* copy_output_to_h_in = new tiramisu::computation(comp->get_name() + "_copy_output_to_h_in", v,
                                                            expr(o_call, "tiramisu_copy_vector", {
                                                              call_arguments[5], // h_in
                                                              call_arguments[6], // h_out
                                                              expr(v[number_of_loops - 2]), // layer number (l)*
                                                              expr(v[number_of_loops - 1]), // sequence number (s)
                                                            }, p_int32));

            tiramisu::computation* copy_final_cell_state = new tiramisu::computation(comp->get_name() + "copy_final_cell_state", v_weights_and_cell_state,
                                                            expr(o_call, "tiramisu_flexnlp_copy_cell_state_to_host", {
                                                              call_arguments[7], // c_in
                                                              expr(v[number_of_loops - 2]), // layer number (l)
                                                              call_arguments[8] // device number
                                                            }, p_int32));

            // Prepare the needed loop levels
            int after_level_comp;
            if (succ) // If comp is not the last computation, and has one after it (succ is not NULL).
              after_level_comp = this->sched_graph[comp][succ]; // Get loop level that is set for comp and succ by (succ->after(comp, after_level_comp))
            int before_level_comp = this->sched_graph[pred][comp];
            // Get loop numbers equivalento the two innermost loops
            std::vector<int> dimensions =
                    comp->get_loop_level_numbers_from_dimension_names({v[number_of_loops - 2].get_name(), v[number_of_loops - 1].get_name()});


            // Reschedule the computations
            copy_weights->between(*pred, before_level_comp, *comp, dimensions[0]);

            copy_cell_state->between(*copy_weights, dimensions[0], *comp, dimensions[0]);

            copy_input->between(*copy_cell_state, v[number_of_loops - 2], *comp, v[number_of_loops - 1]);

            copy_input2->between(*copy_input, v[number_of_loops - 1], *comp, v[number_of_loops - 1]);

            if (succ) // If comp has a successor
              copy_output->between(*comp, dimensions[1], *succ, after_level_comp);
            else // If comp is the last computation
              copy_output->after(*comp, dimensions[1]);

            if (succ) // If comp has a successor
              copy_output_to_h_in->between(*copy_output, dimensions[1], *succ, after_level_comp);
            else // If comp is the last computation
              copy_output_to_h_in->after(*copy_output, dimensions[1]);


            if (succ) // If comp has a successor
              copy_final_cell_state->between(*copy_output_to_h_in, dimensions[0], *succ, after_level_comp);
            else // If comp is the last computation
              copy_final_cell_state->after(*copy_output_to_h_in, dimensions[0]);

            // Storing computations in the dummy buffer
            copy_weights->store_in(b_dummy_flexnlp, {0});
            copy_cell_state->store_in(b_dummy_flexnlp, {0});
            copy_output_to_h_in->store_in(b_dummy_flexnlp, {0});

            copy_input->store_in(b_dummy_flexnlp, {0});
            copy_input2->store_in(b_dummy_flexnlp, {0});
            copy_output->store_in(b_dummy_flexnlp, {0});
            copy_final_cell_state->store_in(b_dummy_flexnlp, {0});

          }
        }else if (e.get_name() == "tiramisu_flexnlp_gru"){ // GRU case
          std::cout << "GRU not supported yet by Tiramisu-FlexNLP" << std::endl;
        }else{

        }
    }
  }
}

void tiramisu::function::gen_halide_bug_workaround_computations(){
    // PART I : Go through input buffers (arguments) and make a computation that adds each buffer's first element
    // PART II : Go through output buffers (arguments) and rewrite the first element at its place (dummy access to avoid Halide discarding the buffer)
    auto b_dummy_input_accesses = new buffer("b_dummy_input_accesses", {1}, p_float32, a_temporary);
    auto tmp_dummy_comp = new input("tmp_dummy_comp", {var("dummy_var", 0, 1)}, p_float32);
    tmp_dummy_comp->store_in(b_dummy_input_accesses, {0});

    auto accumulation_accesses = tiramisu::expr(0.f); // Used to accumulate accesses to each input (to create one dummy access computation)
    computation* first_cpt = this->get_first_cpt(); // get the first computation (to add the dummy computation before it)

    for (auto &buffer : this->get_arguments()){ // Go through the function argument buffers (those passed in the codegen function in the Tiramisu generator)
      int nb_dims = buffer->get_n_dims(); // Get the buffer's number of dimensions
      std::vector<tiramisu::expr> access_vector; // Will contain (0, 0.... 0) access (depending on the buffer's dimension)
      std::vector<tiramisu::var> variables; // A vector that will contains variables var(0, DIM_SIZE) for each of the dimensions
      std::vector<tiramisu::expr> store_vector;

      for (int i =0; i < nb_dims; i++){
        access_vector.push_back(tiramisu::expr(0)); // prepare an access vector to the first element of the buffer
        auto v = new var("i"+std::to_string(i), 0, buffer->get_dim_sizes()[i]); // Create a variable corresponding to the buffer sizes (0-> DIM_SIZE)
        variables.push_back(*v); // create the variables buffer (used for declaring the buffer's associated input)
        store_vector.push_back(expr(*v)); // Create the storing expression (used to map the input to the buffer in store_in)
      }

      // PART I
      if (buffer->get_argument_type() == a_input && // Is an input buffer
          (buffer->get_elements_type() == p_float32 || // Possible types
          buffer->get_elements_type() == p_float64 ||
          buffer->get_elements_type() == p_int8 ||
          buffer->get_elements_type() == p_int32 ||
          buffer->get_elements_type() == p_int64)){

        // Create an input computation associated to the buffer
        auto access_comp = new input("access_"+buffer->get_name(), variables, buffer->get_elements_type());
        // Associate the computation to the buffer
        access_comp->store_in(buffer, store_vector);
        // Accumulate accesses to the different inputs (to make a dummy usage to avoid them being discarded by Halide when unused)
        accumulation_accesses = accumulation_accesses + cast(p_float32, (*access_comp)(access_vector));
      }

      // PART II : for output buffers (copy the first element to itself)
      if (buffer->get_argument_type() == a_output&&( // Is an input buffer
          buffer->get_elements_type() == p_int8 ||
          buffer->get_elements_type() == p_float32)){ // Only float32 type is supported TODO:FLEXNLP Support other types by making a buffer for each output (make a dedicated b_dummy_input_accesses buffer and a dedicated tmp_dummy_comp input for each output buffer)
        // Create an input associated to the buffer
        auto access_comp = new input("access_"+buffer->get_name(), variables, buffer->get_elements_type());

        // Create a computation that copies the first element of the output buffer to the tmp buffer
        auto copy_comp = new computation("copy_comp_"+buffer->get_name(), {}, (*access_comp)(access_vector));

        // Create a computation that copies the tmp buffer content back to the output buffer (doesn't change the original value as we are rewriting the same value)
        auto copy_back_comp = new computation("copy_back_comp_"+buffer->get_name(), {}, (*tmp_dummy_comp)(0));

        // Store the computations
        access_comp->store_in(buffer, store_vector); // Access to the buffer
        copy_comp->store_in(tmp_dummy_comp->get_buffer(), {0}); // tmp contains one element
        copy_back_comp->store_in(buffer, access_vector); // copy back copies back to the (0,0...) location (first element)

        // Schedule the computations (copy first to tmp buffer => Then copy back to the output buffer => then the actual first computation)
        (*copy_comp).then(*copy_back_comp, computation::root)
                 .then(*first_cpt, computation::root);
        first_cpt = copy_comp; // set current first computation to copy_comp which is the new first computation
      }
    }

    // Create the dummy input accesses computation and schedule it, then store it in a dummy buffer
    auto comp = new tiramisu::computation("dummy_input_accesses", {}, accumulation_accesses);
    comp->then(*first_cpt, computation::root);
    comp->store_in(b_dummy_input_accesses,{0});
    first_cpt = comp;
}

void tiramisu::function::codegen(const std::vector<tiramisu::buffer *> &arguments, const std::string obj_filename, const bool gen_cuda_stmt)
{
    if (gen_cuda_stmt)
    {
        if(!this->mapping.empty())
        {
            tiramisu::computation* c1 = this->get_first_cpt();
            tiramisu::computation* c2 = this->get_last_cpt();
            Automatic_communication(c1,c2);
        }
        else
            DEBUG(3, tiramisu::str_dump("You must specify the corresponding CPU buffer to each GPU buffer else you should do the communication manually"));
    }
    this->set_arguments(arguments);
    this->lift_dist_comps();
    this->gen_time_space_domain();
    this->gen_isl_ast();
    if (gen_cuda_stmt) {
        this->gen_cuda_stmt();
    }
    this->gen_halide_stmt();
    this->gen_halide_obj(obj_filename);
}

/*
  General codegen function for more than 2 possible architectures,
  functions specific to architectures are called conditionally on
  the gen_architecture_flag parameter's value.
*/
// TODO:FLEXNLP Fix the bug and delete workaround code ()
#define USE_HALIDE_BUFFERS_BUG_WORKAROUND true
void tiramisu::function::codegen(const std::vector<tiramisu::buffer *> &arguments, const std::string obj_filename, const tiramisu::hardware_architecture_t gen_architecture_flag)
{
    this->set_arguments(arguments);
    if (gen_architecture_flag == tiramisu::hardware_architecture_t::arch_nvidia_gpu ||
        gen_architecture_flag == tiramisu::hardware_architecture_t::arch_flexnlp)
    {
        if(!this->mapping.empty())
        {
            tiramisu::computation* c1 = this->get_first_cpt();
            tiramisu::computation* c2 = this->get_last_cpt();
            Automatic_communication(c1,c2);
        }
        else
            DEBUG(3, tiramisu::str_dump("You must specify the corresponding CPU buffer to each GPU buffer else you should do the communication manually"));
    }

    // TODO:OMIT
    if (false && gen_architecture_flag == tiramisu::hardware_architecture_t::arch_flexnlp)
        this->gen_flexnlp_autocopy();

    if (USE_HALIDE_BUFFERS_BUG_WORKAROUND)
        this->gen_halide_bug_workaround_computations();

    this->lift_dist_comps();
    this->gen_time_space_domain();
    this->gen_isl_ast();
    if (gen_architecture_flag == tiramisu::hardware_architecture_t::arch_nvidia_gpu){
        this->gen_cuda_stmt();
    }
    this->gen_halide_stmt();
    this->gen_halide_obj(obj_filename, gen_architecture_flag);
}

const std::vector<std::string> tiramisu::function::get_invariant_names() const
{
    const std::vector<tiramisu::constant> inv = this->get_invariants();
    std::vector<std::string> inv_str;

    for (int i = 0; i < inv.size(); i++)
        inv_str.push_back(inv[i].get_name());

    return inv_str;
}

void tiramisu::function::performe_full_dependency_analysis()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);
    // align schedules and order schedules
    this->align_schedules();
    this->gen_ordering_schedules();
    // could save default schedules and order here
    this->calculate_dep_flow();
    
    DEBUG_INDENT(-4);

}

bool tiramisu::function::check_legality_for_function()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);
    assert(this->dep_read_after_write!=NULL);

    isl_union_map * all_deps = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->dep_read_after_write));

    all_deps = isl_union_map_union(all_deps,
        isl_union_map_range_factor_domain(isl_union_map_copy(this->dep_write_after_read)));

    all_deps = isl_union_map_union(all_deps, 
        isl_union_map_range_factor_domain(isl_union_map_copy(this->dep_write_after_write)));

    isl_union_map * universe_of_all_deps = isl_union_map_universe(all_deps);

    std::vector<isl_map *> all_basic_maps;
    
    auto f = [](isl_map * bmap,void * user) { 

        std::vector<isl_map *>& myName = *reinterpret_cast<std::vector<isl_map*>*>(user);
     
        myName.push_back(bmap);
        return isl_stat_ok;
    };
    
    isl_stat (*fun_ptr)(isl_map * p,void * m) = (f);

    isl_union_map_foreach_map(universe_of_all_deps,fun_ptr,(void * ) &all_basic_maps);

    isl_set * left_hs = NULL;
    isl_set * right_hs = NULL; // hand side

    computation * left_comp = NULL;
    computation * right_comp = NULL;

    std::string left_computation_name =  "";
    std::string right_computation_name = "";

    bool over_all_legality = true;
    
    for(auto& space_dep:all_basic_maps)
    {
        DEBUG(3, tiramisu::str_dump(" the map of deps is  "+std::string(isl_map_to_str(space_dep))));

        left_hs = isl_map_domain(isl_map_copy(space_dep));
        right_hs = isl_map_range(isl_map_copy(space_dep));

        left_computation_name =  isl_space_get_tuple_name(
            isl_set_get_space(left_hs),isl_dim_set);

        right_computation_name =  isl_space_get_tuple_name(
            isl_set_get_space(right_hs),isl_dim_set);

        DEBUG(3, tiramisu::str_dump(" checking legality of dependences "+left_computation_name+" -> "+right_computation_name));
        
        left_comp = this->get_computation_by_name(left_computation_name)[0];
        right_comp = this->get_computation_by_name(right_computation_name)[0];

        if( left_comp->involved_subset_of_dependencies_is_legal(right_comp) == false)
        {
            over_all_legality = false;
            break;
        }
    }

    DEBUG_INDENT(-4);

    isl_union_map_free(universe_of_all_deps);

    return over_all_legality;
}

bool tiramisu::function::check_partial_legality_in_function(std::vector<tiramisu::computation* > involved_computations)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(this->dep_read_after_write!=NULL);
    

    isl_union_map * all_deps = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->dep_read_after_write));

    all_deps = isl_union_map_union(all_deps,
        isl_union_map_range_factor_domain(isl_union_map_copy(this->dep_write_after_read)));

    all_deps = isl_union_map_union(all_deps, 
        isl_union_map_range_factor_domain(isl_union_map_copy(this->dep_write_after_write)));

    isl_union_map * universe_of_all_deps = isl_union_map_universe(all_deps);

    std::string empty ="{}";

    isl_union_set * domains_of_involved_computations = isl_union_set_read_from_str(this->get_isl_ctx(),empty.c_str());

    for(auto const& computation : involved_computations)
    {
        isl_set * computation_domain = isl_map_domain(isl_map_copy(computation->get_schedule()));

        domains_of_involved_computations = isl_union_set_union(
            domains_of_involved_computations,
            isl_union_set_from_set(computation_domain)
        );
    }

    domains_of_involved_computations = isl_union_set_universe(domains_of_involved_computations);

    DEBUG(3, tiramisu::str_dump(" The Set of all involved computations  "+std::string(isl_union_set_to_str(domains_of_involved_computations))));

    universe_of_all_deps = isl_union_map_intersect_domain(universe_of_all_deps,isl_union_set_copy(domains_of_involved_computations));

    universe_of_all_deps = isl_union_map_intersect_range(universe_of_all_deps,domains_of_involved_computations);

    DEBUG(3, tiramisu::str_dump(" The dependencies subject to legality checks are : "+std::string(isl_union_map_to_str(universe_of_all_deps))));

    std::vector<isl_map *> all_basic_maps;
    
    auto f = [](isl_map * bmap,void * user) { 

        std::vector<isl_map *>& myName = *reinterpret_cast<std::vector<isl_map*>*>(user);
     
        myName.push_back(bmap);
        return isl_stat_ok;
    };
    
    isl_stat (*fun_ptr)(isl_map * p,void * m) = (f);

    isl_union_map_foreach_map(universe_of_all_deps,fun_ptr,(void * ) &all_basic_maps);

    isl_set * left_hs = NULL;
    isl_set * right_hs = NULL; // hand side

    computation * left_comp = NULL;
    computation * right_comp = NULL;

    std::string left_computation_name =  "";
    std::string right_computation_name = "";

    bool over_all_legality = true;
    
    for(auto& space_dep:all_basic_maps)
    {
        DEBUG(3, tiramisu::str_dump(" the map of deps is  "+std::string(isl_map_to_str(space_dep))));

        left_hs = isl_map_domain(isl_map_copy(space_dep));
        right_hs = isl_map_range(isl_map_copy(space_dep));

        left_computation_name =  isl_space_get_tuple_name(
            isl_set_get_space(left_hs),isl_dim_set);

        right_computation_name =  isl_space_get_tuple_name(
            isl_set_get_space(right_hs),isl_dim_set);

        DEBUG(3, tiramisu::str_dump(" checking legality of dependences "+left_computation_name+" -> "+right_computation_name));
        
        left_comp = this->get_computation_by_name(left_computation_name)[0];
        right_comp = this->get_computation_by_name(right_computation_name)[0];

        if( left_comp->involved_subset_of_dependencies_is_legal(right_comp) == false )
        {
            over_all_legality = false;
            break;
        }
    }

    isl_union_map_free(universe_of_all_deps);

    DEBUG_INDENT(-4);

    return over_all_legality;

}


void tiramisu::function::prepare_schedules_for_legality_checks(bool reset_static_dimesion)
{
    this->align_schedules();

    if(reset_static_dimesion == true)
    {
        this->reset_all_static_dims_to_zero();
    }

    this->gen_ordering_schedules();
}
bool tiramisu::function::loop_interchnage_is_legal(int i, int j, std::vector<tiramisu::computation *> fuzed_computations)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);
    
    DEBUG(3, tiramisu::str_dump("interchange check for level : "+std::to_string(i)+" and level "+std::to_string(j)));

    bool result = true;

    for(auto& computation : fuzed_computations)
    {
        // get upper and lower bound for first node from set first_node_upper and first_node_lower
        // get upper and lower bound for second node from set second_node_upper and second_node_lower
        isl_set *iter_domain = computation->get_iteration_domain();
        int nb_iterators = isl_set_dim(iter_domain, isl_dim_set);
        
        assert(i < nb_iterators && j < nb_iterators);
        // Get the iterator name of the first interchange parameter loop.
        std::string first_level_name = isl_set_get_dim_name(iter_domain, isl_dim_set, i);

        // Get both bounds for the second interchange parameter loop.
        std::string second_level_low_bound = utility::get_bound(iter_domain, j, false).to_str();
        std::string second_level_up_bound = utility::get_bound(iter_domain, j, true).to_str();
        // In all the nodes that are inbetween the two levels to interchange check:
        for (int level = i; level < j; ++level)
        {
            // Get the information of the current node.
            std::string name = isl_set_get_dim_name(iter_domain, isl_dim_set, level);
            std::string low_bound = utility::get_bound(iter_domain, level, false).to_str();
            std::string up_bound = utility::get_bound(iter_domain, level, true).to_str();

            //  If the second node bounds depend on the current node iterator
            if(second_level_up_bound.find(name) != std::string::npos || second_level_low_bound.find(name) != std::string::npos){
                return false;
            }
            //  If the current node bounds depend on the first node iterator
            if(low_bound.find(first_level_name) != std::string::npos || up_bound.find(first_level_name) != std::string::npos){
                return false;
            }
        }
    }
    DEBUG_INDENT(-4);
    return result;
}
bool tiramisu::function::loop_unrolling_is_legal(tiramisu::var i , std::vector<tiramisu::computation *> fuzed_computations)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(i.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->dep_read_after_write != NULL );
    assert(this->dep_write_after_write != NULL );
    assert(this->dep_write_after_read != NULL );
    assert(fuzed_computations.size()>0);

    computation * first_computation = fuzed_computations[0];
    
    DEBUG(3, tiramisu::str_dump(" unrolling check for var : "+i.get_name()));

    std::vector<std::string> original_loop_level_names = first_computation->get_loop_level_names();
    std::vector<int> dimensions =
        first_computation->get_loop_level_numbers_from_dimension_names({i.get_name()});

    first_computation->check_dimensions_validity(dimensions);
 
    bool result = true;

    for(auto& computation:fuzed_computations)
    {
        if( computation->unrolling_is_legal(i)== false)
        {
            result = false;
            break;
        }
    }
    DEBUG_INDENT(-4);

    return result;
}

bool tiramisu::function::loop_parallelization_is_legal(tiramisu::var par_dim_var, std::vector<tiramisu::computation *> fuzed_computations )
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(par_dim_var.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->dep_read_after_write != NULL );
    assert(this->dep_write_after_write != NULL );
    assert(this->dep_write_after_read != NULL );
    assert(fuzed_computations.size()>0);

    computation * first_computation = fuzed_computations[0];
    
    DEBUG(3, tiramisu::str_dump(" var parallelization check is : "+par_dim_var.get_name()));

    std::vector<std::string> original_loop_level_names = first_computation->get_loop_level_names();

    std::vector<int> dimensions =
        first_computation->get_loop_level_numbers_from_dimension_names({par_dim_var.get_name()});

    first_computation->check_dimensions_validity(dimensions);

    bool result = this->loop_parallelization_is_legal(dimensions[0],fuzed_computations);

    DEBUG_INDENT(-4);

    return result;
}


bool tiramisu::function::loop_parallelization_is_legal(int dim_parallel , std::vector<tiramisu::computation *> fuzed_computations)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);
    assert(!this->get_name().empty());
    assert(this->dep_read_after_write != NULL );
    assert(this->dep_write_after_write != NULL );
    assert(this->dep_write_after_read != NULL );
    assert(fuzed_computations.size()>0);

    computation * first_computation = fuzed_computations[0];
    
    std::vector<std::string> original_loop_level_names = first_computation->get_loop_level_names();

    int par_dim = tiramisu::loop_level_into_dynamic_dimension(dim_parallel);

    DEBUG(3, tiramisu::str_dump(" par dim number is : "+std::to_string(par_dim)));

    // Extracting deps

     isl_union_map * read_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->dep_read_after_write));

    isl_union_map * write_after_read_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->dep_write_after_read));

    isl_union_map * write_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->dep_write_after_write));

    isl_union_map * all_deps = isl_union_map_union(
        read_after_write_dep,
        write_after_read_dep
        );

    // all the deps in 1 union map
    all_deps = isl_union_map_union(all_deps,write_after_write_dep);

    DEBUG(3, tiramisu::str_dump(" all the dependencies involved are : "+std::string(isl_union_map_to_str(all_deps))));

    // all current schedules in 1 union map
    std::string empty_union = "{}";
    std::string empty_time  = "";

    isl_union_map * schedules = isl_union_map_read_from_str(this->get_isl_ctx(),empty_union.c_str());

    isl_map * schedule_itr = NULL;

    for( auto& computation: fuzed_computations)
    {
        schedule_itr = isl_map_copy(computation->get_schedule());

        schedule_itr = isl_map_set_tuple_name(schedule_itr,isl_dim_out,empty_time.c_str());

        schedules = isl_union_map_union(schedules,isl_union_map_from_map(schedule_itr));

    }

    DEBUG(3, tiramisu::str_dump(" all the used schedules are  : "+std::string(isl_union_map_to_str(schedules))));

    // application to discard unused dep & represent them in their time space

    all_deps = isl_union_map_apply_range(all_deps,isl_union_map_copy(schedules));

    all_deps = isl_union_map_apply_domain(all_deps,isl_union_map_copy(schedules));

    DEBUG(3, tiramisu::str_dump(" all the used dependencies union map are  : "+std::string(isl_union_map_to_str(all_deps))));

    if(isl_union_map_is_empty(all_deps))
    {
        DEBUG(3, tiramisu::str_dump(" No dependencies , parallelism is legal by default "));
        DEBUG_INDENT(-4);
        return true;

    }

    isl_map * equation_map = isl_map_from_union_map(all_deps);

    DEBUG(3, tiramisu::str_dump(" all the used dependencies after transformed to map are  : "+std::string(isl_map_to_str(equation_map))));

    bool overall_legality = false;

    /*
        isl_equate adds restriction that both domain and range positions are equal
        we suppose that legality of the lexicographical order is checked elsewhere, so we only need to check for loop caried dependencies.
        if adding equation of == between input set & output set of map for a dimension strictly before the parallel one is empty means: 
            dep is not a carried one for the parallel loop lvl, so parallelism is legal.

        else 
            if all previous equations added does not make the map empty then the last possibility is:
                dep is within the same loop iteration, parallel is true ( true if equate doesn't make the map empty)
                else it's false
                
    */
    for(int i=0;i<par_dim;i++)
    {
        equation_map = isl_map_equate(equation_map,isl_dim_in,i,isl_dim_out,i);

        DEBUG(10, tiramisu::str_dump(" --> remaining deps at itr "+std::to_string(par_dim)+" : "+std::string(isl_map_to_str(equation_map))));

        if(isl_map_is_empty(equation_map))
        {
            overall_legality = true;
            DEBUG(10, tiramisu::str_dump(" parallelization is legal "));
            break;
        }
    
    }


    if(!overall_legality)
    {
        isl_map * equation_map_final = isl_map_equate(isl_map_copy(equation_map),isl_dim_in,par_dim,isl_dim_out,par_dim);

        DEBUG(10, tiramisu::str_dump(" --> remaining deps at itr "+std::to_string(par_dim)+" : "+std::string(isl_map_to_str(equation_map_final))));

        if(isl_map_is_equal(equation_map,equation_map_final) == isl_bool_false)
        {
            overall_legality = false;
            DEBUG(3, tiramisu::str_dump(" parallelization is illegal "));
        }
        else{
            overall_legality = true;
            DEBUG(3, tiramisu::str_dump(" parallelization is legal "));
        }
        isl_map_free(equation_map_final);
    }
    
    isl_map_free(equation_map);
    isl_union_map_free(schedules);


    DEBUG_INDENT(-4); 
    return overall_legality;


}

bool tiramisu::function::loop_vectorization_is_legal(tiramisu::var i , std::vector<tiramisu::computation *> fuzed_computations)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(i.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->dep_read_after_write != NULL );
    assert(this->dep_write_after_write != NULL );
    assert(this->dep_write_after_read != NULL );
    assert(fuzed_computations.size()>0);

    DEBUG(3, tiramisu::str_dump(" vectorization check for var : "+i.get_name()));

    bool result = this->loop_unrolling_is_legal(i,fuzed_computations) 
                && this->loop_parallelization_is_legal(i,fuzed_computations);

    DEBUG(3, tiramisu::str_dump(" vectorization legality is : "+result));

    DEBUG_INDENT(-4);

    return result;
}

void tiramisu::function::reset_all_static_dims_to_zero()
{   
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    for(auto computation:this->get_computations())
    {
        isl_map * schedule = isl_map_copy(computation->get_schedule());
        isl_space * range_space = isl_set_get_space(isl_map_range(isl_map_copy(schedule)));

        isl_map * transformation_map = isl_map_universe(isl_space_map_from_set(range_space));

        int m1  = isl_map_dim(schedule, isl_dim_out);

        // pos 0 is static always 0 by default, the we will have static then dynamic for all the rest.
        transformation_map = isl_map_fix_si(transformation_map,isl_dim_out,0,0);

        for(int i=1; i<m1; i++)
        {
            if(i%2 == 1)
            {// case of static dimension, fix position to 0.
                transformation_map = isl_map_fix_si(transformation_map,isl_dim_out,i,0);
            }
            else
            {// equate input and output in case of dynamic dimensions
                transformation_map = isl_map_equate(transformation_map,isl_dim_out,i,isl_dim_in,i);
            }
        }
        DEBUG(3, tiramisu::str_dump(" Initial schedule before initialization of beta dimensions : "+std::string(isl_map_to_str(schedule))));
        DEBUG(3, tiramisu::str_dump(" Transformation Map : "+std::string(isl_map_to_str(transformation_map))));

        schedule = isl_map_apply_range(schedule,transformation_map);

        DEBUG(3, tiramisu::str_dump(" Initialized Schedule : "+std::string(isl_map_to_str(schedule))));

        computation->set_schedule(schedule);

    }

    DEBUG_INDENT(-4);
}

std::vector<std::tuple<tiramisu::var,int>> function::correcting_loop_fusion_with_shifting(std::vector<tiramisu::computation*> previous_computations, 
                                                                                tiramisu::computation current_computation,
                                                                                std::vector<tiramisu::var> vars_subjected_to_shifting)
{
    DEBUG_FCT_NAME(3);

    DEBUG_INDENT(4);

    assert(this->dep_read_after_write != NULL );
    assert(this->dep_write_after_write != NULL );
    assert(this->dep_write_after_read != NULL );
    assert(!current_computation.get_name().empty());
    assert(previous_computations.size() > 0);
    assert(!previous_computations[0]->get_name().empty());
    assert(!vars_subjected_to_shifting.empty());

    std::vector<std::string> loops_names;

    DEBUG(3, tiramisu::str_dump(" Loops included in shifting correcting for target computation are : "));

    for(auto variable : vars_subjected_to_shifting)
    {
        assert(variable.get_name().length() > 0);
        loops_names.push_back(variable.get_name());
        DEBUG(3, tiramisu::str_dump(variable.get_name()));
        
    }
    //mapping dynamic loop number into the var itself
    std::unordered_map<int,tiramisu::var> dynamic_var_mapping;
    
    std::vector<std::string> original_loop_level_names = current_computation.get_loop_level_names();

    std::vector<int> dimensions =
        current_computation.get_loop_level_numbers_from_dimension_names(loops_names);

    int schedule_dim_number = 0;

    int max_dynamic_number = 0;

    std::vector<int> all_schedule_dim_numbers;

    for(int i=0;i<dimensions.size();i++)
    {
        schedule_dim_number = tiramisu::loop_level_into_dynamic_dimension(dimensions[i]);
        dynamic_var_mapping[schedule_dim_number] = vars_subjected_to_shifting[i];
        all_schedule_dim_numbers.push_back(schedule_dim_number);
        DEBUG(3, tiramisu::str_dump(" -> "+vars_subjected_to_shifting[i].get_name()+" lvl number in schedule is : "+std::to_string(schedule_dim_number)));

        if(schedule_dim_number > max_dynamic_number)
        {
            max_dynamic_number = schedule_dim_number;
        }
    }

    std::sort(all_schedule_dim_numbers.begin(), all_schedule_dim_numbers.end()); 

    isl_map * current_schedule =  isl_map_copy(current_computation.get_schedule());

    // Extract schedules from vector

    std::string empty_union = "{}";

    isl_union_map * previous_schedules = isl_union_map_read_from_str(this->get_isl_ctx(),empty_union.c_str());

    isl_map * schedule_itr = NULL;

    int m1 = isl_map_dim(current_schedule, isl_dim_out);

    int m2 = 0;

    std::string empty_time="";

    for( auto computation: previous_computations)
    {
        schedule_itr = isl_map_copy(computation->get_schedule());

        m2 = isl_map_dim(schedule_itr, isl_dim_out);

        DEBUG(3, tiramisu::str_dump(" schedule as origin : "+std::string(isl_map_to_str(schedule_itr))));

        assert(m1 == m2);

        previous_schedules = isl_union_map_union(previous_schedules,isl_union_map_from_map(isl_map_copy(schedule_itr)));

    }
    DEBUG(3, tiramisu::str_dump(" the current computation schedule (subject to shifting) : "+std::string(isl_map_to_str(current_schedule))));
    DEBUG(3, tiramisu::str_dump(" the previous computations schedule : "+std::string(isl_union_map_to_str(previous_schedules))));

    isl_union_map * all_schedules =  isl_union_map_copy(previous_schedules);

    all_schedules = isl_union_map_union(
        all_schedules,
        isl_union_map_from_map(isl_map_copy(current_schedule)));

    DEBUG(3, tiramisu::str_dump(" union of schedules : "+std::string(isl_union_map_to_str(all_schedules))));

    isl_union_map * all_deps = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->dep_read_after_write));

    all_deps = isl_union_map_union(
        all_deps,
        isl_union_map_range_factor_domain(isl_union_map_copy(this->dep_write_after_read))
    );

    all_deps = isl_union_map_union(
        all_deps,
        isl_union_map_range_factor_domain(isl_union_map_copy(this->dep_write_after_write))
    );

    all_deps = isl_union_map_apply_range(
        all_deps,
        isl_union_map_copy(all_schedules));

    all_deps = isl_union_map_apply_domain(
        all_deps,
        isl_union_map_copy(all_schedules));


    std::string const_str = "[";
    std::string domain_str = "{"+current_computation.get_name()+"[";

    for(int i=0;i<m1;i++) // isl_map current_schedule output size
    {
        const_str += "cx" + std::to_string(i);
        domain_str += "cx" + std::to_string(i);

        if(i != (m1-1))
        {
            const_str += ",";
            domain_str += ",";
        }

    }
    const_str+="]";
    domain_str+="]}";

    /**
     * Create identity map foreach of the previous computations,
     * The map should map Computation to timestamp
    */
    std::unordered_map<std::string,isl_map*> name_unificator_map;

    for(auto computation: previous_computations)
    {
        std::string unify = "{"+computation->get_name()+const_str+"->"+const_str+"}";
        isl_map * unificator_map = isl_map_read_from_str(this->get_isl_ctx(),unify.c_str());
        name_unificator_map[computation->get_name()] = unificator_map;
    }

    std::string complete_current_cst = const_str+"->"+domain_str;

    isl_union_set * current_fixed_cst = isl_union_set_read_from_str(this->get_isl_ctx(),complete_current_cst.c_str());

    DEBUG(3, tiramisu::str_dump(" fixed current set : "+std::string(isl_union_set_to_str(current_fixed_cst))));

    isl_union_map * dep_constants1 = isl_union_map_intersect_domain(
        isl_union_map_copy(all_deps),
        isl_union_set_copy(current_fixed_cst)
        );
    isl_union_map * dep_constants2 = isl_union_map_intersect_range(
        isl_union_map_copy(all_deps),
        isl_union_set_copy(current_fixed_cst)
        );

    isl_union_set * origin_set_space = isl_union_set_read_from_str(this->get_isl_ctx(),domain_str.c_str());

    dep_constants1 = isl_union_map_subtract_range(dep_constants1,isl_union_set_copy(origin_set_space));
    dep_constants2 = isl_union_map_subtract_domain(dep_constants2,isl_union_set_copy(origin_set_space));

    // this map contain all dep which the origin computation is in either domain, range or both
    dep_constants1 = isl_union_map_union(dep_constants1,dep_constants2);

    DEBUG(3, tiramisu::str_dump(" Involved dep graph used target->origin or origin->target : "+std::string(isl_union_map_to_str(dep_constants1))));

    /**
     * iterate over all the space of dependencies and extract all basic maps inside all_basic_maps
     * 
    */
    isl_union_map * universe_of_all_deps = isl_union_map_universe(isl_union_map_copy(dep_constants1));

    // vector of maps
    std::vector<isl_map *> maps_cases;

    // vector of basic maps
    std::vector<isl_basic_map *> all_basic_maps;// contains basic maps 
    
    // iterator to extract maps from union_map
    auto f_maps = [](isl_map * bmap,void * user) { 

        std::vector<isl_map *>& myName = *reinterpret_cast<std::vector<isl_map*>*>(user);
     
        myName.push_back(bmap);
        return isl_stat_ok;
    };
    
    isl_stat (*fun_ptr_map)(isl_map * p,void * m) = (f_maps);

    // iterate and extract maps from basic maps
    isl_union_map_foreach_map(universe_of_all_deps,fun_ptr_map,(void * ) &maps_cases);


    // iterator to extract basic maps from map
    auto f = [](isl_basic_map * bmap,void * user) { 

        std::vector<isl_basic_map *>& myName = *reinterpret_cast<std::vector<isl_basic_map*>*>(user);
     
        myName.push_back(bmap);
        return isl_stat_ok;
    };
    // iterator as a function 
    isl_stat (*fun_ptr)(isl_basic_map * p,void * m) = (f);



    DEBUG(5, tiramisu::str_dump(" Extracting basic maps starting map per map : "));

    isl_map * my_map1 =NULL;

    for(auto& space_dep:maps_cases)
    {
        my_map1 = isl_union_map_extract_map(dep_constants1,isl_map_get_space(space_dep));

        DEBUG(5, tiramisu::str_dump(" basic Maps extracted : "+std::string(isl_map_to_str(my_map1))));

        // extract basic maps into all_basic_maps
        isl_map_foreach_basic_map(my_map1,fun_ptr,(void * ) &all_basic_maps);

    }

    /**
     * All deps extracted in all_basic_maps Vector.
     * 
     * In order for all previous dependencies to be correct, we need to always have domain < range.
     * The idea is to define dims n1,n2...  in a map {[n1,n2]->domain_current[cst1+n1,cst2+n2]} 
     * We will the solve the problem of minimal n1,n2 values that helps solve all the "domain < range" dependencies.
    */

    /**
     * In all this code, origin refers to previous computations that serves as references and constraints for shifting.
     * Target refers to current computation that would be shifted
    */

    // preparing the solution map ...

    // preparing solution set {[n1,n2]}
    // addition map {[n1,n2]->Target[0,n1,0,n2]} : involved dimensions of variables have ni, 0 otherwise.

    std::string condition_ni_pos = "";
    std::string set_ni = "[";
    std::string zero_set = "[";

    for(int i=0;i<all_schedule_dim_numbers.size();i++)
    {
        set_ni +="n"+std::to_string(i);
        zero_set+="0";

        condition_ni_pos +=" 0<=n"+std::to_string(i); 

        if(i != (all_schedule_dim_numbers.size()-1))
        {
            set_ni +=",";
            condition_ni_pos+=" and ";
            zero_set+=",";
        }
    }
    set_ni +="]";
    zero_set+="]"; 

    int index_of_involved_vars = 0;

    if(index_of_involved_vars == all_schedule_dim_numbers.size())
    {
                index_of_involved_vars = -1;
    }

    std::string addition_set = "[";

    for(int i=0; i<m1; i++)//dim out size
    {
        if((index_of_involved_vars!= - 1) && (all_schedule_dim_numbers[index_of_involved_vars] == i ))
        {
            addition_set +="n"+std::to_string(index_of_involved_vars);
            index_of_involved_vars++;

            if(index_of_involved_vars == all_schedule_dim_numbers.size())
            {
                index_of_involved_vars = -1;
            }
        }
        else
        {
            addition_set+="0";
        }

        if( i != (m1-1))
        {
            addition_set+=",";
        }
    }
    addition_set+="]";

    std::string addition_map_str = const_str+"->{"+set_ni+"->"+addition_set+"}";
    std::string solution_set_str = +"{"+set_ni+":"+condition_ni_pos+"}";

    std::string zero_set_str = "{"+zero_set+"}";

    isl_set * zero_set_isl = isl_set_read_from_str(this->get_isl_ctx(),zero_set_str.c_str());

    isl_map * addition_map = isl_map_read_from_str(this->get_isl_ctx(),addition_map_str.c_str());

    if (my_map1 != NULL)
        addition_map = isl_map_gist_params(addition_map,isl_map_params(my_map1));

    isl_set * solution_set = isl_set_read_from_str(this->get_isl_ctx(),solution_set_str.c_str());

    isl_set * real_solution_set = isl_set_read_from_str(this->get_isl_ctx(),solution_set_str.c_str());

    DEBUG(5, tiramisu::str_dump(" Solution set is : "+std::string(isl_set_to_str(solution_set))));

    DEBUG(5, tiramisu::str_dump(" Addition map is : "+std::string(isl_map_to_str(addition_map))));
    // would be used in sum op to create dependent map

    DEBUG(5, tiramisu::str_dump(" Zeros set is : "+std::string(isl_set_to_str(zero_set_isl))));

    std::string empty_name = "";

    std::string in_name ="";
    std::string out_name="";
    
    std::string unify_target = "{"+current_computation.get_name()+const_str+"->"+const_str+"}";
   
    isl_map * unify_map_target = isl_map_read_from_str(this->get_isl_ctx(),unify_target.c_str());

    isl_map * unify_map_origin = NULL;

    bool aborted_fusion = false;

    for(auto dependency:all_basic_maps)
    {
        in_name = std::string(isl_basic_map_get_tuple_name(dependency,isl_dim_in));
        out_name = std::string(isl_basic_map_get_tuple_name(dependency,isl_dim_out));

        isl_map * dependency_map = isl_map_from_basic_map(isl_basic_map_copy(dependency));

        DEBUG(5, tiramisu::str_dump(" -----> dependendency : "+std::string(isl_map_to_str(dependency_map))));

        if(in_name != current_computation.get_name())
        { // domain is origin,  and range is target(current).

            unify_map_origin = name_unificator_map[in_name];
            
            dependency_map = isl_map_apply_range(dependency_map,isl_map_copy(unify_map_target));
            dependency_map = isl_map_apply_domain(dependency_map,isl_map_copy(unify_map_origin));
            //dep is now in timestamps

            DEBUG(5, tiramisu::str_dump(" -----> dependendency in timestamp : Reference -> target : "+std::string(isl_map_to_str(dependency_map))));

            //check if delta is singleton
            isl_set * deltas = isl_map_deltas(isl_map_copy(dependency_map));
            deltas = isl_set_project_out(deltas,isl_dim_param,0,m1);
            deltas = isl_set_project_out(deltas,isl_dim_set,max_dynamic_number+2,(m1-2)-max_dynamic_number);
            if(!isl_set_is_singleton(deltas))
            {
                DEBUG(5, tiramisu::str_dump(" -### dependendency contains constants !! fusion aborted "));
                aborted_fusion = true;
                break;
            
            }
            isl_set_free(deltas);

            
            // map origin->target
            isl_set * origin_set = isl_map_domain(isl_map_copy(dependency_map));
            isl_set * target_set = isl_map_range(isl_map_copy(dependency_map));

            // origin-free-map is [n1,n2]->[...consts...]
            isl_map * origin_free_map = isl_map_from_domain_and_range(
                isl_set_copy(zero_set_isl),
                isl_set_copy(origin_set)
            );

            isl_map * target_dependant_map = isl_map_from_domain_and_range(
                isl_set_copy(solution_set),
                isl_set_copy(target_set)
            );

            DEBUG(10, tiramisu::str_dump(" Reference map : "+std::string(isl_map_to_str(origin_free_map))));

            target_dependant_map = isl_map_sum(isl_map_copy(target_dependant_map),isl_map_copy(addition_map));

            DEBUG(10, tiramisu::str_dump(" Target map : "+std::string(isl_map_to_str(target_dependant_map))));

            isl_map * iteration_result = isl_map_lex_lt_map(origin_free_map,target_dependant_map);

            DEBUG(5, tiramisu::str_dump(" local solution : "+std::string(isl_map_to_str(iteration_result))));

            isl_set * iteration_result_set = isl_map_range(iteration_result);

            iteration_result_set = isl_set_project_out(iteration_result_set,isl_dim_param,0,m1);

            iteration_result_set = isl_set_coalesce(iteration_result_set);

            DEBUG(5, tiramisu::str_dump(" local set solution : "+std::string(isl_set_to_str(iteration_result_set))));

            real_solution_set = isl_set_intersect(real_solution_set,iteration_result_set);

            DEBUG(5, tiramisu::str_dump(" cumulative solution : "+std::string(isl_set_to_str(real_solution_set))));

            isl_set_free(origin_set);
            isl_set_free(target_set);
        }
        else
        {   //in_name == current_computation

            unify_map_origin = name_unificator_map[out_name];

            dependency_map = isl_map_apply_domain(dependency_map,isl_map_copy(unify_map_target));
            dependency_map = isl_map_apply_range(dependency_map,isl_map_copy(unify_map_origin));
            //dep is now in timestamps

            DEBUG(5, tiramisu::str_dump(" -----> dependendency in timestamp  Target -> Refernce : "+std::string(isl_map_to_str(dependency_map))));

            //check if delta is singleton
            isl_set * deltas = isl_map_deltas(isl_map_copy(dependency_map));
            deltas = isl_set_project_out(deltas,isl_dim_param,0,m1);
            deltas = isl_set_project_out(deltas,isl_dim_set,max_dynamic_number+2,(m1-2)-max_dynamic_number);
            if(!isl_set_is_singleton(deltas))
            {
                DEBUG(5, tiramisu::str_dump(" -### dependendency contains constants !! fusion aborted "));
                aborted_fusion = true;
                break;
            
            }
            isl_set_free(deltas);

            
            // map target -> origin
            isl_set * origin_set = isl_map_range(isl_map_copy(dependency_map));
            isl_set * target_set = isl_map_domain(isl_map_copy(dependency_map));

            // origin-free-map is [n1,n2]->[...consts...]
            isl_map * origin_free_map = isl_map_from_domain_and_range(
                isl_set_copy(zero_set_isl),
                isl_set_copy(origin_set)
            );

            isl_map * target_dependant_map = isl_map_from_domain_and_range(
                isl_set_copy(solution_set),
                isl_set_copy(target_set)
            );

            DEBUG(10, tiramisu::str_dump(" Reference map : "+std::string(isl_map_to_str(origin_free_map))));

            target_dependant_map = isl_map_sum(isl_map_copy(target_dependant_map),isl_map_copy(addition_map));

            DEBUG(10, tiramisu::str_dump(" Target map : "+std::string(isl_map_to_str(target_dependant_map))));

            isl_map * iteration_result  =  isl_map_lex_lt_map(target_dependant_map,origin_free_map);
           
            DEBUG(5, tiramisu::str_dump(" local solution : "+std::string(isl_map_to_str(iteration_result))));

            isl_set * iteration_result_set = isl_map_domain(iteration_result);

            iteration_result_set = isl_set_project_out(iteration_result_set,isl_dim_param,0,m1);

            iteration_result_set = isl_set_coalesce(iteration_result_set);

            DEBUG(5, tiramisu::str_dump(" local set solution : "+std::string(isl_set_to_str(iteration_result_set))));

            real_solution_set = isl_set_intersect(real_solution_set,iteration_result_set);

            DEBUG(5, tiramisu::str_dump(" cumulative solution : "+std::string(isl_set_to_str(real_solution_set))));
            isl_set_free(origin_set);
            isl_set_free(target_set);
        }

        isl_basic_map_free(dependency);
    }

    isl_set * result_set = real_solution_set;


    DEBUG(3, tiramisu::str_dump(" Final result set of n1,n2 ... : "+std::string(isl_set_to_str(result_set))));

    /**
     * Now that solving is over, we either have an empty set : no solutions -> fusion impossible
     * Or : set contain shifting params that we will try and extract the min
    */

    std::vector<std::tuple<tiramisu::var,int>> result_vector;

    if(!aborted_fusion && (isl_set_is_empty(result_set) == isl_bool_false))
    {   

        std::vector<isl_basic_set *> all_basic_set_solutions;

        auto f_set = [](isl_basic_set * bmap,void * user) { 
            std::vector<isl_basic_set *>& myName = *reinterpret_cast<std::vector<isl_basic_set*>*>(user);
            myName.push_back(bmap);
            return isl_stat_ok;
        };

        isl_stat (*fun_ptr_set)(isl_basic_set * p,void * m) = (f_set);

        isl_set_foreach_basic_set(result_set,fun_ptr_set,(void * ) &all_basic_set_solutions);

        isl_set * union_of_correct_solutions = NULL;

        for(auto basic_solution:all_basic_set_solutions)
        {
            isl_set * solution_i = isl_basic_set_lexmin(basic_solution);

            if(isl_set_is_singleton(solution_i) == isl_bool_true)
            {
                DEBUG(10, tiramisu::str_dump(" one valid solution is :"+std::string(isl_set_to_str(solution_i))));
                if(union_of_correct_solutions == NULL)
                {
                    union_of_correct_solutions = isl_set_copy(solution_i);
                }
                else
                {
                    union_of_correct_solutions = isl_set_union(union_of_correct_solutions,solution_i);
                }
            }
            else
            {
                DEBUG(10, tiramisu::str_dump(" incorrect value "));
            }
        }

        DEBUG(3, tiramisu::str_dump(" Union of all shifting solutions  :"+std::string(isl_set_to_str(union_of_correct_solutions))));

        union_of_correct_solutions = isl_set_lexmin(union_of_correct_solutions);

        DEBUG(3, tiramisu::str_dump(" lowest cost choosen solution is :"+std::string(isl_set_to_str(union_of_correct_solutions))));
        
        isl_basic_set * pre_val = isl_basic_set_read_from_str(this->get_isl_ctx(),isl_set_to_str(union_of_correct_solutions));

        int shifting_value = 0;

        for(int i=0; i<all_schedule_dim_numbers.size(); i++)
        {
            isl_val * value = isl_basic_set_dim_max_val( isl_basic_set_copy(pre_val),i);
            shifting_value = isl_val_get_d(value);

            DEBUG(3, tiramisu::str_dump(" Shifting for var :"+dynamic_var_mapping[all_schedule_dim_numbers[i]].get_name()+" "+std::to_string(shifting_value)));

            result_vector.push_back(std::make_tuple(dynamic_var_mapping[all_schedule_dim_numbers[i]],shifting_value));
        }

        isl_basic_set_free(pre_val);

    }
    else
    {
        DEBUG(3, tiramisu::str_dump(" the is fusion impossible "));
    }
    
    name_unificator_map.clear();
    dynamic_var_mapping.clear();
    all_schedule_dim_numbers.clear();
    all_basic_maps.clear();

    isl_union_map_free(all_deps);
    isl_set_free(result_set);
    isl_union_map_free(all_schedules);
    isl_union_map_free(dep_constants1);
    isl_union_map_free(universe_of_all_deps);


    DEBUG_INDENT(-4);

    return result_vector;

}


std::vector<isl_basic_set*> tiramisu::function::compute_legal_skewing(std::vector<tiramisu::computation *> fused_computations, tiramisu::var outer_variable,
                                              tiramisu::var inner_variable, int&  legal_process)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // TO-DO : memory leaks for exists with process != 1
    assert(outer_variable.get_name().length() > 0);
    assert(inner_variable.get_name().length() > 0);
    assert(!this->get_name().empty());

    assert(this->dep_read_after_write != NULL ) ;
    assert(this->dep_write_after_write != NULL ) ;
    assert(this->dep_write_after_read != NULL ) ;
    assert(fused_computations.size()>0) ;

    computation * first_computation = fused_computations[0]  ;
    
    DEBUG(3, tiramisu::str_dump(" skewing solving for : "+outer_variable.get_name()+" and "+inner_variable.get_name()));

    std::vector<std::string> original_loop_level_names = first_computation->get_loop_level_names();

    std::vector<int> dimensions =
        first_computation->get_loop_level_numbers_from_dimension_names({outer_variable.get_name(),inner_variable.get_name()});

    first_computation->check_dimensions_validity(dimensions);

    int outer_dim_full = tiramisu::loop_level_into_dynamic_dimension(dimensions[0]);
    int inner_dim_full = tiramisu::loop_level_into_dynamic_dimension(dimensions[1]);



    DEBUG(3, tiramisu::str_dump(" par dim number is : "+std::to_string(outer_dim_full)+ " and "+std::to_string(inner_dim_full)));

    // extracting deps

     isl_union_map * read_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->dep_read_after_write)) ;

    isl_union_map * write_after_read_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->dep_write_after_read)) ;

    isl_union_map * write_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->dep_write_after_write)) ;


    isl_union_map * all_deps = isl_union_map_union(
        read_after_write_dep,
        write_after_read_dep
        ) ;

    // all the deps in 1 union map
    all_deps = isl_union_map_union(all_deps,write_after_write_dep) ;


    DEBUG(3, tiramisu::str_dump(" all the dependencies are  : "+std::string(isl_union_map_to_str(all_deps))));

    // all current schedules in 1 union map
    std::string empty_union = "{}" ;
    std::string empty_time  = "" ;

    isl_union_map * schedules = isl_union_map_read_from_str(this->get_isl_ctx(),empty_union.c_str()) ;

    isl_map * schedule_one = NULL ;

    for( auto& computation: fused_computations)
    {
        schedule_one = isl_map_copy(computation->get_schedule()) ;

        schedule_one = isl_map_set_tuple_name(schedule_one,isl_dim_out,empty_time.c_str());

        schedules = isl_union_map_union(schedules,isl_union_map_from_map(schedule_one));
    }
    
    DEBUG(3, tiramisu::str_dump(" all the used schedules are  : "+std::string(isl_union_map_to_str(schedules))));

    // application to discard unused dep & modeling them in their time space

    all_deps = isl_union_map_apply_range(all_deps,isl_union_map_copy(schedules));

    all_deps = isl_union_map_apply_domain(all_deps,schedules);

    DEBUG(3, tiramisu::str_dump(" all the used dependencies union map are  : "+std::string(isl_union_map_to_str(all_deps))));

    if(isl_union_map_is_empty(all_deps))
    {
        DEBUG(3, tiramisu::str_dump(" No dependencies to be solved => skewing not necessary & no skewing proposed "));

        legal_process = 0 ; // disables skewing
        DEBUG_INDENT(-4);
        return {NULL,NULL,NULL,NULL,NULL,NULL};
        
    }

    isl_map * equation_map = isl_map_from_union_map(all_deps);

    DEBUG(3, tiramisu::str_dump(" all the used dependencies after transformed to map are  : "+std::string(isl_map_to_str(equation_map))));

    // remove already solved dimensions and project them out
    for(int i=0;i<outer_dim_full;i++)
    {
        equation_map = isl_map_equate(equation_map,isl_dim_in,i,isl_dim_out,i);
        DEBUG(3, tiramisu::str_dump(" --> remaining deps at itr "+std::to_string(i)+" : "+std::string(isl_map_to_str(equation_map))));    
    }

    //equate the middle static dimension [i,_0_,j]
    equation_map = isl_map_equate(equation_map,isl_dim_in,outer_dim_full+1,isl_dim_out,outer_dim_full+1);

    for (int i = outer_dim_full + 2; i < inner_dim_full; i++){
        // equation_map = isl_map_equate(equation_map,isl_dim_in,i,isl_dim_out,i);
        // skip removal
        DEBUG(3, tiramisu::str_dump(" --> remaining deps at itr "+std::to_string(i)+" : "+std::string(isl_map_to_str(equation_map)))); 
    }

    if (inner_dim_full - outer_dim_full - 2 > 0){
        // case where there is more than 1 static dimension in between target loops
        // remove and project them out
        equation_map = isl_map_project_out(equation_map, isl_dim_in,
                outer_dim_full + 2,inner_dim_full - outer_dim_full - 2);
        equation_map = isl_map_project_out(equation_map, isl_dim_out,
                outer_dim_full + 2,inner_dim_full - outer_dim_full - 2);
    }
    DEBUG(3, tiramisu::str_dump(" Map without in between dimensions : "+std::string(isl_map_to_str(equation_map))));

    int left_size = isl_map_dim(equation_map, isl_dim_in);
    int right_size = isl_map_dim(equation_map, isl_dim_out);

    assert(left_size == right_size);

    equation_map = isl_map_project_out(equation_map,isl_dim_in,0,outer_dim_full);
    equation_map = isl_map_project_out(equation_map,isl_dim_out,0,outer_dim_full);

    DEBUG(3, tiramisu::str_dump(" Map without irrelevant dimensions : "+std::string(isl_map_to_str(equation_map))));

    left_size = isl_map_dim(equation_map, isl_dim_in);
    right_size = isl_map_dim(equation_map, isl_dim_out);

    assert(left_size == right_size);

    int number_of_remaining_dims = right_size - 3;// [(i,0,j),10,k,0]

    std::string constant_set = "[";

    for(int i=0; i<number_of_remaining_dims; i++)
    {
        constant_set +="cx"+std::to_string(i);

        if(i != (number_of_remaining_dims - 1))
        {
            constant_set += ",";
        }
    }
    constant_set +="]" ;

    std::string set_of_remaining = constant_set+"->{"+constant_set+"}" ;

    isl_set * set_constant = isl_set_read_from_str(this->get_isl_ctx(),set_of_remaining.c_str());

    DEBUG(3, tiramisu::str_dump(" constant set to intersect domain : "+std::string(isl_set_to_str(set_constant))));
    
    std::vector<isl_basic_map *> all_basic_maps;// contains basic maps 

    auto f = [](isl_basic_map * bmap,void * user) { 

        std::vector<isl_basic_map *>& myName = *reinterpret_cast<std::vector<isl_basic_map*>*>(user);
     
        myName.push_back(bmap);
        return isl_stat_ok;
    };
    
    isl_stat (*fun_ptr)(isl_basic_map * p,void * m) = (f);

    isl_map_foreach_basic_map(equation_map,fun_ptr,(void * ) &all_basic_maps);

    isl_map * remaining_map = NULL;

    isl_set * remaining_domain = NULL;

    isl_set * remaining_range = NULL;

    bool process_legal = true;

    bool must_strongly_solved = false;


    bool upper_domain_involved = true;

    double first_int=0.0;
    double second_int=0.0;
    double tan_value = 0.0;


    std::string normal_map_str = "{ [0,j]->[1,0]: j>0; [i,0]->[0,1]: i!=0 ; [i,j]->[j,-i] : j>0 and i!=0 ; [i,j]->[-j,i] : j<0 and i!=0 }";

    isl_map * normal_map_calculator = isl_map_read_from_str(this->get_isl_ctx(),normal_map_str.c_str());

    std::string upper_domain_str = "{[a,b] : a>0 and b>0}"; 

    // a set used to compute the domain of gamma & sigma 
    std::string upper_domain_real_domain = "{[a,b] : a>=0 and b>=0 and a+b>0}";

    isl_basic_set * upper_weakly = isl_basic_set_read_from_str(this->get_isl_ctx(),upper_domain_str.c_str());
    isl_basic_set * upper_weakly_gamma_sigma = isl_basic_set_read_from_str(this->get_isl_ctx(),upper_domain_real_domain.c_str());

    isl_basic_set * upper_strongly = isl_basic_set_read_from_str(this->get_isl_ctx(),upper_domain_str.c_str());


    std::string lower_domain_str = "{[a,b] : a>0 and b<0 }";

    isl_basic_set * lower_weakly = isl_basic_set_read_from_str(this->get_isl_ctx(),lower_domain_str.c_str());

    isl_basic_set * lower_strongly = isl_basic_set_read_from_str(this->get_isl_ctx(),lower_domain_str.c_str());


    std::string outer_most_parallelism_str = "{[a,b] : a>0 }";

    isl_basic_set * outer_most_parallelism = isl_basic_set_read_from_str(this->get_isl_ctx(),outer_most_parallelism_str.c_str());

    bool outer_impossible = false;

    /**
     * strongly solve for parallelism & weakly solve for legality
    */
   int iteration = 0;

    for(auto dependency:all_basic_maps)
    {
        if(! isl_map_is_identity(isl_map_from_basic_map(isl_basic_map_copy(dependency)))) // remove identity relations 
        {
            DEBUG(3, tiramisu::str_dump(" --> DEPENDENCY & constraints for : "+std::string(isl_basic_map_to_str(dependency))));
            // no free variable or impossible
            
            remaining_map = isl_map_project_out(isl_map_from_basic_map(isl_basic_map_copy(dependency)),isl_dim_out,0,3);

            remaining_map = isl_map_project_out(remaining_map,isl_dim_in,0,3);

            DEBUG(3, tiramisu::str_dump(" --> ---> remaining map no intersect : "+std::string(isl_map_to_str(remaining_map))));

            remaining_map = isl_map_intersect_domain(remaining_map,isl_set_copy(set_constant));

            DEBUG(3, tiramisu::str_dump(" --> ---> remaining map with intersect : "+std::string(isl_map_to_str(remaining_map))));

            isl_map * involved_map = isl_map_project_out(isl_map_from_basic_map(isl_basic_map_copy(dependency)),isl_dim_out,3,number_of_remaining_dims);
            involved_map = isl_map_project_out(involved_map,isl_dim_in,3,number_of_remaining_dims);
            involved_map = isl_map_project_out(involved_map,isl_dim_in,1,1);
            involved_map = isl_map_project_out(involved_map,isl_dim_out,1,1);

            DEBUG(3, tiramisu::str_dump(" --> ---> Applying deltas on map : "+std::string(isl_map_to_str(involved_map))));

            isl_set * delta_result = isl_map_deltas(involved_map);

            if(!isl_set_is_singleton(delta_result))
            {
                process_legal = false;
                DEBUG(3, tiramisu::str_dump(" --> Illegal _END : delta issues "));
                break;
            }

            delta_result =  isl_set_apply(
                    delta_result,
                    isl_map_copy(normal_map_calculator)
                );
            

            DEBUG(3, tiramisu::str_dump(" --> ---> deltas value : "+std::string(isl_set_to_str(delta_result))));

            if(!isl_set_is_empty(delta_result))
            {//normal_map_calculator removes identity equations 

                /**
                 * Deciding weather to strongly solve or weakly solve this dependency
                */
        
                if(isl_map_is_identity(remaining_map))
                { // a case of a reflexive relations : strongly solved-> parallelism , weakly solved -> legality

                    must_strongly_solved = false;
                    DEBUG(3, tiramisu::str_dump(" --> --->  strongly solved for inner-parallelism & weakly for legality "));

                }
                else
                {
                    remaining_domain = isl_map_domain(isl_map_copy(remaining_map));
                    remaining_range = isl_map_range(isl_map_copy(remaining_map));

                    isl_map * remaining_solution = isl_set_lex_gt_set(
                        remaining_domain,
                        remaining_range
                    );

                    if(isl_map_is_empty(remaining_solution))
                    { 

                        must_strongly_solved = false;
                        DEBUG(3, tiramisu::str_dump(" --> --->  strongly solved for inner-parallelism & weakly for legality "));
                    }
                    else
                    { // // dep must be strongly solved regardless
                        must_strongly_solved = true;

                        DEBUG(3, tiramisu::str_dump(" --> ---> must be strongly solved regardless (no outer-parallism)"));

                        //outermost parallism is impossible
                        outer_impossible = true;
                    }
                    isl_map_free(remaining_solution);
                }

                /**
                 * deciding weather this dependency involves upper domain or lower_domain
                */

                isl_val * value1 = isl_basic_set_dim_max_val( isl_basic_set_read_from_str(this->get_isl_ctx(),isl_set_to_str(delta_result)),0);
                isl_val * value2 = isl_basic_set_dim_max_val( isl_basic_set_read_from_str(this->get_isl_ctx(),isl_set_to_str(delta_result)),1);
                first_int = isl_val_get_d(value1);
                second_int = isl_val_get_d(value2);

                isl_val_free(value1);
                isl_val_free(value2);

                int first = (int)(first_int); 
                int second = (int)(second_int); 

                DEBUG(3, tiramisu::str_dump(" --> ---> first delta value "+std::to_string(first_int)));
                DEBUG(3, tiramisu::str_dump(" --> ---> second delta value "+std::to_string(second_int)));

                DEBUG(3, tiramisu::str_dump(" --> ---> first delta value "+std::to_string(first)));
                DEBUG(3, tiramisu::str_dump(" --> ---> second delta value "+std::to_string(second)));

                if(first == 0)
                { // ((first == 0) || ((first == 1) && (second == 0)))
                    upper_domain_involved = true;
                }
                else{
                    tan_value = second_int / first_int ;

                    DEBUG(3, tiramisu::str_dump(" --> ---> tan(x) value "+std::to_string(tan_value)));

                    if(tan_value > 0.0)
                    {
                        upper_domain_involved = true;
                    }
                    else
                    {
                        upper_domain_involved = false;
                    }
                }

                /**
                 * adding the constraints for upper or lower
                */
                
                if(upper_domain_involved)
                {
                    DEBUG(3, tiramisu::str_dump(" --> ---> UPPER Domain constraint "));

                    if(must_strongly_solved)
                    {   //either dep is unsatisfied totally or fully satisfied

                        //upper strongly solved

                        isl_space * space = isl_basic_set_get_space(upper_weakly);
                        isl_constraint * constraint = isl_constraint_alloc_inequality(isl_local_space_from_space(space));

                        constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,0, second);
                        constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,1, -first);
                        constraint = isl_constraint_set_constant_si(constraint,-1);

                        upper_weakly = isl_basic_set_add_constraint(upper_weakly,isl_constraint_copy(constraint));
                        upper_weakly_gamma_sigma = isl_basic_set_add_constraint(upper_weakly_gamma_sigma,isl_constraint_copy(constraint));
                        upper_strongly = isl_basic_set_add_constraint(upper_strongly,constraint);
                        
                    }
                    else
                    {   //upper 2 cases

                        isl_space * space = isl_basic_set_get_space(upper_weakly);
                        isl_constraint * constraint = isl_constraint_alloc_inequality(isl_local_space_from_space(space));

                        constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,0, second );
                        constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,1, -first);

                        isl_constraint * strong_constraint = isl_constraint_copy(constraint);
                        strong_constraint = isl_constraint_set_constant_si(strong_constraint,-1);

                        upper_strongly = isl_basic_set_add_constraint(upper_strongly,strong_constraint);
                        upper_weakly_gamma_sigma = isl_basic_set_add_constraint(upper_weakly_gamma_sigma,isl_constraint_copy(constraint));
                        upper_weakly = isl_basic_set_add_constraint(upper_weakly,constraint);
                    }

                    DEBUG(3, tiramisu::str_dump(" --> ---> new upper strongly domain is "+std::string(isl_basic_set_to_str(upper_weakly))));
                    DEBUG(3, tiramisu::str_dump(" --> ---> new upper weakly domain is "+std::string(isl_basic_set_to_str(upper_strongly))));
                        
                }
                else
                {
                    DEBUG(3, tiramisu::str_dump(" --> ---> lower Domain constraint "));

                    if(must_strongly_solved)
                    {
                          //lower strongly solved

                        isl_space * space = isl_basic_set_get_space(lower_weakly);
                        isl_constraint * constraint = isl_constraint_alloc_inequality(isl_local_space_from_space(space));

                        constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,0,- second);
                        constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,1, first);
                        constraint = isl_constraint_set_constant_si(constraint,-1);

                        lower_weakly = isl_basic_set_add_constraint(lower_weakly,isl_constraint_copy(constraint));
                        lower_strongly =  isl_basic_set_add_constraint(lower_strongly,constraint);
                        
                        
                    }
                    else
                    {   //lower 2cases

                        isl_space * space = isl_basic_set_get_space(lower_weakly);
                        isl_constraint * constraint = isl_constraint_alloc_inequality(isl_local_space_from_space(isl_space_copy(space)));

                        constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,0,- second);
                        constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,1, first);
                        
                        isl_constraint * strong_constraint = isl_constraint_copy(constraint);
                        strong_constraint = isl_constraint_set_constant_si(strong_constraint,-1);

                        lower_strongly = isl_basic_set_add_constraint(lower_strongly,strong_constraint);
                        lower_weakly = isl_basic_set_add_constraint(lower_weakly,constraint);

                    }
                    DEBUG(3, tiramisu::str_dump(" --> ---> new lower strongly domain is "+std::string(isl_basic_set_to_str(lower_strongly))));
                    DEBUG(3, tiramisu::str_dump(" --> ---> new lower weakly domain is "+std::string(isl_basic_set_to_str(lower_weakly))));
                    
                }

                if(!outer_impossible)
                {
                    isl_space * space = isl_basic_set_get_space(outer_most_parallelism);
                    isl_constraint * constraint = isl_constraint_alloc_equality(isl_local_space_from_space(space));

                    constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,0, second );
                    constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,1, -first);

                    outer_most_parallelism = isl_basic_set_add_constraint(outer_most_parallelism,constraint);

                    if(isl_basic_set_is_empty(outer_most_parallelism))
                    {
                        outer_impossible = true;
                        DEBUG(3, tiramisu::str_dump(" --> Outer most parallism impossible "));
                    }
                }

                iteration++;
            }
           
            
            isl_set_free(delta_result);
            isl_map_free(remaining_map);

        }
        isl_basic_map_free(dependency);
        
    }


    if((iteration == 0) || (isl_map_is_empty(equation_map)))
    {
        legal_process = 0;
        DEBUG(3, tiramisu::str_dump(" Lack of dependencies within scope "));
    }
    else
    {   
        if(process_legal){
            legal_process = 1;
             DEBUG(3, tiramisu::str_dump(" Correct Process "));
        }
        else
        {
            legal_process = -1;
            DEBUG(3, tiramisu::str_dump(" insolvable dependencies, incorrect Process "));
        }
    }

    isl_map_free(normal_map_calculator);
    isl_map_free(equation_map);
    isl_set_free(set_constant);
    all_basic_maps.clear();

    if(outer_impossible)
    { //make empty
        DEBUG(3, tiramisu::str_dump(" empty outermost parallism "+std::string(isl_space_to_str(isl_basic_set_get_space(outer_most_parallelism)))));
        outer_most_parallelism = isl_basic_set_empty(isl_basic_set_get_space(outer_most_parallelism));
    }
        DEBUG(3, tiramisu::str_dump(" Upper weakly : "+std::string(isl_basic_set_to_str(upper_weakly))));
        DEBUG(3, tiramisu::str_dump(" Upper strongly : "+std::string(isl_basic_set_to_str(upper_strongly))));
        DEBUG(3, tiramisu::str_dump(" lower weakly : "+std::string(isl_basic_set_to_str(lower_weakly))));
        DEBUG(3, tiramisu::str_dump(" lower strongly : "+std::string(isl_basic_set_to_str(lower_strongly))));
        DEBUG(3, tiramisu::str_dump(" outermost parallism : "+std::string(isl_basic_set_to_str(outer_most_parallelism))));
        DEBUG(3, tiramisu::str_dump(" gamma sigma set : "+std::string(isl_basic_set_to_str(upper_weakly_gamma_sigma))));

    DEBUG_INDENT(-4);

    return {upper_weakly,upper_strongly,lower_weakly,lower_strongly,outer_most_parallelism,upper_weakly_gamma_sigma} ;

}


std::tuple<
      std::vector<std::pair<int,int>>,
      std::vector<std::pair<int,int>>,
      std::vector<std::pair<int,int>>> tiramisu::function::skewing_local_solver(std::vector<tiramisu::computation *> fused_computations,
                                                            tiramisu::var outer_variable,tiramisu::var inner_variable, int nb_parallel)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(outer_variable.get_name().length() > 0);
    assert(inner_variable.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->dep_read_after_write != NULL );
    assert(this->dep_write_after_write != NULL );
    assert(this->dep_write_after_read != NULL );
    assert(fused_computations.size()>0);

    isl_basic_set * upper_strongly = NULL;
    isl_basic_set * upper_weakly = NULL;

    isl_basic_set * lower_strongly = NULL;
    isl_basic_set * lower_weakly = NULL;

    isl_basic_set * parallism = NULL;

    int process = -1;

    std::vector<std::pair<int,int>> locality;
    std::vector<std::pair<int,int>> outermost;
    std::vector<std::pair<int,int>> innermost;

    auto result_vector = this->compute_legal_skewing(fused_computations,outer_variable,inner_variable,process);

    if(process == 1)
    {
        assert(result_vector.size() == 6);
        upper_weakly = result_vector[0];
        upper_strongly = result_vector[1];
        lower_weakly = result_vector[2];
        lower_strongly = result_vector[3];
        parallism = result_vector[4];
        DEBUG(3, tiramisu::str_dump(" EXTRACTING Values of alpha & beta : "));

        DEBUG(3, tiramisu::str_dump(" Upper weakly : "+std::string(isl_basic_set_to_str(upper_weakly))));
        DEBUG(3, tiramisu::str_dump(" Upper strongly : "+std::string(isl_basic_set_to_str(upper_strongly))));
        DEBUG(3, tiramisu::str_dump(" lower weakly : "+std::string(isl_basic_set_to_str(lower_weakly))));
        DEBUG(3, tiramisu::str_dump(" lower strongly : "+std::string(isl_basic_set_to_str(lower_strongly))));
        DEBUG(3, tiramisu::str_dump(" outermost parallism : "+std::string(isl_basic_set_to_str(parallism))));

        bool once_used = false;

        /**
         * Solving locality
        */

       isl_set * upper_intersect = isl_set_subtract(
           isl_set_from_basic_set(isl_basic_set_copy(upper_weakly)),
           isl_set_from_basic_set(isl_basic_set_copy(upper_strongly))
       );

       isl_set * lower_intersect = isl_set_subtract(
           isl_set_from_basic_set(isl_basic_set_copy(lower_weakly)),
           isl_set_from_basic_set(isl_basic_set_copy(lower_strongly))
       );

       DEBUG(3, tiramisu::str_dump(" substracted locality lower "+std::string(isl_set_to_str(lower_intersect))));
       DEBUG(3, tiramisu::str_dump(" substracted locality upper "+std::string(isl_set_to_str(upper_intersect))));

       if(!isl_set_is_empty(upper_intersect))
       {
           upper_intersect = isl_set_lexmin(upper_intersect);
           DEBUG(3, tiramisu::str_dump(" choosen locality upper "+std::string(isl_set_to_str(upper_intersect))));
           isl_basic_set * result = isl_set_polyhedral_hull(upper_intersect);

           DEBUG(3, tiramisu::str_dump(" polyhedral hull is :"+std::string(isl_basic_set_to_str(result))));
           
           isl_val * value1 = isl_basic_set_dim_max_val( isl_basic_set_copy(result),0);
           isl_val * value2 = isl_basic_set_dim_max_val( isl_basic_set_copy(result),1);

           int locality_var1 = isl_val_get_d(value1);
           int locality_var2 = isl_val_get_d(value2);

           DEBUG(3, tiramisu::str_dump(" skewing upper locality is (alpha,beta) = ("+std::to_string(locality_var1)+","+std::to_string(locality_var2)+")"));

           locality.push_back(std::make_pair(locality_var1,locality_var2));

           isl_basic_set_free(result);
           isl_val_free(value1);
           isl_val_free(value2);

           once_used = !once_used;
       }
       else
       {
           isl_set_free(upper_intersect);
       }

       if(!isl_set_is_empty(lower_intersect))
       {
           lower_intersect = isl_set_lexmin(lower_intersect);
           DEBUG(3, tiramisu::str_dump(" choosen locality lower "+std::string(isl_set_to_str(lower_intersect))));
           isl_basic_set * result = isl_set_polyhedral_hull(lower_intersect);

           DEBUG(3, tiramisu::str_dump(" polyhedral hull is :"+std::string(isl_basic_set_to_str(result))));
           
           isl_val * value1 = isl_basic_set_dim_max_val( isl_basic_set_copy(result),0);
           isl_val * value2 = isl_basic_set_dim_max_val( isl_basic_set_copy(result),1);

           int locality_var1 = isl_val_get_d(value1);
           int locality_var2 = isl_val_get_d(value2);

           DEBUG(3, tiramisu::str_dump(" skewing lower locality is (alpha,beta) = ("+std::to_string(locality_var1)+","+std::to_string(locality_var2)+")"));


           locality.push_back(std::make_pair(locality_var1,locality_var2));

           isl_basic_set_free(result);
           isl_val_free(value1);
           isl_val_free(value2);

           once_used = !once_used;
       }
       else
       {
           isl_set_free(lower_intersect);
       }
       

        /**
         * Solving outermost parallelism 
        */
        
        if((!isl_basic_set_is_empty(parallism)) && once_used)
        {
           isl_set * isl_outer_sol = isl_basic_set_lexmin(parallism);
           DEBUG(3, tiramisu::str_dump(" choosen outer parallism "+std::string(isl_set_to_str(isl_outer_sol))));
           isl_basic_set * result = isl_set_polyhedral_hull(isl_outer_sol);

           DEBUG(3, tiramisu::str_dump(" polyhedral hull is :"+std::string(isl_basic_set_to_str(result))));
           
           isl_val * value1 = isl_basic_set_dim_max_val( isl_basic_set_copy(result),0);
           isl_val * value2 = isl_basic_set_dim_max_val( isl_basic_set_copy(result),1);

           int locality_var1 = isl_val_get_d(value1);
           int locality_var2 = isl_val_get_d(value2);

           DEBUG(3, tiramisu::str_dump(" skewing outer_parallelism is (alpha,beta) = ("+std::to_string(locality_var1)+","+std::to_string(locality_var2)+")"));

           outermost.push_back(std::make_pair(locality_var1,locality_var2));

           isl_basic_set_free(result);
           isl_val_free(value1);
           isl_val_free(value2);
        }

        /**
         * Solving the parallelism 
         * */
        isl_set * upper_set = isl_set_from_basic_set(upper_strongly);

        std::string upper_new_set_str = "{[a,b]:a>0 and b>0}";

        //extracting from upper domain
        int i = 0;
        
        while((i < nb_parallel) && (!isl_set_is_empty(upper_set)))
        {
            DEBUG(3, tiramisu::str_dump("# upper inner parallism solution set :"+std::string(isl_set_to_str(upper_set))));

            isl_set * solution = isl_set_lexmin(isl_set_copy(upper_set));
            DEBUG(3, tiramisu::str_dump(" choosen inner parallism "+std::string(isl_set_to_str(solution))));

            isl_basic_set * result = isl_set_polyhedral_hull(solution);

            DEBUG(3, tiramisu::str_dump(" polyhedral hull is :"+std::string(isl_basic_set_to_str(result))));
           
            isl_val * value1 = isl_basic_set_dim_max_val(isl_basic_set_copy(result),0);
            isl_val * value2 = isl_basic_set_dim_max_val(result,1);

            int locality_var1 = isl_val_get_d(value1);
            int locality_var2 = isl_val_get_d(value2);

            DEBUG(3, tiramisu::str_dump(" skewing upper inner_parallelism is (alpha,beta) = ("+std::to_string(locality_var1)+","+std::to_string(locality_var2)+")"));

            innermost.push_back(std::make_pair(locality_var1,locality_var2));

            // adding new constraint a!=b as to avoid same solution twice
            isl_space * space = isl_set_get_space(upper_set);

            isl_set * new_set = isl_set_read_from_str(this->get_isl_ctx(),upper_new_set_str.c_str());

            DEBUG(3, tiramisu::str_dump(" new set  :"+std::string(isl_set_to_str(new_set))));

            isl_constraint * constraint = isl_constraint_alloc_equality(isl_local_space_from_space(space));

            constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,0, locality_var2);
            constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,1, -locality_var1);

            new_set = isl_set_add_constraint(new_set,constraint);

            DEBUG(3, tiramisu::str_dump(" new set with constrainte :"+std::string(isl_set_to_str(new_set))));

            upper_set = isl_set_subtract(upper_set,new_set);

            upper_set = isl_set_coalesce(upper_set);

            isl_val_free(value1);
            isl_val_free(value2);

            i++;
        }

        i = 0;
        //extracting from lower domain

        isl_set * lower_set = isl_set_from_basic_set(lower_strongly);

        std::string lower_new_set_str = "{[a,b]:a>0 and b<0}";

        while((i < nb_parallel) && (!isl_set_is_empty(lower_set)))
        {
            DEBUG(3, tiramisu::str_dump("# lower inner parallism solution set :"+std::string(isl_set_to_str(lower_set))));

            isl_set * solution = isl_set_lexmin(isl_set_copy(lower_set));

            if(isl_set_is_empty(solution) != isl_bool_false)
            {
                isl_set_free(solution);
                break;
            }

            DEBUG(3, tiramisu::str_dump(" choosen inner parallism "+std::string(isl_set_to_str(solution))));

            isl_basic_set * result = isl_set_polyhedral_hull(solution);

            DEBUG(3, tiramisu::str_dump(" polyhedral hull is :"+std::string(isl_basic_set_to_str(result))));
           
            isl_val * value1 = isl_basic_set_dim_max_val(isl_basic_set_copy(result),0);
            isl_val * value2 = isl_basic_set_dim_max_val(result,1);

            int locality_var1 = isl_val_get_d(value1);
            int locality_var2 = isl_val_get_d(value2);

            DEBUG(3, tiramisu::str_dump(" skewing lower inner_parallelism is (alpha,beta) = ("+std::to_string(locality_var1)+","+std::to_string(locality_var2)+")"));

            innermost.push_back(std::make_pair(locality_var1,locality_var2));

            // adding new constraint a!=b as to avoid same solution twice
            isl_space * space = isl_set_get_space(lower_set);

            isl_set * new_set = isl_set_read_from_str(this->get_isl_ctx(),lower_new_set_str.c_str());

            DEBUG(3, tiramisu::str_dump(" new set  :"+std::string(isl_set_to_str(new_set))));

            isl_constraint * constraint = isl_constraint_alloc_equality(isl_local_space_from_space(space));

            constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,0, locality_var2);
            constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,1, -locality_var1);

            new_set = isl_set_add_constraint(new_set,constraint);

            DEBUG(3, tiramisu::str_dump(" new set with constrainte :"+std::string(isl_set_to_str(new_set))));

            lower_set = isl_set_subtract(lower_set,new_set);

            lower_set = isl_set_coalesce(lower_set);

            isl_val_free(value1);
            isl_val_free(value2);

            i++;
        }

        isl_set_free(lower_set);
        isl_set_free(upper_set);
        


    }
     
    DEBUG_INDENT(-4);

    return std::make_tuple(outermost,innermost,locality);

}
/**
 * Given skewing parameters alpha & beta, this method computes the corresponding parameters for gamma and sigma
 * to enforce the condition Det(A)=1 while also making the dependencies positive and enabling tiling.
 * Legal_domain is the isl_set for the correct values of alpha and beta.
 * This set is proven to be exactly the same for gamma and sigma 
 * values is a tuple providing <alpha, beta, gamma, sigma> were will use alpha & beta given to fill gamma & sigma.
 * This function returns true if it was able to find gamma and sigma in the positive domain, false otherwise
*/
bool compute_gamma_sigma_values(isl_basic_set * legal_domain, std::tuple<int,int,int,int>& values)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_set * domain = isl_set_from_basic_set(isl_basic_set_copy(legal_domain));
    // add constraint alpha*gamma - beta*sigma = 1

    isl_space * space = isl_set_get_space(domain);

    isl_constraint * constraint = isl_constraint_alloc_equality(isl_local_space_from_space(space));

    constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_set, 0, -std::get<1>(values));
    constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_set, 1, std::get<0>(values));
    constraint = isl_constraint_set_constant_si(constraint, -1);
    domain = isl_set_add_constraint(domain, constraint);

    DEBUG(3, tiramisu::str_dump(" For alpha & beta  :" + std::to_string(std::get<0>(values)) + " & " +
         std::to_string(std::get<1>(values))));

    DEBUG(3, tiramisu::str_dump(" set of gamma & sigma  :"+std::string(isl_set_to_str(domain))));

    // extract the min parameters for gamma and sigma
    isl_set * result = isl_set_lexmin(domain);
    DEBUG(3, tiramisu::str_dump(" choosen gamma & sigma "+std::string(isl_set_to_str(result))));
    isl_basic_set * result_basic = isl_set_polyhedral_hull(result);

    DEBUG(3, tiramisu::str_dump(" polyhedral hull is :"+std::string(isl_basic_set_to_str(result_basic))));

    if(isl_basic_set_is_empty(result_basic)){
        DEBUG_INDENT(-4);
        return false;
    }
    
    isl_val * gamma_val = isl_basic_set_dim_max_val(isl_basic_set_copy(result_basic), 0);
    isl_val * sigma_val = isl_basic_set_dim_max_val(isl_basic_set_copy(result_basic), 1);

    int gamma = isl_val_get_d(gamma_val);
    int sigma = isl_val_get_d(sigma_val);

    std::get<2>(values) = gamma;
    std::get<3>(values) = sigma;

    isl_basic_set_free(result_basic);
    isl_val_free(gamma_val);
    isl_val_free(sigma_val);

    DEBUG_INDENT(-4);
    return true;
}

std::tuple<
      std::vector<std::tuple<int,int,int,int>>,
      std::vector<std::tuple<int,int,int,int>>,
      std::vector<std::tuple<int,int,int,int>>> tiramisu::function::skewing_local_solver_positive(std::vector<tiramisu::computation *> fused_computations,
                                                          tiramisu::var outer_variable,tiramisu::var inner_variable, int nb_parallel)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(outer_variable.get_name().length() > 0);
    assert(inner_variable.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->dep_read_after_write != NULL );
    assert(this->dep_write_after_write != NULL );
    assert(this->dep_write_after_read != NULL );
    assert(fused_computations.size()>0);

    isl_basic_set * upper_strongly = NULL;
    isl_basic_set * upper_weakly = NULL;

    // set used to computed gamma & sigma
    isl_basic_set * positive_secondary_set = NULL;

    isl_basic_set * parallism = NULL;

    int process = -1;

    std::vector<std::tuple<int,int,int,int>> locality;
    std::vector<std::tuple<int,int,int,int>> outermost;
    std::vector<std::tuple<int,int,int,int>> innermost;
    std::vector<std::tuple<int,int,int,int>> identity;

    auto result_vector = this->compute_legal_skewing(fused_computations,outer_variable,inner_variable,process);

    if(process == 1)
    {
        assert(result_vector.size() == 6);
        upper_weakly = result_vector[0];

        identity.push_back(std::make_tuple(1, 0, 0, 0));

        // set is exactly similar to constraints for alpha and beta
        positive_secondary_set = result_vector[5];
        upper_strongly = result_vector[1];
        parallism = result_vector[4];
        DEBUG(3, tiramisu::str_dump(" EXTRACTING Values of alpha & beta : "));

        DEBUG(3, tiramisu::str_dump(" Upper weakly : "+std::string(isl_basic_set_to_str(upper_weakly))));
        DEBUG(3, tiramisu::str_dump(" Upper strongly : "+std::string(isl_basic_set_to_str(upper_strongly))));
        DEBUG(3, tiramisu::str_dump(" outermost parallism : "+std::string(isl_basic_set_to_str(parallism))));


        /**
         * Solving locality
        */

       isl_set * upper_intersect = isl_set_subtract(
           isl_set_from_basic_set(upper_weakly),
           isl_set_from_basic_set(isl_basic_set_copy(upper_strongly))
       );

       DEBUG(3, tiramisu::str_dump(" substracted locality upper "+std::string(isl_set_to_str(upper_intersect))));

       if(!isl_set_is_empty(upper_intersect))
       {
           upper_intersect = isl_set_lexmin(upper_intersect);
           DEBUG(3, tiramisu::str_dump(" choosen locality upper "+std::string(isl_set_to_str(upper_intersect))));
           isl_basic_set * result = isl_set_polyhedral_hull(upper_intersect);

           DEBUG(3, tiramisu::str_dump(" polyhedral hull is :"+std::string(isl_basic_set_to_str(result))));
           
           isl_val * value1 = isl_basic_set_dim_max_val( isl_basic_set_copy(result),0);
           isl_val * value2 = isl_basic_set_dim_max_val( isl_basic_set_copy(result),1);

           int locality_var1 = isl_val_get_d(value1);
           int locality_var2 = isl_val_get_d(value2);

           DEBUG(3, tiramisu::str_dump(" skewing upper locality is (alpha,beta) = ("+std::to_string(locality_var1)+","+std::to_string(locality_var2)+")"));

           locality.push_back(std::make_tuple(locality_var1, locality_var2, 0, 0));

           isl_basic_set_free(result);
           isl_val_free(value1);
           isl_val_free(value2);
       }
       else
       {
           isl_set_free(upper_intersect);
       }

        /**
         * Solving outermost parallelism 
        */
        
        if(!isl_basic_set_is_empty(parallism))
        {
           isl_set * isl_outer_sol = isl_basic_set_lexmin(parallism);
           DEBUG(3, tiramisu::str_dump(" choosen outer parallism "+std::string(isl_set_to_str(isl_outer_sol))));
           isl_basic_set * result = isl_set_polyhedral_hull(isl_outer_sol);

           DEBUG(3, tiramisu::str_dump(" polyhedral hull is :"+std::string(isl_basic_set_to_str(result))));
           
           isl_val * value1 = isl_basic_set_dim_max_val( isl_basic_set_copy(result),0);
           isl_val * value2 = isl_basic_set_dim_max_val( isl_basic_set_copy(result),1);

           int locality_var1 = isl_val_get_d(value1);
           int locality_var2 = isl_val_get_d(value2);

           DEBUG(3, tiramisu::str_dump(" skewing outer_parallelism is (alpha,beta) = ("+std::to_string(locality_var1)+","+std::to_string(locality_var2)+")"));

           outermost.push_back(std::make_tuple(locality_var1, locality_var2, 0, 0));

           isl_basic_set_free(result);
           isl_val_free(value1);
           isl_val_free(value2);
        }

        /**
         * Solving the parallelism 
         * */
        isl_set * upper_set = isl_set_from_basic_set(upper_strongly);

        std::string upper_new_set_str = "{[a,b]:a>0 and b>0}";

        //extracting from upper domain
        int i = 0;
        
        while((i < nb_parallel) && (!isl_set_is_empty(upper_set)))
        {
            DEBUG(3, tiramisu::str_dump("# upper inner parallism solution set :"+std::string(isl_set_to_str(upper_set))));

            isl_set * solution = isl_set_lexmin(isl_set_copy(upper_set));
            DEBUG(3, tiramisu::str_dump(" choosen inner parallism "+std::string(isl_set_to_str(solution))));

            isl_basic_set * result = isl_set_polyhedral_hull(solution);

            DEBUG(3, tiramisu::str_dump(" polyhedral hull is :"+std::string(isl_basic_set_to_str(result))));
           
            isl_val * value1 = isl_basic_set_dim_max_val(isl_basic_set_copy(result),0);
            isl_val * value2 = isl_basic_set_dim_max_val(result,1);

            int locality_var1 = isl_val_get_d(value1);
            int locality_var2 = isl_val_get_d(value2);

            DEBUG(3, tiramisu::str_dump(" skewing upper inner_parallelism is (alpha,beta) = ("+std::to_string(locality_var1)+","+std::to_string(locality_var2)+")"));

            innermost.push_back(std::make_tuple(locality_var1, locality_var2, 0, 0));

            // adding new constraint a!=b as to avoid same solution twice
            isl_space * space = isl_set_get_space(upper_set);

            isl_set * new_set = isl_set_read_from_str(this->get_isl_ctx(),upper_new_set_str.c_str());

            DEBUG(3, tiramisu::str_dump(" new set  :"+std::string(isl_set_to_str(new_set))));

            isl_constraint * constraint = isl_constraint_alloc_equality(isl_local_space_from_space(space));

            constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,0, locality_var2);
            constraint = isl_constraint_set_coefficient_si(constraint,isl_dim_set,1, -locality_var1);
            new_set = isl_set_add_constraint(new_set,constraint);

            DEBUG(3, tiramisu::str_dump(" new set with constraint :"+std::string(isl_set_to_str(new_set))));

            upper_set = isl_set_subtract(upper_set,new_set);

            upper_set = isl_set_coalesce(upper_set);

            isl_val_free(value1);
            isl_val_free(value2);

            i++;
        }

        isl_set_free(upper_set);

        // compute gamma & sigma for all produced alpha & beta
        // it's possible not to find gamma & sigma for outmost parameters
        if(outermost.size() > 0){
            if(! compute_gamma_sigma_values(positive_secondary_set, outermost[0])){
                outermost.clear();
            }
        }

        for (auto& params :innermost){
            compute_gamma_sigma_values(positive_secondary_set, params);
        }
        // we know we will not be able to find gamma & sigma in the positive domain 
        // while using these locality alpha & beta
        /*
        for (auto& params :locality){
            compute_gamma_sigma_values(positive_secondary_set, params);
        }
        */
        for (auto& params :identity){
            compute_gamma_sigma_values(positive_secondary_set, params);
        }

        isl_basic_set_free(positive_secondary_set);
        
    }
    else if (process == 0) // no dependencies to be solved
    {
        identity.push_back(std::make_tuple(1, 0, 0, 1));
    }
     
    DEBUG_INDENT(-4);

    return std::make_tuple(outermost,innermost,identity);

}


std::tuple<
      std::vector<std::tuple<int,int,int,int,int,int,int,int,int>>,
      std::vector<std::tuple<int,int,int,int,int,int,int,int,int>>,
      std::vector<std::tuple<int,int,int,int,int,int,int,int,int>>> tiramisu::function::skewing_local_3D_solver_positive(
                                                  std::vector<tiramisu::computation *> fused_computations, tiramisu::var var_outer,
                                                  tiramisu::var var2, tiramisu::var var_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(var_outer.get_name().length() > 0);
    assert(var_inner.get_name().length() > 0);
    assert(var2.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->dep_read_after_write != NULL );
    assert(this->dep_write_after_write != NULL );
    assert(this->dep_write_after_read != NULL );
    assert(fused_computations.size()>0);

    computation * first_computation = fused_computations[0]  ;
    
    DEBUG(3, tiramisu::str_dump(" skewing solving for : " + var_outer.get_name() + 
        " and "+var2.get_name() +" and "+var_inner.get_name()));

    std::vector<std::string> original_loop_level_names = first_computation->get_loop_level_names();

    std::vector<int> dimensions =
        first_computation->get_loop_level_numbers_from_dimension_names(
            {var_outer.get_name(), var2.get_name(), var_inner.get_name()});

    first_computation->check_dimensions_validity(dimensions);

    int out_dim = tiramisu::loop_level_into_dynamic_dimension(dimensions[0]);
    int mid_dim = tiramisu::loop_level_into_dynamic_dimension(dimensions[1]);
    int inner_dim = tiramisu::loop_level_into_dynamic_dimension(dimensions[2]);

    assert(mid_dim == out_dim + 2);
    assert(inner_dim == mid_dim + 2);

    // create a vector of the iterators
    std::vector<tiramisu::var> target_variables;
    target_variables.push_back(var_outer);
    target_variables.push_back(var2);
    target_variables.push_back(var_inner);


    // backup all schedules of computations
    std::vector<isl_map*> schedules_backups;

    for(auto const& computation : fused_computations)
    {
        schedules_backups.push_back(isl_map_copy(computation->get_schedule()));
    }

    bool solvable_solutions = true;
    // compute the 3d skewings for 3 dimensions

    // results (would be filled if solutions were found)
    std::vector<std::tuple<int,int,int,int,int,int,int,int,int>> outer_params;
    std::vector<std::tuple<int,int,int,int,int,int,int,int,int>> inner_params;
    std::vector<std::tuple<int,int,int,int,int,int,int,int,int>> tiling_params;
    
    std::string parameters = "{[i0,i1,i2]->[i0,i1,i2]}";
    isl_basic_map * parameters_map = isl_basic_map_read_from_str(this->get_isl_ctx(),parameters.c_str());

    for (int j_inner = target_variables.size() - 1; solvable_solutions && (j_inner >= 1); j_inner --)
    {
        for (int i_outer = j_inner - 1; i_outer >= 0; i_outer --)
        {
            auto result_tmp = this->skewing_local_solver_positive(
                fused_computations, target_variables[i_outer], target_variables[j_inner], 1);
            // special case for the last iteration
            if ((i_outer == 0) && (j_inner == 1)){
                // generate inner parallelism or outer parralism
                if (std::get<0>(result_tmp).size() > 0){
                    isl_basic_map * outer_map = NULL;
                    
                    auto identity_values = std::get<0>(result_tmp)[0];
                    DEBUG(3, tiramisu::str_dump(" outer skewing iteration is : " + std::to_string(std::get<0>(identity_values)) + 
                        "," + std::to_string(std::get<1>(identity_values)) + "," +std::to_string(std::get<2>(identity_values)) +
                        "," + std::to_string(std::get<3>(identity_values)) ));
                    
                    int index_different = 0;
                    while((index_different == i_outer) || (index_different == j_inner))
                    {
                        index_different++;
                    }
                    assert(index_different < 4);

                    std::string transformation_in = "{[i0,i1,i2]->[i_0,i_1,i_2] : " 
                        " i_" + std::to_string(index_different) + " = i" + std::to_string(index_different) +" and "
                        "i_" + std::to_string(i_outer) + " = i"+std::to_string(i_outer)+"*" + std::to_string(std::get<0>(identity_values)) +
                            " + i" + std::to_string(j_inner)+"*" + std::to_string(std::get<1>(identity_values)) +" and "
                        "i_" + std::to_string(j_inner) + " = i"+std::to_string(i_outer)+"*" + std::to_string(std::get<2>(identity_values)) +
                            " + i" + std::to_string(j_inner)+"*" + std::to_string(std::get<3>(identity_values)) +" } ";
                    isl_basic_map * transformation = isl_basic_map_read_from_str(this->get_isl_ctx(),transformation_in.c_str());

                    DEBUG(10, tiramisu::str_dump(" Current corresponding transformation :", isl_basic_map_to_str(transformation)));
                    outer_map = isl_basic_map_apply_range(isl_basic_map_copy(parameters_map), transformation);
                    DEBUG(10, tiramisu::str_dump(" outer-parallelism transformation :", isl_basic_map_to_str(outer_map))); 

                    outer_params.push_back(this->extract_3d_skewing_params(outer_map));         

                }
                if (std::get<1>(result_tmp).size() > 0){
                    isl_basic_map * inner_map = NULL;
                    
                    auto identity_values = std::get<1>(result_tmp)[0];
                    DEBUG(3, tiramisu::str_dump(" inner skewing iteration is : " + std::to_string(std::get<0>(identity_values)) + 
                        "," + std::to_string(std::get<1>(identity_values)) + "," +std::to_string(std::get<2>(identity_values)) +
                        "," + std::to_string(std::get<3>(identity_values)) ));
                    
                    int index_different = 0;
                    while((index_different == i_outer) || (index_different == j_inner))
                    {
                        index_different++;
                    }
                    assert(index_different < 4);

                    std::string transformation_in = "{[i0,i1,i2]->[i_0,i_1,i_2] : " 
                        " i_" + std::to_string(index_different) + " = i" + std::to_string(index_different) +" and "
                        "i_" + std::to_string(i_outer) + " = i"+std::to_string(i_outer)+"*" + std::to_string(std::get<0>(identity_values)) +
                            " + i" + std::to_string(j_inner)+"*" + std::to_string(std::get<1>(identity_values)) +" and "
                        "i_" + std::to_string(j_inner) + " = i"+std::to_string(i_outer)+"*" + std::to_string(std::get<2>(identity_values)) +
                            " + i" + std::to_string(j_inner)+"*" + std::to_string(std::get<3>(identity_values)) +" } ";
                    isl_basic_map * transformation = isl_basic_map_read_from_str(this->get_isl_ctx(),transformation_in.c_str());

                    DEBUG(10, tiramisu::str_dump(" Current corresponding transformation :", isl_basic_map_to_str(transformation)));
                    inner_map = isl_basic_map_apply_range(isl_basic_map_copy(parameters_map), transformation);
                    DEBUG(10, tiramisu::str_dump(" inner-parallelism transformation :", isl_basic_map_to_str(inner_map)));

                    inner_params.push_back(this->extract_3d_skewing_params(inner_map));        
                    
                }
                if (std::get<2>(result_tmp).size() > 0){
                    isl_basic_map * tiling_map = NULL;
                    
                    auto identity_values = std::get<2>(result_tmp)[0];
                    DEBUG(3, tiramisu::str_dump(" positive-dependencies skewing iteration is : " + std::to_string(std::get<0>(identity_values)) + 
                        "," + std::to_string(std::get<1>(identity_values)) + "," +std::to_string(std::get<2>(identity_values)) +
                        "," + std::to_string(std::get<3>(identity_values)) ));
                    
                    int index_different = 0;
                    while((index_different == i_outer) || (index_different == j_inner))
                    {
                        index_different++;
                    }
                    assert(index_different < 4);

                    std::string transformation_in = "{[i0,i1,i2]->[i_0,i_1,i_2] : " 
                        " i_" + std::to_string(index_different) + " = i" + std::to_string(index_different) +" and "
                        "i_" + std::to_string(i_outer) + " = i"+std::to_string(i_outer)+"*" + std::to_string(std::get<0>(identity_values)) +
                            " + i" + std::to_string(j_inner)+"*" + std::to_string(std::get<1>(identity_values)) +" and "
                        "i_" + std::to_string(j_inner) + " = i"+std::to_string(i_outer)+"*" + std::to_string(std::get<2>(identity_values)) +
                            " + i" + std::to_string(j_inner)+"*" + std::to_string(std::get<3>(identity_values)) +" } ";
                    isl_basic_map * transformation = isl_basic_map_read_from_str(this->get_isl_ctx(),transformation_in.c_str());

                    DEBUG(10, tiramisu::str_dump(" Current corresponding transformation :", isl_basic_map_to_str(transformation)));
                    tiling_map = isl_basic_map_apply_range(isl_basic_map_copy(parameters_map), transformation);
                    DEBUG(10, tiramisu::str_dump(" positive-dependencies transformation :", isl_basic_map_to_str(tiling_map)));     

                    tiling_params.push_back(this->extract_3d_skewing_params(tiling_map));      
                    
                }

            }
            else
            {
                auto identity_vector = std::get<2>(result_tmp);
                if (identity_vector.size() == 0){
                    DEBUG(10, tiramisu::str_dump(" unable to find 3d skewing solution ")); 
                    solvable_solutions = false;
                    break;
                }
                auto identity_values = identity_vector[0];
                DEBUG(3, tiramisu::str_dump(" skewing iteration is : " + std::to_string(std::get<0>(identity_values)) + 
                    "," + std::to_string(std::get<1>(identity_values)) + "," +std::to_string(std::get<2>(identity_values)) +
                    "," + std::to_string(std::get<3>(identity_values)) ));
                
                int index_different = 0;
                while((index_different == i_outer) || (index_different == j_inner))
                {
                    index_different++;
                }
                assert(index_different < 4);

                std::string transformation_in = "{[i0,i1,i2]->[i_0,i_1,i_2] : " 
                    " i_" + std::to_string(index_different) + " = i" + std::to_string(index_different) +" and "
                    "i_" + std::to_string(i_outer) + " = i"+std::to_string(i_outer)+"*" + std::to_string(std::get<0>(identity_values)) +
                        " + i" + std::to_string(j_inner)+"*" + std::to_string(std::get<1>(identity_values)) +" and "
                    "i_" + std::to_string(j_inner) + " = i"+std::to_string(i_outer)+"*" + std::to_string(std::get<2>(identity_values)) +
                        " + i" + std::to_string(j_inner)+"*" + std::to_string(std::get<3>(identity_values)) +" } ";
                isl_basic_map * transformation = isl_basic_map_read_from_str(this->get_isl_ctx(),transformation_in.c_str());

                DEBUG(10, tiramisu::str_dump(" Current corresponding transformation :", isl_basic_map_to_str(transformation)));
                parameters_map = isl_basic_map_apply_range(parameters_map, transformation);
                DEBUG(10, tiramisu::str_dump(" Current overall transformation :", isl_basic_map_to_str(parameters_map)));

                // apply the skewing and rename the variables

                tiramisu::var var1(std::string(target_variables[i_outer].get_name())+"_");
                tiramisu::var var2(std::string(target_variables[j_inner].get_name())+"_");
                
                for (auto& computation : fused_computations)
                {
                    computation->skew(target_variables[i_outer],target_variables[j_inner],
                        std::get<0>(identity_values),
                        std::get<1>(identity_values),
                        std::get<2>(identity_values),
                        std::get<3>(identity_values),
                        var1, var2);
                }
            target_variables[i_outer] = var1;
            target_variables[j_inner] = var2;
            }
        }
    }

    // Restore all computation schedules
    assert(fused_computations.size() == schedules_backups.size());
    int index = 0;
    for(auto const& computation : fused_computations)
    {   
        isl_map_free(computation->get_schedule());
        computation->set_schedule(schedules_backups[index]);
        index ++;
    }

    isl_basic_map_free(parameters_map);
    DEBUG_INDENT(-4);

    return std::make_tuple(outer_params,inner_params,tiling_params);
}

/**
 * Method made to extract coefficents from isl_basic_map
 * For instance {[i0,i1,i2]->[2i0+i2,i1,i2]} for first dimension i0 return {2,0,1}
 * **/
std::vector<int> tiramisu::function::extract_transformation_coeffcients(isl_basic_map * transformation, int position)
{
    int input_dim = isl_basic_map_dim(transformation, isl_dim_in);
    int output_dim = isl_basic_map_dim(transformation, isl_dim_out);

    assert(input_dim == output_dim);
    assert(position < input_dim);

    std::string set_extractor = "{[";

    for (int i=0; i < input_dim; i++){
        if (i == position){
            set_extractor += "1";
        }
        else{
            set_extractor += "0";
        }

        if (i < (input_dim -1)){
            set_extractor += ",";
        }
    }
    set_extractor += "]}";

    isl_basic_set * set = isl_basic_set_read_from_str(this->get_isl_ctx(), set_extractor.c_str());

    set = isl_basic_set_apply(set, isl_basic_map_copy(transformation));

    std::vector<int> result;

    for (int i=0; i < input_dim; i++)
    {
        isl_val * value_isl = isl_basic_set_dim_max_val( isl_basic_set_copy(set),i);

        int value = isl_val_get_d(value_isl);

        result.push_back(value);

    }
    return result;
}

std::tuple<int,int,int,int,int,int,int,int,int> tiramisu::function::extract_3d_skewing_params(
    isl_basic_map * transformation)
{   
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int input_dim = isl_basic_map_dim(transformation, isl_dim_in);
    int output_dim = isl_basic_map_dim(transformation, isl_dim_out);
    assert(input_dim == output_dim);
    assert(input_dim == 3);

    std::vector<int> v1 = this->extract_transformation_coeffcients(transformation, 0);
    std::vector<int> v2 = this->extract_transformation_coeffcients(transformation, 1);
    std::vector<int> v3 = this->extract_transformation_coeffcients(transformation, 2);

    assert(v1.size() == 3);
    assert(v2.size() == 3);
    assert(v3.size() == 3);

    DEBUG(10, tiramisu::str_dump(" Extracted parameters are : " + 
    std::to_string(v1[0])+ " "+ std::to_string(v2[0]) +" "+ std::to_string(v3[0]) +" | "
    + std::to_string(v1[1])+ " "+ std::to_string(v2[1]) +" "+ std::to_string(v3[1]) +" | "
    + std::to_string(v1[2])+ " "+ std::to_string(v2[2]) +" "+ std::to_string(v3[2])
    ));

    DEBUG_INDENT(-4);
    
    return std::make_tuple(v1[0],v2[0],v3[0],
                           v1[1],v2[1],v3[1],
                           v1[2],v2[2],v3[2]);
}


std::vector<int> function::get_potentiel_vectorizable_loop_level(std::vector<tiramisu::computation *> involved_computations)
{
    DEBUG_INDENT(4);
    std::vector<int> result;

    for(auto const& computation:involved_computations)
    {
        int val = computation->get_potentiel_vectorizable_loop_level();
        if(val != -1)
        {
            result.push_back(val);
        }
    }


    DEBUG_INDENT(-4);
    
    return result;


}



}
