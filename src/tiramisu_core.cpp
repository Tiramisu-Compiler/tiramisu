#include <isl/ctx.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/id.h>
#include <isl/constraint.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <cmath>  
#include <regex>

#ifdef _WIN32
#include <iso646.h>
#endif

namespace tiramisu
{
int send::next_msg_tag = 0;
std::set<int> tiramisu::xfer_prop::xfer_prop_ids;
// Used for the generation of new variable names.
int id_counter = 0;
static int next_dim_name = 0;

bool global::auto_data_mapping = false;
primitive_t global::loop_iterator_type = p_int32;
function *global::implicit_fct;
std::unordered_map<std::string, var> var::declared_vars;
const var computation::root = var("root");

std::string generate_new_variable_name();
void project_out_static_dimensions(isl_set*& set);

tiramisu::expr traverse_expr_and_replace_non_affine_accesses(tiramisu::computation *comp,
                                                             const tiramisu::expr &exp);

tiramisu::expr tiramisu_expr_from_isl_ast_expr(isl_ast_expr *isl_expr);

/**
  * Add a dimension to the range of a map in the specified position.
  * Assume that the name of the new dimension is equal to the name of the corresponding
  * dimension in the domain of the map.
  * Add a constraint that indicates that the added dim is equal to a constant.
  */
isl_map *isl_map_add_dim_and_eq_constraint(isl_map *map, int dim_pos, int constant);

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


//********************************************************

void init(std::string fct_name)
{
    function *fct = new function(fct_name);
    global::set_implicit_function(fct);

    // Do all the rest of the initialization.
    init();
}


void init()
{
    // Set default tiramisu options.
    global::set_default_tiramisu_options();
}

void codegen(const std::vector<tiramisu::buffer *> &arguments, const std::string obj_filename, const bool gen_cuda_stmt)
{
    function *fct = global::get_implicit_function();
    fct->codegen(arguments, obj_filename, gen_cuda_stmt);
}

void codegen_select_schedule_number(int schedule_number,
                      const std::vector<tiramisu::buffer *> &arguments, const std::string obj_filename, const bool gen_cuda_stmt )
                      {
                            function *fct = global::get_implicit_function();
                             fct->codegen_select_schedule_number(schedule_number,arguments, obj_filename, gen_cuda_stmt);
                      }

void codegen_write_potential_schedules(std::string& path_name,
                      const std::vector<tiramisu::buffer *> &arguments, const std::string obj_filename, const bool gen_cuda_stmt ){
                          function *fct = global::get_implicit_function();
                        fct->codegen_write_potential_schedules(path_name,arguments, obj_filename, gen_cuda_stmt);
                      }

//********************************************************

isl_set *tiramisu::computation::get_iteration_domains_of_all_definitions()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::string name = this->get_name();
    assert(name.size() > 0);
    isl_set *result = NULL;
    isl_space *space = NULL;

    assert(isl_set_is_empty(this->get_iteration_domain()) == isl_bool_false);
    space = isl_set_get_space(this->get_iteration_domain());
    assert(space != NULL);
    result = isl_set_empty(space);

    std::vector<tiramisu::computation *> computations =
        this->get_function()->get_computation_by_name(name);

    for (auto c : computations)
    {
        if (c->should_schedule_this_computation())
        {
            isl_set *c_iter_space = isl_set_copy(c->get_iteration_domain());
            result = isl_set_union(c_iter_space, result);
        }
    }

    DEBUG_INDENT(-4);

    return result;
}

constant* function::get_invariant_by_name(std::string name) const
{
    assert(!name.empty());

    DEBUG(10, tiramisu::str_dump("Searching invariant " + name));

    tiramisu::constant *res;
    tiramisu::constant *comp;

    for (int i = 0; i < this->get_invariants().size(); i++)
    {
	comp = (constant *) &(this->get_invariants()[i]);
        if (name == comp->get_name())
        {
            res = comp;
        }
    }

    if (res == NULL)
    {
        DEBUG(10, tiramisu::str_dump("Invariant not found."));
    }
    else
    {
        DEBUG(10, tiramisu::str_dump("Invariant found."));
    }

    return res;
}

bool tiramisu::computation::has_multiple_definitions()
{
    bool is_update = false;

    std::string name = this->get_name();
    assert(name.size() > 0);

    std::vector<tiramisu::computation *> computations =
        this->get_function()->get_computation_by_name(name);

    if (computations.size() > 1)
    {
        is_update = true;
    }

    if (this->get_updates().size() > 1)
        is_update = true;

    if (this->get_first_definition() != NULL)
        if (this->get_first_definition()->get_updates().size() > 1)
        is_update = true;

    return is_update;
}

tiramisu::computation *computation::get_root_of_definition_tree()
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    DEBUG(10, tiramisu::str_dump("Getting the root of the definition tree of this computation: " + this->get_name()));
    DEBUG(10, tiramisu::str_dump("This computation has an ID = " + std::to_string(this->definition_ID)));

    tiramisu::computation *root = this;

    // We know that any child definition has an ID > 0 (since only the root has
    // an ID == 0). So we keep traversing the tree up from the leaf to the root
    // until we find the root. The root is identified by ID == 0.
    while (root->definition_ID > 0)
    {
        root = root->get_first_definition();
        DEBUG(10, tiramisu::str_dump("This computation is: " + root->get_name()));
        DEBUG(10, tiramisu::str_dump("This computation has an ID = " + std::to_string(root->definition_ID)));
    }

    DEBUG(10, tiramisu::str_dump("The root of the tree of updates is: " + root->get_name()));

    DEBUG_INDENT(-4);

    return root;
}

void tiramisu::computation::add_definitions(std::string iteration_domain_str,
        tiramisu::expr e,
        bool schedule_this_computation, tiramisu::primitive_t t,
        tiramisu::function *fct)
{
    tiramisu::computation *new_c = new tiramisu::computation(iteration_domain_str, e,
                                                      schedule_this_computation, t, fct);
    new_c->is_first = false;
    new_c->first_definition = this;
    new_c->is_let = this->is_let;
    new_c->definition_ID = this->definitions_number;
    this->definitions_number++;

    if (new_c->get_expr().is_equal(this->get_expr()))
    {
        // Copy the associated let statements to the new definition.
        new_c->associated_let_stmts = this->associated_let_stmts;
    }

    this->updates.push_back(new_c);
}

/*void tiramisu::compututationn::operator=(tiramisu::expr e)
{
    assert(e.is_defined());

    this->set_expression(e);
    this->data_type = e.get_data_type();
}*/


void tiramisu::computation::rename_computation(std::string new_name)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(this->get_function()->get_computation_by_name(new_name).empty());

    std::string old_name = this->get_name();

    this->set_name(new_name);

    // Rename the iteration domain.
    isl_set *dom = this->get_iteration_domain();
    assert(dom != NULL);
    dom = isl_set_set_tuple_name(dom, new_name.c_str());
    DEBUG(10, tiramisu::str_dump("Setting the iteration domain to ", isl_set_to_str(dom)));
    this->set_iteration_domain(dom);

    // Rename the time-space domain (if it is not NULL)
    dom = this->get_time_processor_domain();
    if (dom != NULL)
    {
        dom = isl_set_set_tuple_name(dom, new_name.c_str());
        DEBUG(10, tiramisu::str_dump("Setting the time-space domain to ", isl_set_to_str(dom)));
        this->time_processor_domain = dom;
    }

    if (this->get_access_relation() != NULL)
    {
        // Rename the access relation of the computation.
        isl_map *access = this->get_access_relation();
        access = isl_map_set_tuple_name(access, isl_dim_in, new_name.c_str());
        DEBUG(10, tiramisu::str_dump("Setting the access relation to ", isl_map_to_str(access)));
        this->set_access(access);
    }

    // Rename the schedule
    isl_map *sched = this->get_schedule();
    sched = isl_map_set_tuple_name(sched, isl_dim_in, new_name.c_str());
    sched = isl_map_set_tuple_name(sched, isl_dim_out, new_name.c_str());
    DEBUG(10, tiramisu::str_dump("Setting the schedule relation to ", isl_map_to_str(sched)));
    this->set_schedule(sched);

    // Rename parallel, unroll, vectorize and gpu vectors
    for (auto &pd : this->get_function()->unroll_dimensions)
        if (std::get<0>(pd) == old_name)
            std::get<0>(pd) = new_name;
    for (auto &pd : this->get_function()->parallel_dimensions)
        if (pd.first == old_name)
            pd.first = new_name;
    for (auto &pd : this->get_function()->gpu_block_dimensions)
        if (pd.first == old_name)
            pd.first = new_name;
    for (auto &pd : this->get_function()->gpu_thread_dimensions)
        if (pd.first == old_name)
            pd.first = new_name;
    for (auto &pd : this->get_function()->vector_dimensions)
        if (std::get<0>(pd) == old_name)
            std::get<0>(pd) = new_name;

    DEBUG_INDENT(-4);
}

std::string generate_new_variable_name()
{
    return "t" + std::to_string(id_counter++);
}

std::string generate_new_computation_name()
{
    return "C" + std::to_string(id_counter++);
}

void computation::tag_gpu_level(tiramisu::var L0_var, tiramisu::var L1_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    std::vector<int> dimensions =
    this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];

    this->tag_gpu_level(L0, L1);

    DEBUG_INDENT(-4);
}

void computation::tag_gpu_level(tiramisu::var L0_var, tiramisu::var L1_var,
        tiramisu::var L2_var, tiramisu::var L3_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(L2_var.get_name().length() > 0);
    assert(L3_var.get_name().length() > 0);

    std::vector<int> dimensions =
    this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name(),
                                                       L2_var.get_name(), L3_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];
    int L2 = dimensions[2];
    int L3 = dimensions[3];

    this->tag_gpu_level(L0, L1, L2, L3);

    DEBUG_INDENT(-4);
}

void computation::tag_gpu_level(tiramisu::var L0_var, tiramisu::var L1_var,
        tiramisu::var L2_var, tiramisu::var L3_var,
        tiramisu::var L4_var, tiramisu::var L5_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(L2_var.get_name().length() > 0);
    assert(L3_var.get_name().length() > 0);
    assert(L4_var.get_name().length() > 0);
    assert(L5_var.get_name().length() > 0);

    std::vector<int> dimensions =
    this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name(),
                                                       L2_var.get_name(), L3_var.get_name(),
                                                       L4_var.get_name(), L5_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];
    int L2 = dimensions[2];
    int L3 = dimensions[3];
    int L4 = dimensions[4];
    int L5 = dimensions[5];

    this->tag_gpu_level(L0, L1, L2, L3, L4, L5);

    DEBUG_INDENT(-4);
}

/**
  * Methods for the computation class.
  */
void tiramisu::computation::parallelize(tiramisu::var par_dim_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(par_dim_var.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({par_dim_var.get_name()});
    this->check_dimensions_validity(dimensions);

    int par_dim = dimensions[0];
    this->tag_parallel_level(par_dim);

    DEBUG_INDENT(-4);
}

/*
the dependency (1,1) vectors means [i,j] is used bt [i+1,j+1] which means this dep is scheduled before = true 
*/

bool is_dependency_scheduled_before_self(isl_map * dependency,isl_space * space )
{
   

    isl_map * inf_map = isl_map_lex_lt(space) ;

    

    isl_map * intersection = isl_map_intersect(inf_map,dependency) ;

    if (isl_map_is_empty(intersection))
        return false ;
    else
    {
        return true ;
    }

}

bool tiramisu::computation::parallelization_is_legal(tiramisu::var par_dim_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(par_dim_var.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);
    assert(this->get_function()->dep_read_after_write != NULL ) ;
    assert(this->get_function()->dep_write_after_write != NULL ) ;
    assert(this->get_function()->dep_write_after_read != NULL ) ;

    
    DEBUG(3, tiramisu::str_dump(" var parallelization check is : "+par_dim_var.get_name()));

    
     std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({par_dim_var.get_name()});

    this->check_dimensions_validity(dimensions);
    

    int par_dim = tiramisu::loop_level_into_dynamic_dimension(dimensions[0]);


    DEBUG(3, tiramisu::str_dump(" par dim number is : "+std::to_string(par_dim)));


     // get the schedules to timestamps 
    isl_map * schedule = isl_map_copy(this->schedule) ;

    std::string empty_time  = "" ;

    schedule = isl_map_set_tuple_name(schedule,isl_dim_out,empty_time.c_str()) ;

    DEBUG(3, tiramisu::str_dump(" the schedule to time stamp is  : "+std::string(isl_map_to_str(schedule))));

    // extracting deps

     isl_union_map * read_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_read_after_write)) ;

    isl_union_map * write_after_read_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_write_after_read)) ;

    isl_union_map * write_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_write_after_write)) ;



    isl_union_map * all_deps = isl_union_map_union(
        read_after_write_dep,
        write_after_read_dep
        ) ;

    // all the deps in 1 union map
    all_deps = isl_union_map_union(all_deps,write_after_write_dep) ;


    DEBUG(3, tiramisu::str_dump(" all the dependencies are  : "+std::string(isl_union_map_to_str(all_deps))));

    // all current schedules in 1 union map
    isl_union_map * schedules = isl_union_map_from_map(schedule) ;

    DEBUG(3, tiramisu::str_dump(" all the used schedules are  : "+std::string(isl_union_map_to_str(schedules))));

    // application to discard unused dep & modelize them in thier time space

    all_deps = isl_union_map_apply_range(all_deps,isl_union_map_copy( schedules )) ;

    all_deps = isl_union_map_apply_domain(all_deps,isl_union_map_copy( schedules )) ;

    DEBUG(3, tiramisu::str_dump(" all the used dependencies union map are  : "+std::string(isl_union_map_to_str(all_deps))));

    isl_map * equation_map = isl_map_from_union_map(all_deps) ;

    DEBUG(3, tiramisu::str_dump(" all the used dependencies after transformed to map are  : "+std::string(isl_map_to_str(equation_map))));

    bool over_all_legality = false ;

    /*
        equate adds restriction that both elements are equal
        we suppose that legality is checked elsewhere ; so we need to check for loop caried dependencies only
        if adding equation of == between input set & output set of map for a dimention strictly before the parallel one is empty means : 
            dep is not a carried one for the parallel loop lvl

        else 
            if all previous equations added does not make the map empty then the last possiblity is:
                dep is within the same loop iteration; then parallel is true ( true if equate doesnt make the map empty)
                else it's false
                
    */
    for(int i=0;i<par_dim;i++)
    {
        equation_map = isl_map_equate(equation_map,isl_dim_in,i,isl_dim_out,i) ;

        if(isl_map_is_empty(equation_map))
        {
            over_all_legality = true ;
            DEBUG(10, tiramisu::str_dump(" parallalization is legal "));
            break ;
        }
    
    }

    if(!over_all_legality)
    {
        equation_map = isl_map_equate(equation_map,isl_dim_in,par_dim,isl_dim_out,par_dim) ;

        if(isl_map_is_empty(equation_map))
        {
            over_all_legality = false ;
            DEBUG(3, tiramisu::str_dump(" parallalization is illegal "));
        }
        else{
            over_all_legality = true ;
            DEBUG(3, tiramisu::str_dump(" parallalization is legal "));
        }
    }


    DEBUG_INDENT(-4); 
    return over_all_legality ;
  

}

bool tiramisu::computation::parallelization_is_legal(tiramisu::var par_dim_var,std::vector<tiramisu::computation *> fuze_statments)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(par_dim_var.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);
    assert(this->get_function()->dep_read_after_write != NULL ) ;
    assert(this->get_function()->dep_write_after_write != NULL ) ;
    assert(this->get_function()->dep_write_after_read != NULL ) ;

    
    DEBUG(3, tiramisu::str_dump(" var parallelization check is : "+par_dim_var.get_name()));

    
     std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({par_dim_var.get_name()});

    this->check_dimensions_validity(dimensions);
    

    int par_dim = tiramisu::loop_level_into_dynamic_dimension(dimensions[0]);
    
    DEBUG(3, tiramisu::str_dump(" par dim number is : "+std::to_string(par_dim)));



     // get the schedules to timestamps 
    isl_map * schedule = isl_map_copy(this->schedule) ;

    std::string empty_time  = "" ;

    schedule = isl_map_set_tuple_name(schedule,isl_dim_out,empty_time.c_str()) ;

    DEBUG(3, tiramisu::str_dump(" the schedule to time stamp is  : "+std::string(isl_map_to_str(schedule))));

    // extracting deps

     isl_union_map * read_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_read_after_write)) ;

    isl_union_map * write_after_read_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_write_after_read)) ;

    isl_union_map * write_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_write_after_write)) ;



    isl_union_map * all_deps = isl_union_map_union(
        read_after_write_dep,
        write_after_read_dep
        ) ;

    // all the deps in 1 union map
    all_deps = isl_union_map_union(all_deps,write_after_write_dep) ;


    DEBUG(3, tiramisu::str_dump(" all the dependencies are  : "+std::string(isl_union_map_to_str(all_deps))));

    // all current schedules in 1 union map
    isl_union_map * schedules = isl_union_map_from_map(schedule) ;


    //all schedules adding 

    isl_map * sche = NULL ;

    for( auto& computation: fuze_statments)
    {
        sche = isl_map_copy(computation->get_schedule()) ;

        sche = isl_map_set_tuple_name(sche,isl_dim_out,empty_time.c_str()) ;

        schedules = isl_union_map_union(schedules,isl_union_map_from_map(sche)) ;

    }


    DEBUG(3, tiramisu::str_dump(" all the used schedules are  : "+std::string(isl_union_map_to_str(schedules))));

    // application to discard unused dep & modelize them in thier time space

    all_deps = isl_union_map_apply_range(all_deps,isl_union_map_copy( schedules )) ;

    all_deps = isl_union_map_apply_domain(all_deps,isl_union_map_copy( schedules )) ;

    DEBUG(3, tiramisu::str_dump(" all the used dependencies union map are  : "+std::string(isl_union_map_to_str(all_deps))));

    isl_map * equation_map = isl_map_from_union_map(all_deps) ;

    DEBUG(3, tiramisu::str_dump(" all the used dependencies after transformed to map are  : "+std::string(isl_map_to_str(equation_map))));

    bool over_all_legality = false ;

    /*
        equate adds restriction that both elements are equal
        we suppose that legality is checked elsewhere ; so we need to check for loop caried dependencies only
        if adding equation of == between input set & output set of map for a dimention strictly before the parallel one is empty means : 
            dep is not a carried one for the parallel loop lvl

        else 
            if all previous equations added does not make the map empty then the last possiblity is:
                dep is within the same loop iteration; then parallel is true ( true if equate doesnt make the map empty)
                else it's false
                
    */
    for(int i=0;i<par_dim;i++)
    {
        equation_map = isl_map_equate(equation_map,isl_dim_in,i,isl_dim_out,i) ;

        if(isl_map_is_empty(equation_map))
        {
            over_all_legality = true ;
            DEBUG(10, tiramisu::str_dump(" parallalization is legal "));
            break ;
        }
    
    }

    if(!over_all_legality)
    {
        equation_map = isl_map_equate(equation_map,isl_dim_in,par_dim,isl_dim_out,par_dim) ;

        if(isl_map_is_empty(equation_map))
        {
            over_all_legality = false ;
            DEBUG(3, tiramisu::str_dump(" parallalization is illegal "));
        }
        else{
            over_all_legality = true ;
            DEBUG(3, tiramisu::str_dump(" parallalization is legal "));
        }
    }


    DEBUG_INDENT(-4); 
    return over_all_legality ;
}


bool tiramisu::computation::unrolling_is_legal(tiramisu::var l)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(l.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

     std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({l.get_name()});

    this->check_dimensions_validity(dimensions);

    std::string str_schedule(isl_map_to_str(this->schedule)) ;
    
    DEBUG(10, tiramisu::str_dump(" schedule is "+str_schedule));

    isl_set * time_set = isl_map_range(isl_map_copy(this->schedule)) ;

    int target_dim = tiramisu::loop_level_into_dynamic_dimension(dimensions[0]);

    DEBUG(3, tiramisu::str_dump(" target dim number is : "+std::to_string(target_dim)));

    DEBUG(3, tiramisu::str_dump(" the time set is : "+std::string(isl_set_to_str(time_set))));

    unsigned int n_dim = isl_set_n_dim(time_set) ;

    
    std::string set_n = "[n0";
    std::string set_var = "->{"+this->get_name()+"[" ;

    int iteration = 1 ;
    int dimention_index = 0;
    bool var_found = false ;


    for(unsigned int i=0;i<n_dim;i++)
    {
        if(!var_found)
         {
            set_n +=",n"+std::to_string(iteration) ;

            if(i==target_dim)
            {
                    var_found = true ;
                    set_n += "]" ;
                    set_var += l.get_name() ;
            }
            else{
                set_var +=   "n"+std::to_string(iteration)+"," ;
                dimention_index ++ ;
            }
         }
         else{
             set_var += ",t"+std::to_string(i) ;
         }

         iteration++ ;

    }    

    isl_map * normal_schedule = isl_map_copy(this->schedule) ;

    isl_map * reverse = isl_map_reverse(isl_map_copy(normal_schedule)) ;

    std::string set_complete = set_n + set_var +"]}" ;
    
    DEBUG(10, tiramisu::str_dump(" constructed set to use: "+set_complete));

    isl_set * reversed_set = isl_set_apply(
         isl_set_read_from_str(this->get_ctx(),set_complete.c_str()),
         reverse
         );

    DEBUG(10, tiramisu::str_dump(" to initial set  "+std::string(isl_set_to_str(reversed_set))));

    isl_set * normal_set = isl_set_apply(reversed_set,normal_schedule) ;


    DEBUG(10, tiramisu::str_dump(" dimention number is : "+std::to_string(dimention_index)));

    DEBUG(3, tiramisu::str_dump(" set with applied constraintes : "+std::string(isl_set_to_str(normal_set) )));


    
    isl_pw_aff * max = isl_set_dim_max(isl_set_copy(normal_set),dimention_index) ;
    isl_pw_aff * min = isl_set_dim_min(isl_set_copy(normal_set),dimention_index) ;

    
    // count the number of element that forms the max & min for the specified var
  
    std::cout<<isl_pw_aff_to_str(max) ;

    DEBUG(3, tiramisu::str_dump(" max is  : "+std::string(isl_pw_aff_to_str(max) )));
    DEBUG(3, tiramisu::str_dump(" min is  : "+std::string(isl_pw_aff_to_str(min))));


    int n_piece_max = isl_pw_aff_n_piece(max) ;

    int n_piece_min =  isl_pw_aff_n_piece(min) ;
 
    if((n_piece_max==1)&&(n_piece_min==1))
    {
        DEBUG(3, tiramisu::str_dump(" max & min are both cst urolling legal")) ;
    }
    else{
        DEBUG(3, tiramisu::str_dump(" max & min are not cst unrolling impossible ")) ;
    }

     DEBUG_INDENT(-4); 
    return ((n_piece_max==1)&&(n_piece_min==1)) ;


}

bool tiramisu::computation::vectorization_is_legal(tiramisu::var l)
{

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(l.get_name().length() > 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    DEBUG(3, tiramisu::str_dump(" Vectorization is legal if both parallelization and unrolling are legal ")) ;
    
    bool validity = false; 

    validity = this->unrolling_is_legal(l)
                        &&this->parallelization_is_legal(l) ;

    if(validity)
    {
        DEBUG(3, tiramisu::str_dump(" Vectorization is legal ")) ;
    }
    else{
        DEBUG(3, tiramisu::str_dump(" Vectorization is illegal ")) ;
    }

    DEBUG_INDENT(-4); 
    return validity;
    

    
}


bool tiramisu::computation::applied_schedule_is_legal(){
    
    
    assert(this->schedule!=NULL) ;
    assert(this->get_function()->dep_read_after_write!=NULL) ;
    
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    //isl_union_map * deps = this->get_function()->compute_dep_graph();

    isl_union_map * read_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_read_after_write)) ;

    isl_union_map * write_after_read_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_write_after_read)) ;

    isl_union_map * write_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_write_after_write)) ;


    
    int nb = this->number_of_dims ;
    std::string space_map = "{" + this->get_name() +"[" ;

    for(int i=0;i<nb;i++){
        space_map += "t"+std::to_string(i) ;

        if(i!=nb-1){
                space_map += "," ;
        }
    }

    space_map +=  "]->"+this->get_name()+"[" ;
    
    for(int i=0;i<nb;i++){
        space_map += "tt"+std::to_string(i) ;

        if(i!=nb-1){
                space_map += "," ;
        }
    }
    space_map +="]}" ;

    isl_map * tyy =  isl_map_read_from_str(this->get_ctx(),space_map.c_str()) ;

    isl_space * space = isl_map_get_space(tyy) ;

    isl_map * my_map1 = isl_union_map_extract_map(read_after_write_dep,isl_space_copy(space)) ;

    isl_map * my_map2 = isl_union_map_extract_map(write_after_read_dep,isl_space_copy(space)) ;

    isl_map * my_map3 = isl_union_map_extract_map(write_after_write_dep,isl_space_copy(space)) ;


    DEBUG(3, tiramisu::str_dump(" read after write deps are : "+std::string(isl_map_to_str(my_map1)))) ;


    DEBUG(3, tiramisu::str_dump(" write after read deps are : "+std::string(isl_map_to_str(my_map2)))) ;

    std::vector<isl_basic_map *> all_basic_maps ;
    

    auto f = [](isl_basic_map * bmap,void * user) { 

        std::vector<isl_basic_map *>& myName = *reinterpret_cast<std::vector<isl_basic_map*>*>(user) ;
     
        myName.push_back(bmap) ;
        return isl_stat_ok;
    };
    
    isl_stat (*fun_ptr)(isl_basic_map * p,void * m) = (f) ;

    isl_map_foreach_basic_map(my_map1,fun_ptr,(void * ) &all_basic_maps) ;

    isl_map_foreach_basic_map(my_map2,fun_ptr,(void * ) &all_basic_maps) ;

    isl_map_foreach_basic_map(my_map3,fun_ptr,(void * ) &all_basic_maps) ;

    //collection done
    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::string str_schedule(isl_map_to_str(this->schedule)) ;

    //TO-DO replace compact schedule with normal schedule under condition of no existance of free variables within it for better performence
  
    std::regex r(" *[a-z]+[0-9]* *= *0 *");  
    std::regex r2(" 0 *,") ;
    std::regex r3(", *0 *\\]");
    std::regex r4("\\[0 *,") ;

    std::string sch1 =  std::regex_replace(str_schedule, r," 0"); 
    std::string compact_schedule_b = std::regex_replace(sch1, r2, "") ;
    std::string compact_schedule = std::regex_replace(compact_schedule_b, r3, "]") ;
    std::string compact_schedulef = std::regex_replace(compact_schedule, r4, "[") ;

    DEBUG(10, tiramisu::str_dump(" compact schedule is : "+compact_schedulef)) ;

    isl_map * normal_schedule = isl_map_read_from_str(this->get_ctx(),compact_schedulef.c_str()) ;

    isl_map * reverse = isl_map_reverse(isl_map_copy(normal_schedule)) ;


    std::string set_n = "[n0";
    std::string set_var = "]->{"+this->get_name()+"[n0" ;

    
  
    
    for(int i = 1;i<this->number_of_dims;i++)
    {
         
         
              set_n +=",n"+std::to_string(i) ;
              set_var +=   ",n"+std::to_string(i);
         
     }

    std::string set_complete = set_n + set_var +"]}" ;
   
    DEBUG(10, tiramisu::str_dump(" set to test order is : "+set_complete)) ;
     
    // withing my real computation space of csts
    isl_set * constant_isl_set = isl_set_read_from_str(this->get_ctx(),set_complete.c_str()) ;

    // my point after applying schedule
    isl_set * first_scheduled_point = isl_set_apply(
         isl_set_copy(constant_isl_set),
         isl_map_copy(normal_schedule)
         );
    

    DEBUG(10, tiramisu::str_dump(" schedule time for identity : "+std::string(isl_set_to_str(first_scheduled_point) ))) ;



    bool over_all_legality = true ;

    for(auto& dependecy: all_basic_maps){
        //first sort dep if it's before or after


        DEBUG(10, tiramisu::str_dump("-> current dep : "+std::string(isl_basic_map_to_str(dependecy) ))) ;

        

        if(is_dependency_scheduled_before_self(isl_map_from_basic_map(isl_basic_map_copy(dependecy)),isl_set_get_space(constant_isl_set))){

            

            isl_set * set_after_dep = isl_set_apply(
                isl_set_copy(constant_isl_set),
                isl_map_from_basic_map(dependecy)
                );

           

            DEBUG(10, tiramisu::str_dump(" dep is before self the set mapped is  : "+std::string(isl_set_to_str(set_after_dep) ))) ;


            isl_set * to_initial_result = isl_set_apply(set_after_dep,isl_map_copy(normal_schedule)) ;


            DEBUG(10, tiramisu::str_dump(" the mapped time stamp for target set from dep is : "+std::string(isl_set_to_str(to_initial_result) ))) ;

            isl_map * sup_result = isl_set_lex_lt_set(to_initial_result, isl_set_copy(first_scheduled_point)) ;

            if(isl_map_is_empty(sup_result)){
                    DEBUG(10, tiramisu::str_dump("   this dep is correct  " )) ;
                    
            }
            else{
                    DEBUG(10, tiramisu::str_dump(" dep is not correct  " )) ;
                    over_all_legality = false ;
                    break ;
            }
     

        }
        else{


            isl_set * set_after_dep = isl_set_apply(
                isl_set_copy(constant_isl_set),
                isl_map_from_basic_map(dependecy)
                );


            DEBUG(10, tiramisu::str_dump(" dep is after self the set mapped is  : "+std::string(isl_set_to_str(set_after_dep) ))) ;

            isl_set * to_initial_result = isl_set_apply(set_after_dep,isl_map_copy(normal_schedule)) ;


            DEBUG(10, tiramisu::str_dump(" the mapped time stamp for target set from dep is : "+std::string(isl_set_to_str(to_initial_result) ))) ;

             isl_map * sup_result = isl_set_lex_lt_set(isl_set_copy(first_scheduled_point),to_initial_result) ;

            if(isl_map_is_empty(sup_result)){
                 DEBUG(10, tiramisu::str_dump("   this dep is correct  " )) ;
            }
            else{
                 
                over_all_legality = false ;
                DEBUG(10, tiramisu::str_dump(" dep is not correct  " )) ;
                break ;
            }
            


        }

    }

  

    DEBUG_INDENT(-4);
    
    return over_all_legality ;

}

bool tiramisu::computation::applied_schedule_is_legal(tiramisu::computation * second)
{

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);
    assert(!second->get_name().empty());
    
    assert(this->get_function()->dep_read_after_write != NULL) ;
    
    /* ==========================================
        extarct C0 => Second dependencies from function
    */

    
    isl_union_map * read_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_read_after_write)) ;

    isl_union_map * write_after_read_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_write_after_read)) ;

    isl_union_map * write_after_write_dep = isl_union_map_range_factor_domain(
        isl_union_map_copy(this->get_function()->dep_write_after_write)) ;


        
        //construct the maps of the space needed
    std::string space_map = "{"+this->get_name()+"[" ;

    for(int i = 0 ; i< this->number_of_dims ; i++){

        space_map+="i"+std::to_string(i) ;

        if(i != (this->number_of_dims -1 ))
        {
            space_map +="," ;
        }
    }
    space_map+="]->"+second->get_name()+"[" ;

    for(int i = 0 ; i < second->number_of_dims ; i++){

        space_map+="i"+std::to_string(i) ;

        if(i != (second->number_of_dims -1 ))
        {
            space_map +="," ;
        }
    }
    
    space_map+="]}" ;

    DEBUG(3, tiramisu::str_dump(" the space map is to extract deps : "+space_map));

    isl_space * space = isl_map_get_space(
        isl_map_read_from_str(this->get_ctx(),space_map.c_str())
        ) ;


    isl_map * my_map1 = isl_union_map_extract_map(read_after_write_dep,isl_space_copy(space)) ;

    isl_map * my_map2 = isl_union_map_extract_map(write_after_read_dep,isl_space_copy(space)) ;

    isl_map * my_map3 = isl_union_map_extract_map(write_after_write_dep,isl_space_copy(space)) ;

    DEBUG(10, tiramisu::str_dump(" the extracted deps are : "+std::string(isl_map_to_str(my_map1))));

    DEBUG(10, tiramisu::str_dump(" the extracted deps are : "+std::string(isl_map_to_str(my_map2))));

    DEBUG(10, tiramisu::str_dump(" the extracted deps are : "+std::string(isl_map_to_str(my_map3))));

    std::vector<isl_basic_map *> all_basic_maps ;
    

    auto f = [](isl_basic_map * bmap,void * user) { 

        std::vector<isl_basic_map *>& myName = *reinterpret_cast<std::vector<isl_basic_map*>*>(user) ;
     
        myName.push_back(bmap) ;
        return isl_stat_ok;
    };
    
    isl_stat (*fun_ptr)(isl_basic_map * p,void * m) = (f) ;

    isl_map_foreach_basic_map(my_map1,fun_ptr,(void * ) &all_basic_maps) ;

    isl_map_foreach_basic_map(my_map2,fun_ptr,(void * ) &all_basic_maps) ;

    isl_map_foreach_basic_map(my_map3,fun_ptr,(void * ) &all_basic_maps) ;


   /* ==========================================
        extarct schedul of 2 computation
    */
   assert(this->get_function()->get_schedule()!= NULL) ;

   int m1 = isl_map_dim(this->get_schedule(), isl_dim_out);
   int m2 = isl_map_dim(second->get_schedule(), isl_dim_out);

   assert(m1 == m2) ;

    DEBUG(3, tiramisu::str_dump(" the current schedule of computation "+this->get_name()+" : "+std::string(isl_map_to_str(this->get_schedule()))));
    DEBUG(3, tiramisu::str_dump(" the current schedule of computation "+second->get_name()+" : "+std::string(isl_map_to_str(second->get_schedule()))));

   /* ==========================================
       making schedules comparable by mapping to the same time space
    */
   std::string unificator ="[" ;

   for(int i=0 ;i<m1 ;i++){
       unificator+="i"+std::to_string(i) ;

       if(i != (m1-1))
       {
           unificator+="," ;
       }
   }
   unificator+="]" ;

   std::string this_unificator = "{"+this->get_name()+unificator+"->"+unificator+"}" ;
   std::string second_unificator = "{"+second->get_name()+unificator+"->"+unificator+"}" ;

    isl_map * this_schedule_unif = isl_map_apply_range(
        isl_map_copy(this->schedule),
        isl_map_read_from_str(this->get_ctx(),this_unificator.c_str())
        ) ;

    isl_map * second_schedule_unif = isl_map_apply_range(
        isl_map_copy(second->schedule),
        isl_map_read_from_str(this->get_ctx(),second_unificator.c_str())
        ) ;


    DEBUG(3, tiramisu::str_dump(" first schedule adjusted into timestamp "+std::string(isl_map_to_str(this_schedule_unif))));
    DEBUG(3, tiramisu::str_dump(" second schedule adjusted into timestamp "+std::string(isl_map_to_str(second_schedule_unif))));

    

    std::string s0_set = "[" ;

    for(int i=0 ;i<this->number_of_dims;i++)
    {
        s0_set+="n"+std::to_string(i) ;
        if(i != (this->number_of_dims -1 ))
        {
            s0_set +="," ;
        }
    }
    s0_set +="]->{"+this->get_name()+"[" ;

    for(int i=0 ;i<this->number_of_dims;i++)
    {
        s0_set+="n"+std::to_string(i) ;
        if(i != (this->number_of_dims -1 ))
        {
            s0_set +="," ;
        }
    }
    s0_set +="]}" ;


    DEBUG(3, tiramisu::str_dump(" initial set of first computation is : "+s0_set));

    isl_set * first_set = isl_set_read_from_str(this->get_ctx(),s0_set.c_str()) ;// in S0

    
   /* ==========================================
        always check that S0 is lex lesser than S1 , for that S1<S0 need always to be an empty relation 
    */
 
    bool over_all_corectness = true ;

    DEBUG(10, tiramisu::str_dump(" check the respect of previous deps nature start : "));
    
    for (auto& depandancy:all_basic_maps){

        DEBUG(10, tiramisu::str_dump(" the depandancy is : "+std::string(isl_basic_map_to_str(depandancy))));


        isl_set * second_set = isl_set_apply(
            isl_set_copy(first_set),
            isl_map_from_basic_map(isl_basic_map_copy(depandancy))
            ) ;
        // in S1

        isl_set * time_first = isl_set_apply(isl_set_copy(first_set),isl_map_copy(this_schedule_unif)) ;
        isl_set * time_second = isl_set_apply(isl_set_copy(second_set),isl_map_copy(second_schedule_unif)) ;

        isl_map * result_sup = isl_set_lex_ge_set(
            isl_set_copy(time_first),
            isl_set_copy(time_second)
        ) ;

        if(!isl_map_is_empty(result_sup))
        {
             
            over_all_corectness = false ;
            DEBUG(10, tiramisu::str_dump(" depandancy is wrong by current schedule "));
            break ;
            
        }
        else{
                DEBUG(10, tiramisu::str_dump(" this depandancy is respected by the current schedule  "));
          
        }

        
    }



    DEBUG_INDENT(-4);

    return over_all_corectness ;
}

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

void tiramisu::computation::tag_vector_level(int dim, int length)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(dim >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);
    assert(length > 0);

    this->get_function()->add_vector_dimension(this->get_name(), dim, length);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_vector_level(tiramisu::var L0_var, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    this->tag_vector_level(L0, v);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_distribute_level(tiramisu::var L)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L.get_name().length() > 0);
    std::vector<int> dimensions =
            this->get_loop_level_numbers_from_dimension_names({L.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    this->tag_distribute_level(L0);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_distribute_level(int L)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);

    this->get_function()->add_distributed_dimension(this->get_name(), L);
    this->get_function()->_needs_rank_call = true;

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_parallel_level(tiramisu::var L0_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    this->tag_parallel_level(L0);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_unroll_level(tiramisu::var L0_var)
{
	this->tag_unroll_level(L0_var, 0);
}

void tiramisu::computation::tag_unroll_level(tiramisu::var L0_var, int factor)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    this->tag_unroll_level(L0, factor);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::tag_unroll_level(int level)
{
	this->tag_unroll_level(level, 0);
}

void tiramisu::computation::tag_unroll_level(int level, int factor)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(level >= 0);
    assert(!this->get_name().empty());
    assert(this->get_function() != NULL);
    assert(factor >= 0);

    this->get_function()->add_unroll_dimension(this->get_name(), level, factor);

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
    new_c->is_let = this->is_let;

    DEBUG_INDENT(-4);

    return new_c;
}

isl_map *isl_map_set_const_dim(isl_map *map, int dim_pos, int val);

std::string computation::get_dimension_name_for_loop_level(int loop_level)
{
    int dim = loop_level_into_dynamic_dimension(loop_level);
    std::string name = isl_map_get_dim_name(this->get_schedule(), isl_dim_out, dim);
    assert(name.size() > 0);
    return name;
}

/*
 * Separate creates a new computation that has exactly the same name
 * and the same iteration domain but the two computations would have
 * different schedules.
 * The schedule of the original computation would restrict it to the
 * domain where the computation is full. The schedule of the separated
 * (new) computation would restrict it to the partial domain (i.e.,
 * the remaining part).
 *
 * Example, if we have a computation
 * {S0[i]: 0<=i<N}
 *
 * The schedule of the original (full) computation would be
 * {S0[i]->S0[0, 0, i, 0]: 0<=i<v*floor(N/v)}
 *
 * The schedule of the separated (partial) computation would be
 * {S0[i]->S0[0, 10, i, 0]: v*floor(N/v)<=i<N}
 *
 * Design choices:
 * - We cannot actually change the iteration domain because this will not compose
 * with the other trasnformations that are expressed are schedules. So we have
 * to actually express the separation transformation using the schedule.
 * - At the same time, we want to be able to manipulate the separated computation
 * and schedule it, so we want to access it with get_update(ID), to make that simple
 * we create a new computation. That is better than just keeping the same original
 * computation and addin a new schedule to it for the separated computation.
 */
void tiramisu::computation::separate(int dim, tiramisu::expr N, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Separating the computation at level " + std::to_string(dim)));

    DEBUG(3, tiramisu::str_dump("Generating the time-space domain."));
    this->gen_time_space_domain();


    //////////////////////////////////////////////////////////////////////////////

    // We create the constraint (i < v*floor(N/v))
    DEBUG(3, tiramisu::str_dump("Constructing the constraint (i<v*floor(N/v))"));
    DEBUG(3, tiramisu::str_dump("Removing any cast operator in N."));
    std::string N_without_cast = N.to_str();
    while (N_without_cast.find("cast") != std::string::npos) // while there is a "cast" in the expression
    {
        // Remove "cast" from the string, we do not need it.
        // An alternative to this would be to actually mutate the expression N and remove the cast
        // operator, but that is more time consuming to implement than replacing the string directly.
        int pos = N_without_cast.find("cast");
        N_without_cast = N_without_cast.erase(pos, 4);
    }

    std::string constraint;
    constraint = "";
    for (int i=0; i<isl_map_dim(this->get_schedule(), isl_dim_param); i++)
    {
        if (i==0)
            constraint += "[";
        constraint += isl_map_get_dim_name(this->get_schedule(), isl_dim_param, i);
        if (i!=isl_map_dim(this->get_schedule(), isl_dim_param)-1)
            constraint += ",";
        else
            constraint += "]->";
    }
    constraint += "{" + this->get_name() + "[0,";
    for (int i=1; i<isl_map_dim(this->get_schedule(), isl_dim_out); i++)
    {
        if ((i%2==0) && (isl_map_has_dim_name(this->get_schedule(), isl_dim_out, i)==true))
            constraint += isl_map_get_dim_name(this->get_schedule(), isl_dim_out, i);
        else
            constraint += "o" + std::to_string(i);
        if (i != isl_map_dim(this->get_schedule(), isl_dim_out)-1)
            constraint += ",";
    }
    constraint += "]: ";

    std::string constraint1 = constraint +
                                this->get_dimension_name_for_loop_level(dim) + " < (" + std::to_string(v) + "*(floor((" + N_without_cast + ")/" + std::to_string(v) + ")))}";
    DEBUG(3, tiramisu::str_dump("The constraint is:" + constraint1));

    // We create the constraint (i >= v*floor(N/v))
    DEBUG(3, tiramisu::str_dump("Constructing the constraint (i>=v*(floor(N/v)))"));
    std::string constraint2 = constraint +
                                this->get_dimension_name_for_loop_level(dim) + " >= (" + std::to_string(v) + "*(floor((" + N_without_cast + ")/" + std::to_string(v) + ")))}";
    DEBUG(3, tiramisu::str_dump("The constraint is:" + constraint2));

    //////////////////////////////////////////////////////////////////////////////

    isl_set *constraint2_isl = isl_set_read_from_str(this->get_ctx(), constraint2.c_str());

    if (isl_set_is_empty(isl_map_range(isl_map_intersect_range(isl_map_copy(this->get_schedule()), constraint2_isl))) == false)
    {
        DEBUG(3, tiramisu::str_dump("The separate computation is not empty."));

        // Create the separated computation.
        // First, create the domain of the separated computation (which is identical to
        // the domain of the original computation). Both also have the same name.
        // TODO: create copy functions for all the classes so that we can copy the objects
        // we need to have this->get_expr().copy()

        std::string domain_str = std::string(isl_set_to_str(this->get_iteration_domain()));
        this->add_definitions(domain_str,
            this->get_expr(),
            this->should_schedule_this_computation(),
            this->get_data_type(),
            this->get_function());

        // Set the schedule of the newly created computation (separated
        // computation) to be equal to the schedule of the original computation.
        isl_map *new_schedule = isl_map_copy(this->get_schedule());
        this->get_last_update().set_schedule(new_schedule);

        // Create the access relation of the separated computation.
        if (this->get_access_relation() != NULL)
        {
            DEBUG(3, tiramisu::str_dump("Creating the access function of the separated computation.\n"));
            this->get_last_update().set_access(isl_map_copy(this->get_access_relation()));

            DEBUG(3, tiramisu::str_dump("Access of the separated computation:",
                                        isl_map_to_str(this->get_last_update().get_access_relation())));
        }

        this->get_last_update().add_schedule_constraint("", constraint2.c_str());

        // Mark the separated computation to be executed after the original (full)
        // computation.
        this->get_last_update().after(*this, dim);

        DEBUG(3, tiramisu::str_dump("The separate computation:"); this->get_last_update().dump());
    }
    else
    {
        DEBUG(3, tiramisu::str_dump("The separate computation is empty. Thus not added."));
    }

    this->add_schedule_constraint("", constraint1.c_str());

    DEBUG(3, tiramisu::str_dump("The original computation:"); this->dump());

    DEBUG_INDENT(-4);
}

void tiramisu::computation::separate_at(var _level, std::vector<tiramisu::expr> _separate_points, tiramisu::expr _max)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Separating the computation at level " + _level.get_name()));

    DEBUG(3, tiramisu::str_dump("Generating the time-space domain."));
    this->gen_time_space_domain();

    std::vector<int> dimensions =
            this->get_loop_level_numbers_from_dimension_names({_level.get_name()});
    int level = dimensions[0];

    //////////////////////////////////////////////////////////////////////////////

    std::vector<tiramisu::constant> separate_points;
    for (auto p : _separate_points) {
        separate_points.push_back(tiramisu::constant("c" + std::to_string(id_counter++), p, p.get_data_type(), true,
                                                     NULL, 0, this->get_function()));
    }
    tiramisu::constant max("c" + std::to_string(id_counter++), _max, _max.get_data_type(), true, NULL, 0,
                           this->get_function());

    // We create the constraint (i < separate_point)
    DEBUG(3, tiramisu::str_dump("Constructing the constraint (i<middle)"));
    std::string constraint;
    constraint = "";
    // get the constants
    for (int i=0; i<isl_map_dim(this->get_schedule(), isl_dim_param); i++)
    {
        if (i==0) {
            constraint += "[" + max.get_name() + ",";
            for (auto separate_point : separate_points) {
                constraint += separate_point.get_name() + ",";
            }
        }
        constraint += isl_map_get_dim_name(this->get_schedule(), isl_dim_param, i);
        if (i!=isl_map_dim(this->get_schedule(), isl_dim_param)-1)
            constraint += ",";
        else
            constraint += "]->";
    }
    if (isl_map_dim(this->get_schedule(), isl_dim_param) == 0) {
        // Need to add in these constants
        constraint += "[" + max.get_name();
        for (auto separate_point : separate_points) {
            constraint += ", " +  separate_point.get_name() ;
        }
        constraint += "]->";
    }
    constraint += "{" + this->get_name() + "[0,";
    for (int i=1; i<isl_map_dim(this->get_schedule(), isl_dim_out); i++)
    {
        if ((i%2==0) && (isl_map_has_dim_name(this->get_schedule(), isl_dim_out, i)==true))
            constraint += isl_map_get_dim_name(this->get_schedule(), isl_dim_out, i);
        else
            constraint += "o" + std::to_string(i);
        if (i != isl_map_dim(this->get_schedule(), isl_dim_out)-1)
            constraint += ",";
    }
    constraint += "]: ";
    std::vector<std::string> constraints;
    // This is the first constraint
    std::string constraint1 = constraint +
                              this->get_dimension_name_for_loop_level(level) + " < " + separate_points[0].get_name() + "}";
    DEBUG(3, tiramisu::str_dump("The constraint is:" + constraint1));

    // We create the constraint (i >= separate_point). This is the last constraint
    DEBUG(3, tiramisu::str_dump("Constructing the constraint (i>=middle)"));
    std::string constraintn = constraint +
                              this->get_dimension_name_for_loop_level(level) + " >= " +
                              separate_points[separate_points.size() - 1].get_name() + " and " +
                              this->get_dimension_name_for_loop_level(level) + " < " + max.get_name() + "}";
    DEBUG(3, tiramisu::str_dump("The constraint is:" + constraintn));


    // create the intermediate constraints
    for (int i = 1; i < separate_points.size(); i++) {
        std::string cons = constraint +
                           this->get_dimension_name_for_loop_level(level) + " >= " + separate_points[i-1].get_name() + " and ";
        cons += this->get_dimension_name_for_loop_level(level) + " < " + separate_points[i].get_name() + "}";
        constraints.push_back(cons);
    }
    constraints.push_back(constraintn);
    //////////////////////////////////////////////////////////////////////////////

    for (std::string cons : constraints) {
        isl_set *cons_isl = isl_set_read_from_str(this->get_ctx(), cons.c_str());
        if (isl_set_is_empty(
                isl_map_range(isl_map_intersect_range(isl_map_copy(this->get_schedule()), cons_isl))) == false) {
            DEBUG(3, tiramisu::str_dump("The separate computation is not empty."));

            // Create the separated computation.
            // First, create the domain of the separated computation (which is identical to
            // the domain of the original computation). Both also have the same name.
            // TODO: create copy functions for all the classes so that we can copy the objects
            // we need to have this->get_expr().copy()
            int last_update_computation = this->get_updates().size();

            std::string domain_str = std::string(isl_set_to_str(this->get_iteration_domain()));
            this->add_definitions(domain_str,
                                  this->get_expr(),
                                  this->should_schedule_this_computation(),
                                  this->get_data_type(),
                                  this->get_function());

            // Set the schedule of the newly created computation (separated
            // computation) to be equal to the schedule of the original computation.
            isl_map *new_schedule = isl_map_copy(this->get_schedule());
            this->get_update(last_update_computation).set_schedule(new_schedule);

            // Create the access relation of the separated computation (by replacing its name).
            if (this->get_access_relation() != NULL) {
                DEBUG(3, tiramisu::str_dump("Creating the access function of the separated computation.\n"));
                this->get_update(last_update_computation).set_access(isl_map_copy(this->get_access_relation()));

                DEBUG(3, tiramisu::str_dump("Access of the separated computation:",
                                            isl_map_to_str(
                                                    this->get_update(last_update_computation).get_access_relation())));
            }

            this->get_update(last_update_computation).add_schedule_constraint("", cons.c_str());

            DEBUG(3, tiramisu::str_dump("The separate computation:");
                    this->get_update(last_update_computation).dump());
        } else {
            DEBUG(3, tiramisu::str_dump("The separate computation is empty. Thus not added."));
        }
    }

    this->add_schedule_constraint("", constraint1.c_str());

    DEBUG(3, tiramisu::str_dump("The original computation:"); this->dump());

    // rename all the updates by adding '_<ctr>' to the end of the name
    int ctr = 0;
    for (auto comp : this->get_updates()) {
        comp->rename_computation(comp->get_name() + "_" + std::to_string(ctr++));
    }
    DEBUG_INDENT(-4);
}

void tiramisu::computation::set_iteration_domain(isl_set *domain)
{
    this->iteration_domain = domain;
}

std::string utility::get_parameters_list(isl_set *set)
{
    std::string list = "";

    assert(set != NULL);

    for (int i = 0; i < isl_set_dim(set, isl_dim_param); i++)
    {
        list += isl_set_get_dim_name(set, isl_dim_param, i);
        if ((i != isl_set_dim(set, isl_dim_param) - 1))
        {
            list += ",";
        }
    }

    return list;
}

int utility::get_extent(isl_set *set, int dim)
{
    tiramisu::expr lower_bound = tiramisu::utility::get_bound(set, dim, false);
    tiramisu::expr upper_bound = tiramisu::utility::get_bound(set, dim, true);

    if(lower_bound.get_expr_type() != tiramisu::e_val or upper_bound.get_expr_type() != tiramisu::e_val)
        ERROR("Check if the context is set for constants of distributed dimension", true);

    return upper_bound.get_int_val() - lower_bound.get_int_val() + 1;
}

tiramisu::constant *tiramisu::computation::create_separator(const tiramisu::expr &loop_upper_bound, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    /*
     * Create a new Tiramisu constant M = v*floor(N/v). This is the biggest
     * multiple of w that is still smaller than N.  Add this constant to
     * the list of invariants.
     */
    primitive_t f_type, i_type, u_type;
    if (global::get_loop_iterator_data_type() == p_int32)
    {
        f_type = p_float32;
        i_type = p_int32;
        u_type = p_uint32;
    }
    else
    {
        f_type = p_float64;
        i_type = p_int64;
        u_type = p_uint64;
    }

    std::string separator_name = tiramisu::generate_new_variable_name();
    tiramisu::expr div_expr = tiramisu::expr(o_div, loop_upper_bound, tiramisu::expr(v));
    tiramisu::expr cast_expr = tiramisu::expr(o_cast, f_type, div_expr);
    tiramisu::expr floor_expr = tiramisu::expr(o_floor, cast_expr);
    tiramisu::expr cast2_expr = tiramisu::expr(o_cast, i_type, floor_expr);
    tiramisu::expr separator_expr = tiramisu::expr(o_mul, tiramisu::expr(v), cast2_expr);
    tiramisu::constant *separation_param = new tiramisu::constant(
        separator_name, separator_expr, u_type, true, NULL, 0, this->get_function());

    DEBUG_INDENT(-4);

    return separation_param;
}

tiramisu::buffer *tiramisu::computation::get_automatically_allocated_buffer()
{
    return this->automatically_allocated_buffer;
}

std::vector<tiramisu::expr> computation::compute_buffer_size()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<tiramisu::expr> dim_sizes;

    // If the computation has an update, we first compute the union of all the
    // updates, then we compute the bounds of the union.
    for (int i = 0; i < this->get_iteration_domain_dimensions_number(); i++)
    {
        isl_set *union_iter_domain = isl_set_copy(this->get_update(0).get_iteration_domain());

        for (int j = 1; j < this->get_updates().size(); j++)
        {
            isl_set *iter_domain = isl_set_copy(this->get_update(j).get_iteration_domain());
            union_iter_domain = isl_set_union(union_iter_domain, iter_domain);
        }

        DEBUG(3, tiramisu::str_dump("Extracting bounds of the following set:", isl_set_to_str(union_iter_domain)));
        tiramisu::expr lower = utility::get_bound(union_iter_domain, i, false);
        tiramisu::expr upper = utility::get_bound(union_iter_domain, i, true);
        tiramisu::expr diff = (upper - lower + 1);

        DEBUG(3, tiramisu::str_dump("Buffer dimension size (dim = " + std::to_string(i) + ") : "); diff.dump(false));
        dim_sizes.push_back(diff);
    }

    DEBUG_INDENT(-4);

    return dim_sizes;
}

/**
 * Algorithm:
 * - Compute the size of the buffer:
 *      - TODO: Future work Use the same code that computes the needed area in compute_at,
 *      - TODO: From the needed area, deduce the size by computing the upper
 *              bound and the lower bound and subtracting the two.
 * - declare a buffer with a random name, and with the computed size,
 * - allocate the buffer and get the computation that allocates the buffer,
 * - map the computation to the allocated buffer (one to one mapping),
 * - schedule the computation that allocates the buffer before \p comp
 * at loop level L0,
 * - return the allocation computation.
 */
tiramisu::computation *computation::store_at(tiramisu::computation &comp,
                                             tiramisu::var L0_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    tiramisu::buffer *buff = new tiramisu::buffer("_" + this->name + "_buffer",
            this->compute_buffer_size(),
            this->get_data_type(),
            tiramisu::a_temporary,
            this->get_function());

    this->automatically_allocated_buffer = buff;

    tiramisu::computation *allocation = buff->allocate_at(comp, L0);
    this->store_in(buff);
    if (comp.get_predecessor() != NULL)
        allocation->between(
            *(comp.get_predecessor()),
            L0_var, comp, L0_var);
    else
        allocation->before(comp, L0);

    DEBUG_INDENT(-4);

    return allocation;
}

void tiramisu::computation::vectorize(tiramisu::var L0_var, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());
    this->vectorize(L0_var, v, L0_outer, L0_inner);

    DEBUG_INDENT(-4);
}

void computation::update_names(std::vector<std::string> original_loop_level_names, std::vector<std::string> new_names,
                               int erase_from, int nb_loop_levels_to_erase)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("Original loop level names: "));
    for (auto n: original_loop_level_names)
    {
        DEBUG_NO_NEWLINE_NO_INDENT(3, tiramisu::str_dump(n + " "));
    }
    DEBUG_NEWLINE(3);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("New names: "));
    for (auto n: new_names)
    {
        DEBUG_NO_NEWLINE_NO_INDENT(3, tiramisu::str_dump(n + " "));
    }
    DEBUG_NEWLINE(3);

    DEBUG(3, tiramisu::str_dump("Start erasing from: " + std::to_string(erase_from)));
    DEBUG(3, tiramisu::str_dump("Number of loop levels to erase: " + std::to_string(nb_loop_levels_to_erase)));

    original_loop_level_names.erase(original_loop_level_names.begin() + erase_from, original_loop_level_names.begin() + erase_from + nb_loop_levels_to_erase);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("Original loop level names after erasing loop levels: "));
    for (auto n: original_loop_level_names)
    {
        DEBUG_NO_NEWLINE_NO_INDENT(3, tiramisu::str_dump(n + " "));
    }
    DEBUG_NEWLINE(3);

    original_loop_level_names.insert(original_loop_level_names.begin() + erase_from, new_names.begin(), new_names.end());

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("Original loop level names after inserting the new loop levels: "));
    for (auto n: original_loop_level_names)
    {
        DEBUG_NO_NEWLINE_NO_INDENT(3, tiramisu::str_dump(n + " "));
    }
    DEBUG_NEWLINE(3);

    this->set_loop_level_names(original_loop_level_names);
//    this->name_unnamed_time_space_dimensions();

    DEBUG(3, tiramisu::str_dump("Names updated. New names are: "));
    for (auto n: this->get_loop_level_names())
    {
        DEBUG_NO_NEWLINE_NO_INDENT(3, tiramisu::str_dump(n + " "));
    }
    DEBUG(3, tiramisu::str_dump(""));

    DEBUG_INDENT(-4);
}

void tiramisu::computation::vectorize(tiramisu::var L0_var, int v, tiramisu::var L0_outer, tiramisu::var L0_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    bool split_happened = this->separateAndSplit(L0_var, v, L0_outer, L0_inner);

    if (split_happened)
    {
        // Tag the inner loop after splitting to be vectorized. That loop
        // is supposed to have a constant extent.
        this->get_update(0).tag_vector_level(L0 + 1, v);

        // Replace the original dimension name with two new dimension names
        this->update_names(original_loop_level_names, {L0_outer.get_name(), L0_inner.get_name()}, L0, 1);
    }
    else
    {
        this->get_update(0).tag_vector_level(L0, v);
        this->set_loop_level_names({L0}, {L0_outer.get_name()});

        // Replace the original dimension name with two new dimension names
        this->update_names(original_loop_level_names, {L0_inner.get_name()}, L0, 1);
    }

    this->get_function()->align_schedules();

    DEBUG_INDENT(-4);
}

tiramisu::computation& computation::get_last_update()
{
    return this->get_update(this->get_updates().size()-1);
}

/**
  * Returns all updates the have been defined for this computation using
  * add_definitions. The 0th update is a pointer to this computation.
  */
std::vector<computation*>& tiramisu::computation::get_updates() {
    return this->updates;
}

/**
  * Returns the \p index update that has been added to this computation such that:
  * - If \p index == 0, then this computation is returned.
  * - If \p > 0, then it returns the pth computation added through add_definitions.
  */
computation& tiramisu::computation::get_update(int i)
{
    return *(this->updates[i]);
}

void tiramisu::computation::unroll(tiramisu::var L0_var, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());
    this->unroll(L0_var, v, L0_outer, L0_inner);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::unroll(tiramisu::var L0_var, int v, tiramisu::var L0_outer, tiramisu::var L0_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    bool split_happened = this->separateAndSplit(L0_var, v, L0_outer, L0_inner);

    if (split_happened)
    {
        // Tag the inner loop after splitting to be unrolled. That loop
        // is supposed to have a constant extent.
        this->get_update(0).tag_unroll_level(L0 + 1, v);

        // Replace the original dimension name with two new dimension names
        this->update_names(original_loop_level_names, {L0_outer.get_name(), L0_inner.get_name()}, L0, 1);
    }
    else
    {
        this->get_update(0).tag_unroll_level(L0, v);
        this->set_loop_level_names({L0}, {L0_outer.get_name()});

        // Replace the original dimension name with two new dimension names
        this->update_names(original_loop_level_names, {L0_inner.get_name()}, L0, 1);
    }

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
        {
            std::cout << "\n";
        }
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

    for (auto v : vec)
    {
        res = std::max(v, res);
    }

    return res;
}

bool buffer::has_constant_extents()
{
    bool constant_extent = true;

    for (size_t i = 0; i < this->get_dim_sizes().size(); i++)
    {
        if (this->get_dim_sizes()[i].get_expr_type() != tiramisu::e_val)
        {
            constant_extent = false;
        }
    }

    return constant_extent;
}

tiramisu::computation *buffer::allocate_at(tiramisu::computation &C, tiramisu::var level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(level.get_name().length() > 0);

    std::vector<int> dimensions =
        C.get_loop_level_numbers_from_dimension_names({level.get_name()});

    assert(dimensions.size() == 1);

    int L0 = dimensions[0];

    C.check_dimensions_validity({L0});

    tiramisu::computation *alloc = this->allocate_at(C, L0);

    DEBUG_INDENT(-4);

    return alloc;
}

tiramisu::computation *buffer::allocate_at(tiramisu::computation &C, int level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(level >= tiramisu::computation::root_dimension);
    assert(level < (int) isl_set_dim(C.get_iteration_domain(), isl_dim_set));

    DEBUG(3, tiramisu::str_dump("Computing the iteration domain for the allocate() operation"));
    DEBUG(3, tiramisu::str_dump("Computation name " + C.get_name() + ", Level = " + std::to_string(level)));

    isl_set *iter = C.get_iteration_domains_of_all_definitions();

    DEBUG(3, tiramisu::str_dump(
              "The union of iteration domains of the computations with which we allocate (all their definitions): ",
              isl_set_to_str(iter)));

    int projection_dimension = level + 1;
    if (projection_dimension != 0)
        iter = isl_set_project_out(isl_set_copy(iter),
                                   isl_dim_set,
                                   projection_dimension,
                                   isl_set_dim(iter, isl_dim_set) - projection_dimension);
    else
    {
        iter = isl_set_read_from_str(C.get_ctx(), "{[0]}");
    }
    std::string new_name = "_allocation_" + C.get_name();
    iter = isl_set_set_tuple_name(iter, new_name.c_str());
    std::string iteration_domain_str = isl_set_to_str(iter);

    DEBUG(3, tiramisu::str_dump(
              "Computed iteration domain for the allocate() operation",
              isl_set_to_str(iter)));

    tiramisu::expr *new_expression = new tiramisu::expr(tiramisu::o_allocate, this->get_name());

    DEBUG(3, tiramisu::str_dump("The expression of the allocation operation"); new_expression->dump(false));

    tiramisu::computation *alloc = new tiramisu::computation(iteration_domain_str,
            *new_expression,
            true, p_none, C.get_function());

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

    this->name_unnamed_time_space_dimensions();

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
        DEBUG(3, tiramisu::str_dump("The schedule is : "));
        DEBUG(3, tiramisu::str_dump(isl_map_to_str(sched)));

        sched = isl_map_intersect_domain(isl_map_copy(sched), isl_set_copy(domain_cst));

    }

    if (!range_constraints.empty())
    {
        isl_set *range_cst = isl_set_read_from_str(this->ctx, range_constraints.c_str());

        DEBUG(3, tiramisu::str_dump("Adding the following constraints to the range of the schedule : "));
        DEBUG(3, tiramisu::str_dump(isl_set_to_str(range_cst)));
        DEBUG(3, tiramisu::str_dump("The schedule : ", isl_map_to_str(sched)));

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

void tiramisu::computation::save_schedule_to_default(){
    this->default_schedule = isl_map_copy(this->schedule) ;
}

void tiramisu::computation::set_schedule_to_default(){
    if(this->default_schedule!=NULL){
        this->set_schedule(this->default_schedule) ;
       
    }
}

void computation::set_low_level_schedule(std::string map_str)
{
    assert(!map_str.empty());
    assert(this->ctx != NULL);

    isl_map *map = isl_map_read_from_str(this->ctx, map_str.c_str());
    assert(map != NULL);

    this->set_low_level_schedule(map);
}

void tiramisu::computation::set_low_level_schedule(isl_map *map)
{
    this->fct->use_low_level_scheduling_commands = true;
    this->set_schedule(map);
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

// if multiple const values exist, choose the maximal value among them because we
// want to use this value to know by how much we shift the computations back.
// so we need to figure out the maximal const value and use it to shift the iterations
// backward so that that iteration runs before the consumer.
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
            data->out_constant = std::max(data->out_constant, const_val);
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

/**
 * Set the value \p val for the output dimension \p dim_pos of \p map.
 *
 * Example
 *
 * Assuming the map M = {S[i,j]->[i0,i1,i2]}
 *
 * M = isl_map_set_const_dim(M, 0, 0);
 *
 * Would create the constraint i0=0 and add it to the map.
 * The resulting map is
 *
 * M = {S[i,j]->[i0,i1,i2]: i0=0}
 *
 */
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

    isl_map *identity = isl_set_identity(isl_map_range(isl_map_copy(map)));
    // We need to create a universe of the map (i.e., an unconstrained map)
    // because isl_set_identity() create an identity transformation and
    // inserts the constraints that were in the original set.  We don't
    // want to have those constraints.  We want to have a universe map, i.e.,
    // a map without any constraint.
    identity = isl_map_universe(isl_map_get_space(identity));

    isl_space *sp = isl_map_get_space(identity);
    isl_local_space *lsp = isl_local_space_from_space(isl_space_copy(sp));

    // This loops goes through the output dimensions of the map one by one
    // and adds a constraint for each dimension. IF the dimension is dim_pos
    // it add a constraint of equality to val
    // Otherwise it adds a constraint that keeps the original value, i.e.,
    // (output dimension = input dimension)
    // Example
    //  Assuming that dim_pos = 0, val = 10 and the universe map is
    //  {S[i0,i1]->S[j0,j1]}, this loop produces
    //  {S[i0,i1]->S[j0,j1]: j0=0 and j1=i1}
    //  i.e.,
    //  {S[i0,i1]->S[0,i1]}
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
 * to the loop level 0 is 2, and to 1 it is 4, ...
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

/**
 * Transform a dynamic schedule dimension into the corresponding loop level.
 *
 * In the example below, the loop level that corresponds
 * to the dynamic dimension 2 is 0, and to the dynamic dimension 4 is 1, ...
 *
 * The first dimension is the duplication dimension, the following
 * dimensions are static, dynamic, static, dynamic, ...
 *
 * Loop level               :    -1         0      1      2
 * Schedule dimension number:        0, 1   2  3   4  5   6  7
 * Schedule:                        [0, 0, i1, 0, i2, 0, i3, 0]
 */
int dynamic_dimension_into_loop_level(int dim)
{
    assert(dim % 2 == 0);
    int level = (dim - 2)/2;
    return level;
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
void computation::after_low_level(computation &comp, int level)
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
    DEBUG(3, tiramisu::str_dump("Setting the schedule of ");
          tiramisu::str_dump(this->get_name());
          tiramisu::str_dump(" to be equal to the schedule of ");
          tiramisu::str_dump(comp.get_name());
          tiramisu::str_dump(" at all the dimensions before dimension ");
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

    isl_map *new_sched = NULL;
    for (int i = 1; i<=dim; i=i+2)
    {
        if (i < dim)
        {
            // Get the constant in comp, add +1 to it and set it to sched1
            int order = isl_map_get_static_dim(comp.get_schedule(), i);
            new_sched = isl_map_copy(this->get_schedule());
            new_sched = add_eq_to_schedule_map(i, 0, -1, order, new_sched);
        }
        else // (i == dim)
        {
            // Get the constant in comp, add +1 to it and set it to sched1
            int order = isl_map_get_static_dim(comp.get_schedule(), i);
            new_sched = isl_map_copy(this->get_schedule());
            new_sched = add_eq_to_schedule_map(i, 0, -1, order + 10, new_sched);
        }
        this->set_schedule(new_sched);
    }

    DEBUG(3, tiramisu::str_dump("Schedule adjusted: ",
                                isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void tiramisu::buffer::set_argument_type(tiramisu::argument_t type)
{
        this->argtype = type;
}

tiramisu::computation *tiramisu::computation::get_first_definition()
{
    return first_definition;
}

bool tiramisu::computation::is_first_definition()
{
    return is_first;
}

bool tiramisu::computation::buffer_already_allocated()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    bool allocated = false;

    if (this->get_automatically_allocated_buffer() != NULL)
    {
        DEBUG(3, tiramisu::str_dump("A buffer was already allocated automatically for this computation."));
        allocated = true;
    }
    else
    {
        DEBUG(3, tiramisu::str_dump("No buffer was allocated automatically for this computation."));
    }

    // If this computation is not the first computation, and a buffer has
    // already been allocated for the first definition, then exit.
    if (this->has_multiple_definitions() == true)
    {
        DEBUG(3, tiramisu::str_dump("This computation has multiple definitions."));
        if (this->is_first_definition() == false)
        {
            DEBUG(3, tiramisu::str_dump("This is NOT the first definition of the computation."));
            if (this->get_first_definition()->get_automatically_allocated_buffer() != NULL)
            {
                DEBUG(3, tiramisu::str_dump("A buffer has already been allocated for the first computation."));
                allocated = true;
            }
            else
            {
                DEBUG(3, tiramisu::str_dump("No buffer has already been allocated for the first computation."));
                DEBUG(3, tiramisu::str_dump("Checking whether the other definitions have an automatically allocated buffer."));
                for (auto c: this->get_first_definition()->get_updates())
                if (c->get_automatically_allocated_buffer() != NULL)
                {
                    DEBUG(3, tiramisu::str_dump("One of the other definitions has an automatically allocated buffer."));
                    allocated = true;
                }
                DEBUG(3, tiramisu::str_dump("No other definition has an automatically allocated buffer."));
            }
        }
        else // If any of the other definitions has a buffer, exit.
        {
            DEBUG(3, tiramisu::str_dump("This is the first definition of the computation."));
            DEBUG(3, tiramisu::str_dump("Checking whether the other definitions have an automatically allocated buffer."));
            for (auto c: this->get_updates())
                if (c->get_automatically_allocated_buffer() != NULL)
                {
                    DEBUG(3, tiramisu::str_dump("One of the other definitions has an automatically allocated buffer."));
                    allocated = true;
                }
            DEBUG(3, tiramisu::str_dump("No other definition has an automatically allocated buffer."));
        }
    }
    else
    {
        DEBUG(3, tiramisu::str_dump("This computation has only one definition."));
    }

    DEBUG_INDENT(-4);

    return allocated;
}

void tiramisu::computation::allocate_and_map_buffer_automatically(tiramisu::argument_t type)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Allocating and mapping a buffer automatically."));
    DEBUG(3, tiramisu::str_dump("Computation name: " + this->get_name()));

    // If a buffer is already allocated, exit.
    if (this->buffer_already_allocated() == true)
    {
        DEBUG(3, tiramisu::str_dump("Buffer already allocated."));
        DEBUG_INDENT(-4);
        return;
    }

    // If we reach this point, that means that no buffer has been allocated
    // for this computation or for the other definitions of this computation.
    std::vector<tiramisu::expr> dim_sizes = this->compute_buffer_size();

    tiramisu::buffer *buff = NULL;

    if (this->is_first_definition() == true)
    {
        if (this->get_automatically_allocated_buffer() == NULL)
        {
            DEBUG(3, tiramisu::str_dump("The automatically allocated buffer of this "
                                        "computation is NULL."));
            DEBUG(3, tiramisu::str_dump("Allocating an automatically allocated buffer for "
                                        "this computation."));

            std::string buff_name;
            buff_name = "_" + this->name + "_buffer";
            buff = new tiramisu::buffer(buff_name,
                                dim_sizes,
                                this->get_data_type(),
                                type,
                                this->get_function());
            this->automatically_allocated_buffer = buff;
        }
        else // automatic buffer already allocated.
            return;
    }
    else
    {
        if  (this->get_first_definition()->get_automatically_allocated_buffer() == NULL)
        {
            DEBUG(3, tiramisu::str_dump("The automatically allocated buffer of the first "
                                        "definition of this computation is NULL."));
            DEBUG(3, tiramisu::str_dump("Allocating an automatically allocated buffer of the first "
                                        "definition of this computation."));

            std::string buff_name;
            buff_name = "_" + this->get_first_definition()->name + "_buffer";
            buff = new tiramisu::buffer(buff_name,
                                dim_sizes,
                                this->get_data_type(),
                                type,
                                this->get_function());
            this->automatically_allocated_buffer = buff;
        }
        else // first definition has an allocated array.
            buff = this->get_first_definition()->get_automatically_allocated_buffer();
    }

    assert(buff != NULL);

    this->automatically_allocated_buffer = buff;

    tiramisu::computation *allocation;
    if (type == tiramisu::a_temporary)
    {
        allocation = buff->allocate_at(*this, computation::root_dimension);
        allocation->set_name("_allocation_" + this->name);
        // Schedule all allocations at the beginning
        this->get_function()->automatically_allocated.push_back(allocation);
        this->get_function()->starting_computations.erase(allocation);
    }

    this->store_in(buff);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::after(computation &comp, tiramisu::var level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(level.get_name().length() > 0);

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({level.get_name()});

    assert(dimensions.size() == 1);

    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
        level.get_name() + " is " + std::to_string(dimensions[0])));

    this->after(comp, dimensions[0]);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::after(computation &comp, int level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Scheduling " + this->get_name() + " to be executed after " +
                                comp.get_name() + " at level " + std::to_string(level)));
                                
    auto &graph = this->get_function()->sched_graph;

    auto &edges = graph[&comp];

    auto level_it = edges.find(this);

    if (level_it != edges.end())
    {
        if (level_it->second > level)
        {
            level = level_it->second;
        }
    }

    edges[this] = level;

    this->get_function()->starting_computations.erase(this);

    this->get_function()->sched_graph_reversed[this][&comp] = level;

    assert(this->get_function()->sched_graph_reversed[this].size() < 2 &&
            "Node has more than one predecessor.");

    DEBUG(10, tiramisu::str_dump("sched_graph[" + comp.get_name() + ", " +
                                 this->get_name() + "] = " + std::to_string(level)));

    DEBUG_INDENT(-4);
}


void tiramisu::computation::after_change(computation &comp,int level)
{
     DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Scheduling dynamic change " + this->get_name() + " to be executed after " +
                                comp.get_name() + " at level " + std::to_string(level)));
                                


    this->get_function()->sched_graph[&comp][this] = level ;

   

    this->get_function()->sched_graph_reversed[this][&comp] = level;

    

    DEBUG_INDENT(-4);
}



void tiramisu::computation::after_change(computation &comp,tiramisu::var level)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(level.get_name().length() > 0);

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({level.get_name()});

    assert(dimensions.size() == 1);

    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
        level.get_name() + " is " + std::to_string(dimensions[0])));

    this->after_change(comp, dimensions[0]);

    DEBUG_INDENT(-4);
}

void computation::before(computation &comp, tiramisu::var dim)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

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

void computation::between(computation &before_c, int before_dim, computation &after_c, int after_dim)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(before_dim >= computation::root_dimension);
    assert(after_dim >= computation::root_dimension);

    this->check_dimensions_validity({before_dim, after_dim});

    DEBUG(3, tiramisu::str_dump("Scheduling " + this->get_name() + " between " +
                                before_c.get_name() + " and " + after_c.get_name()));

    auto f = this->get_function();

    if (f->sched_graph[&before_c].find(&after_c) != f->sched_graph[&before_c].end()) {
        DEBUG(3, tiramisu::str_dump("Removing pre-existing edge"));
        f->sched_graph[&before_c].erase(&after_c);
        f->sched_graph_reversed[&after_c].erase(&before_c);
    }

    this->after(before_c, before_dim);
    after_c.after(*this, after_dim);

    DEBUG_INDENT(-4);
}

void computation::between(computation &before_c, tiramisu::var before_dim_var, computation &after_c, tiramisu::var after_dim_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(before_dim_var.get_name().length() > 0);
    assert(after_dim_var.get_name().length() > 0);

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({before_dim_var.get_name(), after_dim_var.get_name()});

    this->check_dimensions_validity(dimensions);

    this->between(before_c, dimensions[0], after_c, dimensions[1]);

    DEBUG_INDENT(-4);
}

computation& computation::then(computation &next_computation, tiramisu::var dim)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    next_computation.after(*this, dim);

    DEBUG_INDENT(-4);

    return next_computation;
}

computation& computation::then(computation &next_computation, int dim)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    next_computation.after(*this, dim);

    DEBUG_INDENT(-4);

    return next_computation;
}

void computation::gpu_tile(tiramisu::var L0_var, tiramisu::var L1_var, int sizeX, int sizeY)
{
    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(),
                                                           L1_var.get_name()});

    assert(dimensions.size() == 2);

    int L0 = dimensions[0];
    int L1 = dimensions[1];

    this->check_dimensions_validity({L0, L1});

    assert(L0 >= 0);
    assert(L1 >= 0);
    assert((L1 == L0 + 1));
    assert(sizeX > 0);
    assert(sizeY > 0);

    this->tile(L0, L1, sizeX, sizeY);
    this->tag_gpu_block_level(L0, L1);
    this->tag_gpu_thread_level(L0 + 2, L1 + 2);
    this->thread_block_shape.push_back(sizeX);
    this->thread_block_shape.push_back(sizeY);
}

void computation::gpu_tile(tiramisu::var L0, tiramisu::var L1, int sizeX, int sizeY,
                           tiramisu::var L0_outer, tiramisu::var L1_outer,
                           tiramisu::var L0_inner, tiramisu::var L1_inner)
{
    this->tile(L0, L1, sizeX, sizeY, L0_outer, L1_outer, L0_inner, L1_inner);
    this->tag_gpu_level(L0_outer, L1_outer, L0_inner, L1_inner);
    this->thread_block_shape.push_back(sizeX);
    this->thread_block_shape.push_back(sizeY);
}

void computation::gpu_tile(tiramisu::var L0_var, tiramisu::var L1_var, tiramisu::var L2_var, int sizeX, int sizeY, int sizeZ)
{
    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(L2_var.get_name().length() > 0);

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(),
                                                           L1_var.get_name(),
                                                           L2_var.get_name()});

    assert(dimensions.size() == 3);

    int L0 = dimensions[0];
    int L1 = dimensions[1];
    int L2 = dimensions[2];

    this->check_dimensions_validity({L0, L1, L2});

    assert((L1 == L0 + 1));
    assert((L2 == L1 + 1));
    assert(sizeX > 0);
    assert(sizeY > 0);
    assert(sizeZ > 0);

    this->tile(L0, L1, L2, sizeX, sizeY, sizeZ);
    this->tag_gpu_block_level(L0, L1, L2);
    this->tag_gpu_thread_level(L0 + 3, L1 + 3, L2 + 3);
    this->thread_block_shape.push_back(sizeX);
    this->thread_block_shape.push_back(sizeY);
    this->thread_block_shape.push_back(sizeZ);
}

void computation::gpu_tile(tiramisu::var L0, tiramisu::var L1, tiramisu::var L2,
                           int sizeX, int sizeY, int sizeZ,
                           tiramisu::var L0_outer, tiramisu::var L1_outer, tiramisu::var L2_outer,
                           tiramisu::var L0_inner, tiramisu::var L1_inner, tiramisu::var L2_inner)
{
    this->tile(L0, L1, L2, sizeX, sizeY, sizeZ, L0_outer, L1_outer, L2_outer, L0_inner, L1_inner, L2_inner);
    this->tag_gpu_level(L0_outer, L1_outer, L2_outer, L0_inner, L1_inner, L2_inner);
    this->thread_block_shape.push_back(sizeX);
    this->thread_block_shape.push_back(sizeY);
    this->thread_block_shape.push_back(sizeZ);
}

void computation::assert_names_not_assigned(
        std::vector<std::string> dimensions)
{
    for (auto const dim: dimensions)
    {
        int d = isl_map_find_dim_by_name(this->get_schedule(), isl_dim_out,
                                         dim.c_str());
        if (d >= 0)
        {
            ERROR("Dimension " + dim + " is already in use.", true);
        }

        d = isl_map_find_dim_by_name(this->get_schedule(), isl_dim_in,
                                     dim.c_str());
        if (d >= 0)
        {
            ERROR("Dimension " + dim + " is already in use.", true);
        }
    }
}

void computation::check_dimensions_validity(std::vector<int> dimensions)
{
    assert(dimensions.size() > 0);

    for (auto const dim: dimensions)
    {
        DEBUG(10, tiramisu::str_dump("Checking the validity of loop level " +
                                     std::to_string(dim)));

        assert(dim >= computation::root_dimension);

        if (loop_level_into_dynamic_dimension(dim) >=
            isl_space_dim(isl_map_get_space(this->get_schedule()),
                          isl_dim_out))
        {
            ERROR("The dynamic dimension " +
                            std::to_string(loop_level_into_dynamic_dimension(dim)) +
                            " is not less than the number of dimensions of the "
                            "time-space domain " +
                            std::to_string(isl_space_dim(isl_map_get_space(
                                    this->get_schedule()), isl_dim_out)), true);
        }
    }
}

void computation::set_loop_level_names(std::vector<std::string> names)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(names.size() > 0);

    DEBUG(3, tiramisu::str_dump("Number of loop levels: " + std::to_string(this->get_loop_levels_number())));
    DEBUG(3, tiramisu::str_dump("Number of names to be set: " + std::to_string(names.size())));

    for (int i = 0; i < names.size(); i++)
    {
        if (isl_map_has_dim_name(this->get_schedule(), isl_dim_out, loop_level_into_dynamic_dimension(i)) == isl_bool_true)
        {
            this->schedule = isl_map_set_dim_name(this->get_schedule(),
                                                  isl_dim_out,
                                                  loop_level_into_dynamic_dimension(i),
                                                  names[i].c_str());
            DEBUG(3, tiramisu::str_dump("Setting the name of loop level " + std::to_string(i) + " into " + names[i].c_str()));
        }
    }

    DEBUG(3, tiramisu::str_dump("The schedule after renaming: ", isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::set_schedule_domain_dim_names(std::vector<int> loop_levels,
        std::vector<std::string> names)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    this->check_dimensions_validity(loop_levels);
    assert(names.size() > 0);
    assert(names.size() == loop_levels.size());

    for (int i = 0; i < loop_levels.size(); i++)
    {
        assert(loop_levels[i] <= isl_map_dim(this->get_schedule(), isl_dim_in));
        this->schedule = isl_map_set_dim_name(this->get_schedule(),
                                              isl_dim_in, loop_levels[i], names[i].c_str());
        DEBUG(3, tiramisu::str_dump("Setting the name of the domain of the schedule dimension " + std::to_string(loop_levels[i]) + " into " + names[i].c_str()));
    }

    DEBUG(3, tiramisu::str_dump("The schedule after renaming: ", isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::set_loop_level_names(std::vector<int> loop_levels,
        std::vector<std::string> names)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    this->check_dimensions_validity(loop_levels);
    assert(names.size() > 0);
    assert(names.size() == loop_levels.size());

    for (int i = 0; i < loop_levels.size(); i++)
    {
        if (loop_level_into_static_dimension(loop_levels[i]) <= isl_map_dim(this->get_schedule(), isl_dim_out))
        {
            this->schedule = isl_map_set_dim_name(this->get_schedule(),
                                                  isl_dim_out,
                                                  loop_level_into_dynamic_dimension(loop_levels[i]),
                                                  names[i].c_str());
            DEBUG(3, tiramisu::str_dump("Setting the name of loop level " + std::to_string(loop_levels[i]) + " into " + names[i].c_str()));
        }
    }

    DEBUG(3, tiramisu::str_dump("The schedule after renaming: ", isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::tile(int L0, int L1, int sizeX, int sizeY)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Check that the two dimensions are consecutive.
    // Tiling only applies on a consecutive band of loop dimensions.
    assert(L1 == L0 + 1);
    assert((sizeX > 0) && (sizeY > 0));
    assert(this->get_iteration_domain() != NULL);
    this->check_dimensions_validity({L0, L1});

    this->split(L0, sizeX);
    this->split(L1 + 1, sizeY);

    this->interchange(L0 + 1, L1 + 1);

    DEBUG_INDENT(-4);
}

std::vector<int> computation::get_loop_level_numbers_from_dimension_names(
        std::vector<std::string> dim_names)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(dim_names.size() > 0);

    std::vector<int> dim_numbers;

    for (auto const dim: dim_names)
    {
        assert(dim.size()>0);

        DEBUG(10, tiramisu::str_dump("Searching for the dimension " + dim));

        if (dim == "root")
        {
            int d = computation::root_dimension;
            dim_numbers.push_back(d);
        }
        else
        {
            int d = isl_map_find_dim_by_name(this->get_schedule(), isl_dim_out,
                                             dim.c_str());
            DEBUG(10, tiramisu::str_dump("Searching in the range of ",
                                         isl_map_to_str(this->get_schedule())));

            if (d < 0)
            {
                ERROR("Dimension " + dim + " not found.", true);
            }

            DEBUG(10, tiramisu::str_dump("Corresponding loop level is " +
                                         std::to_string(dynamic_dimension_into_loop_level(d))));

            dim_numbers.push_back(dynamic_dimension_into_loop_level(d));
        }
    }

    this->check_dimensions_validity(dim_numbers);

    DEBUG_INDENT(-4);

    return dim_numbers;
}

void computation::name_unnamed_time_space_dimensions()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_map *sched = this->get_schedule();

    assert(sched != NULL);

    for (int i = 0; i < this->get_loop_levels_number(); i++)
    {
        if (isl_map_has_dim_name(sched, isl_dim_out, loop_level_into_dynamic_dimension(i)) == isl_bool_false)
            sched = isl_map_set_dim_name(sched, isl_dim_out, loop_level_into_dynamic_dimension(i), generate_new_variable_name().c_str());
    }

    this->set_schedule(sched);

    DEBUG_INDENT(-4);
}

void computation::name_unnamed_iteration_domain_dimensions()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_set *iter = this->get_iteration_domain();

    assert(iter != NULL);

    for (int i = 0; i < this->get_iteration_domain_dimensions_number(); i++)
    {
        if (isl_set_has_dim_name(iter, isl_dim_set, i) == isl_bool_false)
            iter = isl_set_set_dim_name(iter, isl_dim_set, i,
                                        generate_new_variable_name().c_str());
    }

    this->set_iteration_domain(iter);

    DEBUG_INDENT(-4);
}

std::vector<std::string> computation::get_iteration_domain_dimension_names()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    isl_set *iter = this->get_iteration_domain();

    assert(iter != NULL);

    std::vector<std::string> result;

    for (int i = 0; i < this->get_iteration_domain_dimensions_number(); i++)
    {
        if (isl_set_has_dim_name(iter, isl_dim_set, i))
            result.push_back(std::string(isl_set_get_dim_name(iter,
                                                              isl_dim_set, i)));
        else
        {
            ERROR("All iteration domain dimensions must have "
                            "a name.", true);
        }
    }

    assert(result.size() == this->get_iteration_domain_dimensions_number());

    DEBUG_INDENT(-4);

    return result;
}

void computation::tile(tiramisu::var L0, tiramisu::var L1,
        tiramisu::var L2, int sizeX, int sizeY, int sizeZ)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);
    assert(L2.get_name().length() > 0);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L1_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L2_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());
    tiramisu::var L1_inner = tiramisu::var(generate_new_variable_name());
    tiramisu::var L2_inner = tiramisu::var(generate_new_variable_name());

    this->tile(L0, L1, L2, sizeX, sizeY, sizeZ,
               L0_outer, L1_outer, L0_outer, L0_inner, L1_inner, L2_inner);

    DEBUG_INDENT(-4);
}

void computation::tile(tiramisu::var L0, tiramisu::var L1,
        int sizeX, int sizeY)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L1_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());
    tiramisu::var L1_inner = tiramisu::var(generate_new_variable_name());

    this->tile(L0, L1, sizeX, sizeY,
               L0_outer, L1_outer, L0_inner, L1_inner);

    DEBUG_INDENT(-4);
}

void computation::tile(tiramisu::var L0, tiramisu::var L1, tiramisu::var L2,
        int sizeX, int sizeY, int sizeZ,
        tiramisu::var L0_outer, tiramisu::var L1_outer,
        tiramisu::var L2_outer, tiramisu::var L0_inner,
        tiramisu::var L1_inner, tiramisu::var L2_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);
    assert(L2.get_name().length() > 0);
    assert(L0_outer.get_name().length() > 0);
    assert(L1_outer.get_name().length() > 0);
    assert(L2_outer.get_name().length() > 0);
    assert(L0_inner.get_name().length() > 0);
    assert(L1_inner.get_name().length() > 0);
    assert(L2_inner.get_name().length() > 0);

    this->assert_names_not_assigned({L0_outer.get_name(), L1_outer.get_name(),
                                     L2_outer.get_name(), L0_inner.get_name(),
                                     L1_inner.get_name(), L2_inner.get_name()});

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0.get_name(),
                                                           L1.get_name(),
                                                           L2.get_name()});
    assert(dimensions.size() == 3);

    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
                                L0.get_name() + " is " + std::to_string(dimensions[0])));
    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
                                L1.get_name() + " is " + std::to_string(dimensions[1])));
    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
                                L2.get_name() + " is " + std::to_string(dimensions[2])));

    this->tile(dimensions[0], dimensions[1], dimensions[2],
               sizeX, sizeY, sizeZ);

    this->update_names(original_loop_level_names, {L0_outer.get_name(), L1_outer.get_name(), L2_outer.get_name(),
                                                   L0_inner.get_name(), L1_inner.get_name(), L2_inner.get_name()}, dimensions[0], 3);

    DEBUG_INDENT(-4);
}

void computation::tile(tiramisu::var L0, tiramisu::var L1,
      int sizeX, int sizeY,
      tiramisu::var L0_outer, tiramisu::var L1_outer,
      tiramisu::var L0_inner, tiramisu::var L1_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0.get_name().length() > 0);
    assert(L1.get_name().length() > 0);
    assert(L0_outer.get_name().length() > 0);
    assert(L1_outer.get_name().length() > 0);
    assert(L0_inner.get_name().length() > 0);
    assert(L1_inner.get_name().length() > 0);

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    this->assert_names_not_assigned({L0_outer.get_name(), L1_outer.get_name(),
                                     L0_inner.get_name(), L1_inner.get_name()});

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0.get_name(),
                                                           L1.get_name()});
    assert(dimensions.size() == 2);

    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
                                L0.get_name() + " is " + std::to_string(dimensions[0])));
    DEBUG(3, tiramisu::str_dump("The loop level that corresponds to " +
                                L1.get_name() + " is " + std::to_string(dimensions[1])));

    this->tile(dimensions[0], dimensions[1], sizeX, sizeY);

    // Replace the original dimension name with new dimension names
    this->update_names(original_loop_level_names, {L0_outer.get_name(), L1_outer.get_name(), L0_inner.get_name(), L1_inner.get_name()}, dimensions[0], 2);

    DEBUG_INDENT(-4);
}

void computation::tile(int L0, int L1, int L2, int sizeX, int sizeY, int sizeZ)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    // Check that the two dimensions are consecutive.
    // Tiling only applies on a consecutive band of loop dimensions.
    assert(L1 == L0 + 1);
    assert(L2 == L1 + 1);
    assert((sizeX > 0) && (sizeY > 0) && (sizeZ > 0));
    assert(this->get_iteration_domain() != NULL);

    this->check_dimensions_validity({L0, L1, L2});

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


void computation::interchange(tiramisu::var L0_var, tiramisu::var L1_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];

    this->interchange(L0, L1);

    DEBUG_INDENT(-4);
}

/**
 * This function modifies the schedule of the computation so that the two loop
 * levels L0 and L1 are interchanged (swapped).
 */
void computation::interchange(int L0, int L1)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int inDim0 = loop_level_into_dynamic_dimension(L0);
    int inDim1 = loop_level_into_dynamic_dimension(L1);

    assert(inDim0 >= 0);
    assert(inDim0 < isl_space_dim(isl_map_get_space(this->get_schedule()),
                                  isl_dim_out));
    assert(inDim1 >= 0);
    assert(inDim1 < isl_space_dim(isl_map_get_space(this->get_schedule()),
                                  isl_dim_out));

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

    DEBUG(3, tiramisu::str_dump("The schedule of the original computation: "); isl_map_dump(this->get_schedule()));
    DEBUG(3, tiramisu::str_dump("The schedule of the duplicate: "); isl_map_dump(new_c->get_schedule()));

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
void computation::angle_skew(tiramisu::var L0_var, tiramisu::var L1_var,
		       int f_i, int f_j ,bool fuze_plans ,
		       tiramisu::var new_L0_var, tiramisu::var new_L1_var)
{
    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(new_L0_var.get_name().length() > 0);
    assert(new_L1_var.get_name().length() > 0);
    this->assert_names_not_assigned({new_L0_var.get_name(), new_L1_var.get_name()});
     std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name()});

    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];
    this->angle_skew(L0, L1, f_i,f_j ,fuze_plans);
    this->update_names(original_loop_level_names, {new_L0_var.get_name(), new_L1_var.get_name()}, dimensions[0], 2);

}

void computation::angle_skew(int L0 , int L1 , int f_i , int f_j,bool fuze_plans)
{
    if (L0 + 1 != L1)
    {
	ERROR("Loop levels passed to angle_skew() should be consecutive. The first argument to angle_skew() should be the outer loop level.", true);
    }

    //assert(f_i!=0) ;
    //assert(f_j>0) ;

    if(fuze_plans)
    {
        if((f_i!=1)&&(f_j!=1)){
            ERROR(" one of factor must be 1 of a or b", true);
        }
    }
    int dim0 = loop_level_into_dynamic_dimension(L0);
    int dim1 = loop_level_into_dynamic_dimension(L1);

    assert(this->get_schedule() != NULL);
    assert(dim0 >= 0);
    assert(dim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));
    isl_map *schedule = this->get_schedule();
    int duplicate_ID = isl_map_get_static_dim(schedule, 0);

    schedule = isl_map_copy(schedule);
    schedule = isl_map_set_tuple_id(schedule, isl_dim_out,
                                    isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL));

    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(schedule)));
    DEBUG(3, tiramisu::str_dump("Angle _ Skewing dimensions " + std::to_string(dim0)
                                + " and " + std::to_string(dim1)));

    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(schedule)));
    DEBUG(3, tiramisu::str_dump("Angle Skewing dimensions " + std::to_string(dim0)
                                + " and " + std::to_string(dim1)));

    std::string inDim0_str, inDim1_str;

    std::string outDim1_str = generate_new_variable_name();

    std::string outDim0_str = generate_new_variable_name() ;

    int n_dims = isl_map_dim(this->get_schedule(), isl_dim_out);
    std::vector<isl_id *> dimensions;
    std::vector<std::string> dimensions_str;
    std::string map = "{";
    // -----------------------------------------------------------------
    // Preparing a map to skew the duplicate computation.
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

            if (i == dim0)
                inDim0_str = dim_str;
            else if (i == dim1)
                inDim1_str = dim_str;
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    //std::cout<<("\n hello dimentions "+inDim0_str+" / "+inDim1_str+" \n") ;

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
        else if ((i != dim1)&&(i!=dim0))
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else // i==dim1
        {
            if(i==dim1){
                  map = map + outDim1_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim1_str.c_str(), NULL);
            dimensions.push_back(id0);

            }
            else{// i== dim 0 
                  map = map + outDim0_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim0_str.c_str(), NULL);
            dimensions.push_back(id0);

            }
          
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    // f_j not used yet - test 
    // compute all sigma , gamma such j = sigma*i + gamma*j

   if(!fuze_plans)
   {
        int gamma = 0 ;
        int sigma = 1 ;
        bool found = false ;

        assert(f_j!=0) ;
        assert(f_i>=0) ;

        if ( (f_j == 1 )||(f_i ==1 )){

                    gamma = f_i -1 ;
            
        }
        else{ // [b] positif & a un sens

            if((f_j == -1)&&(f_i>1))
            {
                    gamma = 1; sigma = 0 ; //interchange 
            }
            else{

                int i =0 ;
                while((i<10)&&(!found)){
                    if ((  (sigma * f_i )% abs(f_j) ) ==  1){
                            found = true ;
                    }
                    else{
                        sigma ++ ;
                        i++;
                    }
                };

                if(!found){
                    ERROR(" det A too complex without gains ", true);
                }

                gamma = ( (sigma * f_i)-1 ) / f_j ;
            }

         }
        map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID) + " and " +
            outDim0_str + " = (" + inDim0_str + "*"+std::to_string(f_i)+" + "+inDim1_str+"*"+std::to_string(f_j)+" ) and "
            
          +outDim1_str+" = ("+inDim0_str+"*"+std::to_string(gamma)+" + "+inDim1_str+"*"+std::to_string(sigma)+" ) }";
          
    
    }
    else{ 
        // fuze many indepandants plans
            assert(f_i>0);
            assert(f_j>0) ;

            if ( f_i > f_j)
            {
                assert(f_j == 1);
                // change dim1 only
                

                map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID) + " and " +
              outDim0_str + " = ( floor(" + inDim1_str + "/"+std::to_string(f_i)+") +"+inDim0_str+") and  "+outDim1_str+"="+inDim1_str+" }";

            }
            else{
                assert(f_i == 1);
                // implicite interchange

                 map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID) + " and " +
              outDim0_str + " = ( floor(" + inDim0_str + "/"+std::to_string(f_j)+") +"+inDim1_str+") and  "+outDim1_str+"="+inDim0_str+" }";

            }

    }

    /*
    map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID) + " and " +
          outDim0_str + " = ( floor(" + inDim1_str + "/"+std::to_string(f_i)+") +"+inDim0_str+") and  "+outDim1_str+"="+inDim1_str+" }";*/



    DEBUG(3, tiramisu::str_dump("Transformation angle map (string format) : " + map));

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

    if (isl_map_is_injective(transformation_map) == isl_bool_true){
        DEBUG(3, tiramisu::str_dump(" This map is injective  "));
    }
    else{
        DEBUG(3, tiramisu::str_dump(" This map is not injective  "));
    }

    this->set_schedule(schedule);

}

void computation::loop_reversal(tiramisu::var old_var,tiramisu::var new_var)
{

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(old_var.get_name().length() > 0);
    assert(new_var.get_name().length() > 0);
    this->assert_names_not_assigned({new_var.get_name()});

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({old_var.get_name()});

    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    this->loop_reversal(L0);
    this->update_names(original_loop_level_names, {new_var.get_name()}, dimensions[0], 1);

    DEBUG_INDENT(-4);

}

void computation::loop_reversal(int L0)
{
     
    int dim0 = loop_level_into_dynamic_dimension(L0);
    assert(this->get_schedule() != NULL);
    assert(dim0 >= 0);
    assert(dim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));
    isl_map *schedule = this->get_schedule();
    int duplicate_ID = isl_map_get_static_dim(schedule, 0);

    schedule = isl_map_copy(schedule);
    schedule = isl_map_set_tuple_id(schedule, isl_dim_out,
                                    isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL));

    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(schedule)));
    DEBUG(3, tiramisu::str_dump(" reversing the iteration direction " + std::to_string(dim0)));

    std::string inDim0_str;

    //std::string outDim1_str = generate_new_variable_name();

    std::string outDim0_str = generate_new_variable_name() ;

    int n_dims = isl_map_dim(this->get_schedule(), isl_dim_out);
    std::vector<isl_id *> dimensions;
    std::vector<std::string> dimensions_str;
    std::string map = "{";
    // -----------------------------------------------------------------
    // Preparing a map to change computation schedule
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

            if (i == dim0)
                inDim0_str = dim_str;
            
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
        else if (i!=dim0)
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else // i==dim0
        {
          
                  map = map + outDim0_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim0_str.c_str(), NULL);
            dimensions.push_back(id0);

            
          
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID) + " and " +
            outDim0_str + " = ( -1*" + inDim0_str + " ) }";
          
    

    /*
    map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID) + " and " +
          outDim0_str + " = ( floor(" + inDim1_str + "/"+std::to_string(f_i)+") +"+inDim0_str+") and  "+outDim1_str+"="+inDim1_str+" }";*/



    DEBUG(3, tiramisu::str_dump("Transformation 1 var reversed map (string format) : " + map));

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

    

    this->set_schedule(schedule);
}



void computation::skew(tiramisu::var L0_var, tiramisu::var L1_var,
		       int factor,
		       tiramisu::var new_L0_var, tiramisu::var new_L1_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(new_L0_var.get_name().length() > 0);
    assert(new_L1_var.get_name().length() > 0);
    assert(factor >= 1);

    this->assert_names_not_assigned({new_L0_var.get_name(), new_L1_var.get_name()});

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name()});

    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];

    this->skew(L0, L1, factor);

    this->update_names(original_loop_level_names, {new_L0_var.get_name(), new_L1_var.get_name()}, dimensions[0], 2);

    DEBUG_INDENT(-4);
}

void computation::skew(tiramisu::var L0_var, tiramisu::var L1_var, tiramisu::var L2_var,
		       int factor,
		       tiramisu::var new_L0_var, tiramisu::var new_L1_var, tiramisu::var new_L2_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(L2_var.get_name().length() > 0);
    assert(new_L0_var.get_name().length() > 0);
    assert(new_L1_var.get_name().length() > 0);
    assert(new_L2_var.get_name().length() > 0);
    assert(factor >= 1);

    this->assert_names_not_assigned({new_L0_var.get_name(), new_L1_var.get_name(), new_L2_var.get_name()});

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name(), L2_var.get_name()});

    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];
    int L2 = dimensions[2];

    this->skew(L0, L1, L2, factor);

    this->update_names(original_loop_level_names, {new_L0_var.get_name(), new_L1_var.get_name(), new_L2_var.get_name()}, dimensions[0], 3);

    DEBUG_INDENT(-4);
}

void computation::skew(tiramisu::var L0_var, tiramisu::var L1_var, tiramisu::var L2_var, tiramisu::var L3_var,
		       int factor,
		       tiramisu::var new_L0_var, tiramisu::var new_L1_var, tiramisu::var new_L2_var, tiramisu::var new_L3_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(L2_var.get_name().length() > 0);
    assert(L3_var.get_name().length() > 0);
    assert(new_L0_var.get_name().length() > 0);
    assert(new_L1_var.get_name().length() > 0);
    assert(new_L2_var.get_name().length() > 0);
    assert(new_L3_var.get_name().length() > 0);
    assert(factor >= 1);

    this->assert_names_not_assigned({new_L0_var.get_name(), new_L1_var.get_name(),
				     new_L2_var.get_name(), new_L3_var.get_name()});

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name(), L1_var.get_name(),
							   L2_var.get_name(), L3_var.get_name()});

    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    int L1 = dimensions[1];
    int L2 = dimensions[2];
    int L3 = dimensions[3];

    this->skew(L0, L1, L2, L3, factor);

    this->update_names(original_loop_level_names, {new_L0_var.get_name(), new_L1_var.get_name(),
						   new_L2_var.get_name(), new_L3_var.get_name()}, dimensions[0], 4);

    DEBUG_INDENT(-4);
}

void computation::skew(tiramisu::var L0_var, tiramisu::var L1_var, int factor)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(factor >= 1);

    tiramisu::var new_L0_var = tiramisu::var(generate_new_variable_name());
    tiramisu::var new_L1_var = tiramisu::var(generate_new_variable_name());

    this->skew(L0_var, L1_var, factor, new_L0_var, new_L1_var);

    DEBUG_INDENT(-4);
}

void computation::skew(tiramisu::var L0_var, tiramisu::var L1_var, tiramisu::var L2_var,
		       int factor)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(L2_var.get_name().length() > 0);
    assert(factor >= 1);

    tiramisu::var new_L0_var = tiramisu::var(generate_new_variable_name());
    tiramisu::var new_L1_var = tiramisu::var(generate_new_variable_name());
    tiramisu::var new_L2_var = tiramisu::var(generate_new_variable_name());

    this->skew(L0_var, L1_var, L2_var, factor, new_L0_var, new_L1_var, new_L2_var);

    DEBUG_INDENT(-4);
}

void computation::skew(tiramisu::var L0_var, tiramisu::var L1_var,
		       tiramisu::var L2_var, tiramisu::var L3_var, int factor)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    assert(L1_var.get_name().length() > 0);
    assert(L2_var.get_name().length() > 0);
    assert(L3_var.get_name().length() > 0);
    assert(factor >= 1);

    tiramisu::var new_L0_var = tiramisu::var(generate_new_variable_name());
    tiramisu::var new_L1_var = tiramisu::var(generate_new_variable_name());
    tiramisu::var new_L2_var = tiramisu::var(generate_new_variable_name());
    tiramisu::var new_L3_var = tiramisu::var(generate_new_variable_name());

    this->skew(L0_var, L1_var, L2_var, L3_var, factor, new_L0_var, new_L1_var, new_L2_var, new_L3_var);

    DEBUG_INDENT(-4);
}

void computation::skew(int L0, int L1, int factor)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if (L0 + 1 != L1)
    {
	ERROR("Loop levels passed to skew() should be consecutive. The first argument to skew() should be the outer loop level.", true);
    }

    int dim0 = loop_level_into_dynamic_dimension(L0);
    int dim1 = loop_level_into_dynamic_dimension(L1);

    assert(this->get_schedule() != NULL);
    assert(dim0 >= 0);
    assert(dim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));


    DEBUG(3, tiramisu::str_dump("Creating a schedule that skews the two loop levels ");
          tiramisu::str_dump(std::to_string(L0));
          tiramisu::str_dump(" and ");
          tiramisu::str_dump(std::to_string(L1));
          tiramisu::str_dump(" of the computation ");
          tiramisu::str_dump(this->get_name()));

    this->get_function()->align_schedules();
    assert(this->get_schedule() != NULL);

    DEBUG(3, tiramisu::str_dump("Original schedule: ",
                                isl_map_to_str(this->get_schedule())));

    isl_map *schedule = this->get_schedule();
    int duplicate_ID = isl_map_get_static_dim(schedule, 0);

    schedule = isl_map_copy(schedule);
    schedule = isl_map_set_tuple_id(schedule, isl_dim_out,
                                    isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL));

    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(schedule)));
    DEBUG(3, tiramisu::str_dump("Skewing dimensions " + std::to_string(dim0)
                                + " and " + std::to_string(dim1)));

    std::string inDim0_str, inDim1_str;

    std::string outDim1_str = generate_new_variable_name();

    int n_dims = isl_map_dim(this->get_schedule(), isl_dim_out);
    std::vector<isl_id *> dimensions;
    std::vector<std::string> dimensions_str;
    std::string map = "{";

    // -----------------------------------------------------------------
    // Preparing a map to skew the duplicate computation.
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

            if (i == dim0)
                inDim0_str = dim_str;
            else if (i == dim1)
                inDim1_str = dim_str;
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
        else if (i != dim1)
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else
        {
            map = map + outDim1_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim1_str.c_str(), NULL);
            dimensions.push_back(id0);
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID) + " and " +
          outDim1_str + " = (" + std::to_string(factor) + "*" + inDim0_str + "+" + inDim1_str + ")}";

    DEBUG(3, tiramisu::str_dump("Transformation map (string format) : " + map));

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

    this->set_schedule(schedule);
    DEBUG(3, tiramisu::str_dump("Schedule after skewing: ",
                                isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::skew(int L0, int L1, int L2, int factor)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if (L0 + 1 != L1 || L1 + 1 != L2)
    {
	ERROR("Loop levels passed to skew() should be consecutive. The first argument to skew() should be the outer loop level.", true);
    }

    int dim0 = loop_level_into_dynamic_dimension(L0);
    int dim1 = loop_level_into_dynamic_dimension(L1);
    int dim2 = loop_level_into_dynamic_dimension(L2);

    assert(this->get_schedule() != NULL);
    assert(dim0 >= 0);
    assert(dim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));


    DEBUG(3, tiramisu::str_dump("Creating a schedule that skews the loop levels ");
          tiramisu::str_dump(std::to_string(L0));
          tiramisu::str_dump(", ");
          tiramisu::str_dump(std::to_string(L1));
          tiramisu::str_dump(" and ");
          tiramisu::str_dump(std::to_string(L2));
          tiramisu::str_dump(" of the computation ");
          tiramisu::str_dump(this->get_name()));

    this->get_function()->align_schedules();
    assert(this->get_schedule() != NULL);

    DEBUG(3, tiramisu::str_dump("Original schedule: ",
                                isl_map_to_str(this->get_schedule())));

    isl_map *schedule = this->get_schedule();
    int duplicate_ID = isl_map_get_static_dim(schedule, 0);

    schedule = isl_map_copy(schedule);
    schedule = isl_map_set_tuple_id(schedule, isl_dim_out,
                                    isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL));

    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(schedule)));
    DEBUG(3, tiramisu::str_dump("Skewing dimensions " + std::to_string(dim0)
                                + ", " + std::to_string(dim1) + " and " + std::to_string(dim2)));

    std::string inDim0_str, inDim1_str, inDim2_str;

    std::string outDim1_str = generate_new_variable_name();
    std::string outDim2_str = generate_new_variable_name();

    int n_dims = isl_map_dim(this->get_schedule(), isl_dim_out);
    std::vector<isl_id *> dimensions;
    std::vector<std::string> dimensions_str;
    std::string map = "{";

    // -----------------------------------------------------------------
    // Preparing a map to skew the duplicate computation.
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

            if (i == dim0)
                inDim0_str = dim_str;
            else if (i == dim1)
                inDim1_str = dim_str;
            else if (i == dim2)
                inDim2_str = dim_str;
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
        else if (i == dim1)
        {
            map = map + outDim1_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim1_str.c_str(), NULL);
            dimensions.push_back(id0);
        }
        else if (i == dim2)
        {
            map = map + outDim2_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim2_str.c_str(), NULL);
            dimensions.push_back(id0);
        }
        else
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                 this->get_ctx(),
                                 dimensions_str[i].c_str(),
                                 NULL));
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID)
	+ " and " + outDim1_str + " = (" + std::to_string(factor) + "*" + inDim0_str + "+" + inDim1_str + ")"
	+ " and " + outDim2_str + " = (" + std::to_string(factor) + "*" + inDim0_str + "+" + inDim2_str + ")"
	+ "}";

    DEBUG(3, tiramisu::str_dump("Transformation map (string format) : " + map));

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

    this->set_schedule(schedule);
    DEBUG(3, tiramisu::str_dump("Schedule after skewing: ",
                                isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::skew(int L0, int L1, int L2, int L3, int factor)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    if (L0 + 1 != L1 || L1 + 1 != L2 || L2 + 1 != L3)
    {
	ERROR("Loop levels passed to skew() should be consecutive. The first argument to skew() should be the outer loop level.", true);
    }

    int dim0 = loop_level_into_dynamic_dimension(L0);
    int dim1 = loop_level_into_dynamic_dimension(L1);
    int dim2 = loop_level_into_dynamic_dimension(L2);
    int dim3 = loop_level_into_dynamic_dimension(L3);

    assert(this->get_schedule() != NULL);
    assert(dim0 >= 0);
    assert(dim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));


    DEBUG(3, tiramisu::str_dump("Creating a schedule that skews the loop levels ");
          tiramisu::str_dump(std::to_string(L0));
          tiramisu::str_dump(", ");
          tiramisu::str_dump(std::to_string(L1));
          tiramisu::str_dump(", ");
          tiramisu::str_dump(std::to_string(L2));
          tiramisu::str_dump(" and ");
          tiramisu::str_dump(std::to_string(L3));
          tiramisu::str_dump(" of the computation ");
          tiramisu::str_dump(this->get_name()));

    this->get_function()->align_schedules();
    assert(this->get_schedule() != NULL);

    DEBUG(3, tiramisu::str_dump("Original schedule: ",
                                isl_map_to_str(this->get_schedule())));

    isl_map *schedule = this->get_schedule();
    int duplicate_ID = isl_map_get_static_dim(schedule, 0);

    schedule = isl_map_copy(schedule);
    schedule = isl_map_set_tuple_id(schedule, isl_dim_out,
                                    isl_id_alloc(this->get_ctx(), this->get_name().c_str(), NULL));

    DEBUG(3, tiramisu::str_dump("Original schedule: ", isl_map_to_str(schedule)));
    DEBUG(3, tiramisu::str_dump("Skewing dimensions " + std::to_string(dim0)
                                + ", " + std::to_string(dim1) + ", " + std::to_string(dim2) + " and " + std::to_string(dim3)));

    std::string inDim0_str, inDim1_str, inDim2_str, inDim3_str;

    std::string outDim1_str = generate_new_variable_name();
    std::string outDim2_str = generate_new_variable_name();
    std::string outDim3_str = generate_new_variable_name();

    int n_dims = isl_map_dim(this->get_schedule(), isl_dim_out);
    std::vector<isl_id *> dimensions;
    std::vector<std::string> dimensions_str;
    std::string map = "{";

    // -----------------------------------------------------------------
    // Preparing a map to skew the duplicate computation.
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

            if (i == dim0)
                inDim0_str = dim_str;
            else if (i == dim1)
                inDim1_str = dim_str;
            else if (i == dim2)
                inDim2_str = dim_str;
            else if (i == dim3)
                inDim3_str = dim_str;
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
        else if (i == dim1)
        {
            map = map + outDim1_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim1_str.c_str(), NULL);
            dimensions.push_back(id0);
        }
        else if (i == dim2)
        {
            map = map + outDim2_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim2_str.c_str(), NULL);
            dimensions.push_back(id0);
        }
        else if (i == dim3)
        {
            map = map + outDim3_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim3_str.c_str(), NULL);
            dimensions.push_back(id0);
        }
        else
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                 this->get_ctx(),
                                 dimensions_str[i].c_str(),
                                 NULL));
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] : " + dimensions_str[0] + " = " + std::to_string(duplicate_ID)
	+ " and " + outDim1_str + " = (" + std::to_string(factor) + "*" + inDim0_str + "+" + inDim1_str + ")"
	+ " and " + outDim2_str + " = (" + std::to_string(factor) + "*" + inDim0_str + "+" + inDim2_str + ")"
	+ " and " + outDim3_str + " = (" + std::to_string(factor) + "*" + inDim0_str + "+" + inDim3_str + ")"
	+ "}";

    DEBUG(3, tiramisu::str_dump("Transformation map (string format) : " + map));

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

    this->set_schedule(schedule);
    DEBUG(3, tiramisu::str_dump("Schedule after skewing: ",
                                isl_map_to_str(this->get_schedule())));

    DEBUG_INDENT(-4);
}

void computation::shift(tiramisu::var L0_var, int n)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    this->shift(L0, n);

    DEBUG_INDENT(-4);
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
 * Compute the needed area.
 */
std::vector<isl_set *> computation::compute_needed_and_produced(computation &consumer, int L,
        std::vector<std::string> &param_names)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<isl_set *> needed_and_produced;

    // Get the consumer domain and schedule and the producer domain and schedule
    isl_set *consumer_domain = isl_set_copy(consumer.get_iteration_domain());
    isl_map *consumer_sched = isl_map_copy(consumer.get_schedule());
    isl_set *producer_domain = isl_set_copy(this->get_iteration_domain());
    isl_map *producer_sched = isl_map_copy(this->get_schedule());

    // Compute the access relation of the consumer computation.
    std::vector<isl_map *> accesses_vector;
    generator::get_rhs_accesses(consumer.get_function(), &consumer, accesses_vector, false);
    assert(accesses_vector.size() > 0);

    DEBUG(3, tiramisu::str_dump("Computed RHS accesses:"));
    for (auto acc : accesses_vector)
    {
        DEBUG(3, tiramisu::str_dump(isl_map_to_str(acc)));
    }

    DEBUG(3, tiramisu::str_dump("Vector of accesses computed."));

    // Create a union map of the accesses to the producer.
    isl_map *consumer_accesses = NULL;

    DEBUG(10, tiramisu::str_dump("Computing a union map of accesses to the producer."));

    for (const auto a : accesses_vector)
    {
        std::string range_name = isl_map_get_tuple_name(isl_map_copy(a), isl_dim_out);

        if (range_name == this->get_name())
        {
            if (consumer_accesses == NULL)
                consumer_accesses = isl_map_copy(a);
            else
            {
                DEBUG(10, tiramisu::str_dump("consumer_accesses: ", isl_map_to_str(consumer_accesses)));
                DEBUG(10, tiramisu::str_dump("access: ", isl_map_to_str(a)));

                consumer_accesses = isl_map_union(isl_map_copy(a), consumer_accesses);
            }
        }
    }

    DEBUG(10, tiramisu::str_dump("Union computed."));

    DEBUG(10, tiramisu::str_dump("Intersecting the range and the domain of the following consumer_accesses: ", isl_map_to_str(consumer_accesses)));
    DEBUG(10, tiramisu::str_dump("with the following iteration domain: ", isl_set_to_str(this->get_iteration_domain())));

    consumer_accesses = isl_map_intersect_range(consumer_accesses,
                        isl_set_copy(this->get_iteration_domain()));
    consumer_accesses = isl_map_intersect_domain(consumer_accesses,
                        isl_set_copy(consumer.get_iteration_domain()));
    consumer_accesses = this->simplify(consumer_accesses);

    DEBUG(3, tiramisu::str_dump("Accesses after keeping only those that have the producer in the range: "));
    DEBUG(3, tiramisu::str_dump(isl_map_to_str(consumer_accesses)));

    // Simplify
    consumer_domain = this->simplify(consumer_domain);
    consumer_sched = this->simplify(consumer_sched);
    producer_sched = this->simplify(producer_sched);
    producer_domain = this->simplify(producer_domain);

    // Transform, into time-processor, the consumer domain and schedule and the producer domain and schedule and the access relation
    consumer_domain = isl_set_apply(consumer_domain, isl_map_copy(consumer_sched));
    assert(consumer_domain != NULL);
    producer_domain = isl_set_apply(producer_domain, isl_map_copy(producer_sched));
    assert(producer_domain != NULL);

    // Transform the consumer accesses to the time-space domain.
    // For each access of the consumer:
    //    - Apply the schedule of the consumer on the domain of the access,
    //    - Get the producer (range) involved in that access,
    //    - Get the schedule of that producer,
    //    - Apply that schedule on the range of the access,
    //    - Add the resulting schedule to the union representing the result.
    {
        DEBUG(3, tiramisu::str_dump("Applying consumer_sched on the domain of consumer_accesses."));
        DEBUG(3, tiramisu::str_dump("consumer_sched: ", isl_map_to_str(consumer_sched)));
        DEBUG(3, tiramisu::str_dump("consumer_accesses: ", isl_map_to_str(consumer_accesses)));

        consumer_accesses = isl_map_apply_domain(isl_map_copy(consumer_accesses),
                                                 isl_map_copy(consumer_sched));
        assert(consumer_accesses != NULL);

        DEBUG(3, tiramisu::str_dump("Applying it on the range."));

        consumer_accesses = isl_map_apply_range(isl_map_copy(consumer_accesses),
                                                isl_map_copy(producer_sched));
        assert(consumer_accesses != NULL);

        DEBUG(3, tiramisu::str_dump("")); DEBUG(3, tiramisu::str_dump(""));
        DEBUG(3, tiramisu::str_dump("Consumer domain (in time-processor): ",
                                    isl_set_to_str(consumer_domain)));
        DEBUG(3, tiramisu::str_dump("Consumer accesses (in time-processor): ",
                                    isl_map_to_str(consumer_accesses)));
        DEBUG(3, tiramisu::str_dump("Producer domain (in time-processor): ",
                                    isl_set_to_str(producer_domain)));
    }

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


    // Compute needed = consumer_access(consumer_domain)
    isl_set *needed = isl_set_apply(isl_set_copy(consumer_domain), isl_map_copy(consumer_accesses));
    needed = this->simplify(needed);
    DEBUG(3, tiramisu::str_dump("Needed in time-processor = consumer_access(consumer_domain) in time-processor: ",
                                isl_set_to_str(needed)));

    needed_and_produced.push_back(needed);
    needed_and_produced.push_back(producer_domain);

    DEBUG_INDENT(-4);

    return needed_and_produced;
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
// TODO: Test the case when \p consumer does not consume this computation.
void computation::compute_at(computation &consumer, int L)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L > 0);

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

    // Compute needed
    std::vector<std::string> param_names;
    std::vector<isl_set *> needed_and_produced = this->compute_needed_and_produced(consumer, L,
            param_names);
    isl_set *needed = needed_and_produced[0];
    isl_set *producer_domain = needed_and_produced[1];

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

    if (!isl_set_is_empty(missing))
    {
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
        this->updates.push_back(duplicated_computation);
        DEBUG(3, tiramisu::str_dump("Producer duplicated. Dumping the schedule of the original computation."));
        original_computation->dump_schedule();
        DEBUG(3, tiramisu::str_dump("Dumping the schedule of the duplicate computation."));
        duplicated_computation->dump_schedule();

        DEBUG(3, tiramisu::str_dump("Now setting the duplicate with regard to the other computations."));
        original_computation->after((*duplicated_computation), L);
        consumer.after((*original_computation), L);

        // Computing the shift degrees.
        for (int i = 0; i <= L; i++)
            if (shift_degrees[i] != 0)
            {
                DEBUG(3, tiramisu::str_dump("Now shifting the duplicate by " + std::to_string(
                                                shift_degrees[i]) + " at loop level " + std::to_string(i)));
                duplicated_computation->shift(i, shift_degrees[i]);
            }
    }
    else
    {
        tiramisu::computation *original_computation = this;
        consumer.after((*original_computation), L);
    }
    DEBUG(3, tiramisu::str_dump("Dumping the schedule of the producer and consumer."));
    this->dump_schedule();
    consumer.dump_schedule();

    DEBUG_INDENT(-4);
}

/**
  * Wrapper around compute_at(computation &consumer, int L).
  */
void computation::compute_at(computation &consumer, tiramisu::var L_var)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L_var.get_name().size() > 0);

    std::vector<int> dimensions = consumer.get_loop_level_numbers_from_dimension_names({L_var.get_name()});
    assert(dimensions.size() == 1);

    int L = dimensions[0];

    this->compute_at(consumer, L);

    DEBUG_INDENT(-4);
}

/**
 * Return true if \p cst is a simple constraint, i.e., it satisfies the
 * following conditions:
 *  - It involves only the dimension \p dim and does not involve any
 *    other dimension,
 *  - It has 1 as a coefficient for \p dim
 */
bool isl_constraint_is_simple(isl_constraint *cst, int dim)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    bool simple = true;

    isl_space *space = isl_constraint_get_space(cst);
    for (int i = 0; i < isl_space_dim(space, isl_dim_set); i++)
        if (i != dim)
            if (isl_constraint_involves_dims(cst, isl_dim_set, i, 1))
            {
                DEBUG(10, tiramisu::str_dump("Constraint involves multiple dimensions"));
                simple = false;
            }

    isl_val *coeff = isl_constraint_get_coefficient_val(cst, isl_dim_set, dim);
    if ((isl_val_is_negone(coeff) == isl_bool_false) && (isl_val_is_one(coeff) == isl_bool_false))
    {
        DEBUG(10, tiramisu::str_dump("Coefficient of the dimension is not one/negative(one).");
              isl_val_dump(coeff));
        simple = false;
    }

    DEBUG_INDENT(-4);

    return simple;
}


/**
 * Extract a tiramisu expression that represents the bound on the dimension
 * \p dim in the constraint \p cst.
 *
 * If \p upper is true, then the bound is an upper bound, otherwise the bound
 * is a lower bound.
 */
tiramisu::expr extract_tiramisu_expr_from_cst(isl_constraint *cst, int dim, bool upper)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(cst != NULL);

    isl_space *space = isl_constraint_get_space(cst);
    tiramisu::expr e = tiramisu::expr();

    DEBUG(10, tiramisu::str_dump("Computing the expression that correspond to the following constraint at dimension "
                                 + std::to_string(dim) + " : "));
    DEBUG(10, isl_constraint_dump(cst));

    // Add the parameter to the expression
    for (int i = 0; i < isl_space_dim(space, isl_dim_param); i++)
    {
        isl_val *coeff = isl_constraint_get_coefficient_val(cst, isl_dim_param, i);
        if (isl_val_is_zero(coeff) == isl_bool_false)
        {
            const char *name = isl_space_get_dim_name(space, isl_dim_param, i);
            tiramisu::expr param = tiramisu::var(global::get_loop_iterator_data_type(), std::string(name));
            if (isl_val_is_one(coeff) == isl_bool_false)
            {
                long c = isl_val_get_num_si(coeff);

                // For lower bounds, inverse the sign.
                if (upper == false)
                {
                    c = -1 * c;
                }

                param = tiramisu::expr(o_mul,
                                       tiramisu::expr(o_cast, tiramisu::global::get_loop_iterator_data_type(),
                                                      tiramisu::expr((int32_t) c)), param);
            }

            if (e.is_defined() == false)
            {
                e = param;
            }
            else
            {
                e = tiramisu::expr(o_add, e, param);
            }
        }
    }

    isl_val *ct = isl_constraint_get_constant_val(cst);
    if ((isl_val_is_zero(ct) == isl_bool_false) || (e.is_defined() == false))
    {
        long v = isl_val_get_num_si(ct);

        // For lower bounds, inverse the sign.
        if (upper == false)
        {
            v = -1 * v;
        }

        tiramisu::expr c = tiramisu::expr(o_cast, global::get_loop_iterator_data_type(), tiramisu::expr((int32_t) v));

        if (e.is_defined() == false)
        {
            e = c;
        }
        else
        {
            e = tiramisu::expr(o_add, e, c);
        }
    }

    DEBUG(10, tiramisu::str_dump("The expression that correspond to the expression is : ");
          e.dump(false));
    DEBUG_INDENT(-4);

    return e;
}

int compute_recursively_max_AST_depth(isl_ast_node *node)
{
    assert(node != NULL);

    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    int result = -1;

    DEBUG(10, tiramisu::str_dump("Computing maximal AST depth from the following ISL AST node "););
    DEBUG(10, tiramisu::str_dump("\n"); tiramisu::str_dump(std::string(isl_ast_node_to_C_str(node))));

    if (isl_ast_node_get_type(node) == isl_ast_node_block)
    {
        DEBUG(10, tiramisu::str_dump("Computing maximal depth from a block."));

        isl_ast_node_list *list = isl_ast_node_block_get_children(node);
        isl_ast_node *child = isl_ast_node_list_get_ast_node(list, 0);
        result = compute_recursively_max_AST_depth(child);

        for (int i = 1; i < isl_ast_node_list_n_ast_node(list); i++)
        {
            child = isl_ast_node_list_get_ast_node(list, i);
            result = std::max(result, compute_recursively_max_AST_depth(child));
        }
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_for)
    {
        DEBUG(10, tiramisu::str_dump("Computing maximal depth from a for loop."));
        isl_ast_node *body = isl_ast_node_for_get_body(node);
        result = compute_recursively_max_AST_depth(body) + 1;
        isl_ast_node_free(body);
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_user)
    {
        DEBUG(10, tiramisu::str_dump("Reached a user node."));
        result = 1;
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_if)
    {
        DEBUG(10, tiramisu::str_dump("Computing maximal depth from an if conditional."));

        result = compute_recursively_max_AST_depth(isl_ast_node_if_get_then(node));

        if (isl_ast_node_if_has_else(node))
            result = std::max(result, compute_recursively_max_AST_depth(isl_ast_node_if_get_else(node)));
    }
    else
    {
        ERROR("Found an unsupported ISL AST node while computing the maximal AST depth.", true);
    }

    DEBUG(3, tiramisu::str_dump("Current depth = " + std::to_string(result)));
    DEBUG_INDENT(-4);

    return result;
}

/**
  * Traverse recursively the ISL AST tree
  *
  * \p node represents the root of the tree to be traversed.
  *
  * \p dim is the dimension of the loop from which the bounds have to be
  * extracted.
  *
  * \p upper is a boolean that should be set to true to extract
  * the upper bound and false to extract the lower bound.
  */
tiramisu::expr utility::extract_bound_expression(isl_ast_node *node, int dim, bool upper)
{
    assert(node != NULL);
    assert(dim >= 0);

    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::expr result;

    DEBUG(3, tiramisu::str_dump("Extracting bounds from a loop at depth = " + std::to_string(dim)));
    DEBUG(3, tiramisu::str_dump("Extracting bounds from the following ISL AST node "));
    DEBUG(3, tiramisu::str_dump("\n"); tiramisu::str_dump(std::string(isl_ast_node_to_C_str(node))));

    if (isl_ast_node_get_type(node) == isl_ast_node_block)
    {
        ERROR("Currently Tiramisu does not support extracting bounds from blocks.", true);
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_for)
    {
        DEBUG(3, tiramisu::str_dump("Extracting bounds from a for loop."));
        isl_ast_expr *init_bound = isl_ast_node_for_get_init(node);
        isl_ast_expr *upper_bound = isl_ast_node_for_get_cond(node);
        DEBUG(3, tiramisu::str_dump("Lower bound at this level is: " + std::string(isl_ast_expr_to_C_str(init_bound))));
        DEBUG(3, tiramisu::str_dump("Upper bound at this level is: " + std::string(isl_ast_expr_to_C_str(upper_bound))));

        if (dim == 0)
        {
            if (upper)
            {
                isl_ast_expr *cond = isl_ast_node_for_get_cond(node);

                /**
                  * If we have an expression
                  *  i < N
                  * or an expression
                  *  i <= N - 1
                  *
                  * In both cases, the returned bound should be (N-1).
                  */
                if (isl_ast_expr_get_op_type(cond) == isl_ast_op_lt)
                {
                    // Create an expression of "1".
                    isl_val *one = isl_val_one(isl_ast_node_get_ctx(node));
                    // Add 1 to the ISL ast upper bound to transform it into a strinct bound.
                    result = tiramisu_expr_from_isl_ast_expr(isl_ast_expr_sub(isl_ast_expr_get_op_arg(cond, 1),
                                                             isl_ast_expr_from_val(one)));
                }
                else if (isl_ast_expr_get_op_type(cond) == isl_ast_op_le)
                {
                    result = tiramisu_expr_from_isl_ast_expr(isl_ast_expr_get_op_arg(cond, 1));
                }
            }
            else
            {
                isl_ast_expr *init = isl_ast_node_for_get_init(node);
                result = tiramisu_expr_from_isl_ast_expr(init);
            }
        }
        else
        {
            isl_ast_node *body = isl_ast_node_for_get_body(node);
            result = utility::extract_bound_expression(body, dim-1, upper);
            isl_ast_node_free(body);
        }

        assert(result.is_defined());
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_user)
    {
        ERROR("Cannot extract bounds from a isl_ast_user node.", true);
    }
    else if (isl_ast_node_get_type(node) == isl_ast_node_if)
    {
        DEBUG(3, tiramisu::str_dump("If conditional."));

        // tiramisu::expr cond_bound = tiramisu_expr_from_isl_ast_expr(isl_ast_node_if_get_cond(node));
        tiramisu::expr then_bound = utility::extract_bound_expression(isl_ast_node_if_get_then(node), dim, upper);

        tiramisu::expr else_bound;
        if (isl_ast_node_if_has_else(node))
        {
            // else_bound = utility::extract_bound_expression(isl_ast_node_if_get_else(node), dim, upper);
            // result = tiramisu::expr(tiramisu::o_s, cond_bound, then_bound, else_bound);
            ERROR("If Then Else is unsupported in bound extraction.", true);
        }
        else
            result = then_bound; //tiramisu::expr(tiramisu::o_cond, cond_bound, then_bound);
    }

    DEBUG(3, tiramisu::str_dump("Extracted bound:"); result.dump(false));
    DEBUG_INDENT(-4);

    return result;
}

int computation::compute_maximal_AST_depth()
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    this->name_unnamed_time_space_dimensions();
    this->gen_time_space_domain();
    isl_set *set = this->get_trimmed_time_processor_domain();
    assert(set != NULL);

    DEBUG(10, tiramisu::str_dump(std::string("Getting the ") +
                                 " maximal AST depth of the set ",
                                 isl_set_to_str(set)));

    isl_ast_build *ast_build;
    isl_ctx *ctx = isl_set_get_ctx(set);
    ast_build = isl_ast_build_alloc(ctx);

    // Create identity map for set.
    isl_space *sp = isl_set_get_space(set);
    isl_map *sched = isl_map_identity(isl_space_copy(isl_space_map_from_set(sp)));
    sched = isl_map_set_tuple_name(sched, isl_dim_out, "");

    // Generate the AST.
    DEBUG(10, tiramisu::str_dump("Setting ISL AST generator options."));
    isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
    isl_options_get_ast_build_exploit_nested_bounds(ctx);
    isl_options_set_ast_build_group_coscheduled(ctx, 1);
    isl_options_set_ast_build_allow_else(ctx, 1);
    isl_options_set_ast_build_detect_min_max(ctx, 1);

    // Intersect the iteration domain with the domain of the schedule.
    DEBUG(10, tiramisu::str_dump("Generating time-space domain."));
    isl_map *map = isl_map_intersect_domain(isl_map_copy(sched), isl_set_copy(set));

    // Set iterator names
    DEBUG(10, tiramisu::str_dump("Setting the iterator names."));
    int length = isl_map_dim(map, isl_dim_out);
    isl_id_list *iterators = isl_id_list_alloc(ctx, length);

    for (int i = 0; i < length; i++)
    {
        std::string name;
        if (isl_set_has_dim_name(set, isl_dim_set, i) == true)
            name = isl_set_get_dim_name(set, isl_dim_set, i);
        else
            name = generate_new_variable_name();
        isl_id *id = isl_id_alloc(ctx, name.c_str(), NULL);
        iterators = isl_id_list_add(iterators, id);
    }

    ast_build = isl_ast_build_set_iterators(ast_build, iterators);

    isl_ast_node *node = isl_ast_build_node_from_schedule_map(ast_build, isl_union_map_from_map(map));
    int depth = compute_recursively_max_AST_depth(node);
    isl_ast_build_free(ast_build);

    DEBUG(10, tiramisu::str_dump("The maximal AST depth is : " + std::to_string(depth)));
    DEBUG_INDENT(-4);

    return depth;
}

/**
 * - Generate code:
 * - Generate time-processor domain.
 * - Generate an ISL AST.
 * - Traverse the tree until the level \p dim.
 * - Extract the bounds of that level.
 * - During the traversal, assert that the loop is fully nested.
 *
 */
tiramisu::expr utility::get_bound(isl_set *set, int dim, int upper)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    assert(set != NULL);
    assert(dim >= 0);
    assert(dim < isl_space_dim(isl_set_get_space(set), isl_dim_set));
    assert(isl_set_is_empty(set) == isl_bool_false);

    DEBUG(10, tiramisu::str_dump(std::string("Getting the ") + (upper ? "upper" : "lower") +
                                 " bound on the dimension " +
                                 std::to_string(dim) + " of the set ",
                                 isl_set_to_str(set)));

    tiramisu::expr e = tiramisu::expr();
    isl_ast_build *ast_build;
    isl_ctx *ctx = isl_set_get_ctx(set);
    ast_build = isl_ast_build_alloc(ctx);

    // Create identity map for set.
    isl_space *sp = isl_set_get_space(set);
    isl_map *sched = isl_map_identity(isl_space_copy(isl_space_map_from_set(sp)));
    sched = isl_map_set_tuple_name(sched, isl_dim_out, "");

    // Generate the AST.
    DEBUG(3, tiramisu::str_dump("Setting ISL AST generator options."));
    isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
    isl_options_get_ast_build_exploit_nested_bounds(ctx);
    isl_options_set_ast_build_group_coscheduled(ctx, 1);
    isl_options_set_ast_build_allow_else(ctx, 1);
    isl_options_set_ast_build_detect_min_max(ctx, 1);

    // Computing the polyhedral hull of the input set.
    //DEBUG(3, tiramisu::str_dump("Computing the polyhedral hull of the input set."));
    //set = isl_set_from_basic_set(isl_set_affine_hull(isl_set_copy(set)));
    //DEBUG(3, tiramisu::str_dump("The polyhedral hull is: ", isl_set_to_str(set)));

    // Intersect the iteration domain with the domain of the schedule.
    DEBUG(3, tiramisu::str_dump("Generating time-space domain."));
    isl_map *map =
        isl_map_intersect_domain(
            isl_map_copy(sched),
            isl_set_copy(set));

    // Set iterator names
    DEBUG(3, tiramisu::str_dump("Setting the iterator names."));
    int length = isl_map_dim(map, isl_dim_out);
    isl_id_list *iterators = isl_id_list_alloc(ctx, length);

    for (int i = 0; i < length; i++)
    {
        std::string name;
        if (isl_set_has_dim_name(set, isl_dim_set, i) == true)
            name = isl_set_get_dim_name(set, isl_dim_set, i);
        else
            name = generate_new_variable_name();
        isl_id *id = isl_id_alloc(ctx, name.c_str(), NULL);
        iterators = isl_id_list_add(iterators, id);
    }

    ast_build = isl_ast_build_set_iterators(ast_build, iterators);

    isl_ast_node *node = isl_ast_build_node_from_schedule_map(ast_build, isl_union_map_from_map(map));
    e = utility::extract_bound_expression(node, dim, upper);
    isl_ast_build_free(ast_build);

    assert(e.is_defined() && "The computed bound expression is undefined.");
    DEBUG(10, tiramisu::str_dump(std::string("The ") + (upper ? "upper" : "lower") + " bound is : "); e.dump(false));
    DEBUG_INDENT(-4);

    return e;
}

bool computation::separateAndSplit(tiramisu::var L0, int sizeX)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());

    bool split_happened = this->separateAndSplit(L0, sizeX, L0_outer, L0_inner);

    DEBUG_INDENT(-4);

    return split_happened;
}


bool computation::separateAndSplit(int L0, int v)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Applying separateAndSplit on loop level " + std::to_string(L0) + " with a split factor of " + std::to_string(v)));

    this->gen_time_space_domain();

    // Compute the depth before any scheduling.
    int original_depth = this->compute_maximal_AST_depth();

    DEBUG(3, tiramisu::str_dump("Computing upper bound at loop level " + std::to_string(L0)));

    tiramisu::expr loop_upper_bound =
        tiramisu::expr(o_cast, global::get_loop_iterator_data_type(),
                       tiramisu::utility::get_bound(this->get_trimmed_time_processor_domain(), L0, true));

    DEBUG(3, tiramisu::str_dump("Computing lower bound at loop level " + std::to_string(L0)));

    tiramisu::expr loop_lower_bound =
        tiramisu::expr(o_cast, global::get_loop_iterator_data_type(),
                       tiramisu::utility::get_bound(this->get_trimmed_time_processor_domain(), L0, false));

    tiramisu::expr loop_bound = loop_upper_bound - loop_lower_bound +
            tiramisu::expr(o_cast, global::get_loop_iterator_data_type(), tiramisu::expr((int32_t) 1));
    loop_bound = loop_bound.simplify();

    DEBUG(3, tiramisu::str_dump("Loop bound for the loop to be separated and split: "); loop_bound.dump(false));

    /*
     * Separate this computation. That is, create two identical computations
     * where we have the constraint
     *     i < v * floor(loop_bound/v)
     * in the first and
     *     i >= v * floor(loop_bound/v)
     * in the second.
     *
     * The first is called the full computation while the second is called
     * the separated computation.
     * The two computations are identical in every thing except that they have
     * two different schedules.  Their schedule restricts them to a smaller domain
     * (the full or the separated domains) and schedule one after the other.
     */
    this->separate(L0, loop_bound, v);

    // Make a copy of the schedule before splitting so that we revert the
    // schedule if splitting did not have any effect (i.e., did not happen).
    isl_map *sc = isl_map_copy(this->get_schedule());

    /**
     * Split the full computation since the full computation will be vectorized.
     */
    this->get_update(0).split(L0, v);

    // Compute the depth after scheduling.
    int depth = this->compute_maximal_AST_depth();

    bool split_happened = false;
    if (depth == original_depth)
    {
        DEBUG(3, tiramisu::str_dump("Split did not happen."));
        split_happened = false;

	DEBUG(3, tiramisu::str_dump("Cancel splitting."));
	this->set_schedule(sc);
    }
    else
    {
         split_happened = true;
         DEBUG(3, tiramisu::str_dump("Split happenned."));
    }

    this->get_function()->align_schedules();

    DEBUG_INDENT(-4);

    return split_happened;
}


bool computation::separateAndSplit(tiramisu::var L0_var, int v,
            tiramisu::var L0_outer, tiramisu::var L0_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::vector<std::string> original_loop_level_names = this->get_loop_level_names();

    assert(L0_var.get_name().length() > 0);
    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];

    bool split_happened = this->separateAndSplit(L0, v);

    if (split_happened == false)
    {
        // Replace the original dimension name with the name of the outermost loop
        this->update_names(original_loop_level_names, {L0_outer.get_name()}, L0, 1);
    }
    else
    {
        // Replace the original dimension name with two new dimension names
        this->update_names(original_loop_level_names, {L0_outer.get_name(), L0_inner.get_name()}, L0, 1);
    }

    return split_happened;
}

void computation::split(tiramisu::var L0_var, int sizeX)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    tiramisu::var L0_outer = tiramisu::var(generate_new_variable_name());
    tiramisu::var L0_inner = tiramisu::var(generate_new_variable_name());
    this->split(L0_var, sizeX, L0_outer, L0_inner);

    DEBUG_INDENT(-4);
}

void computation::split(tiramisu::var L0_var, int sizeX,
        tiramisu::var L0_outer, tiramisu::var L0_inner)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);

    std::vector<std::string> original_loop_level_names =
        this->get_loop_level_names();

    std::vector<int> dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(dimensions);
    int L0 = dimensions[0];
    this->assert_names_not_assigned({L0_outer.get_name(), L0_inner.get_name()});

    this->split(L0, sizeX);

    this->update_names(original_loop_level_names, {L0_outer.get_name(), L0_inner.get_name()}, L0, 1);

    DEBUG_INDENT(-4);
}

/**
 * Modify the schedule of this computation so that it splits the
 * loop level L0 into two new loop levels.
 * The size of the inner dimension created is sizeX.
 */
void computation::split(int L0, int sizeX)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    int inDim0 = loop_level_into_dynamic_dimension(L0);

    assert(this->get_schedule() != NULL);
    assert(inDim0 >= 0);
    assert(inDim0 < isl_space_dim(isl_map_get_space(this->get_schedule()), isl_dim_out));
    assert(sizeX >= 1);

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
    std::string static_dim_str = generate_new_variable_name();
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
            map = map + outDim0_str + ", " + static_dim_str + ", " + outDim1_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim0_str.c_str(), NULL);
            isl_id *id2 = isl_id_alloc(this->get_ctx(),
                                       static_dim_str.c_str(), NULL);
            isl_id *id1 = isl_id_alloc(this->get_ctx(),
                                       outDim1_str.c_str(), NULL);
            dimensions.push_back(id0);
            dimensions.push_back(id2);
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
          inDim0_str + "%" + std::to_string(sizeX) + ") and " + static_dim_str + " = 0}";

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
    case tiramisu::o_lerp:
        return "lerp";
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
    case tiramisu::o_sinh:
        return "sinh";
    case tiramisu::o_cosh:
        return "cosh";
    case tiramisu::o_tanh:
        return "tanh";
    case tiramisu::o_asinh:
        return "asinh";
    case tiramisu::o_acosh:
        return "acosh";
    case tiramisu::o_atanh:
        return "atanh";
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
        ERROR("Tiramisu op not supported.", true);
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
    case tiramisu::e_sync:
        return "sync";
    default:
        ERROR("Tiramisu type not supported.", true);
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
        ERROR("Tiramisu type not supported.", true);
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
        return "uint32";
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
    case tiramisu::p_wait_ptr:
        return "wait";
    case tiramisu::p_void_ptr:
        return "void *";
    default:
        ERROR("Tiramisu type not supported.", true);
        return "";
    }
}

std::string str_from_is_null(void *ptr)
{
    return (ptr != NULL) ? "Not NULL" : "NULL";
}

tiramisu::buffer::buffer(std::string name, std::vector<tiramisu::expr> dim_sizes,
                         tiramisu::primitive_t type,
                         tiramisu::argument_t argt, tiramisu::function *fct,
                         std::string corr):
                         allocated(false), argtype(argt), auto_allocate(true),
                         automatic_gpu_copy(true), dim_sizes(dim_sizes), fct(fct),
                         name(name), type(type), location(cuda_ast::memory_location::host)
{
    assert(!name.empty() && "Empty buffer name");
    assert(fct != NULL && "Input function is NULL");

    // Check that the buffer does not already exist.
    assert((fct->get_buffers().count(name) == 0) && ("Buffer already exists"));
    if(corr.compare("") != 0)
    {
      assert((fct->get_buffers().count(corr) != 0) && ("No corresponding cpu beffer"));
      fct->add_mapping(std::pair<std::string ,tiramisu::buffer *>(corr,this));
    }
    fct->add_buffer(std::pair<std::string, tiramisu::buffer *>(name, this));
};

void buffer::set_automatic_gpu_copy(bool automatic_gpu_copy)
{
    this->automatic_gpu_copy = automatic_gpu_copy;
}

bool buffer::get_automatic_gpu_copy()
{
    return this->automatic_gpu_copy;
}


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

cuda_ast::memory_location buffer::get_location() const
{
    return this->location;
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
    return this->get_dim_sizes().size();
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
                  << "\", Number of dimensions: " << this->get_n_dims()
                  << std::endl;

        std::cout << "Dimension sizes: ";
        for (const auto &size : dim_sizes)
        {
            size.dump(false);
            std::cout << "    ";
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
    case tiramisu::p_wait_ptr:
        t = Halide::Handle();
        break;
    case tiramisu::p_void_ptr:
        t = Halide::type_of<void *>();
        break;
    default:
        ERROR("Tiramisu type cannot be translated to Halide type.", true);
    }
    return t;
}

//----------------

std::map<std::string, isl_ast_expr *> tiramisu::computation::get_iterators_map()
{
    return this->iterators_map;
}

void tiramisu::computation::set_iterators_map(std::map<std::string, isl_ast_expr *> map)
{
    this->iterators_map = map;
}

tiramisu::expr tiramisu::computation::get_predicate()
{
    return this->predicate;
}

void tiramisu::computation::add_predicate(tiramisu::expr predicate)
{
    this->predicate = predicate;
}

/**
  * Initialize a computation
  *  This is a private function that should not be called explicitly
  * by users.
  */
void tiramisu::computation::init_computation(std::string iteration_space_str,
        tiramisu::function *fction,
        const tiramisu::expr &e,
        bool schedule_this_computation,
        tiramisu::primitive_t t)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Constructing the computation: " + iteration_space_str));

    assert(fction != NULL);
    assert(iteration_space_str.length() > 0 && ("Empty iteration space"));

    // Initialize all the fields to NULL (useful for later asserts)
    access = NULL;
    stmt = Halide::Internal::Stmt();
    time_processor_domain = NULL;
    duplicate_number = 0;
    automatically_allocated_buffer = NULL;
    predicate = tiramisu::expr();
    // In the constructor of computations, we assume that every created
    // computation is the first computation, then, if this computation
    // was created by add_definitions(), we change is_first_definition
    // to false, otherwise we keep it.
    // We do the same for first_definition.
    is_first = true;
    first_definition = NULL;
    this->definitions_number = 1;
    this->definition_ID = 0;
    this->_is_library_call = false;
    this->_is_nonblock_or_async = false;
    this->_drop_rank_iter = false;

    this->lhs_access_type = tiramisu::o_access;
    this->lhs_argument_idx = -1;
    this->rhs_argument_idx = -1;
    this->wait_argument_idx = -1;
    this->_is_library_call = false;
    this->wait_access_map = nullptr;
    this->wait_index_expr = nullptr;

    this->schedule_this_computation = schedule_this_computation;
    this->data_type = t;

    this->ctx = fction->get_isl_ctx();

    iteration_domain = isl_set_read_from_str(ctx, iteration_space_str.c_str());
    this->name_unnamed_iteration_domain_dimensions();
    name = std::string(isl_space_get_tuple_name(isl_set_get_space(iteration_domain),
                       isl_dim_type::isl_dim_set));

    number_of_dims = isl_set_dim(iteration_domain, isl_dim_type::isl_dim_set);
    for (unsigned i = 0; i < number_of_dims; i++) {
        if (isl_set_has_dim_name(iteration_domain, isl_dim_type::isl_dim_set, i)) {
            std::string dim_name(isl_set_get_dim_name(iteration_domain, isl_dim_type::isl_dim_set, i));
            this->access_variables.push_back(make_pair(i, dim_name));
        }
    }

    fct = fction;
    fct->add_computation(this);
    this->set_identity_schedule_based_on_iteration_domain();
    this->set_expression(e);
    this->set_inline(false);

    // Set the names of output dimensions to be equal to the names of iteration domain schedules.
    std::vector<std::string> nms = this->get_iteration_domain_dimension_names();
    // Rename the dimensions of the schedule domain so that when we set the names of
    // the schedule range dimension to be equal to the names of the domain, we do not
    // get a conflict.
    for (int i = 0; i< this->get_iteration_domain_dimensions_number(); i++)
        this->set_schedule_domain_dim_names({i}, {generate_new_variable_name()});
    for (int i = 0; i< nms.size(); i++)
        this->set_loop_level_names({i}, {nms[i]});

    // If there are computations that have already been defined and that
    // have the same name, check that they have constraints over their iteration
    // domains.
    std::vector<tiramisu::computation *> same_name_computations =
        this->get_function()->get_computation_by_name(name);
    if (same_name_computations.size() > 1)
    {
        if (isl_set_plain_is_universe(this->get_iteration_domain()))
        {
            ERROR("Computations defined multiple times should"
                            " have bounds on their iteration domain", true);
        }

        for (auto c : same_name_computations)
        {
            if (isl_set_plain_is_universe(c->get_iteration_domain()))
            {
                ERROR("Computations defined multiple times should"
                                " have bounds on their iteration domain", true);
            }
        }
    }

    this->updates.push_back(this);

    DEBUG_INDENT(-4);
}

/**
 * Dummy constructor for derived classes.
 */
tiramisu::computation::computation()
{
    this->access = NULL;
    this->schedule = NULL;
    this->default_schedule = NULL;
    
    this->stmt = Halide::Internal::Stmt();
    this->time_processor_domain = NULL;
    this->duplicate_number = 0;

    this->schedule_this_computation = false;
    this->data_type = p_none;
    this->expression = tiramisu::expr();

    this->ctx = NULL;

    this->lhs_access_type = tiramisu::o_access;
    this->lhs_argument_idx = -1;
    this->rhs_argument_idx = -1;
    this->wait_argument_idx = -1;
    this->_is_library_call = false;
    this->wait_access_map = nullptr;
    this->wait_index_expr = nullptr;

    this->iteration_domain = NULL;
    this->name = "";
    this->fct = NULL;
    this->is_let = false;
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
computation::computation(std::string iteration_domain_str, tiramisu::expr e,
                         bool schedule_this_computation, tiramisu::primitive_t t,
                         tiramisu::function *fct)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    init_computation(iteration_domain_str, fct, e, schedule_this_computation, t);
    is_let = false;

    DEBUG_INDENT(-4);
}

computation::computation(std::string name, std::vector<tiramisu::var> iterator_variables, tiramisu::expr predicate, tiramisu::expr e, bool schedule_this_computation, primitive_t t)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump(std::string("Constructing ") + std::string(schedule_this_computation?"a scheduled":"an unscheduled") + std::string(" computation.")));
    std::string iteration_space_str = construct_iteration_domain(name, iterator_variables, predicate);
    DEBUG(3, tiramisu::str_dump("Constructed iteration domain: " + iteration_space_str));

    init_computation(iteration_space_str, global::get_implicit_function(), e, schedule_this_computation, t);
    is_let = false;

    // Allocate implicit buffer if possible
    if (t != p_none && t != p_async && t != p_wait_ptr && t != p_void_ptr) {
        bool is_bounded = true;
        std::vector<expr> buffer_size;
        for (const auto &var : iterator_variables) {
            if (var.lower.is_defined() && var.upper.is_defined()) {
                buffer_size.push_back(var.upper - var.lower);
            } else {
                is_bounded = false;
                break;
            }
        }
        if (is_bounded) {
            std::string buffer_name = "_" + this->name + "_" + global::generate_new_buffer_name();
            // TODO: Memory leak in implicit buffers.
            this->store_in(new tiramisu::buffer(buffer_name,
                                                buffer_size,
                                                this->get_data_type(),
                                                a_temporary,
                                                this->get_function()));
        } else {
            DEBUG(3, tiramisu::str_dump("The iterators of computation " + name +
                    " are not bounded. Skipping implicit buffer generation."));
        }
    }

    DEBUG(3, tiramisu::str_dump("Constructed computation: "); this->dump());
    DEBUG_INDENT(-4);
}

computation::computation(std::string name, std::vector<tiramisu::var> iterator_variables, tiramisu::expr e, bool schedule_this_computation, primitive_t t)
    : computation(name, iterator_variables, expr(), e, schedule_this_computation, t) {}

computation::computation(std::string name, std::vector<var> iterator_variables, expr e, bool schedule_this_computation)
        : computation(name, iterator_variables, e, schedule_this_computation, e.get_data_type()) {}

computation::computation(std::vector<var> iterator_variables, expr e, bool schedule_this_computation)
        : computation(generate_new_computation_name(), iterator_variables, e, schedule_this_computation) {}

computation::computation(std::string name, std::vector<var> iterator_variables, tiramisu::expr predicate, tiramisu::expr e)
	: computation(name, iterator_variables, predicate, e, true, e.get_data_type()) {}

computation::computation(std::string name, std::vector<var> iterator_variables, expr e)
        : computation(name, iterator_variables, e, true, e.get_data_type()) {}

computation::computation(std::vector<var> iterator_variables, expr predicate, expr e)
        : computation(generate_new_computation_name(), iterator_variables, predicate, e) {}

computation::computation(std::vector<var> iterator_variables, expr e)
        : computation(generate_new_computation_name(), iterator_variables, e) {}

void tiramisu::computation::unschedule_this_computation() {
    schedule_this_computation = false;
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

buffer *computation::get_buffer() const
{
    if (this->access == NULL)
    {
        return nullptr;
    }

    std::string buffer_name = isl_map_get_tuple_name(this->access, isl_dim_out);
    assert((this->get_function()->get_buffers().count(buffer_name) > 0) && ("Buffer does not exist"));
    return this->get_function()->get_buffers().find(buffer_name)->second;
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

    DEBUG(10, tiramisu::str_dump("Getting the access of the computation " + this->get_name() +
                                 " adapted to time-space."));
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

tiramisu::computation::operator expr()
{
    // assert(this->is_let_stmt() && "Can only use let statements as expressions.");
    return var(this->get_data_type(), this->get_name());
}

/**
  * Return the function where the computation is declared.
  */
tiramisu::function *tiramisu::computation::get_function() const
{
    return fct;
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

computation * computation::get_predecessor() {
    auto &reverse_graph = this->get_function()->sched_graph_reversed[this];

    if (reverse_graph.empty())
        return nullptr;
    return reverse_graph.begin()->first;
}

 computation * computation::get_successor() {
    auto &reverse_graph = this->get_function()->sched_graph[this];
    if (reverse_graph.empty())
        return nullptr;
    return reverse_graph.begin()->first;
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
    return this->is_let;
}

bool tiramisu::computation::is_library_call() const
{
    return this->_is_library_call;
}

bool tiramisu::computation::should_drop_rank_iter() const
{
    return this->_drop_rank_iter;
}

int tiramisu::computation::get_level_to_drop() {
    if (!should_drop_rank_iter()) {
        return -1;
    }
    return get_loop_level_number_from_dimension_name(this->drop_level.get_name());
}

/**
  * Return the name of the computation.
  */
const std::string &tiramisu::computation::get_name() const
{
    return name;
}

/**
  * Return a unique name of computation; made of the following pattern:
  * [computation name]@[computation address in memory]
  */
const std::string tiramisu::computation::get_unique_name() const
{
    std::stringstream namestream;
    namestream << get_name();
    namestream << "@";
    namestream << (void *)this;
    return namestream.str();
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
int tiramisu::computation::get_iteration_domain_dimensions_number()
{
    assert(iteration_domain != NULL);

    return isl_set_n_dim(this->iteration_domain);
}

int tiramisu::computation::get_time_space_dimensions_number()
{
    assert(this->get_schedule() != NULL);

    return isl_map_dim(this->get_schedule(), isl_dim_out);
}

int computation::get_loop_levels_number()
{
    assert(this->get_schedule() != NULL);
    int loop_levels_number = ((isl_map_dim(this->get_schedule(), isl_dim_out)) - 2)/2;

    return loop_levels_number;
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
void tiramisu::computation::gen_time_space_domain()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(this->get_iteration_domain() != NULL);
    assert(this->get_schedule() != NULL);

    DEBUG(3, tiramisu::str_dump("Iteration domain:", isl_set_to_str(this->get_iteration_domain())));

    isl_set *iter = isl_set_copy(this->get_iteration_domain());
    iter = this->intersect_set_with_context(iter);

    DEBUG(3, tiramisu::str_dump("Iteration domain Intersect context:", isl_set_to_str(iter)));

    time_processor_domain = isl_set_apply(
                                iter,
                                isl_map_copy(this->get_schedule()));

    DEBUG(3, tiramisu::str_dump("Schedule:", isl_map_to_str(this->get_schedule())));
    DEBUG(3, tiramisu::str_dump("Generated time-space domain:", isl_set_to_str(time_processor_domain)));

    DEBUG_INDENT(-4);
}

void tiramisu::computation::drop_rank_iter(var level)
{
    this->_drop_rank_iter = true;
    this->drop_level = level;
}

void tiramisu::computation::set_wait_access(std::string access_str) {
    set_wait_access(isl_map_read_from_str(this->get_ctx(), access_str.c_str()));
}

void tiramisu::computation::set_wait_access(isl_map *access) {
    this->wait_access_map = access;
}

void tiramisu::computation::set_access(isl_map *access)
{
    assert(access != NULL);

    this->set_access(isl_map_to_str(access));
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
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Setting access " + access_str + " for computation " + this->get_name()));

    this->access = isl_map_read_from_str(this->ctx, access_str.c_str());

    /**
     * Set the access relations of all the computations that have the same name
     * (duplicates and updates).
     */
    std::vector<tiramisu::computation *> same_name_computations =
        this->get_function()->get_computation_by_name(this->get_name());

    if (same_name_computations.size() > 1)
        for (auto c : same_name_computations)
        {
            c->access = isl_map_read_from_str(this->ctx, access_str.c_str());
        }

    /**
     * Check that if there are other computations that have the same name
     * as this computation, then the access of all of these computations
     * should be the same.
     */
    std::vector<tiramisu::computation *> computations =
        this->get_function()->get_computation_by_name(this->get_name());
    for (auto c : computations)
        if (isl_map_is_equal(this->get_access_relation(), c->get_access_relation()) == isl_bool_false)
        {
            ERROR("Computations that have the same name should also have the same access relation.",
                            true);
        }

    assert(this->access != nullptr && "Set access failed");

    DEBUG_INDENT(-4);
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
    sched = isl_map_intersect_domain(sched, isl_set_copy(this->get_iteration_domain()));
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

std::vector<std::string> computation::get_loop_level_names()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG(3, tiramisu::str_dump("Collecting names of loop levels from the range of the schedule: ", isl_map_to_str(this->get_schedule())));

    std::vector<std::string> names;
    std::string names_to_print_for_debugging = "";

    for (int i = 0; i < this->get_loop_levels_number(); i++)
    {
        std::string dim_name = isl_map_get_dim_name(this->get_schedule(), isl_dim_out, loop_level_into_dynamic_dimension(i));
        names.push_back(dim_name);
        names_to_print_for_debugging += dim_name + " ";
    }

    DEBUG(3, tiramisu::str_dump("Names of loop levels: " + names_to_print_for_debugging));
    DEBUG_INDENT(-4);

    return names;
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

bool tiramisu::computation::is_send() const
{
  return false;
}

bool tiramisu::computation::is_recv() const
{
  return false;
}

bool tiramisu::computation::is_send_recv() const
{
  return false;
}

bool tiramisu::computation::is_wait() const
{
  return false;
}

const std::vector<std::pair<std::string, tiramisu::expr>>
        &tiramisu::computation::get_associated_let_stmts() const
{
    return this->associated_let_stmts;
}

bool tiramisu::computation::has_accesses() const
{
    if ((this->get_expr().get_op_type() == tiramisu::o_access))
        return true;
    else if ((this->get_expr().get_op_type() == tiramisu::o_allocate) ||
            (this->get_expr().get_op_type() == tiramisu::o_free) ||
            (this->get_expr().get_op_type() == tiramisu::o_memcpy) ||
            (this->get_expr().get_expr_type() == tiramisu::e_sync) ||
            (this->is_let_stmt()))
    {
        return false;
    }
    else
    {
        return true;
    }
}

/**
 * Set the expression of the computation.
 */
void tiramisu::computation::set_expression(const tiramisu::expr &e)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("The original expression is: "); e.dump(false));
    DEBUG(3, tiramisu::str_dump(""));

    DEBUG(3, tiramisu::str_dump("Traversing the expression to replace non-affine accesses by a constant definition."));
    tiramisu::expr modified_e = traverse_expr_and_replace_non_affine_accesses(this, e);

    DEBUG_NO_NEWLINE(3, tiramisu::str_dump("The new expression is: "); modified_e.dump(false););
    DEBUG(3, tiramisu::str_dump(""));

    this->expression = modified_e.copy();

    DEBUG_INDENT(-4);
}

void tiramisu::computation::set_inline(bool is_inline) {
    this->is_inline = is_inline;
}

const bool tiramisu::computation::is_inline_computation() const {
    return this->is_inline;
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
void tiramisu::computation::store_in(buffer *buff)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(buff != NULL);

    isl_space *sp = isl_set_get_space(this->get_iteration_domain());
    isl_map *map = isl_map_identity(isl_space_map_from_set(sp));
    map = isl_map_set_tuple_name(map, isl_dim_out, buff->get_name().c_str());
    map = isl_map_coalesce(map);

    DEBUG(3, tiramisu::str_dump("Binding. The following access function is set: ",
                                isl_map_to_str(map)));

    this->set_access(isl_map_to_str(map));

    isl_map_free(map);
    //isl_map_free(this->default_schedule) ;

    DEBUG_INDENT(-4);
}

void tiramisu::computation::store_in(buffer *buff, std::vector<tiramisu::expr> iterators)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(buff != NULL);

    std::string map_str = "[" + utility::get_parameters_list(this->get_iteration_domain()) + "] -> ";
    map_str += "{" + this->get_name() + "[";
    std::vector<std::string> iter_names =
        this->get_iteration_domain_dimension_names();
    for (int i = 0; i < iter_names.size(); i++)
    {
        map_str += iter_names[i];
        if (i < iter_names.size() - 1)
            map_str += ",";
    }
    map_str += "] -> " + buff->get_name() + "[";

    if (iterators.size() == 0)
        map_str += "0";
    else
        for (int i = 0; i < iterators.size(); i++)
        {
            map_str += iterators[i].to_str();
            if (i < iterators.size() - 1)
            map_str += ", ";
        }
    map_str += "]}";

    assert(map_str.size() != 0);

    DEBUG(3, tiramisu::str_dump("Parsing following access statement: ", map_str.c_str()));

    isl_map *map = isl_map_read_from_str(this->get_ctx(), map_str.c_str());
    assert(map != NULL);

    DEBUG(3, tiramisu::str_dump("Binding. The following access function is set: ",
                                isl_map_to_str(map)));

    this->set_access(isl_map_to_str(map));

    isl_map_free(map);

    DEBUG_INDENT(-4);
}

void computation::store_in(std::vector<expr> mapping, std::vector<expr> sizes) {
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    std::string buffer_name = "_" + this->name + "_" + global::generate_new_buffer_name();
    buffer *new_buffer = new tiramisu::buffer(buffer_name,
                                              sizes,
                                              this->get_data_type(),
                                              a_temporary,
                                              this->get_function());
    this->store_in(new_buffer, mapping);

    DEBUG_INDENT(-4);
}

void tiramisu::computation::mark_as_let_statement()
{
    this->is_let = true;
}

void tiramisu::computation::mark_as_library_call()
{
    this->_is_library_call = true;
}

/****************************************************************************
 ****************************************************************************
 ***************************** Constant class *******************************
 ****************************************************************************
 ****************************************************************************/

tiramisu::constant::constant(
    std::string param_name, const tiramisu::expr &param_expr,
    tiramisu::primitive_t t,
    tiramisu::function *func): tiramisu::computation()
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(!param_name.empty() && "Parameter name empty");
    assert((func != NULL) && "Function undefined");

    DEBUG(3, tiramisu::str_dump("Constructing a scheduled, function-wide constant (this is supposed to replace non-scheduled function wide computations."));

    this->set_name(param_name);
    this->set_expression(param_expr);
    this->mark_as_let_statement();
    func->add_invariant(*this);
    this->compute_with_computation = NULL;
    DEBUG(3, tiramisu::str_dump("The constant is function wide, but it is scheduled.  Its name is : "));
    DEBUG(3, tiramisu::str_dump(this->get_name()));
    std::string iter_str = "{" + this->get_name() + "[0]}";
    DEBUG(3, tiramisu::str_dump("Computed iteration space for the constant assignment" + iter_str));
    init_computation(iter_str, func, param_expr, true, t);
    DEBUG_NO_NEWLINE(10, tiramisu::str_dump("The computation representing the assignment:"); this->dump(true));

    DEBUG_INDENT(-4);
}

std::string tiramisu::computation::construct_iteration_domain(std::string name, std::vector<var> iterator_variables, tiramisu::expr predicate)
{
    tiramisu::function *fct = global::get_implicit_function();

    if (fct == NULL)
    {
        ERROR("An implicit function has to be created by providing a function name to init(NAME). Otherwise the low level API has to be called", true);
    }

    const std::vector<std::string> inv = fct->get_invariant_names();

    std::string iteration_space_str = "";

    if (inv.size() > 0)
        iteration_space_str = "[";

    for (int i = 0; i < inv.size(); i++)
    {
        iteration_space_str += inv[i];
        if (i < inv.size() - 1)
            iteration_space_str += ", ";
    }

    if (inv.size() > 0)
        iteration_space_str += "]->";

    std::string comp_name = name;

    DEBUG(3, tiramisu::str_dump("Creating computation " + comp_name));

    iteration_space_str += "{" + comp_name + "[";
    if (iterator_variables.size() == 0)
        iteration_space_str += "0";
    else
        for (int i = 0; i < iterator_variables.size(); i++)
        {
            var iter = iterator_variables[i];
            iteration_space_str += iter.get_name();
            if (i < iterator_variables.size() - 1)
                iteration_space_str += ", ";
        }

    iteration_space_str += "] ";

    if (iterator_variables.size() != 0)
        iteration_space_str += ": ";

    if (predicate.is_defined())
	 iteration_space_str += predicate.to_str() + " and ";

    bool insert_and = false;
    for (int i = 0; i < iterator_variables.size(); i++)
    {
        var iter = iterator_variables[i];

        if ((insert_and == true && (iter.lower.is_defined() || iter.upper.is_defined())))
        {
            iteration_space_str += " and ";
            insert_and = false;
        }

        if (iter.lower.is_defined() || iter.upper.is_defined())
        {
            iteration_space_str += iter.lower.to_str() + "<=" + iter.get_name() + "<" + iter.upper.to_str();
            insert_and = true;
        }
    }

    iteration_space_str += "}";

    return iteration_space_str;
}

tiramisu::constant::constant(std::string param_name, const tiramisu::expr &param_expr)
        :tiramisu::constant(param_name, param_expr, param_expr.get_data_type(), true, NULL, 0, global::get_implicit_function())
{
}

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
    assert((((function_wide == true) && (with_computation == NULL)) || ((function_wide == false) && (with_computation != NULL))) &&
           "with_computation, should be set only if function_wide is false");

    DEBUG(3, tiramisu::str_dump("Constructing a constant."));

    if (function_wide)
    {
        this->set_name(param_name);
        this->set_expression(param_expr);
        this->mark_as_let_statement();
        this->data_type = t;
        func->add_invariant(*this);
        this->compute_with_computation = NULL;

        DEBUG(3, tiramisu::str_dump("The constant is function wide, its name is : "));
        DEBUG(3, tiramisu::str_dump(this->get_name()));
    }
    else
    {
        assert((with_computation != NULL) &&
               "A valid computation should be provided.");
        assert((at_loop_level >= computation::root_dimension) &&
               "Invalid root dimension.");

        DEBUG(3, tiramisu::str_dump("Consturcting constant at level: " + std::to_string(at_loop_level)));

        this->compute_with_computation = with_computation;
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
        if (with_computation->get_predecessor() != NULL)
            this->between(*(with_computation->get_predecessor()),
                          this->get_dimension_name_for_loop_level(at_loop_level),
                          *with_computation,
                          this->get_dimension_name_for_loop_level(at_loop_level));
        else
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

tiramisu::constant::operator expr()
{
    return var(this->get_data_type(), this->get_name());
    // return this->get_expr();
}

void tiramisu::buffer::set_dim_size(int dim, int size)
{
    assert(dim >= 0);
    assert(dim < this->dim_sizes.size());
    assert(this->dim_sizes.size() > 0);
    assert(size > 0);

    this->dim_sizes[dim] = size;
}

void tiramisu::computation::storage_fold(tiramisu::var L0_var, int factor)
{
    DEBUG_FCT_NAME(3);
    DEBUG_INDENT(4);

    assert(L0_var.get_name().length() > 0);
    std::vector<int> loop_dimensions =
        this->get_loop_level_numbers_from_dimension_names({L0_var.get_name()});
    this->check_dimensions_validity(loop_dimensions);
    int inDim0 = loop_dimensions[0];

    assert(this->get_access_relation() != NULL);
    assert(inDim0 >= 0);
    assert(inDim0 < isl_space_dim(isl_map_get_space(this->get_access_relation()), isl_dim_out));
    assert(factor > 0);

    isl_map *access_relation = this->get_access_relation();
    std::string buffer_name = isl_map_get_tuple_name(access_relation, isl_dim_out);
    tiramisu::buffer *buff_object = this->get_function()->get_buffers().find(buffer_name)->second;
    buff_object->set_dim_size(inDim0, factor);

    access_relation = isl_map_copy(access_relation);

    DEBUG(3, tiramisu::str_dump("Original access relation: ", isl_map_to_str(access_relation)));
    DEBUG(3, tiramisu::str_dump("Folding dimension " + std::to_string(inDim0)
                                + " by a factor of " + std::to_string(factor)));

    std::string inDim0_str;

    std::string outDim0_str = generate_new_variable_name();

    int n_dims = isl_map_dim(access_relation, isl_dim_out);
    std::vector<isl_id *> dimensions;
    std::vector<std::string> dimensions_str;
    std::string map = "{";

    // -----------------------------------------------------------------
    // Preparing a map to split the duplicate computation.
    // -----------------------------------------------------------------

    map = map + this->get_name() + "[";

    for (int i = 0; i < n_dims; i++)
    {
        std::string dim_str = generate_new_variable_name();
        dimensions_str.push_back(dim_str);
        map = map + dim_str;

        if (i == inDim0)
        {
            inDim0_str = dim_str;
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] -> " + buffer_name + "[";

    for (int i = 0; i < n_dims; i++)
    {
        if (i != inDim0)
        {
            map = map + dimensions_str[i];
            dimensions.push_back(isl_id_alloc(
                                     this->get_ctx(),
                                     dimensions_str[i].c_str(),
                                     NULL));
        }
        else
        {
            map = map + outDim0_str;
            isl_id *id0 = isl_id_alloc(this->get_ctx(),
                                       outDim0_str.c_str(), NULL);
            dimensions.push_back(id0);
        }

        if (i != n_dims - 1)
        {
            map = map + ",";
        }
    }

    map = map + "] : " + outDim0_str + " = floor(" + inDim0_str + "%" +
          std::to_string(factor) + ")}";

    isl_map *transformation_map = isl_map_read_from_str(this->get_ctx(), map.c_str());

    for (int i = 0; i < dimensions.size(); i++)
        transformation_map = isl_map_set_dim_id(
                                 transformation_map, isl_dim_out, i, isl_id_copy(dimensions[i]));

    transformation_map = isl_map_set_tuple_id(
                             transformation_map, isl_dim_in,
                             isl_map_get_tuple_id(isl_map_copy(access_relation), isl_dim_out));

    isl_id *id_range = isl_id_alloc(this->get_ctx(), buffer_name.c_str(), NULL);
    transformation_map = isl_map_set_tuple_id(transformation_map, isl_dim_out, id_range);

    DEBUG(3, tiramisu::str_dump("Transformation map : ",
                                isl_map_to_str(transformation_map)));

    access_relation = isl_map_apply_range(isl_map_copy(access_relation),
                                          isl_map_copy(transformation_map));

    DEBUG(3, tiramisu::str_dump("Access relation after storage folding: ",
                                isl_map_to_str(access_relation)));

    this->set_access(access_relation);



    DEBUG_INDENT(-4);
}

tiramisu::xfer_prop::xfer_prop() { }

tiramisu::xfer_prop::xfer_prop(tiramisu::primitive_t dtype,
                               std::initializer_list<tiramisu::xfer_attr> attrs)
        : dtype(dtype), xfer_prop_id(-1) {
    this->attrs.insert(this->attrs.begin(), attrs);
}

tiramisu::xfer_prop::xfer_prop(tiramisu::primitive_t dtype,
                               std::initializer_list<tiramisu::xfer_attr> attrs,
                               int comm_prop_id) : dtype(dtype), xfer_prop_id(comm_prop_id) {
    this->attrs.insert(this->attrs.begin(), attrs);
    if (comm_prop_id != -1) {
        xfer_prop_ids.insert(comm_prop_id);
    }
    xfer_prop_ids.insert(0); // The kernel one. Just make sure it gets in there
}

tiramisu::primitive_t tiramisu::xfer_prop::get_dtype() const {
    return this->dtype;
}

std::string tiramisu::xfer_prop::attr_to_string(tiramisu::xfer_attr attr) {
    switch (attr) {
        case SYNC: return "SYNC";
        case ASYNC: return "ASYNC";
        case MPI: return "MPI";
        case CUDA: return "CUDA";
        case BLOCK: return "BLOCK";
        case NONBLOCK: return "NONBLOCK";
        default: {
            assert(false && "Unknown xfer_prop attr specified.");
            return "";
        }
    }
}

int tiramisu::xfer_prop::get_xfer_prop_id() const {
    return xfer_prop_id;
}

void tiramisu::xfer_prop::add_attr(tiramisu::xfer_attr attr) {
    attrs.push_back(attr);
}

bool tiramisu::xfer_prop::contains_attr(tiramisu::xfer_attr attr) const {
    return attrs.end() != std::find(attrs.begin(), attrs.end(), attr);
}

bool tiramisu::xfer_prop::contains_attrs(std::vector<tiramisu::xfer_attr> attrs) const {
    for (auto attr : attrs) {
        if (this->attrs.end() == std::find(this->attrs.begin(), this->attrs.end(), attr)) {
            return false;
        }
    }
    return true;
}

tiramisu::communicator::communicator() { }

tiramisu::communicator::communicator(std::string iteration_domain_str, tiramisu::expr e,
                                     bool schedule_this_computation, tiramisu::primitive_t data_type,
                                     tiramisu::xfer_prop prop, tiramisu::function *fct) :
        computation(iteration_domain_str, e, schedule_this_computation, data_type, fct), prop(prop) {}

tiramisu::communicator::communicator(std::string iteration_domain_str, tiramisu::expr e, bool schedule_this_computation,
                                     tiramisu::primitive_t data_type, tiramisu::function *fct) :
        computation(iteration_domain_str, e, schedule_this_computation, data_type, fct) {}

void tiramisu::communicator::collapse_many(std::vector<tiramisu::collapse_group> collapse_each) {
    for (auto c : collapse_each) {
        this->collapse(std::get<0>(c), std::get<1>(c), std::get<2>(c), std::get<3>(c));
    }
}

void tiramisu::communicator::add_dim(tiramisu::expr dim)
{
    this->dims.push_back(dim);
}

tiramisu::expr tiramisu::communicator::get_num_elements() const
{
    tiramisu::expr num = expr(1);
    if (!dims.empty()) {
        num = tiramisu::expr(tiramisu::o_cast, dims[0].get_data_type(), num);
    }
    for (std::vector<tiramisu::expr>::const_iterator iter = dims.cbegin(); iter != dims.cend(); iter++) {
        num = *iter * num;
    }
    return num;
}

xfer_prop tiramisu::communicator::get_xfer_props() const
{
    return prop;
}

std::vector<communicator *> tiramisu::communicator::collapse(int level, tiramisu::expr collapse_from_iter,
                                                             tiramisu::expr collapse_until_iter,
                                                             tiramisu::expr num_collapsed)
{

    std::vector<communicator *> ret;
    if (collapse_until_iter.get_expr_type() == tiramisu::e_val && collapse_until_iter.get_int32_value() == -1) {
        this->add_dim(num_collapsed);
        // Instead of fully removing the loop, we modify the collapsed loop to only have a single iteration.
        full_loop_level_collapse(level, collapse_from_iter);
    } else {
        assert(false && "Get rid of this block");
    }

    return ret;
}

std::string create_send_func_name(const xfer_prop chan)
{
    if (chan.contains_attr(MPI)) {
        std::string name = "tiramisu_MPI";
        if (chan.contains_attr(SYNC) && chan.contains_attr(BLOCK)) {
            name += "_Ssend";
        } else if (chan.contains_attr(SYNC) && chan.contains_attr(NONBLOCK)) {
            name += "_Issend";
        } else if (chan.contains_attr(ASYNC) && chan.contains_attr(BLOCK)) {
            name += "_Send";
        } else if (chan.contains_attr(ASYNC) && chan.contains_attr(NONBLOCK)) {
            name += "_Isend";
        }
        switch (chan.get_dtype()) {
            case p_uint8:
                name += "_uint8";
                break;
            case p_uint16:
                name += "_uint16";
                break;
            case p_uint32:
                name += "_uint32";
                break;
            case p_uint64:
                name += "_uint64";
                break;
            case p_int8:
                name += "_int8";
                break;
            case p_int16:
                name += "_int16";
                break;
            case p_int32:
                name += "_int32";
                break;
            case p_int64:
                name += "_int64";
                break;
            case p_float32:
                name += "_f32";
                break;
            case p_float64:
                name += "_f64";
                break;
            default: {
                ERROR("Channel not allowed", 27);
                break;
            }
        }
        return name;
    } else if (chan.contains_attr(CUDA)) {
        std::string name = "tiramisu_cudad_memcpy";
        if (chan.contains_attr(ASYNC)) {
            name += "_async";
        }
        if (chan.contains_attr(CPU2CPU)) {
            name += "_h2h";
        } else if (chan.contains_attr(CPU2GPU)) {
            name += "_h2d";
        } else if (chan.contains_attr(GPU2GPU)) {
            name += "_d2d";
        } else if (chan.contains_attr(GPU2CPU)) {
            name += "_d2h";
        } else {
            assert(false && "Unknown CUDA transfer direction");
        }

        return name;
    }
    assert(false && "Communication must be either MPI or CUDA!");
    return "";
}

tiramisu::send::send(std::string iteration_domain_str, tiramisu::computation *producer, tiramisu::expr rhs,
                     xfer_prop prop, bool schedule_this, std::vector<expr> dims, tiramisu::function *fct) :
        communicator(iteration_domain_str, rhs, schedule_this, prop.get_dtype(),
                     prop, fct), producer(producer),
        msg_tag(tiramisu::expr(next_msg_tag++))
{
    _is_library_call = true;
    library_call_name = create_send_func_name(prop);
    expr mod_rhs(tiramisu::o_address_of, rhs.get_name(), rhs.get_access(), rhs.get_data_type());
    set_expression(mod_rhs);
}

tiramisu::expr tiramisu::send::get_msg_tag() const
{
    return msg_tag;
}

tiramisu::computation *tiramisu::send::get_producer() const
{
    return producer;
}

tiramisu::recv *tiramisu::send::get_matching_recv() const
{
    return matching_recv;
}

void tiramisu::send::set_matching_recv(tiramisu::recv *matching_recv)
{
    this->matching_recv = matching_recv;
}

bool tiramisu::send::is_send() const
{
    return true;
}

void tiramisu::send::add_definitions(std::string iteration_domain_str,
                                     tiramisu::expr e,
                                     bool schedule_this_computation, tiramisu::primitive_t t,
                                     tiramisu::function *fct)
{
    tiramisu::send *new_c = new tiramisu::send(iteration_domain_str, this->producer, e, this->prop,
                                               schedule_this_computation, {}, fct);
    new_c->set_matching_recv(this->get_matching_recv());
    new_c->set_src(this->get_src());
    new_c->is_first = false;
    new_c->first_definition = this;
    this->updates.push_back(new_c);
}

tiramisu::expr tiramisu::send::get_src() const
{
    return src;
}

tiramisu::expr tiramisu::send::get_dest() const
{
    return dest;
}

void tiramisu::send::set_src(tiramisu::expr src)
{
    this->src = src;
}

void tiramisu::send::set_dest(tiramisu::expr dest)
{
    this->dest = dest;
}

void tiramisu::send::override_msg_tag(tiramisu::expr msg_tag)
{
    this->msg_tag = msg_tag;
}

std::string create_recv_func_name(const xfer_prop chan)
{

    if (chan.contains_attr(MPI)) {
        std::string name = "tiramisu_MPI";
        if (chan.contains_attr(BLOCK)) {
            name += "_Recv";
        } else if (chan.contains_attr(NONBLOCK)) {
            name += "_Irecv";
        }
        switch (chan.get_dtype()) {
            case p_uint8:
                name += "_uint8";
                break;
            case p_uint16:
                name += "_uint16";
                break;
            case p_uint32:
                name += "_uint32";
                break;
            case p_uint64:
                name += "_uint64";
                break;
            case p_int8:
                name += "_int8";
                break;
            case p_int16:
                name += "_int16";
                break;
            case p_int32:
                name += "_int32";
                break;
            case p_int64:
                name += "_int64";
                break;
            case p_float32:
                name += "_f32";
                break;
            case p_float64:
                name += "_f64";
                break;
            default:
                ERROR("Channel type not allowed.", 27);
        }
        return name;
    } else {
        assert(false);
        return "";
    }
}

tiramisu::recv::recv(std::string iteration_domain_str, bool schedule_this, tiramisu::xfer_prop prop,
                     tiramisu::function *fct) : communicator(iteration_domain_str, tiramisu::expr(),
                                                             schedule_this, prop.get_dtype(), prop, fct)
{
    _is_library_call = true;
}

send * tiramisu::recv::get_matching_send() const
{
    return matching_send;
}

void tiramisu::recv::set_matching_send(send *matching_send)
{
    this->matching_send = matching_send;
    library_call_name = create_recv_func_name(prop);
}

bool tiramisu::recv::is_recv() const
{
    return true;
}

tiramisu::expr tiramisu::recv::get_src() const
{
    return src;
}

tiramisu::expr tiramisu::recv::get_dest() const
{
    return dest;
}

void tiramisu::recv::set_src(tiramisu::expr src)
{
    this->src = src;
}

void tiramisu::recv::set_dest(tiramisu::expr dest)
{
    this->dest = dest;
}

void tiramisu::recv::override_msg_tag(tiramisu::expr msg_tag)
{
    this->msg_tag = msg_tag;
}

tiramisu::expr tiramisu::recv::get_msg_tag() const
{
    return this->msg_tag;
}

void tiramisu::recv::add_definitions(std::string iteration_domain_str,
                                     tiramisu::expr e,
                                     bool schedule_this_computation, tiramisu::primitive_t t,
                                     tiramisu::function *fct)
{
    tiramisu::recv *new_c = new tiramisu::recv(iteration_domain_str, schedule_this_computation, this->prop, fct);
    new_c->set_matching_send(this->get_matching_send());
    new_c->set_dest(this->get_dest());
    new_c->is_first = false;
    new_c->first_definition = this;
    new_c->is_let = this->is_let;
    new_c->definition_ID = this->definitions_number;
    this->definitions_number++;

    if (new_c->get_expr().is_equal(this->get_expr()))
    {
        // Copy the associated let statements to the new definition.
        new_c->associated_let_stmts = this->associated_let_stmts;
    }

    this->updates.push_back(new_c);
}

tiramisu::send_recv::send_recv(std::string iteration_domain_str, tiramisu::computation *producer,
                               tiramisu::expr rhs, xfer_prop prop, bool schedule_this_computation,
                               std::vector<expr> dims, tiramisu::function *fct) :
        communicator(iteration_domain_str, rhs, schedule_this_computation, prop.get_dtype(), prop, fct)
{
    _is_library_call = true;
    library_call_name = create_send_func_name(prop);
    if (prop.contains_attr(CPU2GPU)) {
        expr mod_rhs(tiramisu::o_address_of, rhs.get_name(), rhs.get_access(), rhs.get_data_type());
        set_expression(mod_rhs);
    } else if (prop.contains_attr(GPU2CPU)) {
        // we will modify this again later
        expr mod_rhs(tiramisu::o_buffer, rhs.get_name(), rhs.get_access(), rhs.get_data_type());
        set_expression(mod_rhs);
    }
}

bool tiramisu::send_recv::is_send_recv() const
{
    return true;
}

tiramisu::wait::wait(tiramisu::expr rhs, xfer_prop prop, tiramisu::function *fct)
        : communicator(), rhs(rhs) {
    assert(rhs.get_op_type() == tiramisu::o_access && "The RHS expression for a wait should be an access!");
    tiramisu::computation *op = fct->get_computation_by_name(rhs.get_name())[0];
    isl_set *dom = isl_set_copy(op->get_iteration_domain());
    std::string new_name = std::string(isl_set_get_tuple_name(dom)) + "_wait";
    dom = isl_set_set_tuple_name(dom, new_name.c_str());
    init_computation(isl_set_to_str(dom), fct, rhs, true, tiramisu::p_async);
    _is_library_call = true;
    this->prop = prop;
}

tiramisu::wait::wait(std::string iteration_domain_str, tiramisu::expr rhs, xfer_prop prop,
                     bool schedule_this,
                     tiramisu::function *fct) : communicator(iteration_domain_str, rhs, schedule_this, tiramisu::p_async, fct), rhs(rhs) {
    _is_library_call = true;
    this->prop = prop;
    computation *comp = fct->get_computation_by_name(rhs.get_name())[0];
    comp->_is_nonblock_or_async = true;
}

std::vector<tiramisu::computation *> tiramisu::wait::get_op_to_wait_on() const {
    std::string op_name = this->get_expr().get_name();
    return this->get_function()->get_computation_by_name(op_name);
}

bool tiramisu::wait::is_wait() const
{
    return true;
}

void tiramisu::wait::add_definitions(std::string iteration_domain_str,
                                     tiramisu::expr e, bool schedule_this_computation, tiramisu::primitive_t t,
                                     tiramisu::function *fct)
{
    tiramisu::computation *new_c = new tiramisu::wait(iteration_domain_str, e, this->prop, schedule_this_computation,
                                                      fct);
    new_c->is_first = false;
    new_c->first_definition = this;
    this->updates.push_back(new_c);
}

void tiramisu::computation::full_loop_level_collapse(int level, tiramisu::expr collapse_from_iter)
{
    std::string collapse_from_iter_repr;
    if (global::get_loop_iterator_data_type() == p_int32) {
        collapse_from_iter_repr = collapse_from_iter.get_expr_type() == tiramisu::e_val ?
                                  std::to_string(collapse_from_iter.get_int32_value()) : collapse_from_iter.get_name();
    } else {
        collapse_from_iter_repr = collapse_from_iter.get_expr_type() == tiramisu::e_val ?
                                  std::to_string(collapse_from_iter.get_int64_value()) : collapse_from_iter.get_name();
    }
    isl_map *sched = this->get_schedule();
    int dim = loop_level_into_dynamic_dimension(level);
    const char *_dim_name = isl_map_get_dim_name(sched, isl_dim_out, dim);
    std::string dim_name;
    if (!_dim_name) { // Since dim names are optional...
        dim_name = "jr" + std::to_string(next_dim_name++);
        sched = isl_map_set_dim_name(sched, isl_dim_out, dim, dim_name.c_str());
    } else {
        dim_name = _dim_name;
    }
    std::string subtract_cst =
            dim_name + " > " + collapse_from_iter_repr; // > because you want a single iteration (iter 0)
    isl_map *ident = isl_set_identity(isl_set_copy(this->get_iteration_domain()));
    ident = isl_map_apply_domain(isl_map_copy(this->get_schedule()), ident);
    ident = isl_map_set_dim_name(ident, isl_dim_out, dim, dim_name.c_str());
    isl_map *universe = isl_map_universe(isl_map_get_space(ident));
    std::string transform_str = isl_map_to_str(universe);
    std::vector<std::string> parts;
    split_string(transform_str, "}", parts);
    transform_str = parts[0] + ": " + subtract_cst + "}";
    isl_map *transform = isl_map_read_from_str(this->get_ctx(), transform_str.c_str());
    if (collapse_from_iter.get_expr_type() != tiramisu::e_val) { // This might be a free variable
        transform = isl_map_add_free_var(collapse_from_iter_repr, transform, this->get_ctx());
    }
    sched = isl_map_subtract(sched, transform);
    this->set_schedule(sched);
}

xfer tiramisu::computation::create_xfer(std::string send_iter_domain, std::string recv_iter_domain,
                                        tiramisu::expr send_dest, tiramisu::expr recv_src,
                                        xfer_prop send_prop, xfer_prop recv_prop,
                                        tiramisu::expr send_expr, tiramisu::function *fct) {
    if (send_prop.contains_attr(MPI)) {
        assert(recv_prop.contains_attr(MPI));
    } else if (send_prop.contains_attr(CUDA)) {
        assert(recv_prop.contains_attr(CUDA));
    }

    assert(send_expr.get_op_type() == tiramisu::o_access);
    tiramisu::computation *producer = fct->get_computation_by_name(send_expr.get_name())[0];

    isl_set *s_iter_domain = isl_set_read_from_str(producer->get_ctx(), send_iter_domain.c_str());
    isl_set *r_iter_domain = isl_set_read_from_str(producer->get_ctx(), recv_iter_domain.c_str());
    tiramisu::send *s = new tiramisu::send(isl_set_to_str(s_iter_domain), producer, send_expr, send_prop, true,
                                           {1}, producer->get_function());
    tiramisu::recv *r = new tiramisu::recv(isl_set_to_str(r_iter_domain), true, recv_prop, fct);
    isl_map *send_sched = s->gen_identity_schedule_for_iteration_domain();
    isl_map *recv_sched = r->gen_identity_schedule_for_iteration_domain();

    s->set_src(expr());
    s->set_dest(send_dest);
    r->set_src(recv_src);
    r->set_dest(expr());

    s->set_schedule(send_sched);
    r->set_schedule(recv_sched);
    s->set_matching_recv(r);
    r->set_matching_send(s);

    tiramisu::xfer c;
    c.s = s;
    c.r = r;
    c.sr = nullptr;

    return c;
}

xfer tiramisu::computation::create_xfer(std::string iter_domain_str, xfer_prop prop, tiramisu::expr expr,
                                        tiramisu::function *fct) {
    assert(expr.get_op_type() == tiramisu::o_access);
    tiramisu::computation *producer = fct->get_computation_by_name(expr.get_name())[0];

    isl_set *iter_domain = isl_set_read_from_str(producer->get_ctx(), iter_domain_str.c_str());
    tiramisu::send_recv *sr = new tiramisu::send_recv(isl_set_to_str(iter_domain), producer, expr, prop, true, {1},
                                                      producer->get_function());
    isl_map *sched = sr->gen_identity_schedule_for_iteration_domain();

    sr->set_schedule(sched);

    tiramisu::xfer c;
    c.s = nullptr;
    c.r = nullptr;
    c.sr = sr;

    return c;
}

void split_string(std::string str, std::string delimiter, std::vector<std::string> &vector)
{
    size_t pos = 0;
    std::string token;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        vector.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    token = str.substr(0, pos);
    vector.push_back(token);
}

isl_map *isl_map_add_free_var(const std::string &free_var_name, isl_map *map, isl_ctx *ctx) {
    isl_map *final_map = nullptr;

    // first, check to see if this variable is actually a free variable. If not, then we don't need to add it.
    int num_domain_dims = isl_map_dim(map, isl_dim_in);
    for (int i = 0; i < num_domain_dims; i++) {
        if (std::strcmp(isl_map_get_dim_name(map, isl_dim_in, i), free_var_name.c_str()) == 0) {
            return map;
        }
    }
    int num_range_dims = isl_map_dim(map, isl_dim_out);
    for (int i = 0; i < num_range_dims; i++) {
        if (std::strcmp(isl_map_get_dim_name(map, isl_dim_out, i), free_var_name.c_str()) == 0) {
            return map;
        }
    }

    std::string map_str = isl_map_to_str(map);
    std::vector<std::string> parts;
    split_string(map_str, "{", parts);
    if (parts[0] != "") { // A free variable already exists, so add this variable to that box
        std::vector<std::string> free_parts;
        split_string(parts[0], "[", free_parts);
        // remove the right bracket
        std::vector<std::string> tmp;
        split_string(free_parts[free_parts.size() - 1], "]", tmp);
        free_parts.insert(free_parts.end(), tmp[0]);
        std::string free_vars = "";
        int ctr = 0;
        for (auto s: free_parts) {
            if (s == free_var_name) {
                // The variable name was already in the box, so we don't actually need to do anything
                return map;
            }
            free_vars += ctr++ == 0 ? s : "," + s;
        }
        free_vars += "," + free_var_name;
        free_vars = "[" + free_vars + "]" + "{" + parts[1];
        final_map = isl_map_read_from_str(ctx, free_vars.c_str());
    } else {
        std::string m = "[" + free_var_name + "]->{" + parts[1];
        final_map = isl_map_read_from_str(ctx, m.c_str());
    }
    assert(final_map && "Adding free param to map resulted in a null isl_map");
    return final_map;
}


expr tiramisu::computation::get_span(int level)
{
    this->check_dimensions_validity({level});
    tiramisu::expr loop_upper_bound =
            tiramisu::utility::get_bound(this->get_trimmed_time_processor_domain(),
                                         level, true);

    tiramisu::expr loop_lower_bound =
            tiramisu::utility::get_bound(this->get_trimmed_time_processor_domain(),
                                         level, false);

    tiramisu::expr loop_bound = loop_upper_bound - loop_lower_bound + value_cast(global::get_loop_iterator_data_type(), 1);
    return loop_bound.simplify();
}

void tiramisu::buffer::tag_gpu_shared() {
    location = cuda_ast::memory_location::shared;
    set_auto_allocate(false);
}

void tiramisu::buffer::tag_gpu_constant() {
    location = cuda_ast::memory_location::constant;
}

void tiramisu::buffer::tag_gpu_global() {
    location = cuda_ast::memory_location::global;
}

void tiramisu::buffer::tag_gpu_local() {
    location = cuda_ast::memory_location::local;
    set_auto_allocate(false);
}

void tiramisu::buffer::tag_gpu_register() {
    bool is_single_val = this->get_n_dims() == 1 && this->get_dim_sizes()[0].get_expr_type() == e_val && this->get_dim_sizes()[0].get_int_val() == 1;
    assert(is_single_val && "Buffer needs to correspond to a single value to be in register");
    location = cuda_ast::memory_location::reg;
    set_auto_allocate(false);
}

std::string get_rank_string_type(tiramisu::rank_t rank_type)
{
    if (rank_type == rank_t::r_sender)
        return "r_snd";
    else
        return "r_rcv";
}

void project_out_static_dimensions(isl_set*& set)
{
    int i = 0;

    std::string tuple_name = isl_set_get_tuple_name(set);

    while(i < isl_set_dim(set, isl_dim_set)) {
        set = isl_set_project_out(set, isl_dim_set, i, 1);
        i++;
    }

    set = isl_set_set_tuple_name(set, tuple_name.c_str());
}

std::vector<std::string> computation::get_trimmed_time_space_domain_dimension_names()
{
    std::vector<std::string> dimensions_names;

    this->gen_time_space_domain();
    isl_set* iteration_domain = this->get_trimmed_time_processor_domain();

    int number_of_dims = isl_set_dim(iteration_domain,isl_dim_set);

    for (int i = 0; i < number_of_dims; i++) {
        if (isl_set_has_dim_name(iteration_domain, isl_dim_set, i))
            dimensions_names.push_back(isl_set_get_dim_name(iteration_domain, isl_dim_set, i));
        else
            dimensions_names.push_back(generate_new_variable_name());
    }

    return dimensions_names;
}

int computation::get_distributed_dimension()
{
    this->gen_time_space_domain();

    int number_of_dimensions = isl_set_dim(this->get_trimmed_time_processor_domain(), isl_dim_set);

    int distributed_dimension = 0;

    while (distributed_dimension < number_of_dimensions and
         !this->get_function()->should_distribute(this->get_name(), distributed_dimension))
        distributed_dimension++;

    if (distributed_dimension < number_of_dimensions)
        return distributed_dimension;
    else
        return -1;//no distributed dimension
}

isl_map* computation::construct_distribution_map(tiramisu::rank_t rank_type)
{
    DEBUG_FCT_NAME(10);
    DEBUG_INDENT(4);

    std::vector<std::string> dimensions_names = this->get_trimmed_time_space_domain_dimension_names();

    int distributed_dimension = this->get_distributed_dimension();

    if (distributed_dimension == -1)
        ERROR("Computation " + this->get_name() + "isn't tagged distributed and used gen_communication().",true);

    if(distributed_dimension > 0)
        ERROR("Generating communication code automatically for inner distributed loops is currently not supported.",true);

    //get the extent of the distributed loop, the number od available ranks should be equal to it
    this->simplify(this->get_iteration_domain());
    isl_set * it_dom = this->get_trimmed_time_processor_domain();
    project_out_static_dimensions(it_dom);
    int number_of_ranks = tiramisu::utility::get_extent(it_dom, distributed_dimension);

    std::string dimensions_string = "";
    for (int i = 0; i < dimensions_names.size(); i++)
    {
        dimensions_string += dimensions_names[i];
        if (i < dimensions_names.size()-1)
            dimensions_string += ",";
    }
    //TODO : this won't give a correct result if distributed_stride%number_of_nodes!=0
    //should be corrected
    std::string rank_name = get_rank_string_type(rank_type);
    std::string params = "[" + rank_name + "]";
    std::string ranks_definition = "0<=" + rank_name + "<" + std::to_string(number_of_ranks);

    std::string domain = this->get_name() + "[" + dimensions_string + "]";

    std::string constraint_on_distributed_dimension = rank_name + "<=" + this->get_dimension_name_for_loop_level(distributed_dimension) + "<" + "(" + rank_name + "+1)";

    std::string distribution_map_string = params + "->{" + domain +"->" + domain + ":"
    + ranks_definition + " and " + constraint_on_distributed_dimension + "}";

    isl_map* distribution_map = isl_map_read_from_str(this->get_ctx(), distribution_map_string.c_str());

    DEBUG(3, tiramisu::str_dump("The distribution map is:"); isl_map_dump(distribution_map));
    DEBUG_INDENT(-4);

    return distribution_map;
}

std::string computation::get_comm_id(rank_t rank_type, int i)
{
    return "b_" + this->get_name() + "_" + std::to_string(i) + "_" + get_rank_string_type(rank_type);
}

isl_set* computation::construct_comm_set(isl_set* set, rank_t rank_type, int comm_id)
{
    int dist_dim = this->get_distributed_dimension();

    set = isl_set_insert_dims(set, isl_dim_set, 0, 1);
    set = isl_set_insert_dims(set, isl_dim_set, 1, 1);

    // If the rank is a receiver, this means that in the set, the iterators will be in
    // this order: dist_dim, r_receiver, r_sender, iterators
    if (rank_type == rank_t::r_receiver)
    {
        set = isl_set_set_dim_name(set, isl_dim_set, 0, get_rank_string_type(rank_t::r_receiver).c_str());
        set = isl_set_set_dim_name(set, isl_dim_set, 1, get_rank_string_type(rank_t::r_sender).c_str());
    }
    // If the rank is a sender, this means that in the set, the iterators will be in
    // this order: dist_dim, r_sender, r_receiver, iterators
    else
    {
        set = isl_set_set_dim_name(set, isl_dim_set, 1, get_rank_string_type(rank_t::r_receiver).c_str());
        set = isl_set_set_dim_name(set, isl_dim_set, 0, get_rank_string_type(rank_t::r_sender).c_str());
    }

    //Add constraints to create a relation between the dim_params and dim_sets
    std::vector <std::string> set_parts;
    split_string(isl_set_to_str(set), "}", set_parts);
    set_parts[0] += " and " + get_rank_string_type(rank_t::r_sender) + "'=" + get_rank_string_type(rank_t::r_sender);
    set_parts[0] += " and " + get_rank_string_type(rank_t::r_receiver) + "'=" + get_rank_string_type(rank_t::r_receiver) + "}";
    set = isl_set_read_from_str(isl_set_get_ctx(set), set_parts[0].c_str());

    //Project out the distributed dimension
    set = isl_set_project_out(set, isl_dim_set, dist_dim + 2, 1);

    //Project out r_receiver from isl_dim_param
    int idx_rrcv= 0;
    while(idx_rrcv < isl_set_dim(set,isl_dim_param) and
        isl_set_get_dim_name(set,isl_dim_param,idx_rrcv) != get_rank_string_type(rank_t::r_receiver)) idx_rrcv++;
    set = isl_set_project_out(set, isl_dim_param, idx_rrcv, 1);

    //Project out r_sender from isl_dim_param
    int idx_rsnd = 0;
    while(idx_rsnd < isl_set_dim(set,isl_dim_param) and
        isl_set_get_dim_name(set,isl_dim_param,idx_rsnd) != get_rank_string_type(rank_t::r_sender)) idx_rsnd++;
    set = isl_set_project_out(set, isl_dim_param, idx_rsnd, 1);

    //Set the name of the set to:
    //If it's a send --> b_r_snd_compName_seqId
    //If it's a receiver --> b_r_rcv_compName_seqId
    return isl_set_set_tuple_name(set, get_comm_id(rank_type, comm_id).c_str());
}

std::unordered_map<std::string, isl_set*> computation::construct_exchange_sets()
{
    //construct distribution map of the receiver
    isl_map* receiver_dist_map = construct_distribution_map(rank_t::r_receiver);

    //Find the set that needs to be computed by the receiver
    isl_set* receiver_to_compute_set = isl_set_apply(isl_set_copy(this->get_trimmed_time_processor_domain()), receiver_dist_map);

    //Find the receiver's needed_sets
    std::vector<isl_map*> rhs_accesses;
    generator::get_rhs_accesses(this->get_function(), this, rhs_accesses, false);

    //map computation name to the receiver needed set of that computation
    std::unordered_map <std::string, isl_set*> receiver_needed;

    for (isl_map* rhs_access : rhs_accesses) {
        //an access has the following structure [params]->{consumer[dims]->producer[dims]:constraints}
        //consumer is the current computation
        //get the name of the producer
        std::string comp_name = isl_map_get_tuple_name(rhs_access, isl_dim_out);
        //apply schedule to consumer
        rhs_access = isl_map_apply_domain(rhs_access, isl_map_copy(get_trimmed_union_of_schedules()));
        //apply schedule to producer
        computation* producer = get_function()->get_computation_by_name(comp_name)[0];
        rhs_access = isl_map_apply_range(rhs_access, isl_map_copy(producer->get_trimmed_union_of_schedules()));
        //tiramisu::str_dump("rhs_access after applying schedule ");isl_map_dump(rhs_access);
        //apply rhs_access
        isl_set* needed_set = isl_set_apply(isl_set_copy(receiver_to_compute_set), rhs_access);
        //check if it should do communication on it
        if(producer->get_distributed_dimension()!=-1){
            if (receiver_needed.find(comp_name) != receiver_needed.end())
                receiver_needed[comp_name] = isl_set_coalesce(isl_set_union(receiver_needed[comp_name], needed_set));
            else
                receiver_needed.insert({comp_name, needed_set});
        }else {
            DEBUG(3, "Computation " + comp_name + "isn't distributed, no communication needed");
        }
    }

    //receiver's owned_sets
    std::unordered_map<std::string,isl_set*> receiver_owned;
    for (auto needed_set : receiver_needed)
    {
        //get computation
        computation* producer = get_function()->get_computation_by_name(needed_set.first)[0];
        //construct distribution map of the receiver
        isl_map* producer_map = producer->construct_distribution_map(rank_t::r_receiver);
        isl_set* producer_to_compute_set = isl_set_apply(isl_set_copy(producer->get_trimmed_time_processor_domain()), producer_map);
        receiver_owned.insert({needed_set.first, producer_to_compute_set});
    }

    //sender's owned set
    std::unordered_map<std::string,isl_set*> sender_owned;
    for (auto needed_set : receiver_needed) {
        //get computation
        computation* producer = get_function()->get_computation_by_name(needed_set.first)[0];
        //construct distribution map of the receiver
        isl_map* producer_map = producer->construct_distribution_map(rank_t::r_sender);
        isl_set* producer_to_compute_set = isl_set_apply(isl_set_copy(producer->get_trimmed_time_processor_domain()), producer_map);
        sender_owned.insert({needed_set.first, producer_to_compute_set});
    }

    //The sets that need to be sent from r_sender -> r_receiver
    //sender_owned intersect (receiver_needed - receiver_owned)
    std::unordered_map<std::string, isl_set*> to_exchange_sets;
    for (auto needed : receiver_needed) {
        isl_set* missing = isl_set_subtract(needed.second, receiver_owned[needed.first]);
        to_exchange_sets.insert({needed.first, isl_set_coalesce(isl_set_intersect(missing, sender_owned[needed.first]))});
    }

    return to_exchange_sets;
}

void computation::gen_communication_code(isl_set*recv_iter_dom, isl_set* send_iter_dom, int comm_id, std::string comp_name)
{
    //creating access_variables
    var r_snd(get_rank_string_type(rank_t::r_sender).c_str());
    var r_rcv(get_rank_string_type(rank_t::r_receiver).c_str());

    //creating new iterators
    std::vector<tiramisu::expr> iterators;
    int idx = 2;
    while (idx < isl_set_dim(recv_iter_dom,isl_dim_set) ) {
        std::string name = generate_new_variable_name();
        recv_iter_dom = isl_set_set_dim_name(recv_iter_dom, isl_dim_set, idx, name.c_str());
        send_iter_dom = isl_set_set_dim_name(send_iter_dom, isl_dim_set, idx, name.c_str());
        iterators.push_back(var(name));
        idx++;
    }

    //creating access
    tiramisu::expr access = tiramisu::expr(op_t::o_access, comp_name,iterators,
    get_function()->get_computation_by_name(comp_name)[0]->get_data_type());

    auto data_type = get_function()->get_computation_by_name(comp_name)[0]->get_data_type();

    xfer data_transfer = computation::create_xfer(
        isl_set_to_str(send_iter_dom),
        isl_set_to_str(recv_iter_dom),
        r_rcv, r_snd,
        xfer_prop(data_type, {MPI, BLOCK, ASYNC}),
        xfer_prop(data_type, {MPI, BLOCK, ASYNC}),
        access, get_function());

    data_transfer.s->tag_distribute_level(r_snd);
    data_transfer.r->tag_distribute_level(r_rcv);

    computation *c = get_function()->get_computation_by_name(this->get_name())[0];

    //schedule communications
    assert(this->get_function()->sched_graph_reversed[this].size() <= 1 &&
            "Node has more than one predecessor.");

    //if predecessor
    if(this->get_predecessor() != nullptr)
    {
         //get level
        int level = this->get_function()->sched_graph_reversed[this][this->get_predecessor()] ;
        //clear
        computation *pred = this->get_predecessor();
        data_transfer.s->between(*pred, level, *c, level);
        data_transfer.r->between(*data_transfer.s, level, *c, level);
    }
    else
    {
        DEBUG(3, tiramisu::str_dump("Communication of "+ this->get_name()+" has no predecessor"));
        data_transfer.s->before(*data_transfer.r, computation::root);
        data_transfer.r->before(*c, computation::root);
    }

    //create send _access_string
    std::string it_string = "";
    for (int i = 0; i < iterators.size(); i++)
    {
        it_string += iterators[i].get_name();
        if(i < iterators.size() - 1) it_string += ',';
    }

    //Each rank will process dim_extent/nb_ranks
    //To fix and test
    int distributed_dimension = this->get_distributed_dimension();
    this->simplify(this->get_iteration_domain());
    isl_set * s= this->get_trimmed_time_processor_domain();
    project_out_static_dimensions(s);
    s = isl_set_project_out(s, isl_dim_set, distributed_dimension, 1);
    int extent = tiramisu::utility::get_extent(s, distributed_dimension);

    //construct string access
    std::string access_string = "{" + get_comm_id(rank_t::r_receiver,comm_id) + "[" + get_rank_string_type(rank_t::r_receiver)
    + "," + get_rank_string_type(rank_t::r_sender) + "," + it_string + "]->" +
    isl_map_get_tuple_name(get_function()->get_computation_by_name(comp_name)[0]->get_access_relation(), isl_dim_out);
    access_string += "[" + std::to_string(extent) + "+" +it_string + "]}";
    data_transfer.r->set_access(access_string);

    /***This works only for outermost loops***/
    //To do: make it work on any level
    //Important : get_bound doesn't work for dim more than one, that's why we project out all other dim
    recv_iter_dom = isl_set_project_out(recv_iter_dom, isl_dim_set, 0, 2);

    //adapt buffer size
    int additional_space = tiramisu::utility::get_extent(recv_iter_dom, 0);

    tiramisu::buffer *buff = this->get_function()->get_buffers().find(isl_map_get_tuple_name(
    get_function()->get_computation_by_name(comp_name)[0]->get_access_relation(), isl_dim_out))->second;

    int size = buff->get_dim_sizes()[0].get_int_val() + additional_space;
    buff->set_dim_size(0, size);
}

void computation::gen_communication()
{
    int comm_id = 0;

    //Sets that needs to be exchanged between ranks sender, receiver
    std::unordered_map<std::string, isl_set*>  to_receive_sets = construct_exchange_sets ();

    for (auto set : to_receive_sets)
    {
        project_out_static_dimensions(set.second);

        DEBUG(3, tiramisu::str_dump("To exchange set after project out:"); isl_set_dump(set.second));

        if(isl_set_is_empty(set.second)) continue;

        isl_set* recv_iter_dom = construct_comm_set(isl_set_copy(set.second), rank_t::r_receiver, comm_id);
        isl_set* send_iter_dom = construct_comm_set(set.second, rank_t::r_sender, comm_id);

        DEBUG(3, tiramisu::str_dump("Send iteration domain:"); isl_set_dump(send_iter_dom));
        DEBUG(3, tiramisu::str_dump("Receive iteration domain:"); isl_set_dump(recv_iter_dom));

        gen_communication_code(recv_iter_dom, send_iter_dom, comm_id, set.first);

        comm_id++;
    }
}

computation *computation::cache_shared(computation &inp, const var &level,
                  const std::vector<int> buffer_shape,
                  const std::vector<expr> copy_offsets,
                  bool pad_buffer)
{
    assert(inp.access_variables.size() == buffer_shape.size() &&
           "Buffer shape should be same as input!");
    assert(inp.access_variables.size() == copy_offsets.size() &&
           "Copy offsets should be same size as input!");

    function *fn = this->get_function();

    // Copy level dimension
    std::vector<int> dimensions = this->get_loop_level_numbers_from_dimension_names({level.get_name()});
    assert(dimensions.size() == 1);
    int copy_level = dimensions[0];

    // Find block/thread dimensions
    std::tuple<int, int, int> block_dims(-1, -1, -1), thread_dims;
    for (const auto &dims : fn->gpu_block_dimensions) {
        if (dims.first == this->get_name()) {
            block_dims = dims.second;
        }
    }
    for (const auto &dims : fn->gpu_thread_dimensions) {
        if (dims.first == this->get_name()) {
            thread_dims = dims.second;
        }
    }
    assert(std::get<0>(block_dims) != -1 && "Computation is not mapped to GPU!");

    // Create shared buffer
    std::string name_prefix = "_" + this->get_name() + "_" + inp.get_name();
    std::vector<expr> buff_shape(buffer_shape.begin(), buffer_shape.end());
    if (pad_buffer) {
        buff_shape[buff_shape.size() - 1] = buff_shape[buff_shape.size() - 1] + 1;
    }
    buffer *buff = new buffer(name_prefix + "_shared",
            buff_shape, inp.get_data_type(), a_temporary, fn);
    buff->tag_gpu_shared();

    // Create new access computation and replace mapping
    std::vector<var> access_variables;
    std::vector<expr> access_exprs;
    for (int i = 0; i < inp.access_variables.size(); i++) {
        var v = var(inp.access_variables[i].second, false);
        access_variables.push_back(v);
        access_exprs.push_back(v % buffer_shape[i]);
    }
    input *new_access = new input(name_prefix + "_access", access_variables, inp.get_data_type());
    new_access->store_in(buff, access_exprs);
    this->set_expression(this->expression.substitute_access(inp.get_name(), new_access->get_name()));

    // Find the level after which kernel starts
    int gpu_level = std::get<0>(thread_dims);
    if (std::get<2>(thread_dims) != -1) {
        gpu_level = std::get<2>(thread_dims);
    } else if (std::get<1>(thread_dims) != -1) {
        gpu_level = std::get<1>(thread_dims);
    }

    // Declare buffer
    isl_set *dec_domain = isl_map_range(isl_map_copy(this->get_schedule()));
    // Project out redundancy dimension
    dec_domain = isl_set_project_out(dec_domain, isl_dim_set, 0, 1);
    std::string dec_name = name_prefix + "_dec";
    dec_domain = isl_set_set_tuple_name(dec_domain, dec_name.c_str());
    project_out_static_dimensions(dec_domain);
    // Project out levels inside kernel
    dec_domain = isl_set_project_out(dec_domain, isl_dim_set, gpu_level + 1,
            isl_set_dim(dec_domain, isl_dim_set) - gpu_level - 1);
    dec_domain = isl_set_set_tuple_name(dec_domain, dec_name.c_str());
    std::string iteration_domain_str = isl_set_to_str(dec_domain);
    DEBUG(3, tiramisu::str_dump("Generated iteration domain for declaration: " + iteration_domain_str));
    computation *buf_dec = new computation(iteration_domain_str, allocate(*buff), true, p_none, fn);
    fn->gpu_block_dimensions.push_back(std::make_pair(buf_dec->get_name(), block_dims));
    fn->gpu_thread_dimensions.push_back(std::make_pair(buf_dec->get_name(), thread_dims));
    isl_set_free(dec_domain);

    // Find number of data points we need to copy and number of threads
    int buffer_size = buffer_shape[0];
    for (int i = 1; i < buffer_shape.size(); i++) {
        buffer_size = buffer_size * buffer_shape[i];
    }
    int thread_pool_size = 1;
    for (int i = 0; i < this->thread_block_shape.size(); i++) {
        thread_pool_size *= this->thread_block_shape[i];
    }

    // Construct iteration domain for copy
    isl_set *copy_domain = isl_map_range(isl_map_copy(this->get_schedule()));
    // Project out redundancy dimension
    copy_domain = isl_set_project_out(copy_domain, isl_dim_set, 0, 1);
    copy_domain = isl_set_set_tuple_name(copy_domain, (name_prefix + "_copy").c_str());
    project_out_static_dimensions(copy_domain);
    // Project out dimensions under copy_level
    copy_domain = isl_set_project_out(copy_domain, isl_dim_set, copy_level + 1,
            isl_set_dim(copy_domain, isl_dim_set) - copy_level - 1);
    isl_set *sync_domain = isl_set_copy(copy_domain);
    // Add a new iterator to the domain in case each thread copies more than one value
    copy_domain = isl_set_add_dims(copy_domain, isl_dim_set, 1);
    std::string copy_iter_name = name_prefix + "_copy_iter";
    copy_domain = isl_set_set_dim_name(copy_domain, isl_dim_set,
            isl_set_dim(copy_domain, isl_dim_set) - 1, copy_iter_name.c_str());
    // Add constraints for the new iterator
    isl_constraint *cst1 = isl_constraint_alloc_inequality(isl_local_space_from_space(isl_set_get_space(copy_domain)));
    cst1 = isl_constraint_set_coefficient_si(cst1, isl_dim_set, copy_level + 1, 1);
    //cst1 = isl_constraint_set_constant_si(cst1, 0);
    copy_domain = isl_set_add_constraint(copy_domain, cst1);
    isl_constraint *cst2 = isl_constraint_alloc_inequality(isl_local_space_from_space(isl_set_get_space(copy_domain)));
    cst2 = isl_constraint_set_coefficient_si(cst2, isl_dim_set, copy_level + 1, -1);
    cst2 = isl_constraint_set_constant_si(cst2, (buffer_size - 1) / thread_pool_size);
    copy_domain = isl_set_add_constraint(copy_domain, cst2);
    copy_domain = isl_set_set_tuple_name(copy_domain, (name_prefix + "_copy").c_str());

    // Find copy index
    expr copy_index = var(isl_set_get_dim_name(copy_domain, isl_dim_set, std::get<0>(thread_dims)), false);
    if (std::get<1>(thread_dims) != -1) {
        std::string dim_name = isl_set_get_dim_name(copy_domain, isl_dim_set, std::get<1>(thread_dims));
        copy_index = copy_index * this->thread_block_shape[1] + var(dim_name, false);
    }
    if (std::get<2>(thread_dims) != -1) {
        std::string dim_name = isl_set_get_dim_name(copy_domain, isl_dim_set, std::get<2>(thread_dims));
        copy_index = copy_index * this->thread_block_shape[2] + var(dim_name, false);
    }
    copy_index = var(copy_iter_name, false) * thread_pool_size + copy_index;

    // Assign copy indices to temporary registers. This is a workaround since
    // using copy_index directly for indexing doesn't seem to work
    std::vector<computation *> copy_index_regs;
    std::vector<computation *> copy_index_decs;
    int denominator = 1;
    for (int i = buffer_shape.size() - 1; i >= 0; i--) {
        isl_set *domain = isl_set_copy(copy_domain);
        domain = isl_set_set_tuple_name(domain, (name_prefix + "_copy_comp" + std::to_string(i)).c_str());
        computation *copy_index_reg = new computation(
                isl_set_to_str(domain),
                copy_index / denominator % buffer_shape[i], true, p_int32, fn);
        copy_index_reg->store_in({expr(0)}, {1});
        copy_index_reg->get_buffer()->tag_gpu_register();
        domain = isl_set_set_tuple_name(domain, (name_prefix + "_copy_dec" + std::to_string(i)).c_str());
        computation *copy_index_dec = new computation(
                isl_set_to_str(domain),
                allocate(*copy_index_reg->get_buffer()), true, p_none, fn);
        isl_set_free(domain);
        denominator *= buffer_shape[i];
        copy_index_regs.insert(copy_index_regs.begin(), copy_index_reg);
        copy_index_decs.insert(copy_index_decs.begin(), copy_index_dec);
        fn->gpu_block_dimensions.push_back(std::make_pair(copy_index_reg->get_name(), block_dims));
        fn->gpu_thread_dimensions.push_back(std::make_pair(copy_index_reg->get_name(), thread_dims));
        fn->gpu_block_dimensions.push_back(std::make_pair(copy_index_dec->get_name(), block_dims));
        fn->gpu_thread_dimensions.push_back(std::make_pair(copy_index_dec->get_name(), thread_dims));
    }

    // Create access patterns for shared memory and input computation
    std::vector<expr> buf_access;
    std::vector<expr> inp_access;
    for (int i = 0; i < buffer_shape.size(); i++) {
        buf_access.push_back(var(copy_index_regs[i]->get_buffer()->get_name(), false));
        inp_access.push_back(var(copy_index_regs[i]->get_buffer()->get_name(), false) + copy_offsets[i]);
    }

    // Add names of index variables as params to copy domain
    for (int i = 0; i < copy_index_regs.size(); i++) {
        copy_domain = isl_set_add_dims(copy_domain, isl_dim_param, 1);
        copy_domain = isl_set_set_dim_name(copy_domain, isl_dim_param,
                isl_set_dim(copy_domain, isl_dim_param) - 1,
                copy_index_regs[i]->get_buffer()->get_name().c_str());
    }

    // Create the copy computation
    std::string copy_domain_str = isl_set_to_str(copy_domain);
    DEBUG(3, tiramisu::str_dump("Generated iteration domain for copy: " + copy_domain_str));
    computation *copy_computation = new computation(copy_domain_str,
            expr(o_access, inp.get_name(), inp_access, inp.get_data_type()),
            true, inp.get_data_type(), fn);
    copy_computation->store_in(buff, buf_access);
    copy_computation->add_predicate(copy_index < buffer_size);
    fn->gpu_block_dimensions.push_back(std::make_pair(copy_computation->get_name(), block_dims));
    fn->gpu_thread_dimensions.push_back(std::make_pair(copy_computation->get_name(), thread_dims));
    isl_set_free(copy_domain);

    // Synchronization
    computation *c_sync1 = new computation(
            isl_set_to_str(isl_set_set_tuple_name(isl_set_copy(sync_domain), (name_prefix + "_sync1").c_str())),
            tiramisu::sync(), true, p_int32, fn);
    computation *c_sync2 = new computation(
            isl_set_to_str(isl_set_set_tuple_name(isl_set_copy(sync_domain), (name_prefix + "_sync2").c_str())),
            tiramisu::sync(), true, p_int32, fn);
    fn->gpu_block_dimensions.push_back(std::make_pair(c_sync1->get_name(), block_dims));
    fn->gpu_thread_dimensions.push_back(std::make_pair(c_sync1->get_name(), thread_dims));
    fn->gpu_block_dimensions.push_back(std::make_pair(c_sync2->get_name(), block_dims));
    fn->gpu_thread_dimensions.push_back(std::make_pair(c_sync2->get_name(), thread_dims));

    // Schedule computations
    {
        // Traverse schedule tree up and find the first computation in the given level
        computation *curr = this;
        computation *pred = curr->get_predecessor();
        while (pred != nullptr && fn->sched_graph[pred][curr] >= copy_level) {
            curr = pred;
            pred = curr->get_predecessor();
        }
        // Schedule
        if (pred != nullptr) {
            c_sync1->between(*pred, fn->sched_graph[pred][curr], *curr, copy_level);
        } else {
            c_sync1->before(*curr, copy_level);
        }
        c_sync2->between(*c_sync1, copy_level, *curr, copy_level);
        copy_computation->between(*c_sync1, copy_level, *c_sync2, copy_level);
        copy_index_decs[0]->between(*copy_computation->get_predecessor(), copy_level, *copy_computation, copy_level + 1);
        for (int i = 1; i < copy_index_decs.size(); i++) {
            copy_index_decs[i]->between(*copy_computation->get_predecessor(), copy_level + 1, *copy_computation, copy_level + 1);
        }
        for (int i = 0; i < copy_index_regs.size(); i++) {
            copy_index_regs[i]->between(*copy_computation->get_predecessor(), copy_level + 1, *copy_computation, copy_level + 1);
        }
    }
    // Schedule buffer declaration
    {
        // Traverse schedule tree up and find the first computation in the same kernel
        computation *curr = this;
        computation *pred = curr->get_predecessor();
        while (pred != nullptr && fn->sched_graph[pred][curr] >= gpu_level) {
            curr = pred;
            pred = curr->get_predecessor();
        }
        // Schedule the declaration to the beginning of the kernel
        if (pred != nullptr) {
            buf_dec->between(*pred, fn->sched_graph[pred][curr], *curr, gpu_level);
        } else {
            buf_dec->before(*curr, gpu_level);
        }
    }

    return new_access;
}

}
