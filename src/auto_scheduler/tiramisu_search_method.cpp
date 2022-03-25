#include <sys/wait.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include <random>

#include <random>
#include <functional>
#include <exception>

#include <stdexcept>
#define TIME_LIMIT 1000
struct UnrollingException : public std::exception {
    const char * what () const throw ()
        {
            return "unrolling error : unrolled loop level is a user node due to dimension error";
        }
};
namespace tiramisu::auto_scheduler
{

void beam_search::search(syntax_tree& ast)
{
//    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
//        ast.clear_new_optimizations();
       
    std::vector<syntax_tree*> children;
        
    // Look for an optimization that can be applied
//    int nb_optims_tried = 0;
//    int nb_explored_optims = ast.nb_explored_optims;

    if(generator_state::initialized == false)
    {
        
        ast.initialize_search_space_optimizations(DEFAULT_OPTIMIZATIONS_ORDER);
        // the optimizations are specified along with the parameters in the generator_state attribute inside the AST.
        assert(generator_state::initialized == true);
    }

    //std::cout<<"TESTED";
    
    while ((!ast.is_search_space_empty()))
    {
        // schedule generation based on generator_state attribute in the AST.
        auto new_children = scheds_gen->generate_schedules(ast);
        

        for(auto& child:new_children)
        {
            child->move_to_next_optimization_target();
        }
        
        children.insert(children.end(), new_children.begin(), new_children.end()); // concatenate

        if  (ast.search_state.is_current_optimization_fully_explored() && !children.empty()) {
            // move to next optimization
            // explores next optimization/alternative
            ast.move_to_next_optimization_target();
            break;
        }
        else
            ast.move_to_next_optimization_target();
    }
       
    // Stop if no more optimizations can be applied
    if (children.size() == 0)
        return ;
       
    // Evaluate children and sort them from smallest to highest evaluation
    

   // evaluate while removing illegal versions
    auto iterator = children.begin();
    while (iterator != children.end())
    {

        (*iterator)->transform_ast();

        if ((*iterator)->ast_is_legal() == false) {

            // print deleted Ast 
            (*iterator)->print_previous_optims();
            std::cout << "\n-----------" << std::endl;
            (*iterator)->print_new_optims();
            (*iterator)->print_ast();
            (*iterator)->print_isl_states();
            std::cout << "\n<illegal>\n";
            delete (*iterator);
            iterator = children.erase(iterator);
        }
        else {

            // evaluate and print Ast
            (*iterator)->print_previous_optims();
            std::cout << "\n-----------" << std::endl;
            (*iterator)->print_new_optims();
            (*iterator)->print_ast();
//            (*iterator)->print_isl_states();
//            (*iterator)->print_computations_accesses();
            std::cout << "\n<legal>\n";

            (*iterator)->evaluation = eval_func->evaluate(*(*iterator));
            std::cout << "Evaluation : " << (*iterator)->evaluation << std::endl << std::endl;


            std::cout << "\n============================================================================================================" << std::endl;

            if ((*iterator)->evaluation < best_evaluation)
            {
                best_evaluation = (*iterator)->evaluation;
                best_ast = (*iterator);
            }

            ++iterator;

        }
        
        nb_explored_schedules++;
    }

    // Stop if we reached the maximum depth
//    if (nb_explored_optims >= max_depth)
//        return ;
        
    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
//    ast_copy->nb_explored_optims = nb_explored_optims;
    children.push_back(ast_copy);

    // Sort children from smallest evaluation to largest
    
//    std::cout<<"\noriginal list\n" ;
//    for (syntax_tree *child : children)
//    {
//        std::cout<<child->evaluation<<"+";
//    }

    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });

    // keep the top 'beam_size' children and delete the rest
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];
    
        
    children.resize(std::min(beam_size, (int)children.size()));

//    std::cout<<"\nremaining list\n" ;
//    for (syntax_tree *child : children)
//    {
//        std::cout<<child->evaluation<<"+";
//    }

    // Search recursively on the best children
    for (syntax_tree *child : children)
    {
        child->search_depth = ast.search_depth + 1;        
        search(*child);
    }
}

int timeout = 0;
int child_done = 0;
int cont = 0;
  void child_handler(int sig)
{
    child_done = 1;
}
int nb_exec = std::stoi(std::getenv("MAX_RUNS"));
void alarm_handler(int sig)
{
    timeout = 1;
}
void sig_usr(int signo){
    cont = 1; 
    
}
/*
multiply two matrices AxB
*/
std::vector<std::vector<int>>  multiply(const std::vector<std::vector<int>> & m1, const std::vector<std::vector<int>> & m2)
{
std::vector<std::vector<int>> result(m1.size(), std::vector<int>(m2.at(0).size()));

    for(std::size_t row = 0; row < result.size(); ++row) {
        for(std::size_t col = 0; col < result.at(0).size(); ++col) {
            for(std::size_t inner = 0; inner < m2.size(); ++inner) {
                result.at(row).at(col) += m1.at(row).at(inner) * m2.at(inner).at(col);
            }
        }
    }
    return result;
}
void beam_search::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
    
    std::vector<syntax_tree*> children;
    if(generator_state::initialized == false)
    {
        ast.initialize_search_space_optimizations(DEFAULT_OPTIMIZATIONS_ORDER);
        // the optimizations are specified along with the parameters in the generator_state attribute inside the AST.
        assert(generator_state::initialized == true);
    }

    //std::cout<<"TESTED 0"<< isl_map_to_str (ast.computations_list[0]->get_schedule())<<std::endl;
    //std::cout<<"TESTED 1 "<< isl_map_to_str (ast.computations_list[1]->get_schedule())<<std::endl;
    //std::cout<<"before search space empty"<<std::endl;
    while ((!ast.is_search_space_empty()))
    {
        // schedule generation based on generator_state attribute in the AST.
        auto new_children = scheds_gen->generate_schedules(ast);
        //std::cout<<"done generating"<<std::endl;
        for(auto& child:new_children)
            child->move_to_next_optimization_target();

        children.insert(children.end(), new_children.begin(), new_children.end()); // concatenate

        if  (ast.search_state.is_current_optimization_fully_explored() && !children.empty()) {
            // move to next optimization
            // explores next optimization/alternative
            ast.move_to_next_optimization_target();
            break;
        }
        else
            ast.move_to_next_optimization_target();
    }
    //children size is: "
    // Stop if no more optimizations can be applied
    if (children.size() == 0)
        return ;

    // Evaluate children and sort them from smallest to highest evaluation
    // evaluate while removing illegal versions
    auto iterator = children.begin();
    while (iterator != children.end())
    {
        bool unrolling_exception_thrown = false;
        (*iterator)->transform_ast();
        if ((*iterator)->ast_is_legal() == false) {
            // print deleted Ast
            (*iterator)->print_previous_optims();
            std::cout << "\n-----------" << std::endl;
            (*iterator)->print_new_optims();
            (*iterator)->print_ast();
            (*iterator)->print_isl_states();
            std::cout << "\n<illegal>\n";
            delete (*iterator);
            iterator = children.erase(iterator);
        }
        else {
            // evaluate and print Ast
            (*iterator)->print_previous_optims();
            std::cout << "\n-----------" << std::endl;
            (*iterator)->print_new_optims();
            (*iterator)->print_ast();
//            (*iterator)->print_isl_states();
//            (*iterator)->print_computations_accesses();
            std::cout << "\n<legal>\n";

            int fd[2];
            
            // create pipe descriptors
	        pipe(fd);
            timeout=0;
            child_done=0;
            cont = false;
            std::vector<float> measurements;
            
            // create a child process to execute the code and get measurements
            // this is added so that we can limit the code generation time for large programs
            pid_t pid;
            pid = fork();

            if (pid == -1) {
                perror("fork failed");
                exit(1);
            } else if (pid == 0) {
                // put get_measurements in a try block in the case of erroneous unrolling
                // this should be added to ast_is_legal but for now it can only be detected at apply_optimization level
                try{
                     measurements = exec_eval->get_measurements(**iterator, false, schedule_timeout,true);
                }
                catch(UnrollingException e){ 
                     // Remove all the optimizations
                    exec_eval->fct->reset_schedules();
                    measurements.clear();
                    measurements.push_back(std::numeric_limits<float>::infinity());
                    unrolling_exception_thrown = true;
                    cont = 1;
                }
                int size =measurements.size();
                float ar[measurements.size()];
                // put the measurements in an array to be sent to the parent process
                for(int i=0;i<measurements.size();i++) ar[i]=measurements.at(i);
                close(fd[0]);
                // send number of measurements to parent process
                write(fd[1], &size, sizeof(size));
                // send measurements to parent process
                write(fd[1], &ar, sizeof(ar));
                // close the pipe
                close(fd[1]);
                // end child process
                _exit(1);
            }

            // set up the signal handlers after forking so the child doesn't inherit them
            // an alarm in the case of the timelimit is reached and the child process was interrupted
            signal(SIGALRM, alarm_handler);
            // an alarm in the case of the child process ending ie. both data generation and evaluation are done
            signal(SIGCHLD, child_handler);
            // an alarm in the case of the data generation ending before the timelimit. We still need to wait for the evaluation to be done. 
            signal(SIGUSR1,sig_usr);
            // install an alarm to be fired after TIME_LIMIT
            
            // set the alarm
            alarm(TIME_LIMIT);
            
            pause();

            if (timeout) {
                
                // if the timeout has been reached    
                int result = waitpid(pid, NULL, WNOHANG);
                if (result == 0) {
                    // child still running, so kill it
                    
                    // remove all the optimizations
                    exec_eval->fct->reset_schedules();
                    measurements.clear();
                    // if the timeout has been reached, put infinity as a measurement
                    measurements.push_back(std::numeric_limits<float>::infinity());
                    // cancel any previously set alarm 
                    alarm(0); 
                    // kill child process
                    kill(pid, 9);
                    
                
                    waitpid(-1,NULL,0);
                } else {
                    // if by the time we detect that the alarm has been raised, the evaluation has been completed
                    // we recieve the measurements from the child 
                    int size = 0;
                    close(fd[1]);
                    read(fd[0], &size, sizeof(int));
                    float ar[size];
                    read(fd[0], &ar, size*sizeof(float));
                    for(int i=0;i<size;i++) measurements.push_back(ar[i]);
                    close(fd[0]);
                    
                }
                   
            }else if (child_done) {
                // the execution of the child is done, both code generation and evaluation
                // we recieve the measurements from the child
                int size =0;
                close(fd[1]);
                read(fd[0], &size, sizeof(int));
                float ar[size];
                read(fd[0], &ar, size*sizeof(float));
                for(int i=0;i<size;i++) measurements.push_back(ar[i]);
                close(fd[0]);
                
                waitpid(-1,NULL,0);
            }else if(cont){
                // execution is not done but the code generation is done. This happens for large programs that take a long time to execute
                // cancel the timeout alarm 
                alarm(0);
                int size=0;
                // we wait for the evalution of the generated code
                while(!child_done){}
                // after evealuation is done, we recieve the measurements from the child
                close(fd[1]);
                read(fd[0], &size, sizeof(int));
                float ar[size];
                read(fd[0], &ar, size*sizeof(float));
                for(int i=0;i<size;i++) measurements.push_back(ar[i]);
                close(fd[0]);
                waitpid(-1,NULL,0);
            }
            
            

                (*iterator)->evaluation = min_eval(measurements);

                parent_trace->add_child_path((*iterator), schedules_annotations->size());


                std::string schedule_annot = evaluate_by_learning_model::get_schedule_json(*(*iterator));

                //remove the last two characters }\n
                schedule_annot.pop_back();
                schedule_annot.pop_back();

                if (std::isfinite((*iterator)->evaluation)) // the evaluation is not finite mean that the schedule didn't run
                    schedule_annot += ", \n\"execution_times\" : " + measurements_to_str(measurements) + "\n}\n";
                else
                    schedule_annot += ", \n\"execution_times\" : null\n}\n";
            if(!unrolling_exception_thrown){
                schedules_annotations->push_back(schedule_annot);

                std::cout << "Schedule number "<< schedules_annotations->size() << std::endl;
                std::cout << "Evaluation : " << (*iterator)->evaluation << std::endl;
                std::cout << "Number of measurements : " << measurements.size() << std::endl;
                std::cout << "===================================" << std::endl << std::endl;

                if (std::isinf((*iterator)->evaluation))
                    std::cerr<< "Evaluation of schedule "<< schedules_annotations->size() <<" failed "<< std::endl;

                if ((*iterator)->evaluation < best_evaluation)
                {
                    best_evaluation = (*iterator)->evaluation;
                    best_ast = (*iterator);
                }
            }
            ++iterator;

        }

        nb_explored_schedules++;
    }

    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
    children.push_back(ast_copy);

    parent_trace->add_child_path(ast_copy, parent_trace->get_candidate_id()); // keeps the same id since it's just copy

    // Sort children from smallest evaluation to largest

    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });

    // keep the top 'beam_size' children and delete the rest
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];


    children.resize(std::min(beam_size, (int)children.size()));
    // Search recursively on the best children
    for (syntax_tree *child : children)
    {
        child->search_depth = ast.search_depth + 1;
        search_save(*child, schedules_annotations, parent_trace->child_mappings[child], schedule_timeout);
    }

}
void beam_search::explore_fusion(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
    //std::cout<<"inside explore fusion"<<std::endl;
    std::vector<syntax_tree*> children;
    std::vector<optimization_type> optims;

    optims.push_back(optimization_type::FUSION);
    
    ast.initialize_search_space_optimizations(optims);


//    std::cout<<"TESTED";
    //std::cout<<"before search space empty in fusion"<<std::endl;
    while ((!ast.is_search_space_empty()))
    {
        // schedule generation based on generator_state attribute in the AST.
        auto new_children = scheds_gen->generate_schedules(ast);

        for(auto& child:new_children)
            child->move_to_next_head();

        children.insert(children.end(), new_children.begin(), new_children.end()); // concatenate

        if  (ast.search_state.is_current_optimization_fully_explored()) {
            // move to next optimization
            // explores next optimization/alternative
            ast.move_to_next_head();
            break;
        }
        else
            ast.move_to_next_head();
    }
    //std::cout<<"children size is: "<<children.size()<<std::endl;

    // Evaluate children and sort them from smallest to highest evaluation
    // evaluate while removing illegal versions
    auto iterator = children.begin();
    while (iterator != children.end())
    {
        (*iterator)->transform_ast();
        if ((*iterator)->ast_is_legal() == false) {
            // print deleted Ast
            (*iterator)->print_previous_optims();
            std::cout << "\n-----------" << std::endl;
            (*iterator)->print_new_optims();
            (*iterator)->print_ast();
            (*iterator)->print_isl_states();
            std::cout << "\n<illegal>\n";
            delete (*iterator);
            iterator = children.erase(iterator);
        }
        else {
            // evaluate and print Ast
            (*iterator)->print_previous_optims();
            std::cout << "\n-----------" << std::endl;
            (*iterator)->print_new_optims();
            (*iterator)->print_ast();
//            (*iterator)->print_isl_states();
//            (*iterator)->print_computations_accesses();
            std::cout << "\n<legal>\n";

            std::vector<float> measurements;
            measurements = exec_eval->get_measurements(**iterator, false, schedule_timeout);
            (*iterator)->evaluation = min_eval(measurements);

            parent_trace->add_child_path((*iterator), schedules_annotations->size());

            std::string schedule_annot = evaluate_by_learning_model::get_schedule_json(*(*iterator));

            //remove the last two characters }\n
            schedule_annot.pop_back();
            schedule_annot.pop_back();

            if (std::isfinite((*iterator)->evaluation)) // the evaluation is not finite mean that the schedule didn't run
                schedule_annot += ", \n\"execution_times\" : " + measurements_to_str(measurements) + "\n}\n";
            else
                schedule_annot += ", \n\"execution_times\" : null\n}\n";

            schedules_annotations->push_back(schedule_annot);

            std::cout << "Schedule number "<< schedules_annotations->size() << std::endl;
            std::cout << "Evaluation : " << (*iterator)->evaluation << std::endl;
            std::cout << "Number of measurements : " << measurements.size() << std::endl;
            std::cout << "===================================" << std::endl << std::endl;

            if (std::isinf((*iterator)->evaluation))
                std::cerr<< "Evaluation of schedule "<< schedules_annotations->size() <<" failed "<< std::endl;

            if ((*iterator)->evaluation < best_evaluation)
            {
                best_evaluation = (*iterator)->evaluation;
                best_ast = (*iterator);
            }

            ++iterator;

        }

        nb_explored_schedules++;
    }

    // Add the current AST to the list of children
    syntax_tree *ast_copy = ast.copy_ast();
    children.push_back(ast_copy);

    parent_trace->add_child_path(ast_copy, parent_trace->get_candidate_id()); // keeps the same id since it's just copy

    // Sort children from smallest evaluation to largest
    //std::cout<<"Schedule of first comp after applying fusion: "<<isl_map_to_str(children.at(0)->computations_list.at(0)->get_schedule())<<std::endl;
    //std::cout<<"Schedule of second comp after applying fusion: "<<isl_map_to_str(children.at(0)->computations_list.at(1)->get_schedule())<<std::endl;
    
    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
        return a->evaluation < b->evaluation;
    });

    // keep the top 'beam_size' children and delete the rest
    for (int i = beam_size; i < children.size(); ++i)
        delete children[i];


    children.resize(std::min(beam_size, (int)children.size()));
    // Search recursively on the best children
    for (syntax_tree *child : children)
    {

        search_save_matrix(*child, schedules_annotations, parent_trace->child_mappings[child], schedule_timeout);
    }

}
/*
return identity matrix
*/
std::vector <  std::vector<int> > get_identity(int depth){
    std::vector <  std::vector<int> >  matrix(depth);
        for(int l = 0; l<matrix.size(); l++){
            matrix.at(l)= std::vector<int>(depth);
            for(int c = 0; c<matrix.size(); c++){
                            if (l!=c ){
                                matrix.at(l).at(c) = 0;
                            }else{
                                matrix.at(l).at(c) = 1;
                            }
            }
        }
        return matrix;
}
// list of matrices to explore at each level of the exploration tree
//std::vector <std::vector < std::vector<int> >> matrices;
// list of hashes of matrices we explored before to avoid repeating schedules 
std::vector<std::size_t> hashes;
void beam_search::search_save_matrix(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
{
    //std::cout<<"starting search save "<<std::endl;
    //     
    std::default_random_engine rand_generator;

    
    std::vector<syntax_tree*> children;
    // list of ASTs to be explored for next level 
    std::vector<syntax_tree*> to_be_explored;

    std::hash<std::string> hasher;

    std::vector<optimization_type> optims;
    optims.push_back(optimization_type::MATRIX);
    
    
    ast.initialize_search_space_optimizations(optims);
    // if this is the roor of the exploration tree 
    // needs to be optimized
    if (ast.search_depth==0){
        //std::cout<<"inside parent of search sapce "<<std::endl;
        std::vector<ast_node*> nodes;
        for(auto root: ast.roots){
            std::vector<ast_node*> nodes;
            root->get_all_nodes(nodes);
            for(auto node : nodes){
                if(node->computations.size()>0){
                    optimization_info optim_info;
                    optim_info.type = optimization_type::MATRIX;
                    node->get_all_computations(optim_info.comps);
                    // for the original schedule, the transformation matrix is the identity
                    //std::cout<<"node->depth: "<<node->depth+1<<std::endl;
                    optim_info.matrix = get_identity(node->depth+1);
                    ast.new_optims.push_back(optim_info);
                    //std::cout<<"pushed optims"<<ast.new_optims.size()<<std::endl;
                    
                }
                
            }
        }
        
        
    }
    
    hashes.push_back(hasher(ast.get_schedule_str()));
    //ast.search_state.optimization_index = 0;
    //std::cout<<"!ast.is_search_space_empty(): "<<!ast.is_search_space_empty()<<std::endl;
    while ((!ast.is_search_space_empty()))
    {
        // schedule generation based on generator_state attribute in the AST.
        auto new_children = scheds_gen->generate_matrices(ast);
        //std::cout<<"new children size: "<<new_children.size()<<std::endl; 
        for(auto& child:new_children)
            child->move_to_next_head();
        
        children.insert(children.end(), new_children.begin(), new_children.end()); // concatenate
        if  (ast.search_state.is_current_optimization_fully_explored() && !children.empty()) {
            // move to next optimization
            // explores next optimization/alternative
            ast.move_to_next_head();
            //std::cout<<"about to break"<<std::endl;
            break;
        }
        else
            ast.move_to_next_head();
    }
    //exec_eval->fct->reset_schedules();
    //std::cout<<"passed while loop with size: "<<children.size()<<std::endl;
   
    
    // hash the parent 
    std::size_t parent_hash=hasher(ast.get_schedule_str());
    // generate the matrices to be explored at this level
    
    
    
    
    
    // stop if no more optimizations can be applied
    if (children.size() == 0) return ;
    


    
    auto iterator = children.begin();
    
    
    
   
    
    std::vector<std::vector<std::vector<int>>> repeated;
    

    // number of matrices explored so far at this level. used to go through the matrices global variable
    int nb_matrices =0;

    syntax_tree *child = *iterator;


    // evaluate children and sort them from smallest to highest evaluation
    // evaluate while removing illegal versions
    while (iterator != children.end())
    {
         
        child = *iterator;
        //std::cout<<"in the cont condition"<<std::endl;
        //std::cout<<"nb_optims prev in child: "<<child->previous_optims.size()<<std::endl;
        //std::cout<<"nb_optims in child: "<<child->new_optims.size()<<std::endl;
        child->transform_ast();
        if (!child->ast_is_legal()) {
            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                // print deleted Ast
                child->print_previous_optims();
                std::cout << "\n-----------" << std::endl;
                child->print_new_optims();
                
                child->print_ast();
                child->print_isl_states();
                std::cout << "\n<illegal>\n";
            }
            delete child;
            iterator = children.erase(iterator);
        }
        else {
            
            
        
            
            
            // hash the legal matrix 
            std::size_t hash=hasher(child->get_schedule_str());
            
            bool repeated = false;
            // check if we explored this matrix before  
            for(std::size_t hashe:hashes){
                if(hashe==hash){
                    delete child;
                    iterator = children.erase(iterator);
                    repeated = true;
                    break;
                    
                }
            }
        
            if(repeated){
                //std::cout<<"this schedule was repeated"<<ast.get_schedule_str()<<std::endl;
                continue;
            } 

            
            // if the matrix is legal and not repeated we add its hash to the list of seen hashes and we start the evaluation 
            hashes.push_back(hash);
              
            // print and evaluate Ast
            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                child->print_previous_optims();
                std::cout << "\n-----------" << std::endl;
                child->print_new_optims();
                child->print_ast();
                child->print_isl_states();
                std::cout << "\n<legal>\n";
                child->print_computations_accesses();
            }
            int fd[2];
            
            // create pipe descriptors
	        pipe(fd);
            timeout=0;
            child_done=0;
            cont = false;
            std::vector<float> measurements;
            
            
            // create a child process to execute the code and get measurements
            // this is added so that we can limit the code generation time for large programs
            pid_t pid;
            pid = fork();

            if (pid == -1) {
                perror("fork failed");
                exit(1);
            } else if (pid == 0) {
                measurements = exec_eval->get_measurements(*child, false, schedule_timeout,true);
                int size =measurements.size();
                float ar[measurements.size()];

                // put the measurements in an array to be sent to the parent process
                //std::cout<<"measuring "<<std::endl;
                for(int i=0;i<measurements.size();i++) ar[i]=measurements.at(i);
                
                close(fd[0]);
                // send number of measurements to parent process
                write(fd[1], &size, sizeof(size));
                // send measurements to parent process
                write(fd[1], &ar, sizeof(ar));
                // close the pipe
                close(fd[1]);
                // end child process
                _exit(1);
            }

            // set up the signal handlers after forking so the child doesn't inherit them
            // an alarm in the case of the timelimit is reached and the child process was interrupted
            signal(SIGALRM, alarm_handler);
            // an alarm in the case of the child process ending ie. both data generation and evaluation are done
            signal(SIGCHLD, child_handler);
            // an alarm in the case of the data generation ending before the timelimit. We still need to wait for the evaluation to be done. 
            signal(SIGUSR1,sig_usr);
            // install an alarm to be fired after TIME_LIMIT
            
            // set the alarm
            alarm(TIME_LIMIT);
            
            pause();

            if (timeout) {
                
                // if the timeout has been reached    
                    
                int result = waitpid(pid, NULL, WNOHANG);
                if (result == 0) {
                    // child still running, so kill it
                
                    // Remove all the optimizations
                    exec_eval->fct->reset_schedules();
                    measurements.clear();
                    // if the timeout has been reached, put infinity as a measurement

                    measurements.push_back(std::numeric_limits<float>::infinity());
                    // cancel any previously set alarm 
                    alarm(0); 
                    // kill child process
                    kill(pid, 9);
                    
                
                    waitpid(-1,NULL,0);
                } else {
                    // if by the time we detect that the alarm has been raised, the evaluation has been completed
                    // we recieve the measurements from the child
                    int size = 0;
                    close(fd[1]);
                    read(fd[0], &size, sizeof(int));
                    float ar[size];
                    read(fd[0], &ar, size*sizeof(float));
                    for(int i=0;i<size;i++) measurements.push_back(ar[i]);
                    close(fd[0]);
                    
                }
                
                
            }else if (child_done) {
                // the execution of the child is done, both code generation and evaluation
                // we recieve the measurements from the child
                int size =0;
                close(fd[1]);
                read(fd[0], &size, sizeof(int));
                float ar[size];
                read(fd[0], &ar, size*sizeof(float));
                for(int i=0;i<size;i++) measurements.push_back(ar[i]);
                close(fd[0]);
                
                waitpid(-1,NULL,0);
            }else if(cont){
                // execution is not done but the code generation is done. This happens for large programs that take a long time to execute
                // cancel the timeout alarm  
                alarm(0);
                int size=0;
                // we wait for the evalution of the generated code
                while(!child_done){}
                // after evealuation is done, we recieve the measurements from the child
                close(fd[1]);
                read(fd[0], &size, sizeof(int));
                float ar[size];
                read(fd[0], &ar, size*sizeof(float));
                for(int i=0;i<size;i++) measurements.push_back(ar[i]);
                
                close(fd[0]);
                waitpid(-1,NULL,0);
            }
            //std::cout<<"before min eval "<<std::endl;       
            child->evaluation = min_eval(measurements);
            //std::cout<<" befpre nb_explored_matrices "<<std::endl;
            if(hash != parent_hash) child->nb_explored_matrices = child->nb_explored_matrices +1; 
            //std::cout<<" nb_explored_matrices "<<std::endl;
            //if(hash != parent_hash) child->nb_explored_matrices = child->nb_explored_matrices +1; 
            
            
            parent_trace->add_child_path(child, schedules_annotations->size());
            //std::cout<<"after child p ath "<<std::endl;
            std::string schedule_annot = evaluate_by_learning_model::get_schedule_json(*child);
            //std::cout<<" after schedule json "<<std::endl;
            //remove the last two characters }\n
            schedule_annot.pop_back();
            schedule_annot.pop_back();
            
            if (std::isfinite(child->evaluation)) // the evaluation is not finite mean that the schedule didn't run
                schedule_annot += ", \n\"execution_times\" : " + measurements_to_str(measurements) + "\n}\n";
            else
                schedule_annot += ", \n\"execution_times\" : null\n}\n";

            schedules_annotations->push_back(schedule_annot);
            //std::cout<<" schedules_annotations->push_back "<<std::endl;
            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
                std::cout << "Schedule number "<< schedules_annotations->size() << std::endl;
                std::cout << "Evaluation : " << child->evaluation << std::endl;
                std::cout << "Number of measurements : " << measurements.size() << std::endl;
                std::cout << "===================================" << std::endl << std::endl;
            }

            if (std::isinf(child->evaluation))
                std::cerr<< "Evaluation of schedule "<< schedules_annotations->size() <<" failed "<< std::endl;

            if (child->evaluation < best_evaluation)
            {
                best_evaluation = child->evaluation;
                best_ast = child;
            }
             
            to_be_explored.push_back(child);
            
            ++iterator;  
            
        }
    }
    syntax_tree *ast_copy = ast.copy_ast();
    //ast_copy->nb_explored_optims = nb_explored_optims;
                    
    to_be_explored.push_back(ast_copy);
    
                    
    parent_trace->add_child_path(ast_copy, parent_trace->get_candidate_id());
    


    
    // Sort children from smallest evaluation to largest
    std::sort(to_be_explored.begin(), to_be_explored.end(), [](syntax_tree *a, syntax_tree *b) {
       return a->evaluation < b->evaluation;
    });

    // shuffle the children so that they are selected a random
    //std::shuffle(std::begin(to_be_explored), std::end(to_be_explored), rand_generator);
    
    // keep the top 'beam_size' children and delete the rest
    for (int i = beam_size; i < to_be_explored.size(); ++i)
       delete to_be_explored[i];
    
    
    

    to_be_explored.resize(std::min(beam_size, (int)to_be_explored.size()));
   
    int nb_comps= ast.get_computations().size();
    //std::cout<<"max levels is:"<<MAX_MAT_DEPTH * nb_comps<<std::endl;
    for (syntax_tree *child : to_be_explored)
    {
        // increment the search depth for the recursive call
        child->search_depth = ast.search_depth + 1;
        // if we are under the maximum depth of matrices to explore then call search_save_matrix recursivly
        if (child->search_depth<MAX_MAT_DEPTH * nb_comps && child->search_depth<=child->nb_explored_matrices){
            //std::cout<<"search saving matrix"<<std::endl;
            
            search_save_matrix(*child, schedules_annotations, parent_trace->child_mappings[child], schedule_timeout);
        }else{
            //std::cout<<"search saving"<<std::endl;
            child->initialize_search_space_optimizations(DEFAULT_OPTIMIZATIONS_ORDER);
            // if we surpassed the MAX_MAT_DEPTH amount of matrices to explore OR we detected the parent of this level through
            // the child->search_depth<=child->nb_explored_matrices condition which means that the search level is greater than the number of applied matrices
            search_save(*child, schedules_annotations, parent_trace->child_mappings[child], schedule_timeout);  
        }
    }
}
//void beam_search::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
//{
//    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
//        ast.clear_new_optimizations();
//
//    std::vector<syntax_tree*> children;
//
//    // Look for an optimization that can be applied
//    int nb_optims_tried = 0;
//    int nb_explored_optims = ast.nb_explored_optims;
//
//    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
//    {
//        optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[nb_explored_optims % NB_OPTIMIZATIONS];
//        children = scheds_gen->generate_schedules(ast, optim_type);
//
//        nb_explored_optims++;
//        nb_optims_tried++;
//    }
//
//    // Stop if no more optimizations can be applied
//    if (children.size() == 0)
//        return ;
//
//    // Evaluate children and sort them from smallest to highest evaluation
//    // evaluate while removing illegal versions
//    auto iterator = children.begin();
//    while (iterator != children.end())
//    {
//        syntax_tree *child = *iterator;
//        child->nb_explored_optims = nb_explored_optims;
//        child->transform_ast();
//
//        if (child->schedule_is_prunable()){
//            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
//                // print deleted Ast
//                child->print_previous_optims();
//                std::cout << "\n-----------" << std::endl;
//                child->print_new_optims();
//                child->print_ast();
//                std::cout << "\n<Schedule pruned>\n";
//            }
//            delete child;
//            iterator = children.erase(iterator);
//        }
//
//        else if (!child->ast_is_legal()) {
//            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
//                // print deleted Ast
//                child->print_previous_optims();
//                std::cout << "\n-----------" << std::endl;
//                child->print_new_optims();
//                child->print_ast();
//                child->print_isl_states();
//                std::cout << "\n<illegal>\n";
//            }
//            delete child;
//            iterator = children.erase(iterator);
//        }
//        else {
//
//            // print and evaluate Ast
//
//            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
//                child->print_previous_optims();
//                std::cout << "\n-----------" << std::endl;
//                child->print_new_optims();
//                child->print_ast();
//                child->print_isl_states();
//                std::cout << "\n<legal>\n";
//                child->print_computations_accesses();
//            }
//
//            std::vector<float> measurements;
//            if (child->can_set_default_evaluation()) { // if yes the child's evaluation is set to a default value
//                measurements = {child->evaluation};
//            }
//            else{
//                measurements = exec_eval->get_measurements(*child, false, schedule_timeout);
//                child->evaluation = min_eval(measurements);
//            }
//
//            parent_trace->add_child_path(child, schedules_annotations->size());
//
//            std::string schedule_annot = evaluate_by_learning_model::get_schedule_json(*child);
//
//            //remove the last two characters }\n
//            schedule_annot.pop_back();
//            schedule_annot.pop_back();
//
//            if (std::isfinite(child->evaluation)) // the evaluation is not finite mean that the schedule didn't run
//                schedule_annot += ", \n\"execution_times\" : " + measurements_to_str(measurements) + "\n}\n";
//            else
//                schedule_annot += ", \n\"execution_times\" : null\n}\n";
//
//            schedules_annotations->push_back(schedule_annot);
//
//            if (std::atoi(read_env_var("AS_VERBOSE"))==1){
//                std::cout << "Schedule number "<< schedules_annotations->size() << std::endl;
//                std::cout << "Evaluation : " << child->evaluation << std::endl;
//                std::cout << "Number of measurements : " << measurements.size() << std::endl;
//                std::cout << "===================================" << std::endl << std::endl;
//            }
//
//            if (std::isinf(child->evaluation))
//                std::cerr<< "Evaluation of schedule "<< schedules_annotations->size() <<" failed "<< std::endl;
//
//            if (child->evaluation < best_evaluation)
//            {
//                best_evaluation = child->evaluation;
//                best_ast = child;
//            }
//
//            ++iterator;
//
//        }
//
//        nb_explored_schedules++;
//    }
//
//    // Stop if we reached the maximum depth
//    if (nb_explored_optims >= max_depth)
//        return ;
//
//    // Add the current AST to the list of children
//    syntax_tree *ast_copy = ast.copy_ast();
//    ast_copy->nb_explored_optims = nb_explored_optims;
//    children.push_back(ast_copy);
//    parent_trace->add_child_path(ast_copy, parent_trace->get_candidate_id()); // keeps the same id since it's just copy
//
//    // Sort children from smallest evaluation to largest
//    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
//        return a->evaluation < b->evaluation;
//    });
//
//    // keep the top 'beam_size' children and delete the rest
//    for (int i = beam_size; i < children.size(); ++i)
//        delete children[i];
//
//    children.resize(std::min(beam_size, (int)children.size()));
//
//    // Search recursively on the best children
//    for (syntax_tree *child : children)
//    {
//        child->search_depth = ast.search_depth + 1;
//        search_save(*child, schedules_annotations, parent_trace->child_mappings[child], schedule_timeout);
//    }
//}

//void mcts::search(syntax_tree& ast)
//{
//    std::default_random_engine rand_generator;
//
//    std::vector<syntax_tree*> samples;
//    std::vector<syntax_tree*> children;
//    std::vector<double> children_evals;
//
//    for (int epoch = 0; epoch < nb_samples; ++epoch)
//    {
//        // Starting from the initial ast, generate optimizations until reaching max_depth
//        syntax_tree *ast_sample = &ast;
//        for (int depth = 0; depth < max_depth; ++depth)
//        {
//            optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[depth % NB_OPTIMIZATIONS];
//            children = scheds_gen->generate_schedules(*ast_sample, optim_type);
//
//            if (children.empty())
//                continue;
//
//            children_evals.clear();
//
//            for (syntax_tree *child : children)
//            {
//                child->transform_ast();
//
//                child->evaluation = eval_func->evaluate(*child);
//                children_evals.push_back(child->evaluation);
//
//                nb_explored_schedules++;
//            }
//
//            // Add the current AST to the list of children
//            children.push_back(ast_sample->copy_ast());
//            children_evals.push_back(ast_sample->evaluation);
//
//            // Sample an AST
//            std::discrete_distribution<int> dist(children_evals.begin(), children_evals.end());
//            ast_sample = children[dist(rand_generator)];
//
//            samples.push_back(ast_sample);
//        }
//    }
//
//    if (samples.empty())
//        return ;
//
//    // Sort schedules with respect to evaluations
//    std::sort(samples.begin(), samples.end(), [](syntax_tree *a, syntax_tree *b) {
//        return a->evaluation < b->evaluation;
//    });
//
//    // Execute top-k schedules and return the best
//    for (int i = 0; i < topk; ++i)
//    {
//        float exec_time = exec_eval->evaluate(*samples[i]);
//        if (exec_time < best_evaluation)
//        {
//            best_evaluation = exec_time;
//            best_ast = samples[i];
//        }
//    }
//}
//
//void mcts::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
//{
//    std::cerr<< "mcts::search_save not yet implemented" << std::endl;
//    exit(1);
//}
//
//// -------------------------------------------------------------------------- //
//
//void beam_search_topk::search(syntax_tree& ast)
//{
//    // Do a beam search
//    beam_search_subroutine(ast);
//
//    // Sort schedules found
//    std::sort(schedules.begin(), schedules.end(), [](syntax_tree *a, syntax_tree *b) {
//        return a->evaluation < b->evaluation;
//    });
//
//    // Execute top-k schedules to find the best
//    for (int i = 0; i < topk; ++i)
//    {
//        float exec_time = exec_eval->evaluate(*schedules[i]);
//        if (exec_time < best_evaluation)
//        {
//            best_evaluation = exec_time;
//            best_ast = schedules[i];
//        }
//    }
//}
//
//void beam_search_topk::search_save(syntax_tree& ast, std::vector<std::string> *schedules_annotations, candidate_trace *parent_trace, float schedule_timeout)
//{
//    std::cerr<< "beam_search_topk::search_save not yet implemented" << std::endl;
//    exit(1);
//}
//
//void beam_search_topk::beam_search_subroutine(syntax_tree& ast)
//{
//    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
//        ast.clear_new_optimizations();
//
//    std::vector<syntax_tree*> children;
//
//    // Look for an optimization that can be applied
//    int nb_optims_tried = 0;
//    int nb_explored_optims = ast.nb_explored_optims;
//
//    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
//    {
//        optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[nb_explored_optims % NB_OPTIMIZATIONS];
//        children = scheds_gen->generate_schedules(ast, optim_type);
//
//        nb_explored_optims++;
//        nb_optims_tried++;
//    }
//
//    // Stop if no more optimizations can be applied
//    if (children.size() == 0)
//        return ;
//
//    // Evaluate children and sort them from smallest to highest evaluation
//    for (syntax_tree *child : children)
//    {
//        child->nb_explored_optims = nb_explored_optims;
//        child->transform_ast();
//
//        child->evaluation = eval_func->evaluate(*child);
//
//        nb_explored_schedules++;
//    }
//
//    // Add the current AST to the list of children
//    syntax_tree *ast_copy = ast.copy_ast();
//    ast_copy->nb_explored_optims = nb_explored_optims;
//    children.push_back(ast_copy);
//
//    // Sort children from smallest evaluation to largest
//    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
//        return a->evaluation < b->evaluation;
//    });
//
//    for (int i = 0; i < beam_size; ++i)
//        schedules.push_back(children[i]);
//
//    // Stop if we reached the maximum depth
//    if (nb_explored_optims >= max_depth)
//        return ;
//
//    // Search recursively on the best children
//    for (int i = beam_size; i < children.size(); ++i)
//        delete children[i];
//
//    children.resize(std::min(beam_size, (int)children.size()));
//
//    for (syntax_tree *child : children)
//    {
//        child->search_depth = ast.search_depth + 1;
//        search(*child);
//    }
//}
//
//void beam_search_accuracy_evaluator::search(syntax_tree& ast)
//{
//    if (ast.nb_explored_optims % NB_OPTIMIZATIONS == 0)
//        ast.clear_new_optimizations();
//
//    std::vector<syntax_tree*> children;
//
//    // Look for an optimization that can be applied
//    int nb_optims_tried = 0;
//    int nb_explored_optims = ast.nb_explored_optims;
//
//    while (children.size() == 0 && nb_optims_tried < NB_OPTIMIZATIONS && nb_explored_optims < max_depth)
//    {
//        optimization_type optim_type = DEFAULT_OPTIMIZATIONS_ORDER[nb_explored_optims % NB_OPTIMIZATIONS];
//        children = scheds_gen->generate_schedules(ast, optim_type);
//
//        nb_explored_optims++;
//        nb_optims_tried++;
//    }
//
//    // Stop if no more optimizations can be applied
//    if (children.size() == 0)
//        return ;
//
//    // Evaluate children and sort them from smallest to highest evaluation
//    for (syntax_tree *child : children)
//    {
//        child->nb_explored_optims = nb_explored_optims;
//        child->transform_ast();
//
//        child->evaluation = eval_func->evaluate(*child);
//
//        // We evaluate both by the model and by execution
//        model_evals_list.push_back(child->evaluation);
//        exec_evals_list.push_back(exec_eval->evaluate(*child));
//
//        if (child->evaluation < best_evaluation)
//        {
//            best_evaluation = child->evaluation;
//            best_ast = child;
//        }
//
//        nb_explored_schedules++;
//    }
//
//    // Stop if we reached the maximum depth
//    if (nb_explored_optims >= max_depth)
//        return ;
//
//    // Add the current AST to the list of children
//    syntax_tree *ast_copy = ast.copy_ast();
//    ast_copy->nb_explored_optims = nb_explored_optims;
//    children.push_back(ast_copy);
//
//    // Sort children from smallest evaluation to largest
//    std::sort(children.begin(), children.end(), [](syntax_tree *a, syntax_tree *b) {
//        return a->evaluation < b->evaluation;
//    });
//
//    // Search recursively on the best children
//    for (int i = beam_size; i < children.size(); ++i)
//        delete children[i];
//
//    children.resize(std::min(beam_size, (int)children.size()));
//
//    for (syntax_tree *child : children)
//    {
//        child->search_depth = ast.search_depth + 1;
//        search(*child);
//    }
//}

}
