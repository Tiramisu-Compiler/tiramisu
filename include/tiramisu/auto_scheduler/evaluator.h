#ifndef _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_
#define _TIRAMISU_AUTO_SCHEDULER_EVALUATOR_

#include "auto_scheduler.h"
#include "utils.h"

namespace tiramisu::auto_scheduler
{

/**
  * An abstract class that represents an evaluation function.
  * Derive this class and implement the method "evaluate" to
  * create new evaluation functions.
  */
class evaluation_function
{
private:
    
protected:
    
public:
    virtual ~evaluation_function() {}
    
    /**
     * Takes as input an abstract syntax tree and returns
     * its evaluation.
     */
    virtual float evaluate(syntax_tree& ast) =0;
    virtual float evaluate(syntax_tree& ast, std::string no_sched_json)=0;
};

/**
 * Evaluate programs by compiling and executing them.
 */
class evaluate_by_execution : public evaluation_function
{
private:

protected:
    /**
     * The following three attributes are parameters for Halide.
     */
    std::vector<Halide::Target::Feature> halide_features = {
        Halide::Target::AVX,
        Halide::Target::SSE41,
        //Halide::Target::AVX2,
	    //Halide::Target::FMA,
        Halide::Target::LargeBuffers
    };
    
    Halide::Target halide_target;
    std::vector<Halide::Argument> halide_arguments;

    
	
	/**
	 * The name of the ".o" to generate (it will contain the compiled Tiramisu program).
	 */
    std::string obj_filename;
    
    /**
     * The command that will be used to execute the wrapper to measure execution time.
     */
    std::string wrapper_cmd;

public:
/**
     * The program to compile and to execute.
     */
	tiramisu::function *fct;
    /**
     * arguments : the input and output buffers of the program.
     */
    evaluate_by_execution(std::vector<tiramisu::buffer*> const& arguments, 
						  std::string const& obj_filename, 
						  std::string const& wrapper_cmd,
						  tiramisu::function *fct = tiramisu::global::get_implicit_function());
    
	/**
	 * Apply the specified optimizations, compile the program and execute it.
	 */
    virtual float evaluate(syntax_tree& ast);
    virtual float evaluate(syntax_tree& ast, std::string no_sched_json);

    /**
     * Apply the specified optimizations, compile the program and execute it.
     * Returns a vector of measured execution times
     * If the timeout parameter is defined, it stops the execution after MAX_RUNS*timeout seconds
     * If exit_on_timeout is set to true, it raises an error when the timeout is reached and terminates the program
     */
    std::vector<float> get_measurements(syntax_tree &ast,  bool exit_on_timeout = false, float timeout = 0);
};

/**
 * This evaluation function uses system pipes to communicate with an ML model
 * that will evaluate schedules.
 * JSON is used to transfer information about the schedule to evaluate.
 */
class evaluate_by_learning_model : public evaluation_function
{
private:

protected:
    /**
     * The pipe on which to write information about schedules to evaluate.
     */
    FILE *model_write;
    
    /**
     * The pipe on which to read the evaluation of a schedule.
     */
    FILE *model_read;

public:
    /**
     * cmd_path : path to the program containing the ML model.
     * cmd_args : arguments to pass to the program in cmd_path.
     */
    evaluate_by_learning_model(std::string const& cmd_path, std::vector<std::string> const& cmd_args);
    
	/**
	 * Call the model and return its evaluation.
	 */
    virtual float evaluate(syntax_tree& ast, std::string no_sched_json);
    /**
	 * Call the model and return its evaluation.
	 */
    virtual float evaluate(syntax_tree& ast);
    /**
     * Return a JSON representation of the program represented by the AST.
     * Uses the function : represent_computations_from_nodes. 
     */
    static std::string get_program_json(syntax_tree const& ast);
    
    /**
     * A recursive subroutine that represents in JSON the computations of a given tree.
     */
    static void represent_computations_from_nodes(ast_node *node, std::string& computations_json, int& comp_absolute_order);

    /**
    * A recursive subroutine that represents in JSON the expression of a computation given its comp_info object.
    */
    static std::string get_expression_json(const tiramisu::auto_scheduler::computation_info& comp_info, const tiramisu::expr& e);

    /**
    * Represents information about buffers used in an ast in JSON format.
    */
    static std::string get_buffers_json(syntax_tree const& ast);
    
    /**
     * Return a JSON representation of the schedule of the given AST.
     */
    static std::string get_schedule_json(syntax_tree & ast);
    
    // --------------------------------------------------------------------------------- //
    
    /**
     * A recursive subroutine that represents in JSON the loop structure of a given tree.
     * This function is not called by this class, but by the AST class.
     * The result of this function is stored in ast.iterators_json.
     * The function get_program_json uses directly ast.iterators_json to get the JSON of iterators.
     *
     * This is done for the following reason : the autoscheduler needs to change the structure
     * of the AST after each optimization. On the other hand, the structure of the original tree
     * must be passed to the model to get the evaluation of a schedule. So we store in
     * ast.iterators_json the JSON corresponding to the initial structure of the tree, in order
     * to retrieve it later, even though the structure of the AST has changed.
     */
    static void represent_iterators_from_nodes(ast_node *node, std::string& iterators_json);
    
    /**
     * Transform the structure of the given AST into JSON.
     * The model needs this information in the schedule.
     *
     * Like represent_iterators_from_nodes, this function is called by the AST class.
     * AST class calls this function every time the optimization UNFUSE is applied.
     * The result is retrieved in ast.tree_structure_json.
     */
    static std::string get_tree_structure_json(syntax_tree const& ast);
    
    /**
     * A recursive subroutine used by get_tree_structure_json(syntax_tree const& ast).
     */
    static std::string get_tree_structure_json(ast_node *node);
};

class simplified_expr_json_exctractor : public Halide::Internal::IRVisitor
{
private:
    /**
     * A map of Halide_Expr_str:tiramisu_expr where Halide_Expr_str represents a string of a halide Load op and
     * tiramisu_expr rpresents a tiramisu access expr.
     */
    std::map<std::string,tiramisu::expr> accesses_map;

    tiramisu::auto_scheduler::computation_info comp_info;

    /**
     * Return an error message when encountering unsupported expressions
     */
    void error() const
    {
     ERROR("Unsupported operation type encountered while exctracting JSON representaion of expression.", true);
    }


public:
    /**
     * A string corresponing to the resulting json representation of the simplified expression
     */
    std::string expression_json;

    /**
     * A constructor converts the tiramisu computation's expression into a Halide Expr, simplifies it,
     * and constructs the expression JSON reprentation out of the simplified Halide Expr
     */
    explicit simplified_expr_json_exctractor(const tiramisu::auto_scheduler::computation_info& comp_info);

    /**
     * Converts a Halide Expr into a string
     */
    static std::string halide_Expr_to_string(const Halide::Expr& e);

    /**
     * Gets a mapping of Halide_Expr_str:tiramisu_expr where Halide_Expr_str represents a string of a halide Load op and
     * tiramisu_expr rpresents a tiramisu access expr.
     * This is used to avoid the non-trivial task of delinearizing the Halide Load op into an access matrix. The access
     * matrix is extracted using the corresponding tiramisu expr instead.
     */
    void get_access_str_mapping(const tiramisu::expr& e);

    /**
     * Recursively build the JSON representation of the of the Halide Expr e
     */
    std::string get_Expr_json(Halide::Expr e);

protected:
     void visit(const Halide::Internal::IntImm *) override;
     void visit(const Halide::Internal::UIntImm *) override;
     void visit(const Halide::Internal::FloatImm *) override;
     void visit(const Halide::Internal::Cast *) override;
     void visit(const Halide::Internal::Variable *) override;
     void visit(const Halide::Internal::Add *) override;
     void visit(const Halide::Internal::Sub *) override;
     void visit(const Halide::Internal::Mul *) override;
     void visit(const Halide::Internal::Div *) override;
     void visit(const Halide::Internal::Mod *) override;
     void visit(const Halide::Internal::Min *) override;
     void visit(const Halide::Internal::Max *) override;
     void visit(const Halide::Internal::EQ *) override;
     void visit(const Halide::Internal::NE *) override;
     void visit(const Halide::Internal::LT *) override;
     void visit(const Halide::Internal::LE *) override;
     void visit(const Halide::Internal::GT *) override;
     void visit(const Halide::Internal::GE *) override;
     void visit(const Halide::Internal::And *) override;
     void visit(const Halide::Internal::Or *) override;
     void visit(const Halide::Internal::Not *) override;
     void visit(const Halide::Internal::Select *) override;
     void visit(const Halide::Internal::StringImm *) override;
     void visit(const Halide::Internal::AssertStmt *) override;
     void visit(const Halide::Internal::Ramp *) override;
     void visit(const Halide::Internal::Broadcast *) override;
     void visit(const Halide::Internal::IfThenElse *) override;
     void visit(const Halide::Internal::Free *) override;
     void visit(const Halide::Internal::Store *) override;
     void visit(const Halide::Internal::Allocate *) override;
     void visit(const Halide::Internal::Evaluate *) override;
     void visit(const Halide::Internal::Load *) override;
     void visit(const Halide::Internal::Let *) override;
     void visit(const Halide::Internal::LetStmt *) override;
     void visit(const Halide::Internal::For *) override;
     void visit(const Halide::Internal::Call *) override;
     void visit(const Halide::Internal::ProducerConsumer *) override;
     void visit(const Halide::Internal::Block *) override;
     void visit(const Halide::Internal::Provide *) override;
     void visit(const Halide::Internal::Realize *) override;
};


}

#endif
