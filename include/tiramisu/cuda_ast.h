//
// Created by malek on 12/18/17.
//

#ifndef TIRAMISU_CUDA_AST_H
#define TIRAMISU_CUDA_AST_H

#ifndef NVCC_PATH
#define NVCC_PATH "nvcc"
#endif

#define UNARY(op, x) {op, op_data_t{true, 1, (x)}}
#define UNARY_TYPED(op, x, T) {op, op_data_t{true, 2, (x), (T)}}
#define BINARY(op, x) {op, op_data_t{true, 2, (x)}}
#define BINARY_TYPED(op, x, T) {op, op_data_t{true, 2, (x), (T)}}
#define TERNARY(op, x, y) {op, op_data_t{true, 3, (x), (y)}}
#define TERNARY_TYPED(op, x, y, T) {op, op_data_t{true, 3, (x), (y), (T)}}
#define FN_CALL(op, x, n) {op, op_data_t{false, (n), (x)}}
#define FN_CALL_TYPED(op, x, n, T) {op, op_data_t{false, (n), (x), (T)}}

#define UNARY2(op, x) case op: return tiramisu::cuda_ast::op_data_t{true, 1, (x)};
#define UNARY_TYPED2(op, x, T) case op: return tiramisu::cuda_ast::op_data_t{true, 2, (x), (T)};
#define BINARY2(op, x) case op: return tiramisu::cuda_ast::op_data_t{true, 2, (x)};
#define BINARY_TYPED2(op, x, T) case op: return tiramisu::cuda_ast::op_data_t{true, 2, (x), (T)};
#define TERNARY2(op, x, y) case op: return tiramisu::cuda_ast::op_data_t{true, 3, (x), (y)};
#define TERNARY_TYPED2(op, x, y, T) case op: return tiramisu::cuda_ast::op_data_t{true, 3, (x), (y), (T)};
#define FN_CALL2(op, x, n) case op: return tiramisu::cuda_ast::op_data_t{false, (n), (x)};
#define FN_CALL_TYPED2(op, x, n, T) case op: return tiramisu::cuda_ast::op_data_t{false, (n), (x), (T)};

#include <isl/id.h>
#include <tiramisu/type.h>
#include <string>
#include <vector>
#include "utils.h"

namespace tiramisu
{
    struct isl_ast_expr_deleter
    {
        void operator()(isl_ast_expr * p) const {isl_ast_expr_free(p);}
    };
    typedef std::unique_ptr<isl_ast_expr, isl_ast_expr_deleter> isl_ast_expr_ptr;

    struct isl_id_deleter
    {
        void operator()(isl_id * p) const {isl_id_free(p);}
    };
    typedef std::unique_ptr<isl_id, isl_id_deleter> isl_id_ptr;

    struct isl_ast_node_list_deleter
    {
        void operator()(isl_ast_node_list * p) const {isl_ast_node_list_free(p);}
    };
    typedef std::unique_ptr<isl_ast_node_list, isl_ast_node_list_deleter> isl_ast_node_list_ptr;

    struct isl_val_deleter
    {
        void operator()(isl_val * p) const {isl_val_free(p);}
    };
    typedef std::unique_ptr<isl_val, isl_val_deleter> isl_val_ptr;

    class function;
namespace cuda_ast
{
    struct op_data_t
    {
        op_data_t() {}
        op_data_t(bool infix, int arity, std::string && symbol) : infix(infix), arity(arity), symbol(symbol) {}
        op_data_t(bool infix, int arity, std::string && symbol, std::string && next_symbol) : infix(infix), arity(arity), symbol(symbol), next_symbol(next_symbol) {}
        op_data_t(bool infix, int arity, std::string && symbol, primitive_t type) : infix(infix), arity(arity), symbol(symbol), type_preserving(
                false), type(type) {}
        op_data_t(bool infix, int arity, std::string && symbol, std::string && next_symbol, primitive_t type) : infix(infix), arity(arity), symbol(symbol), next_symbol(next_symbol), type_preserving(
                false), type(type) {}

        bool operator==(const op_data_t &rhs) const;

        bool operator!=(const op_data_t &rhs) const;

        bool infix;
        int arity;
        std::string symbol;
        std::string next_symbol = "";
        bool type_preserving = true;
        primitive_t type = p_none;
    };

const op_data_t tiramisu_operation_description(tiramisu::op_t op);

//    const std::unordered_map <tiramisu::op_t , op_data_t> tiramisu_operation_description = {
//        UNARY(o_minus, "-"),
//        FN_CALL(o_floor, "floor", 1),
//        FN_CALL(o_sin, "sin", 1),
//        FN_CALL(o_cos, "cos", 1),
//        FN_CALL(o_tan, "tan", 1),
//        FN_CALL(o_asin, "asin", 1),
//        FN_CALL(o_acos, "acos", 1),
//        FN_CALL(o_atan, "atan", 1),
//        FN_CALL(o_sinh, "sinh", 1),
//        FN_CALL(o_cosh, "cosh", 1),
//        FN_CALL(o_tanh, "tanh", 1),
//        FN_CALL(o_asinh, "asinh", 1),
//        FN_CALL(o_acosh, "acosh", 1),
//        FN_CALL(o_atanh, "atanh", 1),
//        FN_CALL(o_abs, "abs", 1),
//        FN_CALL(o_sqrt, "sqrt", 1),
//        FN_CALL(o_expo, "exp", 1),
//        FN_CALL(o_log, "log", 1),
//        FN_CALL(o_ceil, "ceil", 1),
//        FN_CALL(o_round, "round", 1),
//        FN_CALL(o_trunc, "trunc", 1),
//        BINARY(o_add, "+"),
//        BINARY(o_sub, "-"),
//        BINARY(o_mul, "*"),
//        BINARY(o_div, "/"),
//        BINARY(o_mod, "%"),
//        BINARY(o_logical_and, "&&"),
//        BINARY(o_logical_or, "||"),
//        UNARY(o_logical_not, "!"),
//        BINARY(o_eq, "=="),
//        BINARY(o_ne, "!="),
//        BINARY(o_le, "<="),
//        BINARY(o_lt, "<"),
//        BINARY(o_ge, ">="),
//        BINARY(o_gt, ">"),
//        FN_CALL(o_max, "max", 2),
//        FN_CALL(o_min, "min", 2),
//        BINARY(o_right_shift, ">>"),
//        BINARY(o_left_shift, "<<"),
//        TERNARY(o_select, "?", ":"),
//        FN_CALL(o_lerp, "lerp", 3),
//    };

const op_data_t isl_operation_description(isl_ast_op_type op);

//    const std::unordered_map <isl_ast_op_type , op_data_t> isl_operation_description = {
//        BINARY_TYPED(isl_ast_op_and, "&&", p_boolean),
//        BINARY_TYPED(isl_ast_op_and_then, "&&", p_boolean),
//        BINARY_TYPED(isl_ast_op_or, "||", p_boolean),
//        BINARY_TYPED(isl_ast_op_or_else, "||", p_boolean),
//        FN_CALL(isl_ast_op_max, "max", 2),
//        FN_CALL(isl_ast_op_min, "min", 2),
//        UNARY(isl_ast_op_minus, "-"),
//        BINARY(isl_ast_op_add, "+"),
//        BINARY(isl_ast_op_sub, "-"),
//        BINARY(isl_ast_op_mul, "*"),
//        BINARY(isl_ast_op_div, "/"),
//        BINARY(isl_ast_op_fdiv_q, "/"),
//        BINARY(isl_ast_op_pdiv_q, "/"),
//        BINARY(isl_ast_op_pdiv_r, "%"),
//        BINARY(isl_ast_op_zdiv_r, "%"),
//        TERNARY(isl_ast_op_cond, "?", ":"),
//        FN_CALL(isl_ast_op_select, "lerp", 3),
//        BINARY_TYPED(isl_ast_op_eq, "==", p_boolean),
//        BINARY_TYPED(isl_ast_op_le, "<=", p_boolean),
//        BINARY_TYPED(isl_ast_op_lt, "<", p_boolean),
//        BINARY_TYPED(isl_ast_op_ge, ">=", p_boolean),
//        BINARY_TYPED(isl_ast_op_gt, ">", p_boolean),
//    };

    const std::string tiramisu_type_to_cuda_type(tiramisu::primitive_t t);


//    const std::unordered_map <tiramisu::primitive_t, std::string> tiramisu_type_to_cuda_type = {
//            {p_none, "void"},
//            {p_boolean, "bool"},
//            {p_int8, "int8_t"},
//            {p_uint8, "uint8_t"},
//            {p_int16, "int16_t"},
//            {p_uint16, "uint16_t"},
//            {p_int32, "int32_t"},
//            {p_uint32, "uint32_t"},
//            {p_int64, "int64_t"},
//            {p_uint64, "uint64_t"},
//            {p_float32, "float"},
//            {p_float64, "double"},
//    };
enum class memory_location
{
    host,
    global,
    shared,
    local,
    constant,
    reg,
};

class abstract_node {

};

    class statement;
    typedef std::shared_ptr<statement> statement_ptr;



    struct gpu_iterator;
class statement : public abstract_node {
public:
    primitive_t get_type() const;
    std::string print();
    virtual void print_body(std::stringstream &ss, const std::string &base);
    virtual void print(std::stringstream &ss, const std::string &base) = 0;
    virtual std::pair<statement_ptr, statement_ptr> extract_min_cap();
    // TODO implement in more subclasses
    virtual statement_ptr replace_iterators(std::unordered_map<std::string, gpu_iterator> & iterators);
    virtual std::unordered_set<std::string> extract_scalars();

protected:

    explicit statement(primitive_t type);
//    template <std::function<statement_ptr(statement_ptr)> F>
//    virtual statement_ptr apply()
//    {}

private:
    tiramisu::primitive_t type;
};

class cast : public statement {
    statement_ptr to_be_cast;
public:
    cast(primitive_t type, statement_ptr stmt);
    void print(std::stringstream &ss, const std::string &base) override ;
    statement_ptr replace_iterators(std::unordered_map<std::string, gpu_iterator> & iterators) override;
    std::unordered_set<std::string> extract_scalars() override;

};

class block : public statement {
private:
    std::vector<statement_ptr> elements;

public:
    void print(std::stringstream &ss, const std::string &base) override;
    void print_body(std::stringstream &ss, const std::string &base) override ;

public:
    block();

    virtual ~block();

    void add_statement(statement_ptr stmt);

};

class abstract_identifier : public statement
{
protected:
    abstract_identifier(primitive_t type, const std::string &name, memory_location location);

public:
    const std::string &get_name() const;
    memory_location get_location() const;
    virtual void print_declaration(std::stringstream &ss, const std::string &base) = 0;
    virtual bool is_buffer() const;


private:
    std::string name;
    cuda_ast::memory_location location;

public:

};
    typedef std::shared_ptr<abstract_identifier> abstract_identifier_ptr;

class buffer : public abstract_identifier
{
public:
    buffer(primitive_t type, const std::string &name, memory_location location, const std::vector<statement_ptr> &size);
    void print(std::stringstream &ss, const std::string &base) override;
    void print_declaration(std::stringstream &ss, const std::string &base) override;
    void print_size(std::stringstream &ss, const std::string &base, const std::string &seperator);
    bool is_buffer() const override;


private:
    std::vector<statement_ptr> size;
};
    typedef std::shared_ptr<buffer> buffer_ptr;

class scalar : public abstract_identifier
{
    bool is_const;
public:
    scalar(primitive_t type, const std::string &name, memory_location location);
    scalar(primitive_t type, const std::string &name, memory_location location, bool is_const);

public:
    void print(std::stringstream &ss, const std::string &base) override;
    void print_declaration(std::stringstream &ss, const std::string &base) override;
    statement_ptr replace_iterators(std::unordered_map<std::string, gpu_iterator> & iterators) override;
    std::unordered_set<std::string> extract_scalars() override;
};
    typedef std::shared_ptr<scalar> scalar_ptr;

class value;
typedef std::shared_ptr<value> value_ptr;

class value : public statement
{
public:

    explicit value(const tiramisu::expr & expr);
    explicit value(uint8_t val);
    explicit value(int8_t val);
    explicit value(uint16_t val);
    explicit value(int16_t val);
    explicit value(uint32_t val);
    explicit value(int32_t val);
    explicit value(uint64_t val);
    explicit value(int64_t val);
    explicit value(float val);
    explicit value(double val);

    value_ptr copy();

public:
    void print(std::stringstream &ss, const std::string &base) override;
    statement_ptr replace_iterators(std::unordered_map<std::string, gpu_iterator> & iterators) override;

private:

    /**
      * The value.
      */
    union
    {
        uint8_t     u8_val;
        int8_t      i8_val;
        uint16_t    u16_val;
        int16_t     i16_val;
        uint32_t    u32_val;
        int32_t     i32_val;
        uint64_t    u64_val;
        int64_t     i64_val;
        float       f32_val;
        double      f64_val;
    };
};

class assignment : public statement
{
protected:
    explicit assignment(primitive_t type);

public:
    virtual void print_declaration(std::stringstream &ss, const std::string &base);
};
    typedef std::shared_ptr<assignment> assignment_ptr;

class scalar_assignment : public assignment
{
    scalar_ptr m_scalar;
    statement_ptr m_rhs;
public:
    scalar_assignment(scalar_ptr scalar, statement_ptr rhs);
    void print(std::stringstream &ss, const std::string &base) override;
    void print_declaration(std::stringstream &ss, const std::string &base) override;

};

class buffer_assignment : public assignment
{
    cuda_ast::buffer_ptr m_buffer;
    cuda_ast::statement_ptr m_index_access;
    cuda_ast::statement_ptr m_rhs;
public:
    void print(std::stringstream &ss, const std::string &base) override;
public:
    buffer_assignment(cuda_ast::buffer_ptr buffer, statement_ptr index_access, statement_ptr rhs);
};

class function_call : public statement
{
public:
    function_call(primitive_t type, const std::string &name, const std::vector<statement_ptr> &arguments);
public:
    void print(std::stringstream &ss, const std::string &base) override;
    std::pair<statement_ptr, statement_ptr> extract_min_cap() override;
    statement_ptr replace_iterators(std::unordered_map<std::string, gpu_iterator> & iterators) override;
    std::unordered_set<std::string> extract_scalars() override;

private:
    std::string name;
    std::vector<statement_ptr> arguments;
};

class for_loop : public statement
{
public:
    for_loop(statement_ptr initialization, statement_ptr condition, statement_ptr incrementer, statement_ptr body);

public:
    void print(std::stringstream &ss, const std::string &base) override;

private:
    statement_ptr initial_value;
    statement_ptr condition;
    statement_ptr incrementer;
    statement_ptr body;
};

class if_condition : public statement
{
public:
    if_condition(statement_ptr condition, statement_ptr then_body, statement_ptr else_body);
    if_condition(statement_ptr condition, statement_ptr then_body);

public:
    void print(std::stringstream &ss, const std::string &base) override;

private:
    statement_ptr condition;
    statement_ptr then_body;
    bool has_else;
    statement_ptr else_body;
};

class buffer_access : public statement
{
public:
    buffer_access(buffer_ptr accessed, const std::vector<statement_ptr> &access);

public:
    void print(std::stringstream &ss, const std::string &base) override;
    statement_ptr replace_iterators(std::unordered_map<std::string, gpu_iterator> & iterators) override;
    std::unordered_set<std::string> extract_scalars() override;

private:
    buffer_ptr accessed;
    std::vector<cuda_ast::statement_ptr> access;
};

class op : public statement
{

protected:
    op(primitive_t type, const std::vector<statement_ptr> & operands);
    std::vector<statement_ptr> m_operands;
    std::unordered_set<std::string> extract_scalars() override;

};

class unary : public op
{
public:
    unary(primitive_t type, statement_ptr operand, std::string &&op_symbol);
public:
    void print(std::stringstream &ss, const std::string &base) override;
    statement_ptr replace_iterators(std::unordered_map<std::string, gpu_iterator> & iterators) override;

private:
    std::string m_op_symbol;
};

class binary : public op
{
public:
    binary(primitive_t type, statement_ptr operand_1, statement_ptr operand_2, std::string &&op_symbol);

public:
    void print(std::stringstream &ss, const std::string &base) override;
    statement_ptr replace_iterators(std::unordered_map<std::string, gpu_iterator> & iterators) override;
private:
    std::string m_op_symbol;
};

class ternary : public op
{
public:
    ternary(primitive_t type, statement_ptr operand_1, statement_ptr operand_2, statement_ptr operand_3, std::string &&op_symbol_1,  std::string &&op_symbol_2);

public:
    void print(std::stringstream &ss, const std::string &base) override;
    statement_ptr replace_iterators(std::unordered_map<std::string, gpu_iterator> & iterators) override;
private:
    std::string m_op_symbol_1;
    std::string m_op_symbol_2;

};

//class assignment : public statement
//{
//public:
//    assignment(primitive_t type, abstract_identifier *identifier, statement *value);
//
//private:
//    abstract_identifier * identifier;
//    statement * value;
//};

class declaration : public statement
{
public:
    explicit declaration (abstract_identifier_ptr id);
    explicit declaration (assignment_ptr asgmnt);
    void print(std::stringstream &ss, const std::string &base) override;


private:
    bool is_initialized;
    abstract_identifier_ptr id;
    assignment_ptr asgmnt;
};

class sync : public statement
{
public:
    sync();
    void print(std::stringstream &ss, const std::string &base) override;
};

typedef std::unordered_map<std::string, std::pair<tiramisu::primitive_t, cuda_ast::memory_location> > scalar_data_t;

struct gpu_iterator
{
    enum class type_t {
        THREAD,
        BLOCK
    } type;
    enum class dimension_t{
        x = 0,
        y,
        z
    } dimension;
    statement_ptr size;
    // returns a simplified name; __tx__, __ty__, __tz__, __bx__, __by__, __bz__
    std::string simplified_name();
};

class gpu_iterator_read : public statement
{
private:
    gpu_iterator it;
    bool simplified;
public:
    explicit gpu_iterator_read(gpu_iterator it);
    explicit gpu_iterator_read(gpu_iterator it, bool simplified);
    void print(std::stringstream &ss, const std::string &base) override;
};


        class kernel_call;
        class kernel_definition;

class return_statement : public statement
{
private:
    statement_ptr return_value;
public:
    explicit return_statement(statement_ptr return_value);
    void print(std::stringstream &ss, const std::string &base) override;
};
class host_function : public statement
{
public:
    host_function(primitive_t type, std::string name, const std::vector<abstract_identifier_ptr> &arguments, statement_ptr body);
    void print(std::stringstream &ss, const std::string &base) override;
    void set_pointer_return(bool val = true);

private:
    bool pointer_return;
    std::string name;
    statement_ptr body;
    std::vector<abstract_identifier_ptr> arguments;
};

class kernel
{
    friend class kernel_call;
    friend class kernel_definition;
private:
    struct dim3d_t
    {
        statement_ptr x, y, z;
        dim3d_t();

        void set(gpu_iterator::dimension_t dim, statement_ptr size);

    };
    dim3d_t block_dimensions;
    dim3d_t thread_dimensions;
    std::map<std::string, scalar_ptr> used_constants;
    std::map<std::string, buffer_ptr> used_buffers;
    statement_ptr body;
    static int kernel_count;
    int kernel_number;
public:
    kernel();
    void set_dimension(gpu_iterator dimension);
    void set_body(statement_ptr body);
    std::string get_name() const;
    std::string get_wrapper_name() const;
    static constexpr auto wrapper_return_type = p_int32;
    void add_used_scalar(scalar_ptr scalar);
    void add_used_buffer(buffer_ptr buffer);
    std::vector<abstract_identifier_ptr> get_arguments();

};
typedef std::shared_ptr<kernel> kernel_ptr;

class kernel_call : public statement
{
public:
    explicit kernel_call (kernel_ptr kernel);
    void print(std::stringstream &ss, const std::string &base) override;


private:
    kernel_ptr kernel;

};

class kernel_definition : public statement
{
public:
    explicit kernel_definition(kernel_ptr kernel);
    void print(std::stringstream &ss, const std::string &base) override;

private:
    kernel_ptr kernel;
};

class memcpy : public statement
{
public:
    memcpy(buffer_ptr from, buffer_ptr to);
    void print(std::stringstream &ss, const std::string &base) override;
private:
    buffer_ptr from, to;
};


class allocate : public statement
{
public:
    allocate(buffer_ptr b);
    void print(std::stringstream &ss, const std::string &base) override;

private:
    buffer_ptr b;
};

class free : public statement
{
public:
    free(buffer_ptr b);
    void print(std::stringstream &ss, const std::string &base) override;

private:
    buffer_ptr b;
};


class generator
{
    friend class tiramisu::function;
private:
    const tiramisu::function &m_fct;
    scalar_data_t m_scalar_data;
    std::unordered_map<std::string, cuda_ast::buffer_ptr> m_buffers;
    cuda_ast::buffer_ptr get_buffer(const std::string & name);
    cuda_ast::statement_ptr parse_tiramisu(const tiramisu::expr & tiramisu_expr);
    int loop_level = 0;
    kernel_ptr current_kernel;
    std::unordered_map<isl_ast_node*, kernel_ptr> iterator_to_kernel_map;
    std::vector<kernel_ptr> kernels;
    // Will be set to true as soon as GPU computation is encountered, and set to false as soon as GPU loop is exited
    bool in_kernel = false;
    std::vector<std::string> iterator_stack;
    std::vector<cuda_ast::statement_ptr> iterator_upper_bound;
    std::vector<cuda_ast::statement_ptr> iterator_lower_bound;
    std::vector<cuda_ast::statement_ptr> kernel_simplified_vars;
    // A mapping from iterator name to GPU info
    std::unordered_map<std::string, cuda_ast::gpu_iterator> gpu_iterators;
    std::vector<cuda_ast::statement_ptr> gpu_conditions;
    std::unordered_set<std::string> gpu_local;
    cuda_ast::gpu_iterator get_gpu_condition(gpu_iterator::type_t type, gpu_iterator::dimension_t dim,
                                                 cuda_ast::statement_ptr lower_bound,
                                                 cuda_ast::statement_ptr upper_bound);
    statement_ptr get_scalar_from_name(std::string name);
    std::unordered_map<computation *, std::vector<isl_ast_expr*>> index_exprs;
public:
    explicit generator(tiramisu::function &fct);

    statement_ptr cuda_stmt_from_isl_node(isl_ast_node *node);
    statement_ptr cuda_stmt_handle_isl_for(isl_ast_node *node);
    statement_ptr cuda_stmt_val_from_for_condition(isl_ast_expr_ptr &expr, isl_ast_node *node);
    statement_ptr cuda_stmt_handle_isl_block(isl_ast_node *node);
    statement_ptr cuda_stmt_handle_isl_if(isl_ast_node *node);
    statement_ptr cuda_stmt_handle_isl_user(isl_ast_node *node);
    cuda_ast::statement_ptr cuda_stmt_handle_isl_expr(isl_ast_expr_ptr &expr, isl_ast_node *node);
    statement_ptr cuda_stmt_handle_isl_op_expr(isl_ast_expr_ptr &expr, isl_ast_node *node);
    void cuda_stmt_foreach_isl_expr_list(isl_ast_expr *node, const std::function<void(int, isl_ast_expr *)> &fn, int start = 0);


    static cuda_ast::value_ptr cuda_stmt_handle_isl_val(isl_val_ptr &node);
};

namespace {

    struct exec_result {
        bool exec_succeeded;
        int result;
        std::string std_out;
        std::string std_err;

        bool fail();

        bool succeed();
    };

}

    class compiler
    {
        std::string code;

        bool compile_cpu_obj(const std::string &filename, const std::string &obj_name) const;
        bool compile_gpu_obj(const std::string &obj_name) const;
        static exec_result exec(const std::string &cmd);

    public:
        std::string get_cpu_obj(const std::string &obj_name) const;
        std::string get_gpu_obj(const std::string &obj_name) const;
        explicit compiler(const std::string &code);
        bool compile(const std::string &obj_name) const;
    };

}

}

#endif //TIRAMISU_CUDA_AST_H
