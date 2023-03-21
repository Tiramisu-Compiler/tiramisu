#include <algorithm>
#include <iostream>
#include <sstream>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <tiramisu/type.h>
#include <tiramisu/expr.h>
#include <Halide.h>

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::map;
using std::set;
using std::vector;

namespace tiramisu
{

tiramisu::primitive_t halide_type_to_tiramisu_type(Type type)
{
    if (type.is_uint())
    {
        if (type.bits() == 8)
        {
            return tiramisu::p_uint8;
        }
        else if (type.bits() == 16)
        {
            return tiramisu::p_uint16;
        }
        else if (type.bits() == 32)
        {
            return tiramisu::p_uint32;
        }
        else
        {
            return tiramisu::p_uint64;
        }
    }
    else if (type.is_int())
    {
        if (type.bits() == 8)
        {
            return tiramisu::p_int8;
        }
        else if (type.bits() == 16)
        {
            return tiramisu::p_int16;
        }
        else if (type.bits() == 32)
        {
            return tiramisu::p_int32;
        }
        else
        {
            return tiramisu::p_int64;
        }
    }
    else if (type.is_float())
    {
        if (type.bits() == 32)
        {
            return tiramisu::p_float32;
        }
        else if (type.bits() == 64)
        {
            return tiramisu::p_float64;
        }
        else
        {
            ERROR("Floats other than 32 and 64 bits are not suppored in Tiramisu.", true);
        }
    }
    else if (type.is_bool())
    {
        return tiramisu::p_boolean;
    }
    else
    {
        ERROR("Halide type cannot be translated to Tiramisu type.", true);
    }
    return tiramisu::p_none;
}

namespace
{

std::string to_string(const std::vector<Expr> &v)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < v.size(); ++i)
    {
        stream << v[i];
        if (i != v.size() - 1)
        {
            stream << ", ";
        }
    }
    stream << "]";
    return stream.str();
}

class HalideToTiramisu : public IRVisitor
{
private:
    const vector<Function> &outputs;
    const map<string, Function> &env;
    const map<string, tiramisu::buffer *> &output_buffers;
    tiramisu::function *func; // Represent one Halide pipeline
    Scope<Expr> &scope; // Scope of the variables

    map<string, tiramisu::buffer *> temporary_buffers;

    void error() const
    {
        ERROR("Can't convert to tiramisu expr.", true);
    }

    void push_loop_dim(const For *op)
    {
        loop_dims.push_back({op->name, op->min, op->extent});
    }

    void pop_loop_dim()
    {
        loop_dims.pop_back();
    }

    string get_loop_bound_vars() const
    {
        std::ostringstream stream;
        stream << "[";
        for (size_t i = 0; i < loop_dims.size(); ++i)
        {
            stream << loop_dims[i].min << ", " << loop_dims[i].extent;
            if (i != loop_dims.size() - 1)
            {
                stream << ", ";
            }
        }
        stream << "]";
        return stream.str();
    }

    string get_loop_bounds() const
    {
        std::ostringstream stream;
        stream << "(";
        for (size_t i = 0; i < loop_dims.size(); ++i)
        {
            stream << loop_dims[i].to_string();
            if (i != loop_dims.size() - 1)
            {
                stream << ") and (";
            }
        }
        stream << ")";
        return stream.str();
    }

    void define_constant(const string &name, Expr value);

public:
    tiramisu::expr expr;
    map<string, tiramisu::computation *> computation_list;
    map<string, tiramisu::constant *> constant_list;

    struct Loop
    {
        std::string name;
        Expr min, extent;

        string to_string() const
        {
            std::ostringstream stream;
            Expr max = simplify(min + extent - 1);
            stream << min << " <= " << name << " <= " << max;
            return stream.str();
        }
    };

    vector<Loop> loop_dims;

    HalideToTiramisu(Scope<Expr> &s,
                     const vector<Function> &outputs,
                     const map<string, Function> &env,
                     const map<string, tiramisu::buffer *> &output_buffers,
                     tiramisu::function *f)
        : outputs(outputs), env(env), output_buffers(output_buffers), func(f), scope(s) {}

    tiramisu::expr mutate(Expr e)
    {
        assert(e.defined() && "HalideToTiramisu can't convert undefined expr\n");
        // For now, substitute in all lets to make life easier (does not substitute in lets in stmt though)
        e = substitute_in_all_lets(e);
        e.accept(this);
        return expr;
    }

    void mutate(Stmt s)
    {
        assert(s.defined() && "HalideToTiramisu can't convert undefined stmt\n");
        // For now, substitute in all lets to make life easier (does not substitute in lets in stmt though)
        s = substitute_in_all_lets(s);
        s.accept(this);
    }

protected:
    void visit(const IntImm *);
    void visit(const UIntImm *);
    void visit(const FloatImm *);
    void visit(const Cast *);
    void visit(const Variable *);
    void visit(const Add *);
    void visit(const Sub *);
    void visit(const Mul *);
    void visit(const Div *);
    void visit(const Mod *);
    void visit(const Min *);
    void visit(const Max *);
    void visit(const EQ *);
    void visit(const NE *);
    void visit(const LT *);
    void visit(const LE *);
    void visit(const GT *);
    void visit(const GE *);
    void visit(const And *);
    void visit(const Or *);
    void visit(const Not *);
    void visit(const Select *);

    void visit(const StringImm *)   { error(); }
    void visit(const AssertStmt *)  { error(); }
    void visit(const Ramp *)        { error(); }
    void visit(const Broadcast *)   { error(); }
    void visit(const IfThenElse *)  { error(); }
    void visit(const Free *)        { error(); }
    //void visit(const Shuffle *);    { error(); }
    //void visit(const Prefetch *);   { error(); }

    void visit(const Store *)       { error(); } // Should pass the unflatten version to Tiramisu
    void visit(const Allocate *)    { error(); } // Should pass the unflatten version to Tiramisu

    void visit(const Evaluate *);
    void visit(const Load *);
    void visit(const Let *);
    void visit(const LetStmt *);
    void visit(const For *);
    void visit(const Call *);
    void visit(const ProducerConsumer *);
    void visit(const Block *);

    void visit(const Provide *);
    void visit(const Realize *);
};

void HalideToTiramisu::visit(const IntImm *op)
{
    if (op->type.bits() == 8)
    {
        expr = tiramisu::expr((int8_t)op->value);
    }
    else if (op->type.bits() == 16)
    {
        expr = tiramisu::expr((int16_t)op->value);
    }
    else if (op->type.bits() == 32)
    {
        expr = tiramisu::expr((int32_t)op->value);
    }
    else
    {
        // 64-bit signed integer
        expr = tiramisu::expr(op->value);
    }
}

void HalideToTiramisu::visit(const UIntImm *op)
{
    if (op->type.bits() == 8)
    {
        expr = tiramisu::expr((uint8_t)op->value);
    }
    else if (op->type.bits() == 16)
    {
        expr = tiramisu::expr((uint16_t)op->value);
    }
    else if (op->type.bits() == 32)
    {
        expr = tiramisu::expr((uint32_t)op->value);
    }
    else
    {
        // 64-bit unsigned integer
        expr = tiramisu::expr(op->value);
    }
}

void HalideToTiramisu::visit(const FloatImm *op)
{
    if (op->type.bits() == 32)
    {
        expr = tiramisu::expr((float)op->value);
    }
    else if (op->type.bits() == 64)
    {
        expr = tiramisu::expr(op->value);
    }
    else
    {
        // Only support 32- and 64-bit integer
        error();
    }
}

void HalideToTiramisu::visit(const Cast *op)
{
    error();
}

void HalideToTiramisu::visit(const Variable *op)
{
    assert(!op->param.defined() && "Can only handle simple variable for now.\n");
    assert(!op->image.defined() && "Can only handle simple variable for now.\n");

    const auto &iter = constant_list.find(op->name);
    if (iter != constant_list.end())
    {
        // It is a reference to variable defined in Let/LetStmt
        // TODO(psuriana): when do we actually generate constant???
        expr = (*iter->second)(0);
    }
    else
    {
        // It is presumably a reference to loop variable
        expr = tiramisu::var(op->name);
    }
}

void HalideToTiramisu::visit(const Add *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = a + b;
}

void HalideToTiramisu::visit(const Sub *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = a - b;
}

void HalideToTiramisu::visit(const Mul *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = a * b;
}

void HalideToTiramisu::visit(const Div *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = a / b;
}

void HalideToTiramisu::visit(const Mod *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = a % b;
}

void HalideToTiramisu::visit(const Min *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = tiramisu::expr(tiramisu::o_min, a, b);
}

void HalideToTiramisu::visit(const Max *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = tiramisu::expr(tiramisu::o_max, a, b);
}

void HalideToTiramisu::visit(const EQ *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = (a == b);
}

void HalideToTiramisu::visit(const NE *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = (a != b);
}

void HalideToTiramisu::visit(const LT *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = (a < b);
}

void HalideToTiramisu::visit(const LE *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = (a <= b);
}

void HalideToTiramisu::visit(const GT *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = (a > b);
}

void HalideToTiramisu::visit(const GE *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = (a >= b);
}

void HalideToTiramisu::visit(const And *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = (a && b);
}

void HalideToTiramisu::visit(const Or *op)
{
    tiramisu::expr a = mutate(op->a);
    tiramisu::expr b = mutate(op->b);
    expr = (a || b);
}

void HalideToTiramisu::visit(const Not *op)
{
    tiramisu::expr a = mutate(op->a);
    expr = !a;
}

void HalideToTiramisu::visit(const Select *op)
{
    tiramisu::expr cond = mutate(op->condition);
    tiramisu::expr t = mutate(op->true_value);
    tiramisu::expr f = mutate(op->false_value);
    expr = tiramisu::expr(tiramisu::o_select, cond, t, f);
}

void HalideToTiramisu::visit(const Let *op)
{
    error(); // Should not have encountered this since we called substitute_in_all_lets before mutating
}

void HalideToTiramisu::visit(const LetStmt *op)
{
    scope.push(op->name, op->value);
    mutate(op->body);
    scope.pop(op->name);
}

void HalideToTiramisu::visit(const ProducerConsumer *op)
{
    assert((op->body.as<Block>() == NULL) && "Does not currently handle update.\n");
    assert((!op->is_producer || (computation_list.find(op->name) == computation_list.end())) &&
           "Found another computation with the same name.\n");

    vector<Loop> old_loop_dims = loop_dims;
    mutate(op->body);
    loop_dims = old_loop_dims;
}

void HalideToTiramisu::define_constant(const string &name, Expr val)
{
    assert((constant_list.find(name) == constant_list.end()) &&
           "Redefinition of lets is not supported right now.\n");

    val = simplify(val);
    tiramisu::expr value = mutate(val);
    tiramisu::constant *c_const = new tiramisu::constant(name, value,
            halide_type_to_tiramisu_type(val.type()), true, NULL, 0, func);
    constant_list.emplace(name, c_const);
}

void HalideToTiramisu::visit(const For *op)
{
    push_loop_dim(op);

    const Variable *min = op->min.as<Variable>();
    assert((min != NULL) && "Min value of a loop should have been a variable.\n");
    const Variable *extent = op->extent.as<Variable>();
    assert((extent != NULL) && "Extent of a loop should have been a variable.\n");

    Expr min_val = scope.get(min->name);
    Expr extent_val = scope.get(extent->name);

    // Substitute it in all references to some other variables in the min/extent val
    map<string, Expr> replacements;
    typename Scope<Expr>::const_iterator iter;
    for (iter = scope.cbegin(); iter != scope.cend(); ++iter)
    {
        if ((iter.name() != min->name) || (iter.name() != extent->name))
        {
            replacements.emplace(iter.name(), iter.value());
        }
    }

    // Do it twice, to make sure we substitute in all variables properly
    min_val = substitute(replacements, min_val);
    min_val = substitute(replacements, min_val);

    extent_val = substitute(replacements, extent_val);
    extent_val = substitute(replacements, extent_val);

    define_constant(min->name, min_val);
    define_constant(extent->name, extent_val);

    mutate(op->body);
    pop_loop_dim();
}

void HalideToTiramisu::visit(const Evaluate *op)
{
    IRVisitor::visit(op);
}

void HalideToTiramisu::visit(const Load *op)
{
    error(); // Load to external buffer is not currently supported
}

void HalideToTiramisu::visit(const Provide *op)
{
    assert((computation_list.find(op->name) == computation_list.end())
           && "Duplicate computation is not currently supported.\n");
    assert((temporary_buffers.count("buff_" + op->name) || output_buffers.count("buff_" + op->name))
           && "The buffer should have been allocated previously.\n");

    for (size_t i = 0; i < op->args.size(); ++i)
    {
        assert((op->args[i].as<Variable>() != NULL)
               && "Expect args of provide to be loop dims for now (doesn't currently handle update).\n");
    }

    assert((op->values.size() == 1) && "Expect 1D store (no tuple) in the Provide node for now.\n");
    vector<tiramisu::expr> values(op->values.size());
    for (size_t i = 0; i < op->values.size(); ++i)
    {
        values[i] = mutate(op->values[i]);
    }

    string dims_str = to_string(op->args);
    string iter_space_str = get_loop_bound_vars() + "->{" + op->name + dims_str + ": " +
                            get_loop_bounds() + "}";
    tiramisu::computation *compute = new tiramisu::computation(
        iter_space_str, values[0], true, halide_type_to_tiramisu_type(op->values[0].type()), func);

    // 1-to-1 mapping to buffer
    string access_str = "{" + op->name + dims_str + "->" + "buff_" + op->name + dims_str + "}";
    compute->set_access(access_str);

    computation_list.emplace(op->name, compute);
}

void HalideToTiramisu::visit(const Realize *op)
{
    // We will ignore the condition on the Realize node for now.

    assert((temporary_buffers.find("buff_" + op->name) == temporary_buffers.end())
           && "Duplicate allocation (i.e. duplicate compute) is not currently supported.\n");

    const auto iter = env.find(op->name);
    assert((iter != env.end()) && "Cannot find function in env.\n");
    bool is_output = false;
    for (Function o : outputs)
    {
        is_output |= o.same_as(iter->second);
    }
    assert(!is_output && "Realize should have been temporary buffer.\n");

    // Assert that the types of all buffer dimensions are the same for now.
    for (size_t i = 1; i < op->types.size(); ++i)
    {
        assert((op->types[i - 1] == op->types[i]) &&
               "Realize node should have the same types for all dimensions for now.\n");
    }

    // Assert that the bounds on the dimensions start from 0 for now.
    for (size_t i = 0; i < op->bounds.size(); ++i)
    {
      //I am not sure this is a valid replacement for is_zero
        assert(is_const_zero(op->bounds[i].min) && "Bound of realize node should start from 0 for now.\n");
    }

    // Create a temporary buffer
    vector<tiramisu::expr> extents(op->bounds.size());
    for (size_t i = 0; i < op->bounds.size(); ++i)
    {
        extents[i] = mutate(op->bounds[i].extent);
    }

    string buffer_name = "buff_" + op->name;
    tiramisu::buffer *produce_buffer = new tiramisu::buffer(
        buffer_name, extents,
        halide_type_to_tiramisu_type(op->types[0]), a_temporary, func);
    temporary_buffers.emplace(buffer_name, produce_buffer);

    mutate(op->body);
}

void HalideToTiramisu::visit(const Call *op)
{
    assert((op->call_type == Call::CallType::Halide) && "Only handle call to halide func for now.\n");

    const auto iter = computation_list.find(op->name);
    assert(iter != computation_list.end() && "Call to computation that does not exist.\n");

    vector<tiramisu::expr> args(op->args.size());
    for (size_t i = 0; i < op->args.size(); ++i)
    {
        args[i] = mutate(op->args[i]);
    }
    expr = (*iter->second)(args);
}

void HalideToTiramisu::visit(const Block *op)
{
    mutate(op->first);
    mutate(op->rest);
}

} // anonymous namespace

tiramisu::HalideCodegenOutput halide_pipeline_to_tiramisu_function(
    Stmt s, const vector<Function> &outputs, const map<string, Function> &env,
    const map<string, vector<int32_t>> &output_buffers_size,
    tiramisu::function *func)
{

    map<string, tiramisu::buffer *> output_buffers;
    Scope<Expr> scope;

    // Allocate the output buffers
    for (const Function &f : outputs)
    {
        const auto iter = output_buffers_size.find(f.name());
        assert(iter != output_buffers_size.end());

        vector<tiramisu::expr> sizes(iter->second.size());
        for (size_t i = 0; i < iter->second.size(); ++i)
        {
            sizes[i] = tiramisu::expr(iter->second[i]);
            scope.push(f.name() + "_min_" + std::to_string(i), make_const(Int(32), 0));
            scope.push(f.name() + "_extent_" + std::to_string(i), make_const(Int(32), iter->second[i]));
        }
        assert(sizes.size() == f.args().size());

        string buffer_name = "buff_" + f.name();
        // TODO(psuriana): should make the buffer data type variable instead of uint8_t always
        tiramisu::buffer *output_buffer = new tiramisu::buffer(
            buffer_name, sizes, p_uint8, a_output, func);
        output_buffers.emplace(buffer_name, output_buffer);
    }

    HalideToTiramisu converter(scope, outputs, env, output_buffers, func);
    converter.mutate(s);
    return tiramisu::HalideCodegenOutput(std::move(converter.computation_list),
                                         std::move(converter.constant_list),
                                         std::move(output_buffers));
}

}
