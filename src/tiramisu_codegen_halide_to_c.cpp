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

string halide_type_to_tiramisu_type_str(Type type)
{
    if (type.is_uint()) {
        if (type.bits() == 8) {
            return "tiramisu::p_uint8";
        } else if (type.bits() == 16) {
            return "tiramisu::p_uint16";
        } else if (type.bits() == 32) {
            return "tiramisu::p_uint32";
        } else {
            return "tiramisu::p_uint64";
        }
    } else if (type.is_int()) {
        if (type.bits() == 8) {
            return "tiramisu::p_int8";
        } else if (type.bits() == 16) {
            return "tiramisu::p_int16";
        } else if (type.bits() == 32) {
            return "tiramisu::p_int32";
        } else {
            return "tiramisu::p_int64";
        }
    } else if (type.is_float()) {
        if (type.bits() == 32) {
            return "tiramisu::p_float32";
        } else if (type.bits() == 64) {
            return "tiramisu::p_float64";
        } else {
            tiramisu::error("Floats other than 32 and 64 bits are not suppored in Coli.", true);
        }
    } else if (type.is_bool()) {
        return "tiramisu::p_boolean";
    } else {
        tiramisu::error("Halide type cannot be translated to Coli type.", true);
    }
    return "tiramisu::p_none";
}

namespace
{

std::string to_string(const std::vector<Expr>& v) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        ss << v[i];
        if (i != v.size() - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

class HalideToC : public IRVisitor {
private:
    const vector<Function> &outputs;
    const map<string, Function> &env;
    const set<string> &output_buffers;
    string func; // Represent one Halide pipeline
    Scope<Expr> &scope; // Scope of the variables

    set<string> temporary_buffers;

    void error() const {
        tiramisu::error("Can't convert to tiramisu expr.", true);
    }

    void push_loop_dim(const For *op) {
        loop_dims.push_back({op->name, op->min, op->extent});
    }

    void pop_loop_dim() {
        loop_dims.pop_back();
    }

    string get_loop_bound_vars() const {
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < loop_dims.size(); ++i) {
            ss << loop_dims[i].min << ", " << loop_dims[i].extent;
            if (i != loop_dims.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }

    string get_loop_bounds() const {
        std::ostringstream ss;
        ss << "(";
        for (size_t i = 0; i < loop_dims.size(); ++i) {
            ss << loop_dims[i].to_string();
            if (i != loop_dims.size() - 1) {
                ss << ") and (";
            }
        }
        ss << ")";
        return ss.str();
    }

    void define_constant(const string &name, Expr value);

public:
    set<string> computation_list;
    set<string> constant_list;

    struct Loop {
        std::string name;
        Expr min, extent;

        string to_string() const {
            std::ostringstream ss;
            Expr max = simplify(min + extent - 1);
            ss << min << " <= " << name << " <= " << max;
            return ss.str();
        }
    };

    vector<Loop> loop_dims;

    HalideToC(Scope<Expr> &s,
              const vector<Function> &outputs,
              const map<string, Function> &env,
              const set<string> &output_buffers,
              const string &f,
              std::ostream &ss)
            : outputs(outputs), env(env), output_buffers(output_buffers), func(f), scope(s), stream(ss), indent(0) {
        ss.setf(std::ios::fixed, std::ios::floatfield);
    }

    void print(Expr e) {
        assert(e.defined() && "HalideToC can't convert undefined expr\n");
        // For now, substitute in all lets to make life easier (does not substitute in lets in stmt though)
        e = substitute_in_all_lets(e);
        e.accept(this);
    }

    void print(Stmt s) {
        assert(s.defined() && "HalideToC can't convert undefined stmt\n");
        // For now, substitute in all lets to make life easier (does not substitute in lets in stmt though)
        s = substitute_in_all_lets(s);
        s.accept(this);
    }

protected:
    /** The stream we're outputting on */
    std::ostream &stream;

    /** The current indentation level, useful for pretty-printing
     * statements */
    int indent;

    /** Emit spaces according to the current indentation level */
    void do_indent();

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

    void visit(const Store *)       { error(); } // Should pass the unflatten version to COLi
    void visit(const Allocate *)    { error(); } // Should pass the unflatten version to COLi

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

void HalideToC::do_indent() {
    for (int i = 0; i < indent; i++) stream << ' ';
}

void HalideToC::visit(const IntImm *op) {
    stream << "tiramisu::expr(";
    if (op->type.bits() == 8) {
        stream << "(int8_t)";
    } else if (op->type.bits() == 16) {
        stream << "(int16_t)";
    } else if (op->type.bits() == 32) {
        stream << "(int32_t)";
    }
    stream << op->value << ")";
}

void HalideToC::visit(const UIntImm *op) {
    stream << "tiramisu::expr(";
    if (op->type.bits() == 8) {
        stream << "(uint8_t)";
    } else if (op->type.bits() == 16) {
        stream << "(uint16_t)";
    } else if (op->type.bits() == 32) {
        stream << "(uint32_t)";
    }
    stream << op->value << ")";
}

void HalideToC::visit(const FloatImm *op) {
    if (op->type.bits() == 32) {
        stream << "tiramisu::expr((float)op->value);";
    } else if (op->type.bits() == 64) {
        stream << "tiramisu::expr(op->value);";
    } else {
        // Only support 32- and 64-bit integer
        error();
    }
}

void HalideToC::visit(const Cast *op) {
    error();
}

void HalideToC::visit(const Variable *op) {
    assert(!op->param.defined() && "Can only handle simple variable for now.\n");
    assert(!op->image.defined() && "Can only handle simple variable for now.\n");

    const auto &iter = constant_list.find(op->name);
    if (iter != constant_list.end()) {
        // It is a reference to variable defined in Let/LetStmt
        //TODO(psuriana): when do we actually generate constant???
        stream << (*iter) << "(0)";
    } else {
        // It is presumably a reference to loop variable
        stream << "tiramisu::idx(" << op->name << ")";
    }
}

void HalideToC::visit(const Add *op) {
    stream << '(';
    print(op->a);
    stream << " + ";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const Sub *op) {
    stream << '(';
    print(op->a);
    stream << " - ";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const Mul *op) {
    stream << '(';
    print(op->a);
    stream << "*";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const Div *op) {
    stream << '(';
    print(op->a);
    stream << "/";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const Mod *op) {
    stream << '(';
    print(op->a);
    stream << " % ";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const Min *op) {
    stream << "tiramisu::expr(tiramisu::o_min, ";
    print(op->a);
    stream << ", ";
    print(op->b);
    stream << ")";
}

void HalideToC::visit(const Max *op) {
    stream << "tiramisu::expr(tiramisu::o_max, ";
    print(op->a);
    stream << ", ";
    print(op->b);
    stream << ")";
}

void HalideToC::visit(const EQ *op) {
    stream << '(';
    print(op->a);
    stream << " == ";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const NE *op) {
    stream << '(';
    print(op->a);
    stream << " != ";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const LT *op) {
    stream << '(';
    print(op->a);
    stream << " < ";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const LE *op) {
    stream << '(';
    print(op->a);
    stream << " <= ";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const GT *op) {
    stream << '(';
    print(op->a);
    stream << " > ";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const GE *op) {
    stream << '(';
    print(op->a);
    stream << " >= ";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const And *op) {
    stream << '(';
    print(op->a);
    stream << " && ";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const Or *op) {
    stream << '(';
    print(op->a);
    stream << " || ";
    print(op->b);
    stream << ')';
}

void HalideToC::visit(const Not *op) {
    stream << '!';
    print(op->a);
}

void HalideToC::visit(const Select *op) {
    stream << "tiramisu::expr(tiramisu::o_cond, ";
    print(op->condition);
    stream << ", ";
    print(op->true_value);
    stream << ", ";
    print(op->false_value);
    stream << ")";
}

void HalideToC::visit(const Let *op) {
    error(); // Should not have encountered this since we called substitute_in_all_lets before mutating
}

void HalideToC::visit(const LetStmt *op) {
    scope.push(op->name, op->value);
    print(op->body);
    scope.pop(op->name);
}

void HalideToC::visit(const ProducerConsumer *op) {
    assert((op->body.as<Block>() == NULL) && "Does not currently handle update.\n");
    assert((!op->is_producer || (computation_list.find(op->name) == computation_list.end())) &&
           "Found another computation with the same name.\n");

    vector<Loop> old_loop_dims = loop_dims;
    print(op->body);
    loop_dims = old_loop_dims;
}

void HalideToC::define_constant(const string &name, Expr val) {
    assert((constant_list.find(name) == constant_list.end()) && "Redefinition of lets is not supported right now.\n");

    val = simplify(val);

    do_indent();

    stream << "tiramisu::constant " << name << "(\"" << name << "\", ";
    print(val);
    stream << ", " << halide_type_to_tiramisu_type_str(val.type())
           << ", true, NULL, 0, &" << func << ");\n";

    constant_list.insert(name);
}

void HalideToC::visit(const For *op) {
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
    for (iter = scope.cbegin(); iter != scope.cend(); ++iter) {
        if ((iter.name() != min->name) || (iter.name() != extent->name)) {
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

    print(op->body);
    pop_loop_dim();
}

void HalideToC::visit(const Evaluate *op) {
    //TODO(psuriana): do nothing for now
}

void HalideToC::visit(const Load *op) {
    error(); // Load to external buffer is not currently supported
}

void HalideToC::visit(const Provide *op) {
    assert((computation_list.find(op->name) == computation_list.end())
           && "Duplicate computation is not currently supported.\n");
    assert((temporary_buffers.count("buff_" + op->name) || output_buffers.count("buff_" + op->name))
           && "The buffer should have been allocated previously.\n");

    for (size_t i = 0; i < op->args.size(); ++i) {
        assert((op->args[i].as<Variable>() != NULL)
               && "Expect args of provide to be loop dims for now (doesn't currently handle update).\n");
    }
    assert((op->values.size() == 1) && "Expect 1D store (no tuple) in the Provide node for now.\n");

    do_indent();

    string dims_str = to_string(op->args);
    string iter_space_str = get_loop_bound_vars() + "->{" + op->name + dims_str + ": " + get_loop_bounds() + "}";

    stream << "tiramisu::computation " << op->name << "(\"" << iter_space_str << "\", ";
    print(op->values[0]);
    stream << ", true, " << halide_type_to_tiramisu_type_str(op->values[0].type())
           << ", &" << func << ");\n";

    // 1-to-1 mapping to buffer
    string access_str = "{" + op->name + dims_str + "->" + "buff_" + op->name + dims_str + "}";
    stream << op->name << ".set_access(\"" << access_str << "\");\n";

    computation_list.insert(op->name);
}

void HalideToC::visit(const Realize *op) {
    // We will ignore the condition on the Realize node for now.

    assert((temporary_buffers.find("buff_" + op->name) == temporary_buffers.end())
           && "Duplicate allocation (i.e. duplicate compute) is not currently supported.\n");

    const auto iter = env.find(op->name);
    assert((iter != env.end()) && "Cannot find function in env.\n");
    bool is_output = false;
    for (Function o : outputs) {
        is_output |= o.same_as(iter->second);
    }
    assert(!is_output && "Realize should have been temporary buffer.\n");

    // Assert that the types of all buffer dimensions are the same for now.
    for (size_t i = 1; i < op->types.size(); ++i) {
        assert((op->types[i-1] == op->types[i]) && "Realize node should have the same types for all dimensions for now.\n");
    }

    // Assert that the bounds on the dimensions start from 0 for now.
    for (size_t i = 0; i < op->bounds.size(); ++i) {
        assert(is_zero(op->bounds[i].min) && "Bound of realize node should start from 0 for now.\n");
    }

    do_indent();

    // Create a temporary buffer

    string buffer_name = "buff_" + op->name;
    stream << "tiramisu::buffer " << buffer_name << "(\"" << buffer_name << "\", "
           << op->bounds.size() << ", ";

    stream << "{";
    for (size_t i = 0; i < op->bounds.size(); ++i) {
        print(op->bounds[i].extent);
        if (i != op->bounds.size() - 1) {
            stream << ", ";
        }
    }
    stream << "}, ";

    stream << halide_type_to_tiramisu_type_str(op->types[0]) << ", NULL, tiramisu::a_temporary, "
           << "&" << func << ");\n";

    temporary_buffers.insert(buffer_name);

    print(op->body);
}

void HalideToC::visit(const Call *op) {
    assert((op->call_type == Call::CallType::Halide) && "Only handle call to halide func for now.\n");

    const auto iter = computation_list.find(op->name);
    assert(iter != computation_list.end() && "Call to computation that does not exist.\n");

    stream << (*iter) << "(";
    for (size_t i = 0; i < op->args.size(); i++) {
        print(op->args[i]);
        if (i < op->args.size() - 1) {
            stream << ", ";
        }
    }
    stream << ")";
}

void HalideToC::visit(const Block *op) {
    print(op->first);
    if (op->rest.defined()) print(op->rest);
}

} // anonymous namespace

void halide_pipeline_to_c(
        Stmt s, const vector<Function> &outputs, const map<string, Function> &env,
        const map<string, vector<int32_t>> &output_buffers_size,
        const string &func) {

    std::ostringstream stream;

    stream << "tiramisu::function " << func << "(\"" << func << "\")" << ";\n";

    set<string> output_buffers;
    Scope<Expr> scope;

    std::ostringstream output_buffers_stream;
    output_buffers_stream << "{";
    // Allocate the output buffers
    for (size_t k = 0; k < outputs.size(); ++k) {
        const Function &f = outputs[k];
        const auto iter = output_buffers_size.find(f.name());
        assert(iter != output_buffers_size.end());
        assert(iter->second.size() == f.args().size());

        std::ostringstream sizes;
        sizes << "{";
        for (size_t i = 0; i < iter->second.size(); ++i) {
            sizes << "tiramisu::expr(" << iter->second[i] << ")";
            scope.push(f.name() + "_min_" + std::to_string(i), make_const(Int(32), 0));
            scope.push(f.name() + "_extent_" + std::to_string(i), make_const(Int(32), iter->second[i]));
            if (i != iter->second.size() - 1) {
                sizes << ", ";
            }
        }
        sizes << "}";

        string buffer_name = "buff_" + f.name();
        //TODO(psuriana): should make the buffer data type variable instead of uint8_t always
        stream << "tiramisu::buffer " << buffer_name << "(\"" << buffer_name << "\", "
               << f.args().size() << ", " << sizes.str() << ", tiramisu::p_uint8, NULL, tiramisu::a_output, "
               << "&" << func << ");\n";
        output_buffers.insert(buffer_name);

        output_buffers_stream << "&" << buffer_name;
        if (k != outputs.size() - 1) {
            output_buffers_stream << ", ";
        }
    }
    output_buffers_stream << "}";

    HalideToC converter(scope, outputs, env, output_buffers, func, stream);
    converter.print(s);

    stream << func << ".set_arguments(" << output_buffers_stream.str() << ");\n";
    stream << func << ".gen_isl_ast();\n";
    stream << func << ".gen_halide_stmt();\n";
    stream << func << ".dump_halide_stmt();\n";
    stream << func << ".gen_halide_obj(\"build/generated_fct_test_06.o\");\n";

    std::cout << "\nCOLi C output:\n\n" << stream.str() << "\n";
}

}
