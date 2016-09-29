#include <algorithm>
#include <iostream>

#include <coli/debug.h>
#include <coli/core.h>
#include <coli/type.h>
#include <coli/expr.h>
#include <Halide.h>

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::map;
using std::vector;

namespace coli
{

coli::primitive_t halide_type_to_coli_type(Type type)
{
    if (type.is_uint()) {
        if (type.bits() == 8) {
            return coli::p_uint8;
        } else if (type.bits() == 16) {
            return coli::p_uint16;
        } else if (type.bits() == 32) {
            return coli::p_uint32;
        } else {
            return coli::p_uint64;
        }
    } else if (type.is_int()) {
        if (type.bits() == 8) {
            return coli::p_int8;
        } else if (type.bits() == 16) {
            return coli::p_int16;
        } else if (type.bits() == 32) {
            return coli::p_int32;
        } else {
            return coli::p_int64;
        }
    } else if (type.is_float()) {
        if (type.bits() == 32) {
            return coli::p_float32;
        } else if (type.bits() == 64) {
            return coli::p_float64;
        } else {
            coli::error("Floats other than 32 and 64 bits are not suppored in Coli.", true);
        }
    } else if (type.is_bool()) {
        return coli::p_boolean;
    } else {
        coli::error("Halide type cannot be translated to Coli type.", true);
    }
    return coli::p_none;
}

namespace
{

class HalideToColi : public IRVisitor {
private:
    const vector<Function> &outputs;
    const map<string, Function> &env;

    void error() {
        coli::error("Can't convert to coli expr.", true);
    }

    void push_loop_dim(const For *op) {
        loop_dims.push_back({op->name, op->min, op->extent});
    }

    void pop_loop_dim() {
        loop_dims.pop_back();
    }

public:
    coli::expr expr;
    map<string, coli::computation> computation_list;
    coli::function func;

    struct Loop {
        std::string name;
        Expr min, extent;
    };

    vector<Loop> loop_dims;

    HalideToColi(const vector<Function> &outputs, const map<string, Function> &env)
        : outputs(outputs), env(env), func("func") {}

    coli::expr mutate(Expr e) {
        assert(e.defined() && "HalideToColi can't convert undefined expr\n");
        e.accept(this);
        return expr;
    }

    void mutate(Stmt s) {
        assert(s.defined() && "HalideToColi can't convert undefined stmt\n");
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

    void visit(const StringImm *)           { error(); }
    void visit(const AssertStmt *)          { error(); }
    void visit(const Evaluate *)            { error(); }
    void visit(const Ramp *)                { error(); }
    void visit(const Broadcast *)           { error(); }
    void visit(const IfThenElse *)          { error(); }
    void visit(const Free *)                { error(); }

    void visit(const Let *);
    void visit(const LetStmt *);
    void visit(const For *);
    void visit(const Load *);
    void visit(const Store *);
    void visit(const Call *);
    void visit(const ProducerConsumer *);
    void visit(const Block *);
    void visit(const Allocate *);

    void visit(const Provide *)             { error(); }
    void visit(const Realize *)             { error(); }
};

void HalideToColi::visit(const IntImm *op) {
    if (op->type.bits() == 8) {
        expr = coli::expr((int8_t)op->value);
    } else if (op->type.bits() == 16) {
        expr = coli::expr((int16_t)op->value);
    } else if (op->type.bits() == 32) {
        expr = coli::expr((int32_t)op->value);
    } else {
        // 64-bit signed integer
        expr = coli::expr(op->value);
    }
}

void HalideToColi::visit(const UIntImm *op) {
    if (op->type.bits() == 8) {
        expr = coli::expr((uint8_t)op->value);
    } else if (op->type.bits() == 16) {
        expr = coli::expr((uint16_t)op->value);
    } else if (op->type.bits() == 32) {
        expr = coli::expr((uint32_t)op->value);
    } else {
        // 64-bit unsigned integer
        expr = coli::expr(op->value);
    }
}

void HalideToColi::visit(const FloatImm *op) {
    if (op->type.bits() == 32) {
        expr = coli::expr((float)op->value);
    } else if (op->type.bits() == 64) {
        expr = coli::expr(op->value);
    } else {
        error();
    }
}

void HalideToColi::visit(const Cast *op) {
    error();
}

void HalideToColi::visit(const Variable *op) {
    //TODO(psuriana)
    error();
    const auto &iter = computation_list.find(op->name);
    if (iter != computation_list.end()) {
        // It is a reference to variable defined in Let/LetStmt or a reference
        // to a buffer
        expr = iter->second(0);
    } else {
        // It is presumably a reference to loop variable

    }
}

void HalideToColi::visit(const Add *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = a + b;
}

void HalideToColi::visit(const Sub *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = a - b;
}

void HalideToColi::visit(const Mul *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = a * b;
}

void HalideToColi::visit(const Div *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = a / b;
}

void HalideToColi::visit(const Mod *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = a % b;
}

void HalideToColi::visit(const Min *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = coli::expr(coli::o_min, a, b);
}

void HalideToColi::visit(const Max *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = coli::expr(coli::o_max, a, b);
}

void HalideToColi::visit(const EQ *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = (a == b);
}

void HalideToColi::visit(const NE *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = (a != b);
}

void HalideToColi::visit(const LT *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = (a < b);
}

void HalideToColi::visit(const LE *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = (a <= b);
}

void HalideToColi::visit(const GT *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = (a > b);
}

void HalideToColi::visit(const GE *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = (a >= b);
}

void HalideToColi::visit(const And *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = (a && b);
}

void HalideToColi::visit(const Or *op) {
    coli::expr a = mutate(op->a);
    coli::expr b = mutate(op->b);
    expr = (a || b);
}

void HalideToColi::visit(const Not *op) {
    coli::expr a = mutate(op->a);
    expr = !a;
}

void HalideToColi::visit(const Select *op) {
    coli::expr cond = mutate(op->condition);
    coli::expr t = mutate(op->true_value);
    coli::expr f = mutate(op->false_value);
    expr = coli::expr(coli::o_cond, cond, t, f);
}

void HalideToColi::visit(const Let *op) {
    assert(is_const(op->value) && "Only support let of constant for now.\n");

    coli::expr value = mutate(op->value);
    //TODO(psuriana): potential segfault here since we're passing stack pointer
    coli::constant c_const(op->name, &value, halide_type_to_coli_type(op->value.type()), true, NULL, 0, &func);
    computation_list.emplace(op->name, c_const);

    coli::expr body = mutate(op->body);
    expr = body;
}

void HalideToColi::visit(const LetStmt *op) {
    assert(is_const(op->value) && "Only support let of constant for now.\n");

    coli::expr value = mutate(op->value);
    //TODO(psuriana): potential segfault here since we're passing stack pointer
    coli::constant c_const(op->name, &value, halide_type_to_coli_type(op->value.type()), true, NULL, 0, &func);
    computation_list.emplace(op->name, c_const);

    mutate(op->body);
}

void HalideToColi::visit(const ProducerConsumer *op) {
    assert(!op->update.defined() && "Does not currently handle update.\n");
    assert((computation_list.count(op->name) == 0) && "Find another computation with the same name.\n");

    vector<Loop> old_loop_dims = loop_dims;
    mutate(op->produce);
    loop_dims = old_loop_dims;
}

void HalideToColi::visit(const For *op) {
    push_loop_dim(op);
    mutate(op->body);
    pop_loop_dim();
}

void HalideToColi::visit(const Load *op) {
    //TODO(psuriana): doesn't handle this right now
    error();
}

void HalideToColi::visit(const Store *op) {
    //TODO(psuriana)
    /*string iter_space_str = "[N]->{c_input[i,j]: 0<=i<N and 0<=j<N}";
    coli::computation compute(iter_space_str, NULL, false, halide_type_to_coli_type(op->value.type()), &func);
    // Map to buffer
    compute.set_access("{" + op->name + "[i,j]->" + "b_" + op->name + "[i,j]}");

    computation_list.emplace(op->name, compute);*/
}

void HalideToColi::visit(const Call *op) {
    assert((op->call_type == Call::CallType::Halide) && "Only handle call to halide func for now.\n");
    assert(computation_list.count(op->name) && "Computation does not exist.\n");

    vector<coli::expr> args(op->args.size());
    for (size_t i = 0; i < op->args.size(); ++i) {
        args[i] = mutate(op->args[i]);
    }
    //TODO(psuriana)
    //expr = computation_list[op->name](args[i]);
}

void HalideToColi::visit(const Block *op) {
    mutate(op->first);
    mutate(op->rest);
}

void HalideToColi::visit(const Allocate *op) {
    Function f = env.find(op->name)->second;
    bool is_output = false;
    for (Function o : outputs) {
        is_output |= o.same_as(f);
    }

    if (!is_output) {
        // Create temporary buffer if it's not an output
        vector<coli::expr> extents(op->extents.size());
        for (size_t i = 0; i < op->extents.size(); ++i) {
            extents[i] = mutate(op->extents[i]);
        }

        coli::buffer produce_buffer(
            "b_" + op->name, extents.size(), extents,
            halide_type_to_coli_type(op->type), NULL, a_temporary, &func);
    }
}

} // anonymous namespace

}
