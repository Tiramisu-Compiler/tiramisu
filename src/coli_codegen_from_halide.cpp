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
    void error() {
        coli::error("Can't convert to coli expr.", true);
    }

public:
    coli::expr expr;
    map<string, coli::computation> computation_list;
    coli::function *fct_ptr;

    HalideToColi() {}

    ~HalideToColi() { fct_ptr = nullptr; }

    coli::expr mutate(Expr e) {
        assert(e.defined() && "HalideToColi can't convert undefined expr\n");
        return expr;
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
    coli::expr value = mutate(op->value);
    coli::computation t("{" + op->name + "[0]}", &value, true,
                        halide_type_to_coli_type(op->value.type()),
                        fct_ptr);
    //TODO(psuriana): not sure???
    /*coli::buffer scalar_t("scalar_t", 1, {coli::expr(1)}, ....)
    t.set_access("{t[0] -> scalar_t[0]}")*/

    computation_list.emplace(op->name, t);
    coli::expr body = mutate(op->body);
    expr = body;
}

void HalideToColi::visit(const LetStmt *op) {
    error();
}

void HalideToColi::visit(const For *op) {
    //TODO(psuriana)
    error();
    //computation f("{f[x,y]: 0<x<N and 0<y<M}", coli::idx("x") + coli::idx("y"), ....);
}

void HalideToColi::visit(const ProducerConsumer *op) {
    error();
}

void HalideToColi::visit(const Load *op) {
    error();
}

void HalideToColi::visit(const Store *op) {
    error();
}

void HalideToColi::visit(const Call *op) {
    //TODO(psuriana): handle call to extern functions, e.g. sin, cos, etc.
    const auto &iter = computation_list.find(op->name);

    if (iter != computation_list.end()) {
        vector<coli::expr> args(op->args.size());
        for (size_t i = 0; i < op->args.size(); ++i) {
            args[i] = mutate(op->args[i]);
        }
        //expr = iter->second(args);
    } else {
        coli::error("Call to " + op->name + " is undefined." , true);
    }
}

void HalideToColi::visit(const Block *op) {
    error();
}

void HalideToColi::visit(const Allocate *op) {
    error();
    //computation c_input("[N]->{c_input[i,j]: 0<=i<N and 0<=j<N}", NULL, false, p_uint8, &blurxy);
}

} // anonymous namespace

}
