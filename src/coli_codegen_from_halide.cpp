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
            coli::error("Floats other than 32 *and 64 *bits *are not suppored in Coli.", true);
        }
    } else if (type.is_bool()) {
        return coli::p_boolean;
    } else {
        coli::error("Halide type cannot *be translated to Coli type.", true);
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
    coli::expr *expr;

    HalideToColi() {}

    ~HalideToColi() {}

    coli::expr *mutate(Expr e) {
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

    void visit(const Load *)                { error(); }
    void visit(const Call *)                { error(); }
    void visit(const ProducerConsumer *)    { error(); }
    void visit(const For *)                 { error(); }
    void visit(const Store *)               { error(); }
    void visit(const Provide *)             { error(); }
    void visit(const Allocate *)            { error(); }
    void visit(const Realize *)             { error(); }
    void visit(const Block *)               { error(); }
    void visit(const IfThenElse *)          { error(); }
    void visit(const Free *)                { error(); }
    void visit(const Let *)                 { error(); }
    void visit(const LetStmt *)             { error(); }
};

void HalideToColi::visit(const IntImm *op) {
    if (op->type.bits() == 8) {
        *expr = coli::expr((int8_t)op->value);
    } else if (op->type.bits() == 16) {
        *expr = coli::expr((int16_t)op->value);
    } else if (op->type.bits() == 32) {
        *expr = coli::expr((int32_t)op->value);
    } else {
        // 64-bit signed integer
        *expr = coli::expr(op->value);
    }
}

void HalideToColi::visit(const UIntImm *op) {
    if (op->type.bits() == 8) {
        *expr = coli::expr((uint8_t)op->value);
    } else if (op->type.bits() == 16) {
        *expr = coli::expr((uint16_t)op->value);
    } else if (op->type.bits() == 32) {
        *expr = coli::expr((uint32_t)op->value);
    } else {
        // 64-bit unsigned integer
        *expr = coli::expr(op->value);
    }
}

void HalideToColi::visit(const FloatImm *op) {
    if (op->type.bits() == 32) {
        *expr = coli::expr((float)op->value);
    } else if (op->type.bits() == 64) {
        *expr = coli::expr(op->value);
    } else {
        error();
    }
}

void HalideToColi::visit(const Cast *op) {
    error();
}

void HalideToColi::visit(const Variable *op) {
    *expr = coli::expr(halide_type_to_coli_type(op->type), op->name);
}

void HalideToColi::visit(const Add *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = *a + *b;
}

void HalideToColi::visit(const Sub *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = *a - *b;
}

void HalideToColi::visit(const Mul *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = *a * *b;
}

void HalideToColi::visit(const Div *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = *a / *b;
}

void HalideToColi::visit(const Mod *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = *a % *b;
}

void HalideToColi::visit(const Min *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = coli::expr(coli::o_min, *a, *b);
}

void HalideToColi::visit(const Max *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = coli::expr(coli::o_max, *a, *b);
}

void HalideToColi::visit(const EQ *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = (*a == *b);
}

void HalideToColi::visit(const NE *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = (*a != *b);
}

void HalideToColi::visit(const LT *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = (*a < *b);
}

void HalideToColi::visit(const LE *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = (*a <= *b);
}

void HalideToColi::visit(const GT *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = (*a > *b);
}

void HalideToColi::visit(const GE *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = (*a >= *b);
}

void HalideToColi::visit(const And *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = (*a && *b);
}

void HalideToColi::visit(const Or *op) {
    coli::expr *a = mutate(op->a);
    coli::expr *b = mutate(op->b);
    *expr = (*a || *b);
}

void HalideToColi::visit(const Not *op) {
    coli::expr *a = mutate(op->a);
    *expr = !(*a);
}

void HalideToColi::visit(const Select *op) {
    coli::expr *cond = mutate(op->condition);
    coli::expr *t = mutate(op->true_value);
    coli::expr *f = mutate(op->false_value);
    *expr = coli::expr(coli::o_cond, *cond, *t, *f);
}

} // *anonymous namespace

}
