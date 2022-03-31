#include "PyExpr.h"


namespace tiramisu {
  namespace PythonBindings {

    template<typename other_t, typename PythonClass>
void add_binary_operators_with(PythonClass &class_instance) {
    using self_t = typename PythonClass::type;
    // If 'other_t' is double, we want to wrap it as an Expr() prior to calling the binary op
    // (so that double literals that lose precision when converted to float issue warnings).
    // For any other type, we just want to leave it as-is.
    // using Promote = typename std::conditional<
    //     std::is_same<other_t, double>::value,
    //     DoubleToExprCheck,
    //     other_t>::type;

#define BINARY_OP(op, method)                                                                  \
    do {                                                                                       \
        class_instance.def(                                                                    \
            "__" #method "__",                                                                 \
            [](const self_t &self, const other_t &other) -> decltype(self op (other)) { \
                auto result = self op (other);                                          \
                return result;                                                                 \
            },                                                                                 \
            py::is_operator());                                                                \
        class_instance.def(                                                                    \
            "__r" #method "__",                                                                \
            [](const self_t &self, const other_t &other) -> decltype((other) op self) { \
                auto result = (other) op self;                                          \
                return result;                                                                 \
            },                                                                                 \
            py::is_operator());                                                                \
    } while (0)

    BINARY_OP(+, add);
    BINARY_OP(-, sub);
    BINARY_OP(*, mul);
    BINARY_OP(/, div);  // TODO: verify only needed for python 2.x (harmless for Python 3.x)
    BINARY_OP(/, truediv);
    BINARY_OP(%, mod);
    BINARY_OP(<<, lshift);
    BINARY_OP(>>, rshift);
    BINARY_OP(<, lt);
    BINARY_OP(<=, le);
    BINARY_OP(==, eq);
    BINARY_OP(!=, ne);
    BINARY_OP(>=, ge);
    BINARY_OP(>, gt);

#undef BINARY_OP

    // const auto floordiv_wrap = [](const self_t &self, const other_t &other) -> decltype(self / Promote(other)) {
    //     static_assert(std::is_same<decltype(self / Promote(other)), Expr>::value, "We expect all operator// overloads to produce Expr");
    //     Expr e = self / Promote(other);
    //     if (e.type().is_float()) {
    //         e = Halide::floor(e);
    //     }
    //     return e;
    // };

    // class_instance
    //     .def("__floordiv__", floordiv_wrap, py::is_operator())
    //     .def("__rfloordiv__", floordiv_wrap, py::is_operator());
    
    }// namespace PythonBindings
    template<typename PythonClass>
void add_binary_operators(PythonClass &class_instance) {
    using self_t = typename PythonClass::type;

    // The order of definitions matters.
    // Python first will try input value as int, then double, then self_t
    // (note that we skip 'float' because we should never encounter that in python;
    // all floating-point literals should be double)
    add_binary_operators_with<self_t>(class_instance);
    //add_binary_operators_with<tiramisu::expr>(class_instance);
    //add_binary_operators_with<double>(class_instance);
    //    add_binary_operators_with<int>(class_instance);

    // Halide::pow() has only an Expr, Expr variant
    // const auto pow_wrap = [](const tiramisu::expr &self, const tiramisu::expr &other) -> decltype(tiramisu::pow(self, other)) {
    //     return tiramisu::pow(self, other);
    // };
    // class_instance
    //     .def("__pow__", pow_wrap, py::is_operator())
    //     .def("__rpow__", pow_wrap, py::is_operator());

    const auto logical_not_wrap = [](const self_t &self) -> decltype(!self) {
        return !self;
    };

    // Define unary operators
    class_instance
        .def(-py::self)  // neg
        .def("logical_not", logical_not_wrap);
}
    
    void define_expr(py::module &m){
      auto expr_class = py::class_<expr>(m, "expr").def(py::init<>())
	      .def(py::init<primitive_t>())
        // for implicitly_convertible
	      .def(py::init<int>())
	      .def(py::init<double>())
        // constant convert
        .def(py::init([](tiramisu::constant &c) -> tiramisu::expr { return (tiramisu::expr) c; }))
        .def("dump", [](const tiramisu::expr &e) -> auto { return e.dump(true); });
	//        .def("__add__", [](tiramisu::expr &l, tiramisu::expr &r) -> auto { return l + r; });

      add_binary_operators(expr_class);
      //operator
      //casts
      //vars
      //buffer
      //cuda syncrnize

      

      py::implicitly_convertible<tiramisu::constant, tiramisu::expr>();
      py::implicitly_convertible<int, tiramisu::expr>();
      py::implicitly_convertible<double, tiramisu::expr>();
    }

  }  // namespace PythonBindings
}  // namespace tiramisu
