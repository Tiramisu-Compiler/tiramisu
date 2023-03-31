#include "PyExpr.h"

namespace tiramisu
{
    namespace PythonBindings
    {

        template <typename other_t, typename PythonClass>
        void add_binary_operators_with(PythonClass &class_instance)
        {
            using self_t = typename PythonClass::type;
            // If 'other_t' is double, we want to wrap it as an Expr() prior to calling the binary op
            // (so that double literals that lose precision when converted to float issue warnings).
            // For any other type, we just want to leave it as-is.
            // using Promote = typename std::conditional<
            //     std::is_same<other_t, double>::value,
            //     DoubleToExprCheck,
            //     other_t>::type;

#define BINARY_OP(op, method)                                                          \
    do                                                                                 \
    {                                                                                  \
        class_instance.def(                                                            \
            "__" #method "__",                                                         \
            [](const self_t &self, const other_t &other) -> decltype(self op(other)) { \
                auto result = self op(other);                                          \
                return result;                                                         \
            },                                                                         \
            py::is_operator(), py::keep_alive<0, 1>(), py::keep_alive<0, 2>());        \
        class_instance.def(                                                            \
            "__r" #method "__",                                                        \
            [](const self_t &self, const other_t &other) -> decltype((other)op self) { \
                auto result = (other)op self;                                          \
                return result;                                                         \
            },                                                                         \
            py::is_operator(), py::keep_alive<0, 1>(), py::keep_alive<0, 2>());        \
    } while (0)

            BINARY_OP(+, add);
            BINARY_OP(-, sub);
            BINARY_OP(*, mul);
            BINARY_OP(/, div); // TODO: verify only needed for python 2.x (harmless for Python 3.x)
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

        } // namespace PythonBindings
        template <typename PythonClass>
        void add_binary_operators(PythonClass &class_instance)
        {
            using self_t = typename PythonClass::type;

            // The order of definitions matters.
            // Python first will try input value as int, then double, then self_t
            // (note that we skip 'float' because we should never encounter that in python;
            // all floating-point literals should be double)
            add_binary_operators_with<self_t>(class_instance);
            // add_binary_operators_with<tiramisu::expr>(class_instance);
            // add_binary_operators_with<double>(class_instance);
            //     add_binary_operators_with<int>(class_instance);

            // Halide::pow() has only an Expr, Expr variant
            // const auto pow_wrap = [](const tiramisu::expr &self, const tiramisu::expr &other) -> decltype(tiramisu::pow(self, other)) {
            //     return tiramisu::pow(self, other);
            // };
            // class_instance
            //     .def("__pow__", pow_wrap, py::is_operator())
            //     .def("__rpow__", pow_wrap, py::is_operator());

            const auto logical_not_wrap = [](const self_t &self) -> decltype(!self)
            {
                return !self;
            };

            // Define unary operators
            class_instance
                .def(-py::self, py::keep_alive<0, 1>()) // neg
                .def("logical_not", logical_not_wrap, py::keep_alive<0, 1>());
        }

        void define_expr(py::module &m)
        {
            auto expr_class = py::class_<expr>(m, "expr").def(py::init<>(), py::return_value_policy::reference).def(py::init<primitive_t>(), py::keep_alive<0, 1>())
                                  // for implicitly_convertible
                                  .def(py::init<int>(), py::keep_alive<0, 1>())
                                  .def(py::init<double>(), py::keep_alive<0, 1>())
                                  // constant convert
                                  .def(py::init([](tiramisu::constant &c) -> tiramisu::expr
                                                { return (tiramisu::expr)c; }),
                                       py::keep_alive<0, 1>())
                                  .def(py::init([](tiramisu::op_t o, tiramisu::primitive_t dtype, tiramisu::expr expr0) -> tiramisu::expr
                                                { return expr(o, dtype, expr0); }),
                                       py::keep_alive<0, 3>())
                                  .def(py::init([](tiramisu::op_t o, tiramisu::expr expr0) -> tiramisu::expr
                                                { return expr(o, expr0); }),
                                       py::keep_alive<0, 2>())
                                  .def(py::init([](tiramisu::op_t o, std::string name) -> tiramisu::expr
                                                { return expr(o, name); }))
                                  .def(py::init([](tiramisu::op_t o, tiramisu::expr expr0, tiramisu::expr expr1) -> tiramisu::expr
                                                { return expr(o, expr0, expr1); }),
                                       py::keep_alive<0, 2>(), py::keep_alive<0, 3>())
                                  .def(py::init([](tiramisu::op_t o, tiramisu::expr expr0, tiramisu::expr expr1, tiramisu::expr expr2) -> tiramisu::expr
                                                { return expr(o, expr0, expr1, expr2); }),
                                       py::keep_alive<0, 2>(), py::keep_alive<0, 3>(), py::keep_alive<0, 4>())
                                  .def(py::init([](tiramisu::op_t o, std::string name, std::vector<tiramisu::expr> vec, tiramisu::primitive_t type) -> tiramisu::expr
                                                { return expr(o, name, vec, type); }),
                                       py::keep_alive<0, 2>())
                                  .def(
                                      "dump", [](const tiramisu::expr &e) -> auto{ return e.dump(true); })
                                  .def(
                                      "get_name", [](tiramisu::expr &e) -> std::string
                                      { return e.get_name(); },
                                      py::return_value_policy::reference)
                                  .def("set_name", [](tiramisu::expr &e, std::string &name) -> void
                                       { return e.set_name(name); })
                                  .def("is_equal", [](tiramisu::expr &e, tiramisu::expr &ep) -> bool
                                       { return e.is_equal(ep); })
                                  .def("__repr__", [](tiramisu::expr &e) -> std::string
                                       { return e.to_str(); })
                                  .def(
                                      "cast", [](tiramisu::expr &e, tiramisu::primitive_t tT) -> tiramisu::expr
                                      { return cast(tT, e); },
                                      py::keep_alive<0, 1>());
            add_binary_operators(expr_class);

            auto memcpy_value = m.def("memcpy", py::overload_cast<const tiramisu::buffer &, const tiramisu::buffer &>(&tiramisu::memcpy));
            auto allocate_value = m.def("allocate", py::overload_cast<const tiramisu::buffer &>(&tiramisu::allocate));
            auto cuda_stream_synchronize_value = m.def("cuda_stream_synchronize", py::overload_cast<>(&tiramisu::cuda_stream_synchronize));

            auto sync_class = py::class_<tiramisu::sync, expr>(m, "sync").def(py::init<>());

            // Integral only operations and value cast...

            // various var acesses

            // #define IMPC(ty)
            //       py::implicitly_convertible<ty, tiramisu::expr>();
            //       expr_class.def("get_" + #ty + "_value", [](tiramisu::expr &e) -> auto {return e.get_})
            // #undef

            py::implicitly_convertible<tiramisu::constant, tiramisu::expr>();
            py::implicitly_convertible<uint8_t, tiramisu::expr>();
            expr_class.def(
                "get_uint8_value", [](tiramisu::expr & e) -> auto{ return e.get_uint8_value(); });
            py::implicitly_convertible<int8_t, tiramisu::expr>();
            expr_class.def(
                "get_int8_value", [](tiramisu::expr & e) -> auto{ return e.get_int8_value(); });
            py::implicitly_convertible<uint16_t, tiramisu::expr>();
            expr_class.def(
                "get_uint16_value", [](tiramisu::expr & e) -> auto{ return e.get_uint16_value(); });
            py::implicitly_convertible<int16_t, tiramisu::expr>();
            expr_class.def(
                "get_int16_value", [](tiramisu::expr & e) -> auto{ return e.get_int16_value(); });
            py::implicitly_convertible<uint32_t, tiramisu::expr>();
            expr_class.def(
                "get_uint32_value", [](tiramisu::expr & e) -> auto{ return e.get_uint32_value(); });
            py::implicitly_convertible<int32_t, tiramisu::expr>();
            expr_class.def(
                "get_int32_value", [](tiramisu::expr & e) -> auto{ return e.get_int32_value(); });
            py::implicitly_convertible<uint64_t, tiramisu::expr>();
            expr_class.def(
                "get_uint64_value", [](tiramisu::expr & e) -> auto{ return e.get_uint64_value(); });
            py::implicitly_convertible<int64_t, tiramisu::expr>();
            expr_class.def(
                "get_int64_value", [](tiramisu::expr & e) -> auto{ return e.get_int64_value(); });
            py::implicitly_convertible<float, tiramisu::expr>();
            expr_class.def(
                "get_float32_value", [](tiramisu::expr & e) -> auto{ return e.get_float32_value(); });
            py::implicitly_convertible<double, tiramisu::expr>();
            expr_class.def(
                "get_float64_value", [](tiramisu::expr & e) -> auto{ return e.get_float64_value(); });
            expr_class.def(
                "get_double_val", [](tiramisu::expr & e) -> auto{ return e.get_double_val(); });
            expr_class.def(
                "get_int_val", [](tiramisu::expr & e) -> auto{ return e.get_int_val(); });
        }

    } // namespace PythonBindings
} // namespace tiramisu
