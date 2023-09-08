#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_atax_SMALL_wrapper.h"


using namespace tiramisu;

int main(int argc, char **argv) {
        tiramisu::init("function_atax_SMALL");

        // -------------------------------------------------------
        // Layer I
        // -------------------------------------------------------

        // Iteration variables
        var i("i", 0, 116), j("j", 0, 124);

        // inputs
        input A("A", {i, j}, p_float64);
        input x("x", {j}, p_float64);

        // Computations
        computation Ax_init("Ax_init", {i}, 0.0);
        computation Ax("Ax", {i, j}, p_float64);
        Ax.set_expression(Ax(i, j) + A(i, j) * x(j));
        computation y_init("y_init", {j}, 0.0);
        computation y("y", {i, j}, p_float64);
        y.set_expression(y(i, j) + A(i, j) * Ax(i, 0));

        // -------------------------------------------------------
        // Layer II
        // -------------------------------------------------------
        y_init.then(Ax_init, computation::root)
         .then(Ax, i)
         .then(y, i);

        // -------------------------------------------------------
        // Layer III
        // -------------------------------------------------------
        // Input Buffers
        buffer b_A("b_A", {116, 124}, p_float64, a_input);
        buffer b_Ax("b_Ax", {116}, p_float64, a_temporary);
        buffer b_x("b_x", {124}, p_float64, a_input);
        buffer b_y("b_y", {124}, p_float64, a_output);

        // Store inputs
        A.store_in(&b_A);
        x.store_in(&b_x);

        // Store computations
        Ax_init.store_in(&b_Ax);
        Ax.store_in(&b_Ax, {i});
        y_init.store_in(&b_y);
        y.store_in(&b_y, {j});

        // -------------------------------------------------------
        // Code Generation
        // -------------------------------------------------------
        tiramisu::codegen({&b_A, &b_x, &b_y}, "function_atax_SMALL.o");

      return 0;
}
