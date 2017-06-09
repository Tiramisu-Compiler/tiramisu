#include "Halide.h"

#include <tiramisu/utils.h>
#include <tiramisu/core.h>

#include <isl/ctx.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/constraint.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <cstdlib>
#include <iostream>

using namespace tiramisu;

/**
 * Test file for low level functions in tiramisu.
 */

#define SET_1D_0   "[N]->{S0[i]: 0 <= i <= N}"
#define SET_1D_1   "[N]->{S0[i]: 0 <= i <  N}"
#define SET_1D_2   "[N]->{S0[i]: 0 <= i <  N-1}"
#define SET_1D_3   "[N]->{S0[i]: 1  < i <  N+1}"
#define SET_1D_4        "{S0[i]: 0 <= i <  10}"
#define SET_1D_5        "{S0[i]: 1  < i <  10}"
#define SET_1D_6 "[N,M]->{S0[i]: 0 <= i <  N+M+2}"
#define SET_1D_7 "[N,M]->{S0[i]: 0 <= i <= N and i <= M}"
#define SET_1D_8 "[N,M]->{S0[i]: M <= i <= N}"
#define SET_1D_9 "[N,M]->{S0[i]: M <  i <= N}"

#define SET_2D_0 "[N,M]->{S0[i,j]: 0<=i<=N and 0<=j<=M}"

#define SET_3D_0 "[N,M,K]->{S0[i,j,k]: 0<=i<=N and 0<=j<=M and 0<=k<=K}"






void test_get_parameters_list(isl_ctx *ctx)
{
    isl_set *set_0 = isl_set_read_from_str(ctx, SET_1D_0);
    std::string list = utility::get_parameters_list(set_0);
    std::string expected_list = "N";
    bool success = (list == expected_list);
    print_test_results("test 0 for get_parameters_list", success);

    set_0 = isl_set_read_from_str(ctx, SET_1D_4);
    list = utility::get_parameters_list(set_0);
    expected_list = "";
    success = (list == expected_list);
    print_test_results("test 1 for get_parameters_list", success);

    set_0 = isl_set_read_from_str(ctx, SET_1D_6);
    list = utility::get_parameters_list(set_0);
    expected_list = "N,M";
    success = (list == expected_list);
    print_test_results("test 2 for get_parameters_list", success);

    set_0 = isl_set_read_from_str(ctx, SET_3D_0);
    list = utility::get_parameters_list(set_0);
    expected_list = "N,M,K";
    success = (list == expected_list);
    print_test_results("test 3 for get_parameters_list", success);
}

void test_get_upper_bounds(isl_ctx *ctx)
{
    isl_set *set_0 = isl_set_read_from_str(ctx, SET_1D_0);
    expr e_0 = utility::get_bound(set_0, 0, true);
    expr expected_e_0 = var(p_int32, "N");
    bool success = e_0.is_equal(expected_e_0);
    print_test_results("test 0 for upper bound", success);

    set_0 = isl_set_read_from_str(ctx, SET_1D_1);
    e_0 = utility::get_bound(set_0, 0, true);
    expr N = tiramisu::var(p_int32, "N");
    expected_e_0 = expr(tiramisu::o_add, N, expr((int32_t) -1));
    success = e_0.is_equal(expected_e_0);
    print_test_results("test 1 for upper bound", success);

    set_0 = isl_set_read_from_str(ctx, SET_1D_2);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = expr(tiramisu::o_add, N, expr((int32_t) -2));
    success = e_0.is_equal(expected_e_0);
    print_test_results("test 2 for upper bound", success);

    set_0 = isl_set_read_from_str(ctx, SET_1D_4);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = expr((int32_t) 9);
    success = e_0.is_equal(expected_e_0);
    print_test_results("test 3 for upper bound", success);

    set_0 = isl_set_read_from_str(ctx, SET_1D_6);
    e_0 = utility::get_bound(set_0, 0, true);
    expr M = tiramisu::var(p_int32, "M");
    expr add1 = expr(tiramisu::o_add, N, M);
    expected_e_0 = expr(tiramisu::o_add, add1, expr((int32_t) 1));
    success = e_0.is_equal(expected_e_0);
    print_test_results("test 4 for upper bound", success);

    set_0 = isl_set_read_from_str(ctx, SET_1D_7);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = expr(o_min, N, M);
    success = e_0.is_equal(expected_e_0);
    print_test_results("test 5 for upper bound", success);

    // Negative test
    set_0 = isl_set_read_from_str(ctx, SET_1D_0);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = M;
    success = ! e_0.is_equal(expected_e_0);
    print_test_results("test 6 for upper bound (negative test)", success);

    // Negative test
    set_0 = isl_set_read_from_str(ctx, SET_1D_0);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = M + N;
    success = ! e_0.is_equal(expected_e_0);
    print_test_results("test 7 for upper bound (negative test)", success);

    set_0 = isl_set_read_from_str(ctx, SET_2D_0);
    e_0 = utility::get_bound(set_0, 1, true);
    expected_e_0 = M;
    success = e_0.is_equal(expected_e_0);
    print_test_results("test 8 for upper bound", success);

    set_0 = isl_set_read_from_str(ctx, SET_2D_0);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = N;
    success = e_0.is_equal(expected_e_0);
    print_test_results("test 9 for upper bound", success);

    set_0 = isl_set_read_from_str(ctx, SET_3D_0);
    e_0 = utility::get_bound(set_0, 1, true);
    expected_e_0 = M;
    success = e_0.is_equal(expected_e_0);
    print_test_results("test 10 for upper bound", success);
}

void test_get_lower_bounds(isl_ctx *ctx)
{
    isl_set *set_0 = isl_set_read_from_str(ctx, SET_1D_0);
    expr e_0 = utility::get_bound(set_0, 0, false);
    expr expected_e_0 = expr((int32_t) 0);
    bool success = e_0.is_equal(expected_e_0);
    print_test_results("test 0 for upper bound", success);

    set_0 = isl_set_read_from_str(ctx, SET_1D_3);
    e_0 = utility::get_bound(set_0, 0, false);
    expected_e_0 = expr((int32_t) 2);
    success = e_0.is_equal(expected_e_0);
    print_test_results("test 1 for upper bound", success);

    set_0 = isl_set_read_from_str(ctx, SET_1D_8);
    e_0 = utility::get_bound(set_0, 0, false);
    expr M = tiramisu::var(p_int32, "M");
    expected_e_0 = 1*M;
    success = e_0.is_equal(expected_e_0);
    print_test_results("test 2 for upper bound", success);

    set_0 = isl_set_read_from_str(ctx, SET_1D_9);
    e_0 = utility::get_bound(set_0, 0, false);
    expected_e_0 = 1*M + 1;
    success = e_0.is_equal(expected_e_0);
    print_test_results("test 3 for upper bound", success);
}

int main(int, char **)
{
    isl_ctx *ctx = isl_ctx_alloc();

    test_get_upper_bounds(ctx);
    test_get_parameters_list(ctx);
    test_get_lower_bounds(ctx);

    return 0;
}
