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

std::vector<std::pair<std::string, bool>> test_results;

/**
 * Test file for low level functions in tiramisu.
 */

#define SIZE	   10
#define FCT_NAME   "name"

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
    tiramisu::global::set_loop_iterator_default_data_type(tiramisu::p_int32);
    tiramisu::str_dump("------------ test_get_parameters_list - test 0 -----------\n");
    isl_set *set_0 = isl_set_read_from_str(ctx, SET_1D_0);
    std::string list = utility::get_parameters_list(set_0);
    std::string expected_list = "N";
    bool success = (list == expected_list);
    test_results.push_back(std::pair<std::string, bool>("test 0 for get_parameters_list", success));

    tiramisu::str_dump("------------ test_get_parameters_list - test 1 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_1D_4);
    list = utility::get_parameters_list(set_0);
    expected_list = "";
    success = (list == expected_list);
    test_results.push_back(std::pair<std::string, bool>("test 1 for get_parameters_list", success));

    tiramisu::str_dump("------------ test_get_parameters_list - test 2 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_1D_6);
    list = utility::get_parameters_list(set_0);
    expected_list = "N,M";
    success = (list == expected_list);
    test_results.push_back(std::pair<std::string, bool>("test 2 for get_parameters_list", success));

    tiramisu::str_dump("------------ test_get_parameters_list - test 3 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_3D_0);
    list = utility::get_parameters_list(set_0);
    expected_list = "N,M,K";
    success = (list == expected_list);
    test_results.push_back(std::pair<std::string, bool>("test 3 for get_parameters_list", success));
}

void test_get_upper_bounds(isl_ctx *ctx)
{
    tiramisu::global::set_loop_iterator_default_data_type(tiramisu::p_int32);
    tiramisu::str_dump("------------ test_get_upper_bounds - test 0 -----------\n");
    isl_set *set_0 = isl_set_read_from_str(ctx, SET_1D_0);
    expr e_0 = utility::get_bound(set_0, 0, true);
    expr expected_e_0 = var(p_int32, "N");
    bool success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 0 for upper bound", success));

    tiramisu::str_dump("------------ test_get_upper_bounds - test 1 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_1D_1);
    e_0 = utility::get_bound(set_0, 0, true);
    expr N = tiramisu::var(p_int32, "N");
    expected_e_0 = expr(tiramisu::o_sub, N, expr((int32_t) 1));
    success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 1 for upper bound", success));

    tiramisu::str_dump("------------ test_get_upper_bounds - test 2 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_1D_2);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = expr(tiramisu::o_sub, N, expr((int32_t) 1));
    expected_e_0 = expr(tiramisu::o_sub, expected_e_0, expr((int32_t) 1));
    success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 2 for upper bound", success));

    tiramisu::str_dump("------------ test_get_upper_bounds - test 3 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_1D_4);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = expr((int32_t) 9);
    success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 3 for upper bound", success));

    tiramisu::str_dump("------------ test_get_upper_bounds - test 4 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_1D_6);
    e_0 = utility::get_bound(set_0, 0, true);
    expr M = tiramisu::var(p_int32, "M");
    expr add1 = expr(tiramisu::o_add, N, M);
    expected_e_0 = expr(tiramisu::o_add, add1, expr((int32_t) 1));
    success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 4 for upper bound", success));

    tiramisu::str_dump("------------ test_get_upper_bounds - test 5 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_1D_7);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = expr(o_min, N, M);
    success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 5 for upper bound", success));

    tiramisu::str_dump("------------ test_get_upper_bounds - test 6 -----------\n");
    // Negative test
    set_0 = isl_set_read_from_str(ctx, SET_1D_0);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = M;
    success = ! e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 6 for upper bound (negative test)", success));

    tiramisu::str_dump("------------ test_get_upper_bounds - test 7 -----------\n");
    // Negative test
    set_0 = isl_set_read_from_str(ctx, SET_1D_0);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = M + N;
    success = ! e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 7 for upper bound (negative test)", success));

    tiramisu::str_dump("------------ test_get_upper_bounds - test 8 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_2D_0);
    e_0 = utility::get_bound(set_0, 1, true);
    expected_e_0 = M;
    success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 8 for upper bound", success));

    tiramisu::str_dump("------------ test_get_upper_bounds - test 9 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_2D_0);
    e_0 = utility::get_bound(set_0, 0, true);
    expected_e_0 = N;
    success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 9 for upper bound", success));

    tiramisu::str_dump("------------ test_get_upper_bounds - test 10 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_3D_0);
    e_0 = utility::get_bound(set_0, 1, true);
    expected_e_0 = M;
    success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 10 for upper bound", success));
}

void test_get_lower_bounds(isl_ctx *ctx)
{
    tiramisu::global::set_loop_iterator_default_data_type(tiramisu::p_int32);
    tiramisu::str_dump("------------ test_get_lower_bounds - test 0 -----------\n");
    isl_set *set_0 = isl_set_read_from_str(ctx, SET_1D_0);
    expr e_0 = utility::get_bound(set_0, 0, false);
    expr expected_e_0 = expr((int32_t) 0);
    bool success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 0 for lower bound", success));

    tiramisu::str_dump("------------ test_get_lower_bounds - test 1 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_1D_3);
    e_0 = utility::get_bound(set_0, 0, false);
    expected_e_0 = expr((int32_t) 2);
    success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 1 for lower bound", success));

    tiramisu::str_dump("------------ test_get_lower_bounds - test 2 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_1D_8);
    e_0 = utility::get_bound(set_0, 0, false);
    expr M = tiramisu::var(p_int32, "M");
    expected_e_0 = M;
    success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 2 for lower bound", success));

    tiramisu::str_dump("------------ test_get_lower_bounds - test 3 -----------\n");
    set_0 = isl_set_read_from_str(ctx, SET_1D_9);
    e_0 = utility::get_bound(set_0, 0, false);
    expected_e_0 = M + 1;
    success = e_0.is_equal(expected_e_0);
    test_results.push_back(std::pair<std::string, bool>("test 3 for lower bound", success));
}

namespace tiramisu
{

int dynamic_dimension_into_loop_level(int dim);

class computation_tester: tiramisu::computation
{
public:
    static void test_get_iteration_domain_dimension_names()
    {
	tiramisu::global::set_default_tiramisu_options();
    tiramisu::global::set_loop_iterator_default_data_type(tiramisu::p_int32);
	tiramisu::function function0(FCT_NAME);
	tiramisu::constant N("N", tiramisu::expr((int32_t) SIZE), p_int32,
			true, NULL, 0, &function0);
	tiramisu::var i("i");
	tiramisu::var j("j");

	tiramisu::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}",
			tiramisu::expr((uint8_t) 0), true, p_uint8,
			&function0);
	tiramisu::computation S1("[N]->{S1[0,j]: 0<=j<N}",
			S0(i,j), true, p_uint8, &function0);
	tiramisu::computation S2("[N]->{S2[j]: 0<=j<N}",
			S0(i,j), true, p_uint8, &function0);
	tiramisu::computation S3("[N]->{S3[0]}",
			S0(i,j), true, p_uint8, &function0);
	tiramisu::computation S4("[N]->{S4[i,ijj,k,l]}",
			S0(i,j), true, p_uint8, &function0);

        tiramisu::str_dump("-------- test_get_iteration_domain_dimension_names "
			   "- test 0 -----------\n");
	std::vector<std::string> names =
		S0.get_iteration_domain_dimension_names();
	bool success = ((names[0] == "i") && (names[1] == "j"));
        test_results.push_back(std::pair<std::string, bool>("test_global: test 0 for get_iteration_domain_dimension_names", success));

	tiramisu::str_dump("-------- test_get_iteration_domain_dimension_names "
			   "- test 1 -----------\n");
	names = S1.get_iteration_domain_dimension_names();
	success = ((names[1] == "j") && (names.size() == 2)) ;
        test_results.push_back(std::pair<std::string, bool>("test_global: test 1 for get_iteration_domain_dimension_names", success));

        tiramisu::str_dump("-------- test_get_iteration_domain_dimension_names "
			   "- test 2 -----------\n");
	names = S2.get_iteration_domain_dimension_names();
	success = ((names[0] == "j") && (names.size() == 1)) ;
        test_results.push_back(std::pair<std::string, bool>("test_global: test 2 for get_iteration_domain_dimension_names", success));

        tiramisu::str_dump("-------- test_get_iteration_domain_dimension_names "
			   "- test 3 -----------\n");
	names = S3.get_iteration_domain_dimension_names();
	success = ((names.size() == 1)) ;
        test_results.push_back(std::pair<std::string, bool>("test_global: test 3 for get_iteration_domain_dimension_names", success));

        tiramisu::str_dump("-------- test_get_iteration_domain_dimension_names "
			   "- test 4 -----------\n");
	names = S4.get_iteration_domain_dimension_names();
	success = ((names.size() == 4) && (names[0] == "i") &&
			(names[1] == "ijj") && (names[2] == "k") &&
			(names[3] == "l")) ;
        test_results.push_back(std::pair<std::string, bool>("test_global: test 4 for get_iteration_domain_dimension_names", success));
    }

    static void test_get_dimension_numbers_from_dimension_names()
    {
	tiramisu::global::set_default_tiramisu_options();
    tiramisu::global::set_loop_iterator_default_data_type(tiramisu::p_int32);
	tiramisu::function function0(FCT_NAME);
	tiramisu::constant N("N", tiramisu::expr((int32_t) SIZE), p_int32,
			true, NULL, 0, &function0);
	tiramisu::var i("i");
	tiramisu::var j("j");

	tiramisu::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}",
			tiramisu::expr((uint8_t) 0), true, p_uint8,
			&function0);
	tiramisu::computation S1("[N]->{S1[0,j]: 0<=j<N}",
			S0(i,j), true, p_uint8, &function0);
	tiramisu::computation S2("[N]->{S2[j]: 0<=j<N}",
			S0(i,j), true, p_uint8, &function0);
	tiramisu::computation S3("[N]->{S3[0]}",
			S0(i,j), true, p_uint8, &function0);
	tiramisu::computation S4("[N]->{S4[i,j,k,l]}",
			S0(i,j), true, p_uint8, &function0);

        tiramisu::str_dump("-------- get_loop_level_numbers_from_dimension_names "
			   "- test 0 -----------\n");
	std::vector<int> numbers =
		S0.get_loop_level_numbers_from_dimension_names({"j"});
	bool success = ((numbers[0] == 1) && (numbers.size() == 1));
        test_results.push_back(std::pair<std::string, bool>("test_global: test 0 for get_loop_level_numbers_from_dimension_names", success));

	tiramisu::str_dump("-------- get_loop_level_numbers_from_dimension_names "
			   "- test 1 -----------\n");
	numbers =
	    S0.get_loop_level_numbers_from_dimension_names({"j", "i"});
	success = ((numbers[0] == 1) &&
			(numbers[1] == 0) &&
			(numbers.size() == 2));
        test_results.push_back(std::pair<std::string, bool>("test_global: test 1 for get_loop_level_numbers_from_dimension_names", success));

	tiramisu::str_dump("-------- get_loop_level_numbers_from_dimension_names "
			   "- test 2 -----------\n");
	numbers =
	    S0.get_loop_level_numbers_from_dimension_names({"i", "j"});
	success = ((numbers[0] == 0) &&
			(numbers[1] == 1) &&
			(numbers.size() == 2));
        test_results.push_back(std::pair<std::string, bool>("test_global: test 2 for get_loop_level_numbers_from_dimension_names", success));

	tiramisu::str_dump("-------- get_loop_level_numbers_from_dimension_names "
			   "- test 3 -----------\n");
	numbers =
	    S1.get_loop_level_numbers_from_dimension_names({"j"});
	success = ((numbers[0] == 1) &&
			(numbers.size() == 1));
        test_results.push_back(std::pair<std::string, bool>("test_global: test 3 for get_loop_level_numbers_from_dimension_names", success));

	tiramisu::str_dump("-------- get_loop_level_numbers_from_dimension_names "
			   "- test 4 -----------\n");
	numbers =
	    S2.get_loop_level_numbers_from_dimension_names({"j"});
	success = ((numbers[0] == 0) &&
		   (numbers.size() == 1));
        test_results.push_back(std::pair<std::string, bool>("test_global: test 4 for get_loop_level_numbers_from_dimension_names", success));

	tiramisu::str_dump("-------- get_loop_level_numbers_from_dimension_names "
			   "- test 5 -----------\n");
	numbers =
	    S2.get_loop_level_numbers_from_dimension_names({"root"});
	success = ((numbers[0] == computation::root_dimension) &&
		   (numbers.size() == 1));
        test_results.push_back(std::pair<std::string, bool>("test_global: test 5 for get_loop_level_numbers_from_dimension_names", success));
    }

    static void test_dynamic_dimension_into_loop_level()
    {
	tiramisu::str_dump("-------- get_dynamic_dimension_into_loop_level "
			   "- test 0 -----------\n");
	bool success = (dynamic_dimension_into_loop_level(2) == 0);
        test_results.push_back(std::pair<std::string, bool>("test_global: test 0 for get_dynamic_dimension_into_loop_level", success));

	tiramisu::str_dump("-------- get_dynamic_dimension_into_loop_level "
			   "- test 1 -----------\n");
	success = (dynamic_dimension_into_loop_level(4) == 1);
	test_results.push_back(std::pair<std::string, bool>("test_global: test 1 for get_dynamic_dimension_into_loop_level", success));

	tiramisu::str_dump("-------- get_dynamic_dimension_into_loop_level "
			   "- test 2 -----------\n");
	success = (dynamic_dimension_into_loop_level(6) == 2);
	test_results.push_back(std::pair<std::string, bool>("test_global: test 2 for get_dynamic_dimension_into_loop_level", success));
    }

    static void test_names_functions()
    {
	tiramisu::global::set_default_tiramisu_options();
    tiramisu::global::set_loop_iterator_default_data_type(tiramisu::p_int32);
	tiramisu::function function0(FCT_NAME);
	tiramisu::constant N("N", tiramisu::expr((int32_t) SIZE), p_int32, true, NULL, 0, &function0);
	tiramisu::var i("i");
	tiramisu::var j("j");

	tiramisu::computation S0("[N]->{S0[i,j]: 0<=i<N and 0<=j<N}", tiramisu::expr((uint8_t) 0), true, p_uint8, &function0);
	tiramisu::computation S1("[N]->{S1[i0,i1,i2,i3]: 0<=i0<N and 0<=i1<N and 0<=i2<N and 0<=i3<N}", tiramisu::expr((uint8_t) 0), true, p_uint8, &function0);
	tiramisu::computation S2("[N]->{S2[j]: 0<=j<N}", tiramisu::expr((uint8_t) 0), true, p_uint8, &function0);
	tiramisu::computation S3("[N]->{S3[0]}", tiramisu::expr((uint8_t) 0), true, p_uint8, &function0);

	tiramisu::str_dump("-------- get_loop_levels_number -----------\n");
	bool success = (S0.get_loop_levels_number() == 2 &&
		        S1.get_loop_levels_number() == 4 &&
			S2.get_loop_levels_number() == 1 &&
			S3.get_loop_levels_number() == 1);
        test_results.push_back(std::pair<std::string, bool>("test_global: get_loop_levels_number", success));

	tiramisu::str_dump("-------- get_iteration_domain_dimensions_number -----------\n");
	success = (S0.get_loop_levels_number() == 2 &&
		        S1.get_loop_levels_number() == 4 &&
			S2.get_loop_levels_number() == 1 &&
			S3.get_loop_levels_number() == 1);
        test_results.push_back(std::pair<std::string, bool>("test_global: get_iteration_domain_dimensions_number", success));

	tiramisu::str_dump("-------- get_loop_level_names -----------\n");
	success = (S0.get_loop_level_names()[0] == "i" && S0.get_loop_level_names()[1] == "j" &&
			S1.get_loop_level_names()[0] == "i0" && S1.get_loop_level_names()[1] == "i1" &&
			S1.get_loop_level_names()[2] == "i2" && S1.get_loop_level_names()[3] == "i3" &&
			S2.get_loop_level_names()[0] == "j");
        test_results.push_back(std::pair<std::string, bool>("test_global: get_loop_level_names", success));

	tiramisu::str_dump("-------- set_loop_level_names -----------\n");
	S0.set_loop_level_names({"x0", "x1"});
        success = (S0.get_loop_level_names()[0] == "x0" && S0.get_loop_level_names()[1] == "x1");
        test_results.push_back(std::pair<std::string, bool>("test_global: set_loop_level_names", success));

        tiramisu::str_dump("-------- S0.tile(x0,x1, 8,8, i0,j0,i1,j1) -----------\n");
	S0.tile(tiramisu::var("x0"), tiramisu::var("x1"), 8, 8, tiramisu::var("i0"), tiramisu::var("j0"), tiramisu::var("i1"), tiramisu::var("j1"));
	success = (S0.get_loop_level_names()[0] == "i0" && S0.get_loop_level_names()[1] == "j0" &&
		   S0.get_loop_level_names()[2] == "i1" && S0.get_loop_level_names()[3] == "j1");
        test_results.push_back(std::pair<std::string, bool>("test_global: S0.tile(x0,x1, 8,8, i0,j0,i1,j1)", success));

        tiramisu::str_dump("-------- S0.vectorize(j1, 8, j10, j11) -----------\n");
	S0.vectorize(tiramisu::var("j1"), 8, tiramisu::var("j10"), tiramisu::var("j11"));
	success = (S0.get_loop_levels_number() == 5 &&
		S0.get_loop_level_names()[0] == "i0" && S0.get_loop_level_names()[1] == "j0" &&
		S0.get_loop_level_names()[2] == "i1" && S0.get_loop_level_names()[3] == "j10" &&
		S0.get_loop_level_names()[4] == "j11");
        test_results.push_back(std::pair<std::string, bool>("test_global: S0.vectorize(j1, 8, j10, j11)", success));

        tiramisu::str_dump("-------- S2.vectorize(j, 8, j0, j1) -----------\n");
	S2.vectorize(tiramisu::var("j"), 8, tiramisu::var("j0"), tiramisu::var("j1"));
	success = (S2.get_loop_level_names()[0] == "j0" && S2.get_loop_level_names()[1] == "j1");
        test_results.push_back(std::pair<std::string, bool>("test_global: S2.vectorize(j, 8, j0, j1)", success));
    }
};

}

int main(int, char **)
{
    isl_ctx *ctx = isl_ctx_alloc();

    test_get_upper_bounds(ctx);
    test_get_parameters_list(ctx);
    test_get_lower_bounds(ctx);
    computation_tester::test_get_iteration_domain_dimension_names();
    computation_tester::test_get_dimension_numbers_from_dimension_names();
    computation_tester::test_dynamic_dimension_into_loop_level();
    computation_tester::test_names_functions();

    for (auto const res: test_results)
    {
	print_test_results(res.first, res.second);
    }

    return 0;
}
