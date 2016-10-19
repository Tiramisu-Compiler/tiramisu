#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <coli/debug.h>
#include <coli/core.h>

#include <string.h>
#include <Halide.h>

using namespace Halide;
using namespace Halide::Internal;
using std::map;
using std::string;
using std::vector;

/*
let f.s0.y.max = ((f.min.1 + f.extent.1) - 1)
let f.s0.y.min = f.min.1
let f.s0.x.max = ((f.min.0 + f.extent.0) - 1)
let f.s0.x.min = f.min.0
producer f {
    let f.s0.y.loop_max = f.s0.y.max
    let f.s0.y.loop_min = f.s0.y.min
    let f.s0.y.loop_extent = ((f.s0.y.max + 1) - f.s0.y.min)
    let f.s0.x.loop_max = f.s0.x.max
    let f.s0.x.loop_min = f.s0.x.min
    let f.s0.x.loop_extent = ((f.s0.x.max + 1) - f.s0.x.min)
    for (f.s0.y, f.s0.y.loop_min, f.s0.y.loop_extent) {
        for (f.s0.x, f.s0.x.loop_min, f.s0.x.loop_extent) {
            f(f.s0.x, f.s0.y) = 13
        }
    }
}
consume f {
    0
}*/

void generate_function_1(std::string name, int size)
{
    coli::global::set_default_coli_options();

    Var x("x"), y("y");
    Func halide_f("f");
    halide_f(x, y) = cast<uint8_t>(13);

    Expr f_min_0 = Variable::make(Int(32), "f_min_0");
    Expr f_extent_0 = Variable::make(Int(32), "f_extent_0");
    Expr f_min_1 = Variable::make(Int(32), "f_min_1");
    Expr f_extent_1 = Variable::make(Int(32), "f_extent_1");

    Expr y_loop_min = Variable::make(Int(32), "f_s0_y_loop_min");
    Expr y_loop_extent = Variable::make(Int(32), "f_s0_y_loop_extent");
    Expr x_loop_min = Variable::make(Int(32), "f_s0_x_loop_min");
    Expr x_loop_extent = Variable::make(Int(32), "f_s0_x_loop_extent");

    Expr y_max = Variable::make(Int(32), "f_s0_y_max");
    Expr y_min = Variable::make(Int(32), "f_s0_y_min");
    Expr x_max = Variable::make(Int(32), "f_s0_x_max");
    Expr x_min = Variable::make(Int(32), "f_s0_x_min");

    Expr f_s0_x = Variable::make(Int(32), "f_s0_x");
    Expr f_s0_y = Variable::make(Int(32), "f_s0_y");

    Stmt producer = Provide::make("f", {make_const(UInt(8), 13)}, {f_s0_x, f_s0_y});
    producer = For::make("f_s0_y", y_loop_min, y_loop_extent, ForType::Serial, DeviceAPI::None, producer);
    producer = For::make("f_s0_x", x_loop_min, x_loop_extent, ForType::Serial, DeviceAPI::None, producer);
    producer = LetStmt::make("f_s0_x_loop_extent", ((x_max + 1) - x_min), producer);
    producer = LetStmt::make("f_s0_x_loop_min", x_min, producer);
    producer = LetStmt::make("f_s0_x_loop_max", x_max, producer);
    producer = LetStmt::make("f_s0_y_loop_extent", ((y_max + 1) - y_min), producer);
    producer = LetStmt::make("f_s0_y_loop_min", y_min, producer);
    producer = LetStmt::make("f_s0_y_loop_max", y_max, producer);

    Stmt consumer = Evaluate::make(make_const(UInt(8), 0));

    Stmt s = Block::make(ProducerConsumer::make("f", true, producer),
                         ProducerConsumer::make("f", false, consumer));
    s = LetStmt::make("f_s0_x_min", f_min_0, s);
    s = LetStmt::make("f_s0_x_max", ((f_min_0 + f_extent_0) - 1), s);
    s = LetStmt::make("f_s0_y_min", f_min_1, s);
    s = LetStmt::make("f_s0_y_max", ((f_min_1 + f_extent_1) - 1), s);

    std::cout << "Test Halide Stmt:\n" << s << "\n\n";

    map<string, Function> env = { {"f", halide_f.function()} };
    vector<int32_t> f_size = {size, size};
    map<string, vector<int32_t>> output_buffers_size = { {"f", f_size} };

    coli::halide_pipeline_to_c(s, {halide_f.function()}, env, output_buffers_size, name);

    // Generated COLi
    /*coli::function test_c_gen("test_c_gen");
    coli::buffer buff_f("buff_f", 2, {coli::expr(10), coli::expr(10)}, coli::p_uint8, NULL, coli::a_output, &test_c_gen);
    coli::constant f_s0_x_loop_min("f_s0_x_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &test_c_gen);
    coli::constant f_s0_x_loop_extent("f_s0_x_loop_extent", coli::expr((int32_t)10), coli::p_int32, true, NULL, 0, &test_c_gen);
    coli::constant f_s0_y_loop_min("f_s0_y_loop_min", coli::expr((int32_t)0), coli::p_int32, true, NULL, 0, &test_c_gen);
    coli::constant f_s0_y_loop_extent("f_s0_y_loop_extent", coli::expr((int32_t)10), coli::p_int32, true, NULL, 0, &test_c_gen);
    coli::computation f("[f_s0_x_loop_min, f_s0_x_loop_extent, f_s0_y_loop_min, f_s0_y_loop_extent]->{f[f_s0_x, f_s0_y]: (f_s0_x_loop_min <= f_s0_x <= ((f_s0_x_loop_min + f_s0_x_loop_extent) + -1)) and (f_s0_y_loop_min <= f_s0_y <= ((f_s0_y_loop_min + f_s0_y_loop_extent) + -1))}", coli::expr((uint8_t)13), true, coli::p_uint8, &test_c_gen);
    f.set_access("{f[f_s0_x, f_s0_y]->buff_f[f_s0_x, f_s0_y]}");
    test_c_gen.set_arguments({&buff_f});
    test_c_gen.gen_isl_ast();
    test_c_gen.gen_halide_stmt();
    test_c_gen.dump_halide_stmt();
    test_c_gen.gen_halide_obj("build/generated_fct_test_06.o");*/
}


int main(int argc, char **argv)
{
    generate_function_1("test_c_gen", 10);

    return 0;
}
