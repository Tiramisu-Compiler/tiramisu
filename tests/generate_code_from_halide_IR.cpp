#include <DebugIR.h>
#include <Halide.h>
#include <iostream>
#include <fstream>

namespace Halide {
namespace Internal {

using std::ostream;
using std::endl;
using std::string;
using std::vector;
using std::ostringstream;
using std::map;

#ifndef GENERATE_C
#ifndef GENERATE_X86
#ifndef GENERATE_DEBUGING_CODE
//#define GENERATE_C
//#define GENERATE_X86
#define GENERATE_DEBUGING_CODE
#endif
#endif
#endif

void foo()
{
    // Generate arguments
    Argument buffer_arg("buf", Argument::OutputBuffer, Int(32), 3);
    vector<Argument> args(1);
    args[0] = buffer_arg;

    // Loop iterators
    Var x("x");

    Expr e = Add::make(Expr(1), Expr(1));
    Stmt s = Store::make("buf", e, x, Parameter());
    s = Halide::Internal::For::make("x", Expr(0), Expr(10), Halide::Internal::ForType::Serial,
		  		    Halide::DeviceAPI::Host, s);

    Module m("", get_host_target());
    m.append(LoweredFunc("test1", args, s, LoweredFunc::External));

#ifdef GENERATE_C
    ostringstream source;
    {
        CodeGen_C cg(source, CodeGen_C::CImplementation);
        cg.compile(m);
    }
    std::ofstream myfile;
    myfile.open("generated_C_pgm.cpp"); 
    string src = source.str();
    myfile << src << std::endl;
    // std::cout << src << std::endl;
    myfile.close();
#endif

#ifdef GENERATE_X86
    Target target;
    target.os = Target::OSX;
    target.arch = Target::X86;
    target.bits = 64;
    std::vector<Target::Feature> x86_features;
    x86_features.push_back(Target::AVX);
    x86_features.push_back(Target::SSE41);
    target.set_features(x86_features);
    CodeGen_X86 cg(target);
    cg.compile(m);
#endif

#ifdef GENERATE_DEBUGING_CODE
    Halide::Internal::IRPrinter pr(std::cout);
    pr.print(s);
#endif

}

}
}

int main(int argc, char **argv)
{
	Halide::Internal::foo();

	return 0;
}
