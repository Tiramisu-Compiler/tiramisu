#include <stdio.h>
#include "Halide.h"
#include <iostream>
#include <string>

#define SIZE 10

void generate_function(std::string f_name, int size, int pos, int val)
{
	std::string buff_name = "myOutBuff";

        halide_dimension_t *shape = new halide_dimension_t[1];
        shape[0].min = 0;
        shape[0].extent = 10; 
        shape[0].stride = 1;
	Halide::Buffer<> buffer = Halide::Buffer<>(Halide::UInt(8), NULL, 1, shape, buff_name);

	Halide::Internal::Parameter p = Halide::Internal::Parameter(Halide::UInt(8), true, 1, buff_name);
	p.set_buffer(buffer);

	Halide::Expr halideTrue = Halide::Internal::const_true();
	Halide::Internal::Stmt s = Halide::Internal::Store::make(buff_name, Halide::Expr(val), Halide::Expr(pos), p, halideTrue);

	std::cout << s << std::endl;
	s = unpack_buffers(s);

	std::vector<Halide::Argument> args_vect;
	Halide::Argument arg(buff_name, Halide::Argument::Kind::OutputBuffer, Halide::UInt(8), 1);
       	args_vect.push_back(arg);

	Halide::Target target = Halide::get_host_target();
  
	Halide::Module m(f_name, target);
        Halide::Internal::LoweredFunc ss = Halide::Internal::LoweredFunc(f_name, args_vect, s, Halide::Internal::LoweredFunc::External);
        std::cout << ss << std::endl;
 
	m.append(ss);
        Halide::Outputs output = Halide::Outputs().object(f_name + ".o");
  	m.compile(output);
  	m.compile(Halide::Outputs().c_header(f_name + ".h"));
}







int main(int argc, char* argv[]){

	int pos = 5;
	int val = 3;

	generate_function("my_func", SIZE, pos, val);

	return 0;

}
