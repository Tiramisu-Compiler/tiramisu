#include <tiramisu/tiramisu.h>

#include <string.h>
#include "configure.h"

using namespace tiramisu;

void set_parameters(int &vec_len, int &o_block, int &y_block, int p_N)
{
	if (p_N > 32)
    	{
            vec_len = 32;
            y_block = 32;
    	}
    	else
    	{
            vec_len = std::min(16, p_N);
            y_block = 8;
    	}

    	o_block = 4;

	if (C11 || C12)
	{
		vec_len = 16;
		y_block = 6;
		o_block = 4;
	}
	else if (C13)
	{
		vec_len = 32;
		y_block = 6;
		o_block = 2;
	}
	else if (C21)
	{
		vec_len = 4;
		y_block = 12;
		o_block = 4;
	}
	else if (C22)
	{
		vec_len = 4;
		y_block = 6;
		o_block = 4;
	}
	else if (C23)
	{
		vec_len = 4;
		y_block = 3;
		o_block = 2;
	}
	else if (C41)
	{
		vec_len = 4;
		y_block = 6;
		o_block = 16;
	}
	else if (C42)
	{
		vec_len = 4;
		y_block = 6;
		o_block = 32;
	}
	else if (C43)
	{
		vec_len = 16;
		y_block = 6;
		o_block = 32;
	}
	else if (C61)
	{
		vec_len = 8;
		y_block = 16;
		o_block = 16;
	}
	else if (C62 || C63)
	{
		vec_len = 8;
		y_block = 32;
		o_block = 32;
	}
	else if (C71)
	{
		vec_len = 8;
		y_block = 32;
		o_block = 16;
	}
	else if (C72)
	{
		vec_len = 8;
		y_block = 32;
		o_block = 32;
	}
	else if (C73)
	{
		vec_len = 8;
		y_block = 8;
		o_block = 8;
	}
	else if (C91 || C92 || C93)
	{
		vec_len = 2;
		y_block = 32;
		o_block = 32;
	}
	else if (C101)
	{
		vec_len = 2;
		y_block = 32;
		o_block = 8;
	}
	else if (C102)
	{
		vec_len = 2;
		y_block = 32;
		o_block = 32;
	}
	else if (C103)
	{
		vec_len = 2;
		y_block = 64;
		o_block = 64;
	}
}

void generate(int p_N, int p_K, int p_BATCH_SIZE, int p_FIn, int p_FOut)
{
    init("conv_tiramisu");

    // N: parameters[0]
    // K: parameters[1]
    // FIn: parameters[2]
    // FOut: parameters[3]
    // BATCH_SIZE: parameters[4]

    var i("i", 0, p_K);
    input parameters("parameters",{i}, p_int32);

    constant N_INPUT("N_INPUT", parameters(0) + parameters(1)); // The input is padded, so its size is N + K instead of N.
    constant C_N("C_N", parameters(0));
    constant C_K("C_K", parameters(1));
    constant C_FIn("C_FIn", parameters(2));
    constant C_FOut("C_FOut", parameters(3));
    constant C_BATCH_SIZE("C_BATCH_SIZE", parameters(4));

    var x("x", 0, C_N), y("y", 0, C_N), z("z", 0, C_FOut), n("n", 0, C_BATCH_SIZE ); // Iterators for the conv computations.
    var k_x("k_x", 0, C_K), k_y("k_y", 0, C_K), k_z("k_z", 0, C_FIn); // Iterators for the kernel (filter).

    // Input computations
    input c_input("c_input",{"n", "k_z", "y", "x"}, {C_BATCH_SIZE, C_FIn, N_INPUT, N_INPUT} , p_float32);
    input bias("bias", {"z"}, {C_FOut}, p_float32);
    input filter("filter", {z, k_z , k_y, k_x}, p_float32);

    // First conv computations
    computation conv_init("conv_init",{n, z, y, x}, bias(z));
    computation conv("conv",{n, z, y, x, k_z, k_y, k_x }, conv_init(n, z, y, x) + filter(z, k_z, k_y, k_x) * c_input(n, k_z, y + k_y, x + k_x));
    
    global::get_implicit_function()->add_context_constraints("[C_N, C_K, C_FIn, C_FOut, C_BATCH_SIZE]->{:C_N>1 and C_K>1 and C_FOut>1 and C_FIn>0 and C_BATCH_SIZE>1 and C_K=5}"); // C_FIn%16=0 and C_N%16=0}");

    // Layer II
    int vec_len;
    int y_block;
    int o_block;

    set_parameters(vec_len, o_block, y_block, p_N);
 
    conv_init.then(conv, y);

    conv.parallelize(n);

    // n, z,   y,   x, k_z, k_y,   k_x,
    conv.interchange(x, k_z);
    conv.interchange(y, k_z);
    // n, z, (k_z,  y),   x,  k_y, k_x,

    var z1("z1"), z2("z2");
    conv.split(z, o_block, z1, z2);
    conv_init.split(z, o_block, z1, z2);
    // n, (z1, z2), k_z,   y,   x, k_y, k_x,

    var k_z1("k_z1"), k_z2("k_z2");
    conv.split(k_z, y_block, k_z1, k_z2);
    // n, z1, z2, k_z1, k_z2, y, x, k_y, k_x,

    conv.vectorize(x, vec_len);
    conv_init.vectorize(y, vec_len);

    if (p_N > 32)
    {
	    conv.unroll(k_y, p_K);
	    conv.unroll(k_x, p_K);
    }

    // Layer III
    buffer conv_buf("conv_buf", {expr(C_BATCH_SIZE), expr(C_FOut), expr(C_N), expr(C_N)}, p_float32, a_output);

    conv_init.store_in(&conv_buf);
    conv.store_in(&conv_buf,{n, z, y, x});

    tiramisu::codegen({parameters.get_buffer(), c_input.get_buffer(), filter.get_buffer(), bias.get_buffer(), &conv_buf},"generated_conv_layer.o");

}

int main(int argc, char **argv)
{
    int nb_sizes;
    int sizes[NB_MAX_SIZES][4];

    fill_sizes_array(sizes, nb_sizes);

    int n = sizes[0][0];
    int batch_size = sizes[0][1];
    int fin = sizes[0][2];
    int fout = sizes[0][3];

    generate(n, 5, batch_size, fin, fout);

    return 0;
}
