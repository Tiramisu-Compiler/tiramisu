#include <iostream>

void coli_str_dump(std::string str)
{
	std::cout << str;
	std::cout.flush();	
}


void coli_error(std::string str, bool exit_program)
{
	std::cerr << "Error in " << __FILE__ << ":"
		  << __LINE__ << " - " << str << std::endl;

	if (exit_program)
		exit(1);
}
