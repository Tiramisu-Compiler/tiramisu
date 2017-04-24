#include <iostream>

namespace tiramisu
{

int tiramisu_indentation = 0;

void str_dump(std::string str)
{
    std::cout << str;
    std::cout.flush();
}

void str_dump(std::string str, const char * str2)
{
    std::cout << str << " " << str2;
    std::cout.flush();
}

void str_dump(const char *str, const char *str2)
{
    std::cout << str << " " << str2;
    std::cout.flush();
}

void print_indentation()
{
    for (int tiramisu_indent = 0; tiramisu_indent < tiramisu::tiramisu_indentation; tiramisu_indent++)
        if (tiramisu_indent % 4 == 0)
            str_dump("|");
        else
            str_dump(" ");
}

void error(std::string str, bool exit_program)
{
    std::cerr << "Error in " << __FILE__ << ":"
              << __LINE__ << " - " << str << std::endl;

    if (exit_program)
    {
        exit(1);
    }
}

}
