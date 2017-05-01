#include "tiramisu/utils.h"

#include <stdexcept>
#include <iomanip>
#include <fstream>


double median(std::vector<std::chrono::duration<double, std::milli>> scores)
{
    double median;
    size_t size = scores.size();

    sort(scores.begin(), scores.end());

    if (size % 2 == 0)
    {
        median = (scores[size / 2 - 1].count() + scores[size / 2].count()) / 2;
    }
    else
    {
        median = scores[size / 2].count();
    }

    return median;
}

void print_time(std::string file_name, std::string kernel_name,
                std::vector<std::string> header_text,
                std::vector<double> time_vector)
{
    std::ofstream file;

    file.open(file_name, std::ios::app);
    file << std::fixed << std::setprecision(6);

    file << kernel_name << " ; ";
    for (auto t : time_vector)
    {
        file << t << " ;";
    }
    file << std::endl;

    std::cout << "Kernel : ";
    for (auto t : header_text)
    {
        std::cout << t << " ;";
    }
    std::cout << std::endl;

    std::cout << kernel_name << " : ";
    for (auto t : time_vector)
    {
        std::cout << t << " ;";
    }
    std::cout << std::endl;

    file.close();
}
