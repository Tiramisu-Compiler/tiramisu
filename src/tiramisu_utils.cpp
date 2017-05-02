#include "tiramisu/utils.h"

#include <stdexcept>
#include <iomanip>
#include <fstream>

using std::string;
using std::vector;

double median(vector<std::chrono::duration<double, std::milli>> scores)
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

string str_identation(int size)
{
    assert(size >= 0);
    std::ostringstream ss;
    for (size_t i = 0; i < size; ++i) {
        ss << " ";
    }
    return ss.str();
}

void print_time(const string &file_name, const string &kernel_name,
                const vector<string> &header_text,
                const vector<double> &time_vector)
{
    std::ofstream file;

    file.open(file_name, std::ios::app);
    file << std::fixed << std::setprecision(6);

    file << kernel_name << "; ";
    for (const auto &t : time_vector)
    {
        file << t << "; ";
    }
    file << std::endl;

    std::cout << "Kernel" << str_identation(14) << ": ";
    for (const auto &t : header_text)
    {
        std::cout << t << str_identation(15 - t.size()) << "; ";
    }
    std::cout << std::endl;

    std::cout << kernel_name << str_identation(20 - kernel_name.size()) << ": ";
    for (const auto &t : time_vector)
    {
        string str = std::to_string(t);
        std::cout << str << str_identation(15 - str.size()) << "; ";
    }
    std::cout << std::endl;

    file.close();
}
