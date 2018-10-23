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

void combine_dist_results(const std::string &test, std::vector<int> dims, int num_ranks) {
    // Figure out the total size
    int total_vals = 1;
    for (auto d : dims) {
        total_vals *= d;
    }
    std::ofstream output_file;
    output_file.open("/tmp/" + test + "_all_ranks.txt");
    for (int rank = 0; rank < num_ranks; rank++) {
        std::ifstream f("/tmp/" + test + "_rank_" + std::to_string(rank) + ".txt");
        if (f.is_open()) {
            std::string line;
            for (int v = 0; v < total_vals; v++) {
                std::getline(f, line);
                output_file << line << std::endl;
            }
        } else {
            assert(false); // TODO put in appropriate error checking
        }
        f.close();
    }
    output_file.flush();
    output_file.close();
}