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

void combine_dist_results(const std::string &test, int split_type,
                          std::vector<std::tuple<int, int, int>> dims_per_rank) {
    // Figure out the total size
    int total_channels = 0;
    int total_cols = 0;
    int total_rows = 0;
    if (split_type == 0) { // rows and cols constant across ranks
        total_cols = std::get<0>(dims_per_rank[0]);
        total_rows = std::get<1>(dims_per_rank[0]);
        for (auto e : dims_per_rank) {
            total_channels += std::get<2>(e);
        }
    } else if (split_type == 1) { // channels and rows constant across ranks
        total_channels = std::get<2>(dims_per_rank[0]);
        total_rows = std::get<1>(dims_per_rank[0]);
        for (auto e : dims_per_rank) {
            total_cols += std::get<0>(e);
        }
    } else if (split_type == 2) { // channels and cols constant across ranks
        total_channels = std::get<2>(dims_per_rank[0]);
        total_cols = std::get<0>(dims_per_rank[0]);
        for (auto e : dims_per_rank) {
            total_rows += std::get<1>(e);
        }
    } else {
        assert(false); // TODO put in appropriate error checking
    }
    std::ofstream output_file;
    output_file.open("/tmp/" + test + "_all_ranks.txt");
    for (int rank = 0; rank < dims_per_rank.size(); rank++) {
        std::ifstream f("/tmp/" + test + "_rank_" + std::to_string(rank) + ".txt");
        if (f.is_open()) {
            std::string line;
            for (int chan = 0; chan < total_channels; chan++) {
                for (int cols = 0; cols < total_cols; cols++) {
                    for (int rows = 0; rows < total_rows; rows++) {
                        std::getline(f, line);
                        output_file << line << std::endl;
                    }
                }
            }
        } else {
            assert(false); // TODO put in appropriate error checking
        }
        f.close();
    }
    output_file.flush();
    output_file.close();
}