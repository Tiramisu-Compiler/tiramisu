#include <iostream>
#include <string>
#include <sstream>
#include "tiramisu_code_generator.h"
#include <random>
#include <cstdlib>

using namespace std;


void read_inputs(int *nb_codes, int *nb_stages, string *default_type_tiramisu, string *default_type_wrapper, double *assignment_prob, double *assignment_input_prob, double *conv_prob, double *same_padding_prob, vector<int> *computations_dimensions, 
                int *nb_inputs,  vector <int> *var_nums, int *offset);
int main() {
    int nb_codes, nb_stages, nb_inputs, offset;
    vector <int> computations_dimensions, var_nums;
    double assignment_prob, assignment_input_prob, conv_prob, same_padding_prob;
    string defaut_type_tiramisu, default_type_wrapper;
    read_inputs(&nb_codes, &nb_stages, &defaut_type_tiramisu, &default_type_wrapper, &assignment_prob, &assignment_input_prob, &conv_prob, &same_padding_prob, &computations_dimensions, &nb_inputs, &var_nums, &offset);


    double num, *padding_probs = new double[1], *computations_probs = new double[2];  

    padding_probs[0] = same_padding_prob;

    computations_probs[0] = assignment_prob;
    computations_probs[1] = assignment_input_prob;

    for (int i = 0; i < nb_codes; ++i){
        num = (double) rand() / (RAND_MAX);
        if (num < conv_prob){
            generate_tiramisu_code_conv(i, nb_stages, padding_probs, &defaut_type_tiramisu, &default_type_wrapper);
        }
        else{
            generate_tiramisu_code_multiple_computations(i, &computations_dimensions, nb_stages, computations_probs, &var_nums, nb_inputs, &defaut_type_tiramisu, &default_type_wrapper, offset);
        }
    }

    return 0;
}

void read_inputs(int *nb_codes, int *nb_stages, string *default_type_tiramisu, string *default_type_wrapper, double *assignment_prob, double *assignment_input_prob, double *conv_prob, double *same_padding_prob, vector<int> *computations_dimensions, 
                int *nb_inputs,  vector <int> *var_nums, int *offset){
    ifstream input_file;
    string line, info;
    unsigned long pos1, pos2;
    input_file.open("inputs.txt");

    //nb_codes
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *nb_codes;
    }

    //nb_stages
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *nb_stages;
    }

    //default_type_tiramisu
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        *default_type_tiramisu = line.substr(pos1 + 1, pos2 - pos1 - 1);
    }

    //default_type_wrapper
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        *default_type_wrapper = line.substr(pos1 + 1, pos2 - pos1 - 1);
    }


    //assignment_prob
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *assignment_prob;
    }

    //assignment_input_prob
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *assignment_input_prob;
    }

    //conv_prob
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *conv_prob;
    }

    //computations_dimensions
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        int number;
        while (info_stream >> number) {
            (*computations_dimensions).push_back(number);
        }

    }

    //nb_inputs
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *nb_inputs;
    }

    //var_nums
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        int number;
        while (info_stream >> number)
            (*var_nums).push_back(number);

    }

    //offset
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *offset;
    }

    //same_padding_prob
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *same_padding_prob;
    }


}
