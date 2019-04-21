#include <iostream>
#include <string>
#include <sstream>
#include "tiramisu_code_generator.h"
#include <random>
#include <cstdlib>

//#define CODES_FROM 25

using namespace std;



void read_inputs(int *nb_codes, int *nb_stages, string *default_type_tiramisu, string *default_type_wrapper, double *assignment_prob, double *assignment_input_prob, double *conv_prob, double *same_padding_prob, int *nb_dims,
            vector<int> *scheduling_commands, bool *all_schedules, int *nb_inputs, int *offset, vector <int> *tile_sizes, vector <int> *unrolling_factors, int *nb_rand_schedules);



int main() {
    int CODES_FROM;
    cin >> CODES_FROM; 
    string fpath = "samples";
    mkdir(fpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    int nb_codes, nb_stages, nb_inputs, offset, nb_rand_schedules, nb_dims;
    vector <int> scheduling_commands, tile_sizes, unrolling_factors;
    double assignment_prob, assignment_input_prob, conv_prob, same_padding_prob, interchange_prob, tiling_prob, unrolling_prob;
    string defaut_type_tiramisu, default_type_wrapper;
    bool all_schedules;

    read_inputs(&nb_codes, &nb_stages, &defaut_type_tiramisu, &default_type_wrapper, &assignment_prob, &assignment_input_prob, &conv_prob, &same_padding_prob, &nb_dims, &scheduling_commands, &all_schedules, &nb_inputs,
                &offset, &tile_sizes, &unrolling_factors, &nb_rand_schedules);

    //init stats
    vector<stats_per_type*> types;
    vector <dim_stats*> stats;

    for (int i = 2; i < 5; ++i) {
        types.clear();
        for (int j = 0; j < 3; ++j) {
            types.push_back(new stats_per_type());
        }
        stats.push_back(new dim_stats(i, types));
    }





    double num, *padding_probs = new double[1], *computations_probs = new double[2];

    padding_probs[0] = same_padding_prob;

    computations_probs[0] = assignment_prob;
    computations_probs[1] = assignment_input_prob;


    int sum0 = 0, sum1 = 0, *nb_sched = new int[2], *nb_gen = new int[2];
    nb_gen[0] = 0;
    nb_gen[1] = 0;

    for (int i = CODES_FROM; i < CODES_FROM + nb_codes; ++i){
        num = (double) rand() / (RAND_MAX);
        if (num < conv_prob){
            generate_tiramisu_code_conv(i, nb_stages, padding_probs, &defaut_type_tiramisu, &default_type_wrapper);
        }
        else{
            generate_tiramisu_code_multiple_computations(i, nb_stages, computations_probs, nb_dims, scheduling_commands, all_schedules, nb_inputs, &defaut_type_tiramisu, &default_type_wrapper, offset, &tile_sizes,
                                                         &unrolling_factors, nb_rand_schedules, &stats);
        }
        sum0 += nb_sched[0];
        sum1 += nb_sched[1];
    }

    return 0;
}

void read_inputs(int *nb_codes, int *nb_stages, string *default_type_tiramisu, string *default_type_wrapper, double *assignment_prob, double *assignment_input_prob, double *conv_prob, double *same_padding_prob, int *nb_dims,
                 vector<int> *scheduling_commands, bool *all_schedules, int *nb_inputs, int *offset, vector <int> *tile_sizes, vector <int> *unrolling_factors, int *nb_rand_schedules){
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

    //nb_dims
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *nb_dims;

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

    //offset
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *offset;
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

    //same_padding_prob
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *same_padding_prob;
    }

    //tile_sizes
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        int number;
        while (info_stream >> number)
            (*tile_sizes).push_back(number);

    }

    //unrolling_factors
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        if (info.compare("all") == 0){
            for (int i = 1; i < MAX_UNROLL_FACTOR; ++i) {
               (*unrolling_factors).push_back(i*2);
                //(*unrolling_factors).push_back((int)pow(2.0, i));
            }
        }
        else {
            stringstream info_stream(info);
            int number;
            while (info_stream >> number)
                (*unrolling_factors).push_back(number);
        }

    }

    //scheduling_commands
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        string word;
        int number = 0;
        while (info_stream >> word){
            if (word.find("interchang") != string::npos) {
                number = INTERCHANGE;
            }
            else {
                if (word.find("til") != string::npos) {
                    number = TILING;
                }
                else {
                    if (word.find("unroll") != string::npos) {
                        number = UNROLL;
                    }
                    else {
                        cout << word + "command not supported" << endl;
                    }
                }
            }
            (*scheduling_commands).push_back(number);
        }

    }



    //schedules
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        string s;
        s = line.substr(pos1 + 1, pos2 - pos1 - 1);
        (s.find("all") != string::npos) ? *all_schedules = true : *all_schedules = false;
    }


    //nb_rand_schedules
    {
        getline(input_file, line);
        pos1 = line.find("\"", 0);
        pos2 = line.find("\"", pos1 + 1);
        info = line.substr(pos1 + 1, pos2 - pos1 - 1);
        stringstream info_stream(info);
        info_stream >> *nb_rand_schedules;
    }


}

