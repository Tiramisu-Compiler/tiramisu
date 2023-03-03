#ifndef _TIRAMISU_AUTO_SCHEDULER_UTILS_
#define _TIRAMISU_AUTO_SCHEDULER_UTILS_

#include <vector>
#include <regex>
#include <tiramisu/auto_scheduler/optimization_info.h>
namespace tiramisu::auto_scheduler
{
// Return true if the string contains a number only
inline bool check_if_number(const std::string s)
{
    char* p;
    strtol(s.c_str(), &p, 10);
    if (*p) {
        return false;
    }
    else {
        return true;
    }
}
/**
 * Return true if an iterator having extent = it_extent can
 * be split perfectly by a factor = split_fact.
 */
inline bool can_split_iterator(std::string up_bound, std::string low_bound, int split_fact, optimization_type opt=optimization_type::TILING)
{
    if(check_if_number(up_bound) && check_if_number(low_bound)){
        int it_extent = stoi(up_bound) - stoi(low_bound)+1;
        return it_extent > split_fact && it_extent % split_fact == 0;
    }else{
        // in the case where one of the bounds is not a constant, we can't apply unrolling.
        if (opt == optimization_type::UNROLLING)
            return false;
    }
    return true;
    
}
/**
 * Returns true if the extent is bigger than split factor,
 * (.i.e more than one iteration would be produced after splitting) 
*/
inline bool can_split_iterator_sup(std::string up_bound, std::string low_bound,  int split_fact)
{
    if(check_if_number(up_bound) && check_if_number(low_bound)){
        int it_extent = stoi(up_bound) - stoi(low_bound)+1;
        return it_extent > split_fact;
    }else{
        return true;
    }
    
}

/**
 * Returns the minimal value of a vector of measurements
 */
inline float min_eval(std::vector<float> measurements)
{
    return *std::min_element(measurements.begin(), measurements.end());
}

/**
 * Formats a vector of measurements into a string
 */
inline std::string measurements_to_str(std::vector<float> measurements)
{
    std::string str_array="[";
    for (float measure: measurements)
        str_array+= " " + std::to_string(measure) + ",";
    str_array.pop_back(); // remove the last ","
    str_array+= "]";

    return str_array;
}

/**
 * return an environment variable value if declared, otherwise returns empty string
 * This function is just for reducing code clutter
 */
inline const char* read_env_var(const char* env_var_name){
    char* value = std::getenv(env_var_name);
    if (value!=nullptr)
        return value;
    else
        return "";
}

}

#endif
