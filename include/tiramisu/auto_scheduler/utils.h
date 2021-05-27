#ifndef _TIRAMISU_AUTO_SCHEDULER_UTILS_
#define _TIRAMISU_AUTO_SCHEDULER_UTILS_

#include <vector>

namespace tiramisu::auto_scheduler
{

/**
 * Return true if an iterator having extent = it_extent can
 * be split perfectly by a factor = split_fact.
 */
inline bool can_split_iterator(int it_extent, int split_fact)
{
    return it_extent > split_fact && it_extent % split_fact == 0;
}

/**
 * Returns true if the extent is bigger than split factor,
 * (.i.e more than one iteration would be produced after splitting) 
*/
inline bool can_split_iterator_sup(int it_extent, int split_fact)
{
    return it_extent > split_fact;
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
