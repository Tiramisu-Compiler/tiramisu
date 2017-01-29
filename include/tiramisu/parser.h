/*
 * parser.h
 *
 *  Created on: Sep 13, 2016
 *      Author: Riyadh.
 */

#ifndef INCLUDE_COLI_PARSER_H_
#define INCLUDE_COLI_PARSER_H_


#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/space.h>

#include <map>
#include <string.h>
#include <stdint.h>

#include <Halide.h>
#include <tiramisu/debug.h>
#include <tiramisu/expr.h>
#include <tiramisu/type.h>

namespace tiramisu
{

namespace parser
{

/**
  * A class to hold parsed tokens of an isl_space.
  * This class is useful in parsing isl_space objects.
  */
class space
{
private:
    std::vector<std::string> constraints;

public:
    std::vector<std::string> dimensions;

    /**
      * Parse an isl_space.
      * The isl_space is a string in the format of ISL.
      */
    space(std::string isl_space_str)
    {
        assert(isl_space_str.empty() == false);
        this->parse(isl_space_str);
    };

    space() {};

    /**
      * Return a string that represents the parsed string.
      */
    std::string get_str() const
    {
        std::string result;

        for (int i=0; i<dimensions.size(); i++)
        {
            if (i != 0)
                result = result + ",";
            result = result + dimensions.at(i);
        }

        return result;
    };

    void add_constraint(std::string cst)
    {
        constraints.push_back(cst);
    }

    const std::vector<std::string> &get_constraints() const
    {
        return constraints;
    }

    void replace(std::string in, std::string out1, std::string out2)
    {
        std::vector<std::string> new_dimensions;

        for (const auto &dim : dimensions)
        {
            if (dim == in)
            {
                new_dimensions.push_back(out1);
                new_dimensions.push_back(out2);
            }
            else
                new_dimensions.push_back(dim);
        }

        dimensions = new_dimensions;
    }

    void parse(std::string space);
    bool empty() const {return dimensions.empty();};
};


/**
 * A class to hold parsed tokens of isl_constraints.
 */
class constraint
{
public:
    std::vector<std::string> constraints;
    constraint() { };

    void parse(std::string str);

    void add(std::string str)
    {
        assert(str.empty() == false);
        constraints.push_back(str);
    }

    void add_constraints(std::vector<std::string> vec)
    {
        for (const auto &cst : vec)
        {
            constraints.push_back(cst);
        }
    }

    std::string get_str() const
    {
        std::string result;

        for (int i=0; i<constraints.size(); i++)
        {
            if (i != 0)
                result = result + " and ";
            result = result + constraints.at(i);
        }

        return result;
    };

    bool empty() const {return constraints.empty();};
};


/**
  * A class to hold parsed tokens of isl_maps.
  */
class map
{
public:
    tiramisu::parser::space parameters;
    std::string domain_name;
    std::string range_name;
    tiramisu::parser::space domain;
    tiramisu::parser::space range;
    tiramisu::parser::constraint constraints;

    map(std::string map_str)
    {
        int map_begin =  map_str.find("{")+1;
        int map_end   =  map_str.find("}")-1;

        assert(map_begin != std::string::npos);
        assert(map_end != std::string::npos);

        int domain_space_begin = map_str.find("[", map_begin)+1;
        int domain_space_begin_pre_bracket = map_str.find("[", map_begin)-1;
        int domain_space_end   = map_str.find("]", map_begin)-1;

        assert(domain_space_begin != std::string::npos);
        assert(domain_space_end != std::string::npos);

        domain_name = map_str.substr(
            map_begin, domain_space_begin_pre_bracket-map_begin+1);

        std::string domain_space_str =
            map_str.substr(domain_space_begin,
                           domain_space_end-domain_space_begin+1);

        domain.parse(domain_space_str);

        int pos_arrow = map_str.find("->", domain_space_end);
        int first_char_after_arrow = pos_arrow + 2;

        assert(pos_arrow != std::string::npos);

        int range_space_begin = map_str.find("[", first_char_after_arrow)+1;
        int range_space_begin_pre_bracket = map_str.find("[", first_char_after_arrow)-1;
        int range_space_end = map_str.find("]", first_char_after_arrow)-1;

        assert(range_space_begin != std::string::npos);
        assert(range_space_end != std::string::npos);

        range_name = map_str.substr(
            first_char_after_arrow,
            range_space_begin_pre_bracket-first_char_after_arrow+1);
        std::string range_space_str = map_str.substr(
            range_space_begin, range_space_end-range_space_begin+1);
        range.parse(range_space_str);
        constraints.add_constraints(range.get_constraints());

        int column_pos = map_str.find(":")+1;

        if (column_pos != std::string::npos)
        {
            std::string constraints_str = map_str.substr(
                column_pos, map_end-column_pos+1);
            constraints.parse(constraints_str);
        }

        DEBUG(2, tiramisu::str_dump("Parsing the map : " + map_str + "\n"));
        DEBUG(2, tiramisu::str_dump("The parsed map  : " + this->get_str() + "\n"));
    };

    std::string get_str() const
    {
        std::string result;

        result = "{" + domain_name + "[" + domain.get_str() + "] ->" +
                 range_name + "[" + range.get_str() + "]";

        if (constraints.empty() == false)
            result = result + " : " + constraints.get_str();

        result = result + " }";

        return result;
    };

    isl_map *get_isl_map(isl_ctx *ctx) const
    {
        return isl_map_read_from_str(ctx, this->get_str().c_str());
    };
};

}
}

#endif /* INCLUDE_COLI_PARSER_H_ */
