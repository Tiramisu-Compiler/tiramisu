#include <tiramisu/expr.h>
#include <tiramisu/core.h>

namespace tiramisu
{
    std::string tiramisu::global::nvcc_bin_dir;

tiramisu::expr& tiramisu::expr::operator=(tiramisu::expr const & e)
{
    this->_operator = e._operator;
    this->op = e.op;
    this->access_vector = e.access_vector;
    this->argument_vector = e.argument_vector;
    this->defined = e.defined;
    this->name = e.name;
    this->dtype = e.dtype;
    this->etype = e.etype;

    // Copy the integer value
    if (e.get_expr_type() == tiramisu::e_val)
    {
        if (e.get_data_type() == tiramisu::p_uint8)
        {
            this->uint8_value = e.get_uint8_value();
        }
        else if (e.get_data_type() == tiramisu::p_int8)
        {
            this->int8_value = e.get_int8_value();
        }
        else if (e.get_data_type() == tiramisu::p_uint16)
        {
            this->uint16_value = e.get_uint16_value();
        }
        else if (e.get_data_type() == tiramisu::p_int16)
        {
            this->int16_value = e.get_int16_value();
        }
        else if (e.get_data_type() == tiramisu::p_uint32)
        {
            this->uint32_value = e.get_uint32_value();
        }
        else if (e.get_data_type() == tiramisu::p_int32)
        {
            this->int32_value = e.get_int32_value();
        }
        else if (e.get_data_type() == tiramisu::p_uint64)
        {
            this->uint64_value = e.get_uint64_value();
        }
        else if (e.get_data_type() == tiramisu::p_int64)
        {
            this->int64_value = e.get_int64_value();
        }
        else if (e.get_data_type() == tiramisu::p_float32)
        {
            this->float32_value = e.get_float32_value();
        }
        else if (e.get_data_type() == tiramisu::p_float64)
        {
            this->float64_value = e.get_float64_value();
        }
    }
    return *this;
}

tiramisu::expr tiramisu::expr::substitute(std::vector<std::pair<var, expr>> substitutions) const
{
    for (auto &substitution: substitutions)
        if (this->is_equal(substitution.first))
            return substitution.second;


    return apply_to_operands([&substitutions](const expr& e){
        return e.substitute(substitutions);
    });
}

tiramisu::expr tiramisu::expr::substitute_access(std::string original, std::string substitute) const {
    expr && result = this->apply_to_operands([&original, &substitute](const expr& e){
        return e.substitute_access(original, substitute);
    });
    if (result.get_op_type() == o_access && result.name == original)
    {
        result.name = substitute;
    }
    return result;
}

tiramisu::var::var(std::string name, bool save)
{
    assert(!name.empty());

    auto declared = var::declared_vars.find(name);

    if (declared != var::declared_vars.end())
    {
        *this = declared->second;
    }
    else
    {
        this->name = name;
        this->etype = tiramisu::e_var;
        this->dtype = global::get_loop_iterator_data_type();
        this->defined = true;
        if (save)
        {
            var::declared_vars.insert(std::make_pair(name, *this));
            DEBUG(3, std::cout << "Saved variable " << this->name << " of type " << str_from_tiramisu_type_primitive(this->dtype));
        }
    }
}

tiramisu::var::var(tiramisu::primitive_t type, std::string name, bool save)
{
    assert(!name.empty());

    auto declared = var::declared_vars.find(name);


    if (declared != var::declared_vars.end())
    {
        assert(declared->second.dtype == type);
        *this = declared->second;
    }
    else
    {
        this->name = name;
        this->etype = tiramisu::e_var;
        this->dtype = type;
        this->defined = true;
        if (save)
        {
            var::declared_vars.insert(std::make_pair(name, *this));
            DEBUG(3, std::cout << "Saved variable " << this->name << " of type " << str_from_tiramisu_type_primitive(this->dtype));
        }
    }
}

tiramisu::expr tiramisu::expr::copy() const
{
    return (*this);
}


std::unordered_map<std::string, var> tiramisu::var::declared_vars;

expr cast(primitive_t tT, const expr & e) {
    if (e.get_data_type() == tT)
        return e;
    return expr{o_cast, tT, e};
}

expr tiramisu::expr::operator+(tiramisu::expr other) const {
    return tiramisu::expr{o_add, *this, other};
}

expr tiramisu::expr::operator-(tiramisu::expr other) const {
    return tiramisu::expr{o_sub, *this, other};
}

expr tiramisu::expr::operator*(tiramisu::expr other) const {
    return tiramisu::expr{o_mul, *this, other};
}

expr tiramisu::expr::operator/(tiramisu::expr other) const {
    return tiramisu::expr{o_div, *this, other};
}

expr tiramisu::expr::operator%(tiramisu::expr other) const {
    return tiramisu::expr{o_mod, *this, other};
}

expr tiramisu::expr::operator>>(tiramisu::expr other) const {
    return tiramisu::expr{o_right_shift, *this, other};
}

expr tiramisu::expr::operator<<(tiramisu::expr other) const {
    return tiramisu::expr{o_left_shift, *this, other};
}

expr memcpy(const buffer &from, const buffer &to) {
    return expr(o_memcpy, var(from.get_name()), var(to.get_name()));
}

expr allocate(const buffer &b)
{
    return expr{o_allocate, b.get_name()};
}

}
