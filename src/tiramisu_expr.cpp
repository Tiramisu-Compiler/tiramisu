#include <tiramisu/expr.h>

namespace tiramisu
{

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

tiramisu::expr tiramisu::expr::substitute(std::vector<std::pair<var, expr>> substitutions)
{
    for (auto &substitution: substitutions)
        if (this->is_equal(substitution.first))
            return substitution.second;

    expr new_expr = this->copy();


    if (new_expr.etype == e_op) {
        for (int i = 0; i < new_expr.op.size(); i++) {
            new_expr.op[i] = new_expr.op[i].substitute(substitutions);
        }
    }

    return new_expr;
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
        this->dtype = global::get_loop_iterator_default_data_type();
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
    tiramisu::expr *e = new tiramisu::expr();
    *e = *this; // use copy assignment

    return (*e);
}


std::unordered_map<std::string, var> tiramisu::var::declared_vars;

expr const & caster<expr>::cast(expr const & val, primitive_t dtype)
{
    assert(val.get_data_type() == dtype);
    return val;
}

var const & caster<var>::cast(var const & val, primitive_t dtype)
{
    assert(val.get_data_type() == dtype);
    return val;
}
}
