#include <tiramisu/expr.h>

namespace tiramisu
{

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
        this->dtype = global::get_loop_iterator_data_type();
        this->defined = true;
        if (save)
            var::declared_vars.insert(std::make_pair(name, *this));
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
            var::declared_vars.insert(std::make_pair(name, *this));
    }
}

std::unordered_map<std::string, var> tiramisu::var::declared_vars;

}
