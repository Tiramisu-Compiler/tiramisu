#ifndef _H_TIRAMISU_COMPLEX_
#define _H_TIRAMISU_COMPLEX_

#include <string>
#include <vector>
#include "util.h" // str_fmt

namespace tiramisu {
  class expr;
  class computation;
}

/**
  * Wrapper around tiramisu::tiramisu::expr
  * with one caveat: undefined complex_expr is assumed to be zero
  */
class complex_expr {
  tiramisu::expr real, imag;

public:
  complex_expr() : real(double(0)), imag(double(0)) {}

  complex_expr(tiramisu::expr r, tiramisu::expr i) : real(r), imag(i) {}

  bool is_zero() const {
    if (!real.is_constant() || 
        !imag.is_constant())
      return false;
    return real.get_float64_value() == 0 && imag.get_float64_value() == 0;
  }


  // FIXME: remove 
  complex_expr(std::pair<tiramisu::expr, tiramisu::expr> r_and_i)
  {
    std::tie(real, imag) = r_and_i;
  }

  operator std::pair<tiramisu::expr, tiramisu::expr>() const
  {
    return {real, imag};
  }

  complex_expr operator+(const complex_expr &other) const
  {
    if (is_zero())
      return other;

    return complex_expr(real + other.real, imag + other.imag);
  }

  complex_expr &operator+=(const complex_expr &other)
  {
    *this = *this + other;
    return *this;
  }

  complex_expr operator-(const complex_expr &other) const
  {
    if (is_zero())
      return other * -1;

    return complex_expr(real - other.real, imag - other.imag);
  }

  complex_expr operator*(const complex_expr &other) const
  {
    if (is_zero())
      return complex_expr();

    tiramisu::expr res_real = real * other.real - imag * other.imag;
    tiramisu::expr res_imag = real * other.imag + imag * other.real;
    return complex_expr(res_real, res_imag);
  }

  complex_expr operator*(tiramisu::expr a) const
  {
    if (is_zero())
      return complex_expr();

    return complex_expr(real * a, imag * a);
  }

  tiramisu::expr get_real() const
  {
    return real;
  }

  tiramisu::expr get_imag() const
  {
    return imag;
  }
};

/**
  * Wrapper around tiramisu::computation
  * The main purpose of this class is allow you to access a pair of (real, imag) tensors
  *   as if they are one single complex tensor.
  */
// NOTE: this leaks memory, but it's fine for our use
class complex_computation {
  tiramisu::computation *real, *imag;

public:
  complex_computation(tiramisu::computation *r, tiramisu::computation *i) : real(r), imag(i) {}

  /**
    * \overload
    **/
  complex_computation(std::string name, std::vector<tiramisu::var> iterators, complex_expr def)
	  : complex_computation(name, iterators, tiramisu::expr(), def) {}

  /**
    * Wrapper around tiramisu::computation's counterpart.
    * This creates two computations -- real and imag.
    */
  complex_computation(
      std::string name,
      std::vector<tiramisu::var> iterators,
      tiramisu::expr predicate,
      complex_expr def)
  {
    real = new tiramisu::computation(str_fmt("%s_r", name.c_str()), iterators, predicate, def.get_real());
    imag = new tiramisu::computation(str_fmt("%s_i", name.c_str()), iterators, predicate, def.get_imag());
  }

  /**
    * Convert a pair of computation to a complex computation
    */
  complex_computation(std::pair<tiramisu::computation *, tiramisu::computation*> &r_and_i)
  {
    std::tie(real, imag) = r_and_i;
  }

  /**
    * This complex computation to a pair of computation
    */
  operator std::pair<tiramisu::computation *, tiramisu::computation *>() 
  {
    return {real, imag};
  }
  
  /**
    * Index into the complex computation
    */
  template<typename ... Idxs>
  complex_expr operator()(Idxs ... idxs)
  {
    return complex_expr((*real)(idxs ...), (*imag)(idxs ...));
  }

  /**
    * Wrapper around tiramisu::computation::add_predicate
    */
  void add_predicate(tiramisu::expr pred)
  {
    real->add_predicate(pred);
    imag->add_predicate(pred);
  }

  tiramisu::computation *get_real()
  {
    return real;
  }

  tiramisu::computation *get_imag()
  {
    return imag;
  }
};

#endif
