
//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// BSD 3-Clause License
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER
// Changelog
//
// Version 0.1
// - Initial version.
//
/////////////////////////////////////////////////////////////////////////

#ifndef YAML_ELEMENT_H
#define YAML_ELEMENT_H
#include <string>
#include <vector>
//! The Mantevo YAML_Element class for registering key-value pairs of performance data

/*!
  Mantevo mini-applications generate a collection of performance data for each run of the executable.  YAML_Element, and
  the related YAML_Doc class, provide a uniform facility for gathering and reporting this data using the YAML text format.
*/
class YAML_Element {
  public:

  //! Default constructor.
  YAML_Element (){key="";value="";}
  //! Construct with known key-value pair
  YAML_Element (const std::string& key_arg, const std::string& value_arg);
  //! Destructor
  ~YAML_Element ();
  //! Key accessor method
  std::string getKey(){return key;}
  //! Add a child element to an element list associated with this element, value of type double
  YAML_Element* add(const std::string& key_arg, double value_arg);
  //! Add a child element to an element list associated with this element, value of type int
  YAML_Element* add(const std::string& key_arg, int value_arg);
#ifndef MINIFE_NO_LONG_LONG
  //! Add a child element to an element list associated with this element, value of type long long
  YAML_Element* add(const std::string& key_arg, long long value_arg);
#endif
  //! Add a child element to an element list associated with this element, value of type size_t
  YAML_Element* add(const std::string& key_arg, size_t value_arg);
  //! Add a child element to an element list associated with this element, value of type string
  YAML_Element* add(const std::string& key_arg, const std::string& value_arg);
  //! get the element in the list with the given key
  YAML_Element* get(const std::string& key_arg);
  std::string printYAML(std::string space);
  
protected:
  std::string key;
  std::string value;
  std::vector<YAML_Element*> children;

private:
  std::string convert_double_to_string(double value_arg);
  std::string convert_int_to_string(int value_arg);
#ifndef MINIFE_NO_LONG_LONG
  std::string convert_long_long_to_string(long long value_arg);
#endif
  std::string convert_size_t_to_string(size_t value_arg);
};
#endif /* YAML_ELEMENT_H */
