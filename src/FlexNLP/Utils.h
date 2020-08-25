#ifndef __UTILS__
#define __UTILS__


#include <vector>
#include <iostream>
#include <string>
#include <math.h>
#include <iomanip>
#include <stdexcept>
#include <random>

// Get Random float Vector with zer0
template < typename T >
inline std::vector<T> GetVec(int size) {
  std::vector<T> vec(size);
  return vec;
}

// Get Random float Vector with normal distribution
inline std::vector<float> GetRandVec(int size,
                                           float mean=0.0, float std=0.3) {
  std::default_random_engine generator (std::random_device{}());
  std::normal_distribution<float> distribution(mean,std);
  std::vector<float> vec(size);
  for (int i = 0; i < size; i++) {
    float tmp = distribution(generator);
    vec[i] = tmp;
  }
  return vec;
}

// Get Random float Matrix with zer0
template < typename T >
inline std::vector<std::vector<T>> GetMat(int rows, int cols) {
  std::vector<std::vector<T>> mat(rows, std::vector<T>(cols));
  return mat;
}

// Get Random float Matrix with normal distribution
inline std::vector<std::vector<float>> GetRandMat(int rows, int cols,
                                           float mean=0.0, float std=0.3) {
  std::default_random_engine generator (std::random_device{}());
  std::normal_distribution<float> distribution(mean,std);
  std::vector<std::vector<float>> mat(rows, std::vector<float>(cols));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float tmp = distribution(generator);
      mat[i][j] = tmp;
    }
  }
  return mat;
}


template<typename Scalar>
inline void PrintVec(const std::vector<Scalar>& data)
{
  for (unsigned int i = 0; i < data.size(); i++)
    std::cout << data[i] << "\t";
  std::cout << std::endl;
}


template<typename Scalar>
inline void PrintMat(const std::vector<std::vector<Scalar>>& data)
{
  for (unsigned int i = 0; i < data.size(); i++)
    PrintVec(data[i]);
  std::cout << std::endl;
}


template<typename Scalar>
inline void PrintMatShape(const std::vector<std::vector<Scalar>>& data) {
  std::cout << "[" << data.size() << ", " << data[0].size() << "]" << std::endl;
  const unsigned int ncols = data[0].size();
  for (unsigned int i = 0; i < data.size(); i++) {
    if( ncols != data[i].size()) throw std::domain_error( "bad #cols" );
  }
}

template<typename Scalar>
inline void CheckMatShape(const std::vector<std::vector<Scalar>>& data) {
  const unsigned int ncols = data[0].size();
  for (unsigned int i = 0; i < data.size(); i++) {
    if( ncols != data[i].size()) throw std::domain_error( "bad #cols" );
  }
}


template<typename Scalar>
inline std::vector<Scalar> MatVecMul(const std::vector<std::vector<Scalar>>& M, const std::vector<Scalar>& v) {
  std::vector<Scalar> out;
  out.resize(M.size(), 0.0);

  for (unsigned int i = 0; i < M.size(); i++) {
    for (unsigned int j = 0; j < v.size(); j++) {
      if( M[i].size() != v.size()) throw std::domain_error( "bad M[i].size() != v.size()" );
      out[i] += M[i][j] * v[j];
    }
  }

  return out;
}



template<typename Scalar>
inline std::vector<Scalar> VecAdd(const std::vector<Scalar>& v1, const std::vector<Scalar>& v2){
  if( v1.size() != v2.size()) throw std::domain_error( "bad add v1.size() != v2.size()" );
  std::vector<Scalar> out;
  out.resize(v1.size(), 0.0);
  for (unsigned int i = 0; i < v1.size(); i++) {
    out[i] = v1[i] + v2[i];
  }

  return out;
}

template<typename Scalar>
inline std::vector<Scalar> VecMul(const std::vector<Scalar>& v1, const std::vector<Scalar>& v2){
  if( v1.size() != v2.size()) throw std::domain_error( "bad mul v1.size() != v2.size()" );
  std::vector<Scalar> out;
  out.resize(v1.size(), 0.0);
  for (unsigned int i = 0; i < v1.size(); i++) {
    out[i] = v1[i] * v2[i];
  }

  return out;
}

template<typename Scalar>
inline std::vector<Scalar> VecTanh(const std::vector<Scalar>& v){
  std::vector<Scalar> out;
  out.resize(v.size(), 0.0);
  for (unsigned int i = 0; i < v.size(); i++) {
    out[i] = tanh(v[i]);
  }

  return out;
}

template<typename Scalar>
inline std::vector<Scalar> VecSigm(const std::vector<Scalar>& v){
  std::vector<Scalar> out;
  out.resize(v.size(), 0.0);
  for (unsigned int i = 0; i < v.size(); i++) {
    out[i] = 1 / (1 + exp(-v[i]));
  }

  return out;
}

#endif