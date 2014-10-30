/* 
  Cuda Templates.

  Copyright (C) 2008 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CUDA_VECTOR_SSIZE1_H
#define CUDA_VECTOR_SSIZE1_H


// #include <cudatemplates/ssize.hpp>


namespace Cuda {

/**
   Specialization of size template for 1D case.
*/
template <>
class SSize<1>: public SSizeBase<1>
{
public:
  /**
     Default constructor.
  */
  SSize() {}

  /**
     Copy constructor with type conversion.
     @param v vector to be copied
  */
  template <class T>
  SSize(const VectorBase<T, 1> &v)
  {
    for(unsigned i = 1; i--;)
      (*this)[i] = v[i];
  }

  /**
     Constructor.
     @param x0 vector component(s)
  */
  SSize(ssize_t x0) { (*this)[0] = x0; }

  // CUDA_VECTOR_OPS(SSize<1>, size_t);
};

}  // namespace Cuda


#endif
