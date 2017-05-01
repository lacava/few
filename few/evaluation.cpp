/* evaluation c++ code
Copyright 2016 William La Cava

This file is part of the FEW library.

The FEW library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The FEW library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
the FEW library. If not, see http://www.gnu.org/licenses/.
*/
#include <iostream>
#include "Eigen/Dense"
#include <Python.h>
using namespace Eigen;
using namespace std;

/*  DEFINE Custom Type Names to make code more readable
    ExtMat :  2-dim matrix/array externally defined (in Python)
*/
typedef Map<ArrayXXd> ExtMat;
typedef ArrayXXd Mat;
typedef ArrayXd Vec;


void evaluate(node n, ExtMat& features, vector<Vec> stack_float, vector<Vec> stack_bool)
{
  //evalute a program node on a given set of data.
  ExtMat F (features, n, d);

  vector<float> stack_float;
  vector<float> stack_bool;
  for (auto n: program){
    // evaluate program nodes on stack
    evaluate(n,F,stack_float,stack_bool);
  }
}

void out(vector<node> program, ExtMat& features, char otype){
  // evaluate program output.
}
