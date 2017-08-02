/*
Copyright 2017 William La Cava

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
// #include <vector>
#include  <random>
#include  <iterator>
#include "Eigen/Dense"
#include <Python.h>

using namespace Eigen;
using namespace std;

/*  DEFINE Custom Type Names to make code more readable
    ExtMat :  2-dim matrix/array externally defined (in Python)
*/
typedef Map<ArrayXXd> ExtMat;
typedef Map<ArrayXi> ExtVec;
// typedef ArrayXd Vec;

double median(const ArrayXd& v) {
	// instantiate a vector
  vector<double> x(v.size());
  x.assign(v.data(),v.data()+v.size());
  // middle element
  size_t n = x.size()/2;
  // sort nth element of array
  nth_element(x.begin(),x.begin()+n,x.end());
  // if evenly sized, return average of middle two elements
	if (x.size() % 2 == 0) {
    nth_element(x.begin(),x.begin()+n-1,x.end());
		return (x[n] + x[n-1]) / 2;
	}
  // otherwise return middle element
  else
		return x[n];
}
double mad(const ArrayXd& x) {
	// returns median absolute deviation (MAD)
	// get median of x
	double x_median = median(x);
  //calculate absolute deviation from median
	ArrayXd dev(x.size());
  for (int i =0; i < x.size(); ++i)
		dev(i) = abs(x(i) - x_median);
  // return median of the absolute deviation
	return median(dev);
}
// random functions


template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    uniform_int_distribution<> dis(0, distance(start, end) - 1);
    advance(start, dis(g));
    return start;
}

// template<typename Iter>
// Iter select_randomly(Iter start, Iter end) {
//     static random_device rd;
//     static mt19937 gen(rd());
//     return select_randomly(start, end, gen);
// }
// random number generator
static random_device rd;
static mt19937 gen(rd());
//extern "C"
void epsilon_lexicase(const ExtMat & F, int n, int d,
  int num_selections,  ExtVec& locs, bool lex_size, ExtVec& sizes)
{
  // training cases
  // ExtMat T (F, n, d);
  // cout << "F size: " << F.rows() << "x" << F.cols() << "\n";
  // cout << "locs size: " << locs.size() << "\n";
  // parent locations
  // ExtVec L (locs, num_selections);
  // get epsilon via median absolute deviations
  ArrayXd epsilon(d);
  //cout << "calculating epsilon\n";
  //for columns of T, calculate epsilon
  for (int i = 0; i<epsilon.size(); ++i)
    epsilon(i) = mad(F.col(i));

  vector<int> ind_locs;
  if(lex_size){
    //randomly select a size from sizes
    int max_index = sizes.size();
    int random_index = rand() % max_index;

    // individual locations
    int j=0;
    for(int i=0;i<max_index;i++){
      if(sizes[i]<=sizes[random_index])
        ind_locs.push_back(i);
    }
    
  }
  else{
    // individual locations
    ind_locs.resize(n);
    iota(ind_locs.begin(),ind_locs.end(),0);
  }

  // temporary winner pool
  vector<int> winner;
  for (int i = 0; i<num_selections; ++i){
    //cout << "selection " << i << "\n";
    // perform selection
    // set candidate locations to those not yet picked
    vector<int> can_locs = ind_locs;
    // set cases
    vector<int> cases(d);
    iota(cases.begin(),cases.end(),0);
    // shuffle cases
    random_shuffle(cases.begin(),cases.end());
    //main loop
    while(can_locs.size()>1 && cases.size() > 0){
      // winner pool
      winner.resize(0);
      // minimum error on case
      double minfit;
      for (int j = 0; j<can_locs.size(); ++j){
        if (j==0 || F(can_locs[j],cases.back())<minfit )
          minfit = F(can_locs[j],cases.back());
        }
      //cout << "minfit: " << minfit << "\n";
      //cout << "epsilon: " << epsilon(cases.back()) <<"\n";
      // for each individual
      // for (int j = 0; j < can_locs.size(); ++j){
      for (auto cl : can_locs){
        // determine whether it passes the case
        //cout << "ind " << cl << "error on case " << cases.back() << ": "
      //  << F(cl,cases.back()) << "\n";
        if (F(cl,cases.back()) <= minfit + epsilon(cases.back())){
          winner.push_back(cl);
          //cout << "<- pass\n";
        }
      }
      // remove top case
      cases.pop_back();
      // reduce pool
      can_locs = winner;

      assert(can_locs.size()!=0);
    }

    // pick a winner from can_locs
    locs(i) = *select_randomly(can_locs.begin(),can_locs.end(),gen);
    // remove the winner from ind_locs
    for (auto l = ind_locs.begin(); l!=ind_locs.end();){
      if (*l == locs(i))
        l = ind_locs.erase(l);
      else
        ++l;
    }
  }
  //cout << "locs pointer: " << locs.data() << "\n locs: " << locs << "\n";
  //cout << "locs: "<< locs << "\n";

}
