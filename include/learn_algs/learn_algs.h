#ifndef LEARN_ALGS_LIB_H
#define LEARN_ALGS_LIB_H

#include <cmath>

typedef double (*FZero_FCN)(double);

double findZero(double initial_point, FZero_FCN Jcst, FZero_FCN gradJ, double GD_GAIN, double GD_COST_TOL, double lambda, double MAX_GD_ITER );

double findZero(double initial_point, FZero_FCN Jcst, FZero_FCN gradJ, double GD_GAIN, double GD_COST_TOL, double lambda, double MAX_GD_ITER, bool& b_max_iter );

#endif