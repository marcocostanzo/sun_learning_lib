#include "learn_algs/learn_algs.h"


double findZero(double initial_point, FZero_FCN Jcst, FZero_FCN gradJ, double GD_GAIN, double GD_COST_TOL, double lambda, double MAX_GD_ITER, bool& b_max_iter ){

    double JJ = Jcst(initial_point);
	bool converge = false;
	int num_iter = 0;

    double x = initial_point;

	while( !converge && (num_iter<MAX_GD_ITER) && !std::isnan(x) ){

		//x = x - GD_GAIN * JJ * (1.0 / gradJ(x) );
        double grad = gradJ(x);
		x = x - GD_GAIN * JJ * (grad / ( grad*grad + lambda ) );
		JJ = Jcst(x);
		converge = (fabs(JJ) < GD_COST_TOL);				
		num_iter++;

	}

    if(num_iter >= MAX_GD_ITER)
        b_max_iter = true;
    else
        b_max_iter = false;

    return x;
}

double findZero(double initial_point, FZero_FCN Jcst, FZero_FCN gradJ, double GD_GAIN, double GD_COST_TOL, double lambda, double MAX_GD_ITER ){
    bool b_max_iter;
    return findZero(initial_point, Jcst, gradJ, GD_GAIN, GD_COST_TOL, lambda, MAX_GD_ITER, b_max_iter );
}