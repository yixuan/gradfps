#ifndef GRADFPS_PROX_LP_H
#define GRADFPS_PROX_LP_H

#include "common.h"

// Proximal operator of 0.5 * ||x||_p^2
void prox_lp_impl(RefConstVec vv, double p, double alpha, RefVec res,
                  double eps = 1e-6, int maxiter = 100, int verbose = 0);

// Proximal operator of 0.5 * ||x||_p^2, applied to a symmetric matrix
void prox_lp_mat_impl(RefConstMat vv, double p, double alpha, RefMat res,
                      double eps = 1e-6, int maxiter = 100, int verbose = 0);


#endif  // GRADFPS_PROX_LP_H
