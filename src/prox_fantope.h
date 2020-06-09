#ifndef GRADFPS_PROX_FANTOPE_H
#define GRADFPS_PROX_FANTOPE_H

#include "common.h"

// min  -tr(SX) + s1 * max(0, eigmax(X)-1) + s2 * |tr(X)-d| + (0.5/alpha) * ||X - U||_F^2
// s.t. X is p.s.d.
//
// Equivalent to
// min  alpha*s1 * max(0, eigmax(X)-1) + alpha*s2 * |tr(X)-d| + 0.5 * ||X - U - alpha*S||_F^2
// s.t. X is p.s.d.
//
// For simplicity we solve
// min  l1 * max(0, eigmax(X)-1) + l2 * |tr(X)-d| + 0.5 * ||X - A||_F^2
// s.t. X is p.s.d.
int prox_fantope_impl(
    RefConstMat A, double l1, double l2, int d, int inc, int maxiter, RefMat res,
    double eps = 1e-3, int verbose = 0
);

// The old implementation, in fact a hard projection
//
// min  -tr(AX) + 0.5 * ||X||_F^2
// s.t. X in Fantope
int prox_fantope_hard_impl(
    RefConstMat A, int d, int inc, int maxiter, RefMat res, double& dsum,
    double eps = 1e-3, int verbose = 0
);


#endif  // GRADFPS_PROX_FANTOPE_H
