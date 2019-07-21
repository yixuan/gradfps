#include "prox_lp.h"

using Rcpp::NumericVector;
using Rcpp::NumericMatrix;

//' Proximal operator of squared Lp norm
//'
//' This function solves the optimization problem
//' \deqn{\min\quad\frac{1}{2}||x||_p^2 + \frac{1}{2\alpha}||x - v||^2}{min  0.5 * ||x||_p^2 + (0.5 / \alpha) * ||x - v||^2}
//'
//' @param v       A numeric vector.
//' @param V       A symmetric matrix.
//' @param p       Norm parameter.
//' @param alpha   Proximal parameter.
//' @param eps     Precision of the result.
//' @param maxiter Maximum number of iterations.
//' @param verbose Level of verbosity.
//'
// [[Rcpp::export]]
NumericVector prox_lp(NumericVector v, double p, double alpha,
                      double eps = 1e-6, int maxiter = 100, int verbose = 0)
{
    const int n = v.length();
    MapConstVec vv(v.begin(), n);

    NumericVector res(n);
    MapVec x(res.begin(), n);

    prox_lp_impl(vv, p, alpha, x, eps, maxiter, verbose);

    return res;
}

//' @rdname prox_lp
//'
// [[Rcpp::export]]
NumericMatrix prox_lp_mat(NumericMatrix V, double p, double alpha,
                          double eps = 1e-6, int maxiter = 100, int verbose = 0)
{
    const int n = V.nrow();
    MapConstMat VV(V.begin(), n, n);

    NumericMatrix res(n, n);
    MapMat X(res.begin(), n, n);

    prox_lp_mat_impl(VV, p, alpha, X, eps, maxiter, verbose);

    return res;
}
