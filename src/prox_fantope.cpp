#include "prox_fantope.h"

// min  -tr(AX) + (0.5 / alpha) * ||X - B||_F^2
// s.t. X in Fantope

// [[Rcpp::export]]
Rcpp::NumericMatrix prox_fantope(MapMat A, MapMat B, double alpha, int d,
                                 double eps = 1e-5, int inc = 1, int max_try = 10,
                                 int verbose = 0)
{
    const int n = A.rows();
    if(A.cols() != n)
        Rcpp::stop("A is not square");
    if(B.rows() != n || B.cols() != n)
        Rcpp::stop("dimensions of A and B do not change");

    MatrixXd mat = B + alpha * A;
    MapConstMat matm(mat.data(), n, n);

    Rcpp::NumericMatrix res(n, n);
    MapMat resm(res.begin(), n, n);
    double dsum;

    prox_fantope_impl(matm, d, inc, max_try, resm, dsum, eps, verbose);

    return res;
}
