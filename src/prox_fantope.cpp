#include "prox_fantope.h"

using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::List;

// [[Rcpp::export]]
NumericMatrix prox_fantope(MapMat v, double alpha, MapMat S, int d, int inc, int max_try)
{
    MatrixXd mat = v + alpha * S;
    MapConstMat matm(mat.data(), mat.rows(), mat.cols());

    NumericMatrix res(v.rows(), v.cols());
    MapMat resm(res.begin(), res.nrow(), res.ncol());

    prox_fantope_impl(matm, d, inc, max_try, resm);

    return res;
}

